"""Regression suite entry point.

End-to-end validation: load Excel truth, run Python Calc engine, compare outputs.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from excel_replica.model.calc_engine import CalcConfig, CalcResults, run_calc


EXCEL_FILE = Path(__file__).resolve().parents[2] / "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"


def _load_excel_calc(file_path: Path) -> pd.DataFrame:
    """Load the Calc sheet from Excel."""
    return pd.read_excel(file_path, sheet_name="Calc", engine="openpyxl")


def _load_excel_measures(file_path: Path) -> pd.DataFrame:
    """Load the Measures sheet from Excel."""
    return pd.read_excel(file_path, sheet_name="Measures", engine="openpyxl")


def _find_column(columns: list, candidates: list) -> str | None:
    """Find first matching column name."""
    for col in columns:
        col_lower = str(col).lower().replace(" ", "").replace("_", "")
        for cand in candidates:
            if cand.lower().replace(" ", "").replace("_", "") in col_lower:
                return col
    return None


def _load_bess_config(file_path: Path) -> CalcConfig:
    """Load BESS parameters from Assumption sheet."""
    df = pd.read_excel(file_path, sheet_name="Assumption", engine="openpyxl", header=None)

    def get_value(row_idx: int, col_idx: int = 4) -> float:
        val = df.iloc[row_idx, col_idx]
        if pd.notna(val) and isinstance(val, (int, float)):
            return float(val)
        # Try adjacent columns
        for c in [5, 3]:
            val = df.iloc[row_idx, c]
            if pd.notna(val) and isinstance(val, (int, float)):
                return float(val)
        return 0.0

    # Row indices from Assumption sheet scan
    total_capacity_kwh = get_value(24)  # Total BESS Storage Capacity
    power_kw = get_value(25)            # Total BESS Power Output
    dod = get_value(26)                 # Depth of Discharge (0.85 = 85%)
    efficiency = get_value(27)          # HalfCycle Efficiency
    min_soc = get_value(38)             # Min Reserve SOC

    # Usable capacity = total * DoD
    usable_capacity_kwh = total_capacity_kwh * dod

    # Efficiency is stored as decimal (0.95), use as-is for charge/discharge
    rt_efficiency = efficiency

    # Tariff rates (VND/kWh) - convert to USD/kWh
    exchange_rate = df.iloc[8, 10] if pd.notna(df.iloc[8, 10]) else 26000.0
    ca_normal_vnd = df.iloc[13, 16] if pd.notna(df.iloc[13, 16]) else 1253.0
    ca_peak_vnd = df.iloc[14, 16] if pd.notna(df.iloc[14, 16]) else 2162.0
    ca_offpeak_vnd = df.iloc[15, 16] if pd.notna(df.iloc[15, 16]) else 843.0

    ca_normal = float(ca_normal_vnd) / float(exchange_rate)
    ca_peak = float(ca_peak_vnd) / float(exchange_rate)
    ca_offpeak = float(ca_offpeak_vnd) / float(exchange_rate)

    print(f"BESS Config: usable_capacity={usable_capacity_kwh} kWh (total={total_capacity_kwh}, DoD={dod}), power={power_kw} kW, eff={rt_efficiency}, min_soc={min_soc} kWh")
    print(f"Tariffs (USD/kWh): peak={ca_peak:.4f}, normal={ca_normal:.4f}, offpeak={ca_offpeak:.4f}")

    return CalcConfig(
        step_hours=1.0,
        bess_capacity_kwh=usable_capacity_kwh,
        bess_power_kw=power_kw,
        bess_efficiency=rt_efficiency,
        min_soc_kwh=min_soc,
        ca_peak=ca_peak,
        ca_normal=ca_normal,
        ca_offpeak=ca_offpeak,
    )


def _extract_calc_truth(calc_df: pd.DataFrame) -> Dict[str, float]:
    """Extract aggregated truth values from Calc sheet."""
    cols = calc_df.columns.tolist()
    truth = {}

    solar_col = _find_column(cols, ["SolarGen_kW", "SolarGen", "Solar Generation"])
    if solar_col:
        truth["solar_gen_mwh"] = calc_df[solar_col].astype(float).sum() / 1000

    discharge_col = _find_column(cols, ["DischargeEnergy_kWh", "Discharge Energy"])
    if discharge_col:
        truth["discharge_mwh"] = calc_df[discharge_col].astype(float).sum() / 1000

    surplus_col = _find_column(cols, ["PowerSurplus_kW", "Power Surplus"])
    if surplus_col:
        truth["power_surplus_mwh"] = calc_df[surplus_col].astype(float).sum() / 1000

    return truth


def _extract_inputs(calc_df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract input arrays from Calc sheet."""
    cols = calc_df.columns.tolist()

    datetime_col = _find_column(cols, ["DateTime", "Date", "Time"])
    datetime_series = pd.to_datetime(calc_df[datetime_col]) if datetime_col else pd.Series(range(len(calc_df)))

    solar_col = _find_column(cols, ["SolarGen_kW", "SolarGen"])
    solar_kw = calc_df[solar_col].astype(float).to_numpy() if solar_col else np.zeros(len(calc_df))

    load_col = _find_column(cols, ["Load_kW", "Load", "Demand"])
    load_kw = calc_df[load_col].astype(float).to_numpy() if load_col else np.zeros(len(calc_df))

    period_col = _find_column(cols, ["TimePeriodFlag", "Time Period"])
    period_flags = calc_df[period_col].astype(str).str.upper().to_numpy() if period_col else np.full(len(calc_df), "N")

    # Use DischargeConditionFlag if available (more restrictive than AllowDischarge)
    discharge_cond_col = _find_column(cols, ["DischargeConditionFlag", "Discharge Condition"])
    if discharge_cond_col:
        allow_discharge = calc_df[discharge_cond_col].astype(bool).to_numpy()
    else:
        allow_col = _find_column(cols, ["AllowDischarge", "Allow Discharge"])
        if allow_col:
            allow_discharge = calc_df[allow_col].astype(bool).to_numpy()
        else:
            allow_discharge = np.isin(period_flags, ["P", "N"])

    return datetime_series, solar_kw, load_kw, period_flags, allow_discharge


def _pct_error(model: float, truth: float) -> float:
    """Calculate percent error."""
    if truth == 0:
        return 0.0 if model == 0 else 100.0
    return abs(model - truth) / abs(truth) * 100


def run_regression_suite(excel_path: Path | None = None) -> Dict[str, float]:
    """Execute end-to-end validation checks.

    Args:
        excel_path: Path to Excel file. Defaults to project Excel file.

    Returns:
        Dictionary of percent errors for each metric.
    """
    file_path = excel_path or EXCEL_FILE

    print(f"Loading Excel: {file_path}")
    calc_df = _load_excel_calc(file_path)
    truth = _extract_calc_truth(calc_df)

    print("Extracting inputs...")
    datetime_series, solar_kw, load_kw, period_flags, allow_discharge = _extract_inputs(calc_df)

    # Load BESS parameters from Assumption sheet
    cfg = _load_bess_config(file_path)

    print("Running Calc engine...")
    results = run_calc(datetime_series, solar_kw, load_kw, period_flags, allow_discharge, cfg)

    print("\n=== Regression Results ===")
    errors = {}
    for key in ["solar_gen_mwh", "discharge_mwh", "power_surplus_mwh"]:
        model_val = results.outputs.get(key.replace("discharge", "discharge"), 0.0)
        if key == "discharge_mwh":
            model_val = results.outputs.get("discharge_mwh", 0.0)
        truth_val = truth.get(key, 0.0)
        err = _pct_error(model_val, truth_val)
        errors[key] = err
        status = "✓" if err < 1.0 else "✗"
        print(f"  {key}: Model={model_val:.2f}, Truth={truth_val:.2f}, Error={err:.2f}% {status}")

    print("\n=== Summary ===")
    max_err = max(errors.values()) if errors else 0.0
    if max_err < 1.0:
        print("All metrics within 1% tolerance. PASS")
    else:
        print(f"Max error: {max_err:.2f}%. FAIL")

    return errors


if __name__ == "__main__":
    run_regression_suite()
