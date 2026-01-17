from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXCEL_FILE = Path(__file__).with_name("AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx")

DEFAULT_BESS_CAPACITY_KWH = 66000.0
DEFAULT_BESS_POWER_KW = 20000.0
DEFAULT_BESS_EFFICIENCY = 0.95  # one-way efficiency

DEFAULT_SOLAR_CAPACITY_KWP = 40360.0
DEFAULT_PERFORMANCE_RATIO = 0.8085913562510872

DEFAULT_PV_DEGRADATION = [
    1.0,
    0.98,
    0.9745,
    0.969,
    0.9635,
    0.958,
    0.9525,
    0.947,
    0.9415,
    0.936,
]

DEFAULT_BATT_DEGRADATION = [
    1.0,
    0.9745,
    0.9375,
    0.9157,
    0.89505,
    0.87435,
    0.85365,
    0.83295,
    0.81225,
    0.79155,
]

DEFAULT_AUGMENTATION_KWH = DEFAULT_BESS_CAPACITY_KWH
AUGMENTATION_YEARS = (11, 21)


@dataclass
class BESSParameters:
    capacity_kwh: float
    power_kw: float
    efficiency: float
    augmentation_kwh: float


@dataclass
class SolarParameters:
    capacity_kwp: float
    performance_ratio: float


@dataclass
class TouFlags:
    peak: Optional[np.ndarray]
    standard: Optional[np.ndarray]
    offpeak: Optional[np.ndarray]
    source: str

    def discharge_allowed(self, index: int) -> bool:
        if self.peak is None or self.standard is None:
            return True
        return bool(self.peak[index] or self.standard[index])


@dataclass
class ModelResult:
    name: str
    summary: Dict[str, float]
    excel_errors: Dict[str, float]
    score: float
    lifetime_summary: Optional[Dict[str, float]] = None
    lifetime_yearly: Optional[pd.DataFrame] = None


def safe_read_excel(file_path: Path, sheet_name: str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    except Exception:
        return None


def find_supply_table_totals(measures_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    header_row_index = None
    header_positions: Dict[str, int] = {}
    required = ["month", "solar", "bess", "grid", "surplus"]

    for idx, row in measures_df.iterrows():
        row_text = [normalize_text(cell) for cell in row.tolist()]
        if "month" in row_text:
            header_positions = {}
            for label in required:
                if label in row_text:
                    header_positions[label] = row_text.index(label)
            if len(header_positions) == len(required):
                header_row_index = idx
                break

    if header_row_index is None:
        return None

    for idx in range(header_row_index + 1, min(header_row_index + 30, len(measures_df))):
        row = measures_df.iloc[idx].tolist()
        label = row[0] if row else None
        if isinstance(label, str) and normalize_text(label) == "total":
            solar_val = first_numeric([row[header_positions["solar"]]])
            bess_val = first_numeric([row[header_positions["bess"]]])
            surplus_val = first_numeric([row[header_positions["surplus"]]])
            if solar_val is None or bess_val is None or surplus_val is None:
                return None
            return {
                "solar_gen_mwh": solar_val,
                "bess_discharge_mwh": bess_val,
                "grid_export_mwh": surplus_val,
            }

    return None


def search_metric_any(measures_df: pd.DataFrame, keyword_sets: List[List[str]]) -> Optional[float]:
    for keywords in keyword_sets:
        value = search_metric(measures_df, keywords)
        if value is not None:
            return value
    return None


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def to_bool(series: pd.Series) -> np.ndarray:
    return series.fillna(0).astype(float).to_numpy() > 0.5


def first_numeric(values: List[object]) -> Optional[float]:
    for value in values:
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            return float(value)
        if isinstance(value, str):
            candidate = value.replace(",", "").strip()
            try:
                return float(candidate)
            except ValueError:
                continue
    return None


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {col: normalize_text(col) for col in columns}
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        for col, col_norm in normalized.items():
            if candidate_norm == col_norm:
                return col
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        for col, col_norm in normalized.items():
            if candidate_norm in col_norm:
                return col
    return None


def pick_flag_column(columns: List[str], include: List[str], exclude: List[str]) -> Optional[str]:
    for col in columns:
        name = normalize_text(col)
        if all(token in name for token in include) and not any(token in name for token in exclude):
            return col
    return None


def load_excel_inputs(file_path: Path) -> Dict[str, Optional[pd.DataFrame]]:
    return {
        "data": safe_read_excel(file_path, sheet_name="Data Input"),
        "calc": safe_read_excel(file_path, sheet_name="Calc"),
        "measures": safe_read_excel(file_path, sheet_name="Measures", header=None),
        "assumption": safe_read_excel(file_path, sheet_name="Assumption", header=None),
        "loss": safe_read_excel(file_path, sheet_name="Loss"),
        "other": safe_read_excel(file_path, sheet_name="Other Input", header=None),
    }


def extract_bess_parameters(assumption_df: Optional[pd.DataFrame]) -> BESSParameters:
    capacity_kwh = DEFAULT_BESS_CAPACITY_KWH
    power_kw = DEFAULT_BESS_POWER_KW
    efficiency = DEFAULT_BESS_EFFICIENCY
    augmentation_kwh = DEFAULT_AUGMENTATION_KWH

    if assumption_df is None:
        return BESSParameters(capacity_kwh, power_kw, efficiency, augmentation_kwh)

    for _, row in assumption_df.iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        label_norm = normalize_text(label)
        value = first_numeric(row.iloc[1:].tolist())
        if value is None:
            continue
        if "bess" in label_norm or "battery" in label_norm:
            if "capacity" in label_norm and "kw" not in label_norm:
                capacity_kwh = value
            elif "power" in label_norm or "mw" in label_norm or "kw" in label_norm:
                power_kw = value
            elif "efficiency" in label_norm:
                efficiency = value
            elif "augmentation" in label_norm or "replacement" in label_norm:
                augmentation_kwh = value

    return BESSParameters(capacity_kwh, power_kw, efficiency, augmentation_kwh)


def extract_solar_parameters(assumption_df: Optional[pd.DataFrame]) -> SolarParameters:
    capacity_kwp = DEFAULT_SOLAR_CAPACITY_KWP
    performance_ratio = DEFAULT_PERFORMANCE_RATIO

    if assumption_df is None:
        return SolarParameters(capacity_kwp, performance_ratio)

    for _, row in assumption_df.iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        label_norm = normalize_text(label)
        value = first_numeric(row.iloc[1:].tolist())
        if value is None:
            continue
        if "solar" in label_norm or "pv" in label_norm:
            if "capacity" in label_norm or "kwp" in label_norm:
                capacity_kwp = value
            elif "performance" in label_norm or "ratio" in label_norm:
                performance_ratio = value

    return SolarParameters(capacity_kwp, performance_ratio)


def extract_hourly_inputs(data_df: pd.DataFrame, solar_params: SolarParameters) -> Tuple[np.ndarray, np.ndarray]:
    columns = data_df.columns.tolist()
    solar_col = find_column(
        columns,
        [
            "SimulationProfile_kW",
            "Simulation Profile",
            "SolarGen_kW",
            "SolarGen",
            "PV",
            "Generation",
        ],
    )
    load_col = find_column(columns, ["Load_kW", "Load", "Demand", "Load Profile"])

    if load_col is None:
        raise ValueError("Load_kW column not found in Data Input sheet.")

    load_kw = data_df[load_col].astype(float).to_numpy()

    if solar_col:
        solar_kw = data_df[solar_col].astype(float).to_numpy()
        return solar_kw, load_kw

    irradiance_col = find_column(columns, ["Irradiation_W/m2", "Irradiation", "GHI", "Solar Irradiance"])
    if irradiance_col:
        irradiance = data_df[irradiance_col].astype(float).to_numpy()
        solar_kw = irradiance * solar_params.capacity_kwp * solar_params.performance_ratio / 1000.0
        return solar_kw, load_kw

    raise ValueError("Solar profile column not found in Data Input sheet.")


def extract_datetime(data_df: pd.DataFrame) -> np.ndarray:
    dt_col = find_column(data_df.columns.tolist(), ["DateTime", "Datetime", "Timestamp", "Time"])
    if dt_col is None:
        return np.arange(len(data_df))
    return pd.to_datetime(data_df[dt_col]).to_numpy()


def extract_demand_target(calc_df: Optional[pd.DataFrame]) -> Optional[float]:
    if calc_df is None:
        return None
    target_col = find_column(calc_df.columns.tolist(), ["DemandTarget_kW", "Demand Target"])
    if target_col is None:
        return None
    series = calc_df[target_col].dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def extract_tou_flags(calc_df: Optional[pd.DataFrame], hours: int) -> TouFlags:
    if calc_df is None:
        return TouFlags(None, None, None, "calc sheet missing")

    columns = calc_df.columns.tolist()
    time_flag_col = find_column(columns, ["TimePeriodFlag", "Time Period Flag", "Time Period"])
    if time_flag_col:
        flags = calc_df[time_flag_col].astype(str).str.upper().fillna("N").to_numpy()
        peak = flags == "P"
        standard = flags == "N"
        offpeak = flags == "O"
        return TouFlags(peak, standard, offpeak, f"timeperiod={time_flag_col}")

    exclude = ["start", "end", "min", "hour", "time"]
    peak_col = pick_flag_column(columns, ["peak"], exclude)
    standard_col = pick_flag_column(columns, ["standard"], exclude)
    offpeak_col = pick_flag_column(columns, ["off"], exclude)

    peak = to_bool(calc_df[peak_col]) if peak_col else None
    standard = to_bool(calc_df[standard_col]) if standard_col else None
    offpeak = to_bool(calc_df[offpeak_col]) if offpeak_col else None

    source = f"peak={peak_col}, standard={standard_col}, offpeak={offpeak_col}"
    if peak is not None and len(peak) != hours:
        peak = None
    if standard is not None and len(standard) != hours:
        standard = None
    if offpeak is not None and len(offpeak) != hours:
        offpeak = None

    return TouFlags(peak, standard, offpeak, source)


def extract_degradation_factors(loss_df: Optional[pd.DataFrame], other_df: Optional[pd.DataFrame]) -> Tuple[List[float], List[float]]:
    pv = None
    battery = None

    for df in [loss_df, other_df]:
        if df is None:
            continue

        for col in df.columns:
            name = normalize_text(col)
            if name == "pv" and pv is None:
                pv = df[col].dropna().astype(float).tolist()
            if name == "battery" and battery is None:
                battery = df[col].dropna().astype(float).tolist()

        if pv or battery:
            continue

        for _, row in df.iterrows():
            label = row.iloc[0]
            if not isinstance(label, str):
                continue
            label_norm = normalize_text(label)
            if "pv" in label_norm and pv is None:
                pv_values = [first_numeric(row.iloc[1:].tolist())]
                pv_values = [val for val in pv_values if val is not None]
                if pv_values:
                    pv = pv_values
            if "battery" in label_norm and battery is None:
                batt_values = [first_numeric(row.iloc[1:].tolist())]
                batt_values = [val for val in batt_values if val is not None]
                if batt_values:
                    battery = batt_values

    if not pv:
        pv = DEFAULT_PV_DEGRADATION
    if not battery:
        battery = DEFAULT_BATT_DEGRADATION

    return pv, battery


def search_metric(measures_df: pd.DataFrame, keywords: List[str]) -> Optional[float]:
    for _, row in measures_df.iterrows():
        row_list = row.tolist()
        for idx, cell in enumerate(row_list):
            if not isinstance(cell, str):
                continue
            cell_norm = normalize_text(cell)
            if all(keyword in cell_norm for keyword in keywords):
                value = first_numeric(row_list[idx + 1 :])
                if value is not None:
                    return value
    return None


def extract_measures_truth(measures_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if measures_df is None:
        raise ValueError("Measures sheet not found in Excel file.")

    supply_totals = find_supply_table_totals(measures_df)
    if supply_totals:
        return supply_totals

    truth = {}
    truth["solar_gen_mwh"] = (
        search_metric_any(
            measures_df,
            [
                ["total", "solar", "generation"],
                ["solar", "generation"],
                ["solar", "gen"],
                ["solar", "mwh"],
            ],
        )
        or 0.0
    )
    truth["grid_export_mwh"] = (
        search_metric_any(
            measures_df,
            [
                ["pv", "surplus"],
                ["surplusenergy"],
                ["surplus", "energy"],
                ["power", "surplus"],
                ["grid", "export"],
                ["export", "mwh"],
            ],
        )
        or 0.0
    )
    truth["bess_discharge_mwh"] = (
        search_metric_any(
            measures_df,
            [
                ["bess-to-load"],
                ["bess", "load"],
                ["bess", "discharge"],
                ["battery", "discharge"],
                ["bess", "mwh"],
            ],
        )
        or 0.0
    )

    return truth


def normalize_measures_truth(truth: Dict[str, float]) -> Tuple[Dict[str, float], Optional[str]]:
    if not truth:
        return truth, None
    max_value = max(truth.values())
    if max_value > 1_000_000:
        scaled = {key: value / 1000.0 for key, value in truth.items()}
        return scaled, "Measures values appear to be in kWh; scaled to MWh for comparison."
    return truth, None


def summarize_measure_candidates(measures_df: Optional[pd.DataFrame]) -> List[str]:
    if measures_df is None:
        return []
    keywords = ["solar", "grid", "export", "bess", "battery", "discharge", "surplus"]
    matches: List[str] = []
    for idx, row in measures_df.iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        label_norm = normalize_text(label)
        if any(keyword in label_norm for keyword in keywords):
            value = first_numeric(row.tolist()[1:])
            matches.append(f"Row {idx}: {label} -> {value}")
            if len(matches) >= 12:
                break
    return matches


def compute_grid_export_excel_bug(solar_kw: np.ndarray, load_kw: np.ndarray) -> np.ndarray:
    """Replicate Excel constraint: Grid Export = Excess Solar - Load."""
    excess_solar_kw = solar_kw - load_kw
    return excess_solar_kw - load_kw


def compute_grid_export_physical(solar_kw: np.ndarray, load_kw: np.ndarray, charge_kw: np.ndarray) -> np.ndarray:
    return np.maximum(solar_kw - load_kw - charge_kw, 0.0)


def aggregate_mwh(values: np.ndarray, clip_negative: bool = False) -> float:
    data = np.maximum(values, 0.0) if clip_negative else values
    return float(np.sum(data) / 1000.0)


def aggregate_energy_kwh(values: np.ndarray) -> float:
    return float(np.sum(values) / 1000.0)


def extract_calc_truth(calc_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], List[str]]:
    warnings: List[str] = []
    if calc_df is None:
        return {}, ["Calc sheet not found; skipping hourly validation."]

    columns = calc_df.columns.tolist()
    solar_col = find_column(columns, ["DirectPVConsumption_kW", "Direct PV Consumption", "Solar Supply"])
    load_col = find_column(columns, ["Load_kW", "Load", "Demand"])
    discharge_col = find_column(columns, ["DischargeEnergy_kWh", "Discharge Energy", "BESS Discharge"])
    grid_export_col = find_column(
        columns,
        [
            "PowerSurplus_kW",
            "Power Surplus",
            "Surplus",
        ],
    )

    truth: Dict[str, float] = {}

    if solar_col:
        truth["solar_gen_mwh"] = aggregate_mwh(calc_df[solar_col].astype(float).to_numpy())
    else:
        warnings.append("Solar generation column not found in Calc sheet.")

    if discharge_col:
        truth["bess_discharge_mwh"] = aggregate_energy_kwh(calc_df[discharge_col].astype(float).to_numpy())
    else:
        warnings.append("BESS discharge column not found in Calc sheet.")

    grid_export_kw: Optional[np.ndarray] = None
    if grid_export_col:
        grid_export_kw = calc_df[grid_export_col].astype(float).to_numpy()
    elif solar_col and load_col:
        grid_export_kw = compute_grid_export_excel_bug(
            calc_df[solar_col].astype(float).to_numpy(),
            calc_df[load_col].astype(float).to_numpy(),
        )
        warnings.append("Grid export column missing; computed using Excel bug formula (Solar - 2*Load).")
    else:
        warnings.append("Grid export could not be computed (missing Solar or Load columns).")

    if grid_export_kw is not None:
        truth["grid_export_mwh"] = aggregate_mwh(grid_export_kw, clip_negative=True)

    return truth, warnings


def simulate_direct_clone(
    solar_kw: np.ndarray,
    load_kw: np.ndarray,
    bess_params: BESSParameters,
    tou_flags: TouFlags,
    demand_target_kw: Optional[float],
) -> Dict[str, object]:
    hours = len(load_kw)
    discharge_kw = np.zeros(hours)
    charge_kwh = np.zeros(hours)
    soc_kwh = np.zeros(hours)

    current_soc = 0.0

    for hour in range(hours):
        net_load_after_solar = load_kw[hour] - solar_kw[hour]

        if solar_kw[hour] > 0 and load_kw[hour] > 0:
            excess_solar_available = min(solar_kw[hour], load_kw[hour])
        else:
            excess_solar_available = 0.0

        headroom = bess_params.capacity_kwh - current_soc
        charge_limit_kw = min(bess_params.power_kw, headroom)

        if excess_solar_available > 0 and headroom > 0:
            available_charge_kw = min(excess_solar_available, charge_limit_kw)
            charge_kwh[hour] = available_charge_kw
            current_soc += charge_kwh[hour] * bess_params.efficiency

        discharge_allowed = tou_flags.discharge_allowed(hour)
        if demand_target_kw is not None:
            discharge_allowed = discharge_allowed and load_kw[hour] > demand_target_kw

        if discharge_allowed and current_soc > 0:
            required_kw = net_load_after_solar - (demand_target_kw or 0.0)
            required_kw = max(required_kw, 0.0)
            discharge = min(required_kw, bess_params.power_kw, current_soc)
            discharge_kw[hour] = discharge
            current_soc -= discharge

        current_soc = max(0.0, min(current_soc, bess_params.capacity_kwh))
        soc_kwh[hour] = current_soc

    grid_load_kw = np.maximum(load_kw - solar_kw - discharge_kw, 0.0)
    grid_export_kw = compute_grid_export_excel_bug(solar_kw, load_kw)

    return {
        "solar_kw": solar_kw,
        "load_kw": load_kw,
        "charge_kwh": charge_kwh,
        "discharge_kw": discharge_kw,
        "soc_kwh": soc_kwh,
        "grid_load_kw": grid_load_kw,
        "grid_export_kw": grid_export_kw,
    }


def simulate_peak_conservative(
    solar_kw: np.ndarray,
    load_kw: np.ndarray,
    bess_params: BESSParameters,
    tou_flags: TouFlags,
) -> Dict[str, object]:
    hours = len(load_kw)
    discharge_kw = np.zeros(hours)
    charge_kwh = np.zeros(hours)
    soc_kwh = np.zeros(hours)
    reserve_kwh = 0.1 * bess_params.capacity_kwh

    current_soc = 0.0

    for hour in range(hours):
        net_load_after_solar = load_kw[hour] - solar_kw[hour]

        if solar_kw[hour] > 0 and load_kw[hour] > 0:
            excess_solar_available = min(solar_kw[hour], load_kw[hour])
        else:
            excess_solar_available = 0.0

        headroom = bess_params.capacity_kwh - current_soc
        charge_limit_kw = min(bess_params.power_kw, headroom)

        if excess_solar_available > 0 and headroom > 0:
            available_charge_kw = min(excess_solar_available, charge_limit_kw)
            charge_kwh[hour] = available_charge_kw
            current_soc += charge_kwh[hour] * bess_params.efficiency

        discharge_allowed = tou_flags.peak[hour] if tou_flags.peak is not None else True
        if discharge_allowed and current_soc > reserve_kwh:
            required_kw = max(net_load_after_solar, 0.0)
            available_soc = current_soc - reserve_kwh
            discharge = min(required_kw, bess_params.power_kw, available_soc)
            discharge_kw[hour] = discharge
            current_soc -= discharge

        current_soc = max(0.0, min(current_soc, bess_params.capacity_kwh))
        soc_kwh[hour] = current_soc

    grid_load_kw = np.maximum(load_kw - solar_kw - discharge_kw, 0.0)
    grid_export_kw = compute_grid_export_physical(solar_kw, load_kw, charge_kwh)

    return {
        "solar_kw": solar_kw,
        "load_kw": load_kw,
        "charge_kwh": charge_kwh,
        "discharge_kw": discharge_kw,
        "soc_kwh": soc_kwh,
        "grid_load_kw": grid_load_kw,
        "grid_export_kw": grid_export_kw,
    }


def summarize_hourly(output: Dict[str, object]) -> Dict[str, float]:
    solar_gen_mwh = aggregate_mwh(output["solar_kw"])
    grid_export_mwh = aggregate_mwh(output["grid_export_kw"], clip_negative=True)
    bess_discharge_mwh = aggregate_mwh(output["discharge_kw"])
    grid_load_mwh = aggregate_mwh(output["grid_load_kw"])

    return {
        "solar_gen_mwh": solar_gen_mwh,
        "grid_export_mwh": grid_export_mwh,
        "bess_discharge_mwh": bess_discharge_mwh,
        "grid_load_mwh": grid_load_mwh,
    }


def calculate_errors(summary: Dict[str, float], excel_truth: Dict[str, float]) -> Dict[str, float]:
    errors = {}
    for key in ["solar_gen_mwh", "grid_export_mwh", "bess_discharge_mwh"]:
        excel_value = excel_truth.get(key, 0.0)
        model_value = summary.get(key, 0.0)
        if excel_value == 0:
            errors[key] = abs(model_value - excel_value)
        else:
            errors[key] = abs(model_value - excel_value) / excel_value
    return errors


def score_errors(errors: Dict[str, float]) -> float:
    return sum(errors.values())


def simulate_lifetime_augmented(
    solar_kw: np.ndarray,
    load_kw: np.ndarray,
    bess_params: BESSParameters,
    tou_flags: TouFlags,
    demand_target_kw: Optional[float],
    pv_degradation: List[float],
    batt_degradation: List[float],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    yearly_rows = []
    total_summary = {
        "solar_gen_mwh": 0.0,
        "grid_export_mwh": 0.0,
        "bess_discharge_mwh": 0.0,
    }

    for year in range(1, 26):
        if year <= len(pv_degradation):
            solar_factor = pv_degradation[year - 1]
        else:
            solar_factor = pv_degradation[-1]

        if year <= len(batt_degradation):
            battery_factor = batt_degradation[year - 1]
        else:
            battery_factor = batt_degradation[-1]

        adjusted_capacity = bess_params.capacity_kwh * battery_factor
        if year in AUGMENTATION_YEARS:
            adjusted_capacity += bess_params.augmentation_kwh

        year_bess = BESSParameters(
            capacity_kwh=adjusted_capacity,
            power_kw=bess_params.power_kw,
            efficiency=bess_params.efficiency,
            augmentation_kwh=bess_params.augmentation_kwh,
        )
        year_solar = solar_kw * solar_factor

        output = simulate_direct_clone(year_solar, load_kw, year_bess, tou_flags, demand_target_kw)
        summary = summarize_hourly(output)

        yearly_rows.append({
            "year": year,
            "solar_gen_mwh": summary["solar_gen_mwh"],
            "grid_export_mwh": summary["grid_export_mwh"],
            "bess_discharge_mwh": summary["bess_discharge_mwh"],
            "capacity_kwh": adjusted_capacity,
            "solar_factor": solar_factor,
            "battery_factor": battery_factor,
        })

        for key in total_summary:
            total_summary[key] += summary[key]

    return total_summary, pd.DataFrame(yearly_rows)


def run_models(
    solar_kw: np.ndarray,
    load_kw: np.ndarray,
    bess_params: BESSParameters,
    tou_flags: TouFlags,
    demand_target_kw: Optional[float],
    pv_degradation: List[float],
    batt_degradation: List[float],
    excel_truth: Dict[str, float],
) -> List[ModelResult]:
    results: List[ModelResult] = []

    model_a_output = simulate_direct_clone(solar_kw, load_kw, bess_params, tou_flags, demand_target_kw)
    model_a_summary = summarize_hourly(model_a_output)
    model_a_errors = calculate_errors(model_a_summary, excel_truth)
    results.append(
        ModelResult(
            name="Model A - Direct Clone",
            summary=model_a_summary,
            excel_errors=model_a_errors,
            score=score_errors(model_a_errors),
        )
    )

    model_b_lifetime, model_b_yearly = simulate_lifetime_augmented(
        solar_kw,
        load_kw,
        bess_params,
        tou_flags,
        demand_target_kw,
        pv_degradation,
        batt_degradation,
    )
    model_b_summary = model_a_summary
    model_b_errors = calculate_errors(model_b_summary, excel_truth)
    results.append(
        ModelResult(
            name="Model B - Lifetime Augmented",
            summary=model_b_summary,
            excel_errors=model_b_errors,
            score=score_errors(model_b_errors),
            lifetime_summary=model_b_lifetime,
            lifetime_yearly=model_b_yearly,
        )
    )

    model_c_output = simulate_peak_conservative(solar_kw, load_kw, bess_params, tou_flags)
    model_c_summary = summarize_hourly(model_c_output)
    model_c_errors = calculate_errors(model_c_summary, excel_truth)
    results.append(
        ModelResult(
            name="Model C - Peak Conservative",
            summary=model_c_summary,
            excel_errors=model_c_errors,
            score=score_errors(model_c_errors),
        )
    )

    return results


def print_results(
    results: List[ModelResult],
    excel_truth: Dict[str, float],
    tou_flags: TouFlags,
    calc_truth: Optional[Dict[str, float]] = None,
    calc_warnings: Optional[List[str]] = None,
    measures_note: Optional[str] = None,
    measures_candidates: Optional[List[str]] = None,
) -> None:
    print("BESS Audit Models - Accuracy Summary")
    print("=" * 60)
    print(f"TOU flags source: {tou_flags.source}")
    if calc_warnings:
        print("Calc sheet validation notes:")
        for note in calc_warnings:
            print(f"  - {note}")
    if calc_warnings:
        print()
    if measures_note:
        print("Measures sheet notes:")
        print(f"  - {measures_note}")
    if measures_candidates:
        print("Measures candidates (check labels if values are zero):")
        for entry in measures_candidates:
            print(f"  - {entry}")
    if measures_note or measures_candidates:
        print()
    print()

    metrics = ["solar_gen_mwh", "grid_export_mwh", "bess_discharge_mwh"]

    for result in sorted(results, key=lambda item: item.score):
        print(result.name)
        print("-" * len(result.name))
        print("Measures sheet truth:")
        for metric in metrics:
            excel_value = excel_truth.get(metric, 0.0)
            model_value = result.summary.get(metric, 0.0)
            error = result.excel_errors.get(metric, 0.0)
            error_pct = error * 100.0 if excel_value else error
            print(f"{metric}: Excel={excel_value:,.2f} | Model={model_value:,.2f} | Error={error_pct:.2f}%")
        if calc_truth:
            print("Calc sheet truth:")
            calc_errors = calculate_errors(result.summary, calc_truth)
            for metric in metrics:
                calc_value = calc_truth.get(metric, 0.0)
                model_value = result.summary.get(metric, 0.0)
                error = calc_errors.get(metric, 0.0)
                error_pct = error * 100.0 if calc_value else error
                print(
                    f"{metric}: Calc={calc_value:,.2f} | Model={model_value:,.2f} | Error={error_pct:.2f}%"
                )
        print(f"Score: {result.score:.4f}")
        if result.lifetime_summary:
            lifetime = result.lifetime_summary
            print("Lifetime totals (25-year):")
            print(
                f"  Solar={lifetime['solar_gen_mwh']:.2f} MWh | "
                f"Grid Export={lifetime['grid_export_mwh']:.2f} MWh | "
                f"BESS Discharge={lifetime['bess_discharge_mwh']:.2f} MWh"
            )
        print()

    best = min(results, key=lambda item: item.score)
    print("Best Model for Final Audit:")
    print(f"-> {best.name}")


def main() -> None:
    inputs = load_excel_inputs(EXCEL_FILE)
    data_df = inputs["data"]

    if data_df is None:
        raise FileNotFoundError("Data Input sheet not found in Excel file.")

    bess_params = extract_bess_parameters(inputs["assumption"])
    solar_params = extract_solar_parameters(inputs["assumption"])
    solar_kw, load_kw = extract_hourly_inputs(data_df, solar_params)
    demand_target_kw = extract_demand_target(inputs["calc"])
    tou_flags = extract_tou_flags(inputs["calc"], len(load_kw))
    pv_degradation, batt_degradation = extract_degradation_factors(inputs["loss"], inputs["other"])
    excel_truth_raw = extract_measures_truth(inputs["measures"])
    excel_truth, measures_note = normalize_measures_truth(excel_truth_raw)
    measures_candidates = None
    if any(value == 0.0 for value in excel_truth.values()):
        measures_candidates = summarize_measure_candidates(inputs["measures"])
    calc_truth, calc_warnings = extract_calc_truth(inputs["calc"])
    if not calc_truth:
        calc_truth = None

    results = run_models(
        solar_kw,
        load_kw,
        bess_params,
        tou_flags,
        demand_target_kw,
        pv_degradation,
        batt_degradation,
        excel_truth,
    )

    print_results(
        results,
        excel_truth,
        tou_flags,
        calc_truth,
        calc_warnings,
        measures_note,
        measures_candidates,
    )


if __name__ == "__main__":
    main()
