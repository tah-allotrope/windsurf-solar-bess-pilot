"""End-to-end pipeline runner for Solar+BESS model.

Combines all modules:
1. Load inputs from Excel
2. Run Calc engine (hourly simulation)
3. Run Lifetime simulation (25-year degradation)
4. Run Financial model (cash flow, IRR, NPV)
5. Run DPPA pricing (optional)
6. Generate summary report
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from excel_replica.model.calc_engine import CalcConfig, CalcResults, run_calc
from excel_replica.model.lifetime import (
    DegradationSchedule,
    LifetimeResults,
    load_degradation_from_excel,
    simulate_lifetime,
)
from excel_replica.model.financial import (
    FinancialConfig,
    FinancialResults,
    TaxHoliday,
    MRASchedule,
    run_financial_model,
)
from excel_replica.model.dppa import (
    DPPAConfig,
    DPPAResults,
    calculate_dppa_hourly,
    load_dppa_config_from_excel,
)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    excel_path: Path
    revenue_per_mwh: float = 49.59  # USD/MWh
    run_dppa: bool = True
    voltage_level_kv: int = 22


@dataclass
class PipelineResults:
    """Results from the full pipeline."""
    calc: CalcResults
    lifetime: LifetimeResults
    financial: FinancialResults
    dppa: Optional[DPPAResults] = None
    summary: Dict[str, float] = None


def load_calc_config(file_path: Path) -> CalcConfig:
    """Load BESS and tariff configuration from Assumption sheet."""
    df = pd.read_excel(file_path, sheet_name="Assumption", engine="openpyxl", header=None)

    def get_value(row_idx: int, col_idx: int = 4) -> float:
        val = df.iloc[row_idx, col_idx]
        if pd.notna(val) and isinstance(val, (int, float)):
            return float(val)
        for c in [5, 3]:
            val = df.iloc[row_idx, c]
            if pd.notna(val) and isinstance(val, (int, float)):
                return float(val)
        return 0.0

    total_capacity_kwh = get_value(24)
    power_kw = get_value(25)
    dod = get_value(26)
    efficiency = get_value(27)
    min_soc = get_value(38)

    usable_capacity_kwh = total_capacity_kwh * dod

    exchange_rate = df.iloc[8, 10] if pd.notna(df.iloc[8, 10]) else 26000.0
    ca_normal_vnd = df.iloc[13, 16] if pd.notna(df.iloc[13, 16]) else 1253.0
    ca_peak_vnd = df.iloc[14, 16] if pd.notna(df.iloc[14, 16]) else 2162.0
    ca_offpeak_vnd = df.iloc[15, 16] if pd.notna(df.iloc[15, 16]) else 843.0

    ca_normal = float(ca_normal_vnd) / float(exchange_rate)
    ca_peak = float(ca_peak_vnd) / float(exchange_rate)
    ca_offpeak = float(ca_offpeak_vnd) / float(exchange_rate)

    return CalcConfig(
        step_hours=1.0,
        bess_capacity_kwh=usable_capacity_kwh,
        bess_power_kw=power_kw,
        bess_efficiency=efficiency,
        min_soc_kwh=min_soc,
        ca_peak=ca_peak,
        ca_normal=ca_normal,
        ca_offpeak=ca_offpeak,
    )


def load_financial_config(file_path: Path) -> FinancialConfig:
    """Load financial configuration from Assumption and Financial sheets."""
    df = pd.read_excel(file_path, sheet_name="Assumption", engine="openpyxl", header=None)
    fin_df = pd.read_excel(file_path, sheet_name="Financial", engine="openpyxl", header=None)

    def get_val(row: int, col: int, default: float) -> float:
        val = df.iloc[row, col]
        if pd.notna(val) and isinstance(val, (int, float)):
            return float(val)
        return default

    def get_fin_val(row: int, col: int, default: float) -> float:
        val = fin_df.iloc[row, col]
        if pd.notna(val) and isinstance(val, (int, float)):
            return float(val)
        return default

    solar_mwp = get_val(17, 10, 40.36)
    bess_mwh = get_val(18, 10, 66.0)

    # Get debt parameters from Financial sheet
    interest_rate = get_fin_val(160, 6, 0.02)  # All-in Interest Rate
    leverage_ratio = 24_584_997 / 49_513_200  # From Excel debt balance

    return FinancialConfig(
        land_cost_usd=get_val(39, 10, 1_200_000),
        bop_cost_usd=get_val(42, 10, 4_843_200),
        pv_cost_usd=solar_mwp * get_val(40, 10, 750_000),
        bess_cost_usd=bess_mwh * get_val(41, 10, 200_000),
        om_pv_usd=242_160,
        om_bess_usd=132_000,
        insurance_pv_usd=75_675,
        insurance_bess_usd=33_000,
        other_opex_usd=161_440,
        land_lease_usd=0,
        leverage_ratio=leverage_ratio,
        debt_tenor_years=15,
        interest_rate=interest_rate,
        discount_rate=0.10,
    )


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Run the full model pipeline.

    Args:
        config: PipelineConfig with Excel path and options.

    Returns:
        PipelineResults with all module outputs.
    """
    print(f"Loading Excel: {config.excel_path}")

    # Load Calc sheet data
    calc_df = pd.read_excel(config.excel_path, sheet_name="Calc", engine="openpyxl")

    datetime_series = pd.to_datetime(calc_df["DateTime"])
    solar_kw = calc_df["SolarGen_kW"].astype(float).to_numpy()
    load_kw = calc_df["Load_kW"].astype(float).to_numpy()
    period_flags = calc_df["TimePeriodFlag"].astype(str).str.upper().to_numpy()

    # Discharge permission
    if "DischargeConditionFlag" in calc_df.columns:
        allow_discharge = calc_df["DischargeConditionFlag"].astype(bool).to_numpy()
    else:
        allow_discharge = np.isin(period_flags, ["P", "N"])

    # Load configurations
    calc_cfg = load_calc_config(config.excel_path)
    fin_cfg = load_financial_config(config.excel_path)

    print(f"BESS: {calc_cfg.bess_capacity_kwh:.0f} kWh, {calc_cfg.bess_power_kw:.0f} kW")
    print(f"CAPEX: ${fin_cfg.land_cost_usd + fin_cfg.bop_cost_usd + fin_cfg.pv_cost_usd + fin_cfg.bess_cost_usd:,.0f}")

    # Step 1: Run Calc engine
    print("\n[1/4] Running Calc engine...")
    calc_results = run_calc(datetime_series, solar_kw, load_kw, period_flags, allow_discharge, calc_cfg)

    # Step 2: Run Lifetime simulation
    print("[2/4] Running Lifetime simulation...")
    degradation = load_degradation_from_excel(config.excel_path)

    year1_outputs = {
        "solar_gen_mwh": calc_results.outputs["solar_gen_mwh"],
        "discharge_mwh": calc_results.outputs["discharge_mwh"],
        "power_surplus_mwh": calc_results.outputs["power_surplus_mwh"],
        "direct_pv_mwh": calc_results.outputs["direct_pv_mwh"],
        "charge_mwh": calc_results.outputs["charge_mwh"],
    }
    lifetime_results = simulate_lifetime(year1_outputs, degradation)

    # Step 3: Run Financial model
    print("[3/4] Running Financial model...")
    financial_results = run_financial_model(
        lifetime_results.yearly,
        revenue_per_mwh=config.revenue_per_mwh,
        cfg=fin_cfg,
    )

    # Step 4: Run DPPA (optional)
    dppa_results = None
    if config.run_dppa:
        print("[4/4] Running DPPA pricing...")
        dppa_cfg = load_dppa_config_from_excel(config.excel_path)
        dppa_results = calculate_dppa_hourly(
            datetime_series, solar_kw, load_kw, period_flags, dppa_cfg, config.voltage_level_kv
        )

    # Build summary
    summary = {
        "solar_gen_mwh_y1": calc_results.outputs["solar_gen_mwh"],
        "discharge_mwh_y1": calc_results.outputs["discharge_mwh"],
        "surplus_mwh_y1": calc_results.outputs["power_surplus_mwh"],
        "total_solar_gen_mwh_25y": lifetime_results.totals["total_solar_gen_mwh"],
        "total_discharge_mwh_25y": lifetime_results.totals["total_discharge_mwh"],
        "project_irr": financial_results.project_irr,
        "equity_irr": financial_results.equity_irr,
        "npv_usd": financial_results.npv,
        "payback_years": financial_results.payback_years,
    }

    if dppa_results:
        summary["dppa_net_revenue_usd"] = dppa_results.totals["total_net_revenue_usd"]

    return PipelineResults(
        calc=calc_results,
        lifetime=lifetime_results,
        financial=financial_results,
        dppa=dppa_results,
        summary=summary,
    )


def print_summary(results: PipelineResults) -> None:
    """Print pipeline results summary."""
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)

    print("\n## Year 1 Energy Metrics")
    print(f"  Solar Generation: {results.summary['solar_gen_mwh_y1']:,.2f} MWh")
    print(f"  BESS Discharge: {results.summary['discharge_mwh_y1']:,.2f} MWh")
    print(f"  Power Surplus: {results.summary['surplus_mwh_y1']:,.2f} MWh")

    print("\n## 25-Year Lifetime Totals")
    print(f"  Total Solar Generation: {results.summary['total_solar_gen_mwh_25y']:,.2f} MWh")
    print(f"  Total BESS Discharge: {results.summary['total_discharge_mwh_25y']:,.2f} MWh")

    print("\n## Financial Metrics")
    print(f"  Project IRR: {results.summary['project_irr']*100:.2f}%")
    print(f"  Equity IRR: {results.summary['equity_irr']*100:.2f}%")
    print(f"  NPV: ${results.summary['npv_usd']:,.0f}")
    print(f"  Payback: {results.summary['payback_years']:.1f} years")

    if "dppa_net_revenue_usd" in results.summary:
        print("\n## DPPA (Year 1)")
        print(f"  Net Revenue: ${results.summary['dppa_net_revenue_usd']:,.2f}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    excel_path = Path(__file__).parent.parent / "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"

    if not excel_path.exists():
        excel_path = Path(r"C:\Users\tukum\CascadeProjects\windsurf-solar-bess-pilot\AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx")

    config = PipelineConfig(
        excel_path=excel_path,
        revenue_per_mwh=49.59,
        run_dppa=True,
    )

    results = run_pipeline(config)
    print_summary(results)

    return results


if __name__ == "__main__":
    main()
