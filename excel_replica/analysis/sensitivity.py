"""Sensitivity analysis module for Solar+BESS model.

Performs parameter sweeps to understand impact on key financial metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import itertools

import numpy as np
import pandas as pd

from excel_replica.run_pipeline import (
    PipelineConfig,
    PipelineResults,
    run_pipeline,
    load_calc_config,
    load_financial_config,
)
from excel_replica.model.financial import FinancialConfig, run_financial_model
from excel_replica.model.lifetime import load_degradation_from_excel, simulate_lifetime


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    excel_path: Path
    base_revenue_per_mwh: float = 49.59
    
    # Parameter ranges (as multipliers of base value)
    revenue_range: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    capex_range: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    opex_range: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    interest_rate_range: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)
    discount_rate_range: Tuple[float, ...] = (0.08, 0.10, 0.12, 0.14)


@dataclass
class SensitivityResult:
    """Result from a single sensitivity run."""
    parameter: str
    value: float
    project_irr: float
    equity_irr: float
    npv: float
    payback_years: float


def run_sensitivity_analysis(config: SensitivityConfig) -> pd.DataFrame:
    """Run sensitivity analysis on key parameters.
    
    Args:
        config: SensitivityConfig with parameter ranges.
        
    Returns:
        DataFrame with sensitivity results.
    """
    results = []
    
    # Load base configuration
    fin_cfg_base = load_financial_config(config.excel_path)
    degradation = load_degradation_from_excel(config.excel_path)
    
    # Run base pipeline to get Year 1 outputs
    base_config = PipelineConfig(excel_path=config.excel_path, run_dppa=False)
    base_results = run_pipeline(base_config)
    
    year1_outputs = {
        "solar_gen_mwh": base_results.calc.outputs["solar_gen_mwh"],
        "discharge_mwh": base_results.calc.outputs["discharge_mwh"],
        "power_surplus_mwh": base_results.calc.outputs["power_surplus_mwh"],
        "direct_pv_mwh": base_results.calc.outputs["direct_pv_mwh"],
        "charge_mwh": base_results.calc.outputs["charge_mwh"],
    }
    lifetime = simulate_lifetime(year1_outputs, degradation)
    
    print("\n=== Running Sensitivity Analysis ===\n")
    
    # 1. Revenue sensitivity
    print("Revenue sensitivity...")
    for mult in config.revenue_range:
        revenue = config.base_revenue_per_mwh * mult
        fin_results = run_financial_model(lifetime.yearly, revenue, fin_cfg_base)
        results.append(SensitivityResult(
            parameter="Revenue",
            value=mult,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    # 2. CAPEX sensitivity
    print("CAPEX sensitivity...")
    for mult in config.capex_range:
        fin_cfg = FinancialConfig(
            land_cost_usd=fin_cfg_base.land_cost_usd * mult,
            bop_cost_usd=fin_cfg_base.bop_cost_usd * mult,
            pv_cost_usd=fin_cfg_base.pv_cost_usd * mult,
            bess_cost_usd=fin_cfg_base.bess_cost_usd * mult,
            om_pv_usd=fin_cfg_base.om_pv_usd,
            om_bess_usd=fin_cfg_base.om_bess_usd,
            insurance_pv_usd=fin_cfg_base.insurance_pv_usd,
            insurance_bess_usd=fin_cfg_base.insurance_bess_usd,
            other_opex_usd=fin_cfg_base.other_opex_usd,
            land_lease_usd=fin_cfg_base.land_lease_usd,
            leverage_ratio=fin_cfg_base.leverage_ratio,
            debt_tenor_years=fin_cfg_base.debt_tenor_years,
            interest_rate=fin_cfg_base.interest_rate,
            discount_rate=fin_cfg_base.discount_rate,
        )
        fin_results = run_financial_model(lifetime.yearly, config.base_revenue_per_mwh, fin_cfg)
        results.append(SensitivityResult(
            parameter="CAPEX",
            value=mult,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    # 3. OPEX sensitivity
    print("OPEX sensitivity...")
    for mult in config.opex_range:
        fin_cfg = FinancialConfig(
            land_cost_usd=fin_cfg_base.land_cost_usd,
            bop_cost_usd=fin_cfg_base.bop_cost_usd,
            pv_cost_usd=fin_cfg_base.pv_cost_usd,
            bess_cost_usd=fin_cfg_base.bess_cost_usd,
            om_pv_usd=fin_cfg_base.om_pv_usd * mult,
            om_bess_usd=fin_cfg_base.om_bess_usd * mult,
            insurance_pv_usd=fin_cfg_base.insurance_pv_usd * mult,
            insurance_bess_usd=fin_cfg_base.insurance_bess_usd * mult,
            other_opex_usd=fin_cfg_base.other_opex_usd * mult,
            land_lease_usd=fin_cfg_base.land_lease_usd,
            leverage_ratio=fin_cfg_base.leverage_ratio,
            debt_tenor_years=fin_cfg_base.debt_tenor_years,
            interest_rate=fin_cfg_base.interest_rate,
            discount_rate=fin_cfg_base.discount_rate,
        )
        fin_results = run_financial_model(lifetime.yearly, config.base_revenue_per_mwh, fin_cfg)
        results.append(SensitivityResult(
            parameter="OPEX",
            value=mult,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    # 4. Interest rate sensitivity
    print("Interest rate sensitivity...")
    for rate in config.interest_rate_range:
        fin_cfg = FinancialConfig(
            land_cost_usd=fin_cfg_base.land_cost_usd,
            bop_cost_usd=fin_cfg_base.bop_cost_usd,
            pv_cost_usd=fin_cfg_base.pv_cost_usd,
            bess_cost_usd=fin_cfg_base.bess_cost_usd,
            om_pv_usd=fin_cfg_base.om_pv_usd,
            om_bess_usd=fin_cfg_base.om_bess_usd,
            insurance_pv_usd=fin_cfg_base.insurance_pv_usd,
            insurance_bess_usd=fin_cfg_base.insurance_bess_usd,
            other_opex_usd=fin_cfg_base.other_opex_usd,
            land_lease_usd=fin_cfg_base.land_lease_usd,
            leverage_ratio=fin_cfg_base.leverage_ratio,
            debt_tenor_years=fin_cfg_base.debt_tenor_years,
            interest_rate=rate,
            discount_rate=fin_cfg_base.discount_rate,
        )
        fin_results = run_financial_model(lifetime.yearly, config.base_revenue_per_mwh, fin_cfg)
        results.append(SensitivityResult(
            parameter="Interest_Rate",
            value=rate,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    # 5. Discount rate sensitivity
    print("Discount rate sensitivity...")
    for rate in config.discount_rate_range:
        fin_cfg = FinancialConfig(
            land_cost_usd=fin_cfg_base.land_cost_usd,
            bop_cost_usd=fin_cfg_base.bop_cost_usd,
            pv_cost_usd=fin_cfg_base.pv_cost_usd,
            bess_cost_usd=fin_cfg_base.bess_cost_usd,
            om_pv_usd=fin_cfg_base.om_pv_usd,
            om_bess_usd=fin_cfg_base.om_bess_usd,
            insurance_pv_usd=fin_cfg_base.insurance_pv_usd,
            insurance_bess_usd=fin_cfg_base.insurance_bess_usd,
            other_opex_usd=fin_cfg_base.other_opex_usd,
            land_lease_usd=fin_cfg_base.land_lease_usd,
            leverage_ratio=fin_cfg_base.leverage_ratio,
            debt_tenor_years=fin_cfg_base.debt_tenor_years,
            interest_rate=fin_cfg_base.interest_rate,
            discount_rate=rate,
        )
        fin_results = run_financial_model(lifetime.yearly, config.base_revenue_per_mwh, fin_cfg)
        results.append(SensitivityResult(
            parameter="Discount_Rate",
            value=rate,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Parameter": r.parameter,
            "Value": r.value,
            "Project_IRR": r.project_irr,
            "Equity_IRR": r.equity_irr,
            "NPV_USD": r.npv,
            "Payback_Years": r.payback_years,
        }
        for r in results
    ])
    
    return df


def generate_tornado_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate tornado chart data showing IRR sensitivity to each parameter.
    
    Args:
        df: Sensitivity results DataFrame.
        
    Returns:
        DataFrame with tornado chart data.
    """
    tornado_data = []
    
    for param in df["Parameter"].unique():
        param_df = df[df["Parameter"] == param]
        base_irr = param_df[param_df["Value"] == 1.0]["Project_IRR"].values
        if len(base_irr) == 0:
            # For absolute values (interest rate, discount rate), use middle value
            base_irr = param_df["Project_IRR"].median()
        else:
            base_irr = base_irr[0]
        
        min_irr = param_df["Project_IRR"].min()
        max_irr = param_df["Project_IRR"].max()
        
        tornado_data.append({
            "Parameter": param,
            "Base_IRR": base_irr,
            "Min_IRR": min_irr,
            "Max_IRR": max_irr,
            "Range": max_irr - min_irr,
        })
    
    return pd.DataFrame(tornado_data).sort_values("Range", ascending=False)


def print_sensitivity_report(df: pd.DataFrame) -> None:
    """Print formatted sensitivity analysis report."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS REPORT")
    print("=" * 70)
    
    for param in df["Parameter"].unique():
        param_df = df[df["Parameter"] == param]
        print(f"\n## {param}")
        print("-" * 50)
        print(f"{'Value':>10} | {'Project IRR':>12} | {'Equity IRR':>12} | {'NPV (USD)':>15}")
        print("-" * 50)
        for _, row in param_df.iterrows():
            val_str = f"{row['Value']:.0%}" if row['Value'] <= 2 else f"{row['Value']:.2%}"
            print(f"{val_str:>10} | {row['Project_IRR']*100:>11.2f}% | {row['Equity_IRR']*100:>11.2f}% | {row['NPV_USD']:>15,.0f}")
    
    # Tornado summary
    tornado = generate_tornado_data(df)
    print("\n" + "=" * 70)
    print("TORNADO CHART DATA (sorted by IRR sensitivity)")
    print("=" * 70)
    print(f"{'Parameter':>15} | {'Min IRR':>10} | {'Max IRR':>10} | {'Range':>10}")
    print("-" * 50)
    for _, row in tornado.iterrows():
        print(f"{row['Parameter']:>15} | {row['Min_IRR']*100:>9.2f}% | {row['Max_IRR']*100:>9.2f}% | {row['Range']*100:>9.2f}%")


def main():
    """Main entry point."""
    excel_path = Path(r"C:\Users\tukum\CascadeProjects\windsurf-solar-bess-pilot\AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx")
    
    config = SensitivityConfig(
        excel_path=excel_path,
        base_revenue_per_mwh=49.59,
    )
    
    df = run_sensitivity_analysis(config)
    
    # Save results
    output_path = Path(__file__).parent / "sensitivity_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print report
    print_sensitivity_report(df)
    
    return df


if __name__ == "__main__":
    main()
