"""Monte Carlo simulation for probabilistic financial analysis.

Runs multiple simulations with randomized input parameters to generate
probability distributions for key financial metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd

from excel_replica.run_pipeline import load_financial_config
from excel_replica.model.financial import FinancialConfig, run_financial_model
from excel_replica.model.lifetime import load_degradation_from_excel, simulate_lifetime


np.random.seed(42)  # Reproducibility


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    excel_path: Path
    n_simulations: int = 1000
    
    # Base values
    base_revenue_per_mwh: float = 49.59
    
    # Uncertainty ranges (mean, std as % of mean)
    revenue_uncertainty: Tuple[float, float] = (1.0, 0.10)  # ±10%
    capex_uncertainty: Tuple[float, float] = (1.0, 0.08)    # ±8%
    opex_uncertainty: Tuple[float, float] = (1.0, 0.12)     # ±12%
    solar_gen_uncertainty: Tuple[float, float] = (1.0, 0.05)  # ±5%
    
    # Year 1 base outputs (will be loaded from pipeline)
    base_solar_gen_mwh: float = 71808.30
    base_discharge_mwh: float = 8677.22


@dataclass
class MonteCarloResult:
    """Result from a single Monte Carlo run."""
    run_id: int
    revenue_mult: float
    capex_mult: float
    opex_mult: float
    solar_mult: float
    project_irr: float
    equity_irr: float
    npv: float
    payback_years: float


def run_monte_carlo(config: MonteCarloConfig) -> pd.DataFrame:
    """Run Monte Carlo simulation.
    
    Args:
        config: MonteCarloConfig with simulation parameters.
        
    Returns:
        DataFrame with all simulation results.
    """
    # Load base configuration
    fin_cfg_base = load_financial_config(config.excel_path)
    degradation = load_degradation_from_excel(config.excel_path)
    
    results = []
    start_time = time.time()
    
    print(f"\n=== Running {config.n_simulations} Monte Carlo Simulations ===\n")
    
    for i in range(config.n_simulations):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (config.n_simulations - i - 1) / rate
            print(f"  Progress: {i+1}/{config.n_simulations} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
        
        # Sample random multipliers
        revenue_mult = np.random.normal(
            config.revenue_uncertainty[0],
            config.revenue_uncertainty[1]
        )
        capex_mult = np.random.normal(
            config.capex_uncertainty[0],
            config.capex_uncertainty[1]
        )
        opex_mult = np.random.normal(
            config.opex_uncertainty[0],
            config.opex_uncertainty[1]
        )
        solar_mult = np.random.normal(
            config.solar_gen_uncertainty[0],
            config.solar_gen_uncertainty[1]
        )
        
        # Clamp to reasonable bounds
        revenue_mult = max(0.5, min(1.5, revenue_mult))
        capex_mult = max(0.7, min(1.3, capex_mult))
        opex_mult = max(0.6, min(1.4, opex_mult))
        solar_mult = max(0.8, min(1.2, solar_mult))
        
        # Apply multipliers to Year 1 outputs
        year1_outputs = {
            "solar_gen_mwh": config.base_solar_gen_mwh * solar_mult,
            "discharge_mwh": config.base_discharge_mwh * solar_mult,
            "power_surplus_mwh": 1087.26 * solar_mult,
            "direct_pv_mwh": 60000.0 * solar_mult,
            "charge_mwh": 9614.65 * solar_mult,
        }
        
        # Run lifetime simulation
        lifetime = simulate_lifetime(year1_outputs, degradation)
        
        # Apply multipliers to financial config
        fin_cfg = FinancialConfig(
            land_cost_usd=fin_cfg_base.land_cost_usd * capex_mult,
            bop_cost_usd=fin_cfg_base.bop_cost_usd * capex_mult,
            pv_cost_usd=fin_cfg_base.pv_cost_usd * capex_mult,
            bess_cost_usd=fin_cfg_base.bess_cost_usd * capex_mult,
            om_pv_usd=fin_cfg_base.om_pv_usd * opex_mult,
            om_bess_usd=fin_cfg_base.om_bess_usd * opex_mult,
            insurance_pv_usd=fin_cfg_base.insurance_pv_usd * opex_mult,
            insurance_bess_usd=fin_cfg_base.insurance_bess_usd * opex_mult,
            other_opex_usd=fin_cfg_base.other_opex_usd * opex_mult,
            land_lease_usd=fin_cfg_base.land_lease_usd,
            leverage_ratio=fin_cfg_base.leverage_ratio,
            debt_tenor_years=fin_cfg_base.debt_tenor_years,
            interest_rate=fin_cfg_base.interest_rate,
            discount_rate=fin_cfg_base.discount_rate,
        )
        
        # Run financial model
        revenue = config.base_revenue_per_mwh * revenue_mult
        fin_results = run_financial_model(lifetime.yearly, revenue, fin_cfg)
        
        results.append(MonteCarloResult(
            run_id=i,
            revenue_mult=revenue_mult,
            capex_mult=capex_mult,
            opex_mult=opex_mult,
            solar_mult=solar_mult,
            project_irr=fin_results.project_irr,
            equity_irr=fin_results.equity_irr,
            npv=fin_results.npv,
            payback_years=fin_results.payback_years,
        ))
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {config.n_simulations} simulations in {elapsed:.1f}s")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Run_ID": r.run_id,
            "Revenue_Mult": r.revenue_mult,
            "CAPEX_Mult": r.capex_mult,
            "OPEX_Mult": r.opex_mult,
            "Solar_Mult": r.solar_mult,
            "Project_IRR": r.project_irr,
            "Equity_IRR": r.equity_irr,
            "NPV_USD": r.npv,
            "Payback_Years": r.payback_years,
        }
        for r in results
    ])
    
    return df


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for Monte Carlo results."""
    stats = {}
    
    for col in ["Project_IRR", "Equity_IRR", "NPV_USD", "Payback_Years"]:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "p5": df[col].quantile(0.05),
            "p25": df[col].quantile(0.25),
            "p50": df[col].quantile(0.50),
            "p75": df[col].quantile(0.75),
            "p95": df[col].quantile(0.95),
        }
    
    return stats


def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> Dict[str, float]:
    """Calculate Value at Risk (VaR) for NPV."""
    alpha = 1 - confidence
    return {
        f"VaR_{int(confidence*100)}": df["NPV_USD"].quantile(alpha),
        f"CVaR_{int(confidence*100)}": df[df["NPV_USD"] <= df["NPV_USD"].quantile(alpha)]["NPV_USD"].mean(),
    }


def print_monte_carlo_report(df: pd.DataFrame) -> None:
    """Print formatted Monte Carlo analysis report."""
    stats = calculate_statistics(df)
    var = calculate_var(df)
    
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION REPORT")
    print("=" * 70)
    print(f"Number of simulations: {len(df)}")
    
    print("\n## Project IRR Distribution")
    print("-" * 50)
    s = stats["Project_IRR"]
    print(f"  Mean: {s['mean']*100:.2f}%")
    print(f"  Std Dev: {s['std']*100:.2f}%")
    print(f"  Range: {s['min']*100:.2f}% to {s['max']*100:.2f}%")
    print(f"  P5/P50/P95: {s['p5']*100:.2f}% / {s['p50']*100:.2f}% / {s['p95']*100:.2f}%")
    
    print("\n## Equity IRR Distribution")
    print("-" * 50)
    s = stats["Equity_IRR"]
    print(f"  Mean: {s['mean']*100:.2f}%")
    print(f"  Std Dev: {s['std']*100:.2f}%")
    print(f"  Range: {s['min']*100:.2f}% to {s['max']*100:.2f}%")
    print(f"  P5/P50/P95: {s['p5']*100:.2f}% / {s['p50']*100:.2f}% / {s['p95']*100:.2f}%")
    
    print("\n## NPV Distribution")
    print("-" * 50)
    s = stats["NPV_USD"]
    print(f"  Mean: ${s['mean']:,.0f}")
    print(f"  Std Dev: ${s['std']:,.0f}")
    print(f"  Range: ${s['min']:,.0f} to ${s['max']:,.0f}")
    print(f"  P5/P50/P95: ${s['p5']:,.0f} / ${s['p50']:,.0f} / ${s['p95']:,.0f}")
    
    print("\n## Risk Metrics")
    print("-" * 50)
    print(f"  VaR (95%): ${var['VaR_95']:,.0f}")
    print(f"  CVaR (95%): ${var['CVaR_95']:,.0f}")
    
    # Probability of positive NPV
    prob_positive_npv = (df["NPV_USD"] > 0).mean() * 100
    print(f"  Probability of Positive NPV: {prob_positive_npv:.1f}%")
    
    # Probability of IRR > hurdle rates
    for hurdle in [0.05, 0.08, 0.10]:
        prob = (df["Project_IRR"] > hurdle).mean() * 100
        print(f"  Probability of Project IRR > {hurdle*100:.0f}%: {prob:.1f}%")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    excel_path = Path(r"C:\Users\tukum\CascadeProjects\windsurf-solar-bess-pilot\AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx")
    
    config = MonteCarloConfig(
        excel_path=excel_path,
        n_simulations=500,  # Reduced for faster execution
        base_revenue_per_mwh=49.59,
        base_solar_gen_mwh=71808.30,
        base_discharge_mwh=8677.22,
    )
    
    df = run_monte_carlo(config)
    
    # Save results
    output_path = Path(__file__).parent / "monte_carlo_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print report
    print_monte_carlo_report(df)
    
    return df


if __name__ == "__main__":
    main()
