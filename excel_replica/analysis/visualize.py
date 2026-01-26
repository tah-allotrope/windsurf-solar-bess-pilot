"""Visualization module for Solar+BESS model analysis.

Generates charts for:
- Sensitivity tornado charts
- Monte Carlo distributions
- Cash flow waterfall
- IRR probability curves
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def plot_tornado_chart(
    sensitivity_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metric: str = "Project_IRR",
) -> None:
    """Generate tornado chart from sensitivity analysis results.
    
    Args:
        sensitivity_df: DataFrame from sensitivity analysis.
        output_path: Path to save chart (optional).
        metric: Metric to plot (Project_IRR, Equity_IRR, NPV_USD).
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    # Calculate ranges for each parameter
    tornado_data = []
    for param in sensitivity_df["Parameter"].unique():
        param_df = sensitivity_df[sensitivity_df["Parameter"] == param]
        base_val = param_df[param_df["Value"] == 1.0][metric].values
        if len(base_val) == 0:
            base_val = param_df[metric].median()
        else:
            base_val = base_val[0]
        
        min_val = param_df[metric].min()
        max_val = param_df[metric].max()
        
        tornado_data.append({
            "Parameter": param,
            "Base": base_val,
            "Low": min_val - base_val,
            "High": max_val - base_val,
            "Range": max_val - min_val,
        })
    
    df = pd.DataFrame(tornado_data).sort_values("Range", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(df))
    
    # Plot bars
    ax.barh(y_pos, df["Low"], align="center", color="red", alpha=0.7, label="Downside")
    ax.barh(y_pos, df["High"], align="center", color="green", alpha=0.7, label="Upside")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Parameter"])
    ax.axvline(x=0, color="black", linewidth=0.5)
    
    if "IRR" in metric:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel(f"{metric} Change")
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
        ax.set_xlabel(f"{metric} Change (USD)")
    
    ax.set_title(f"Sensitivity Analysis - {metric}")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved tornado chart to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_monte_carlo_histogram(
    mc_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metric: str = "Project_IRR",
    bins: int = 30,
) -> None:
    """Generate histogram from Monte Carlo results.
    
    Args:
        mc_df: DataFrame from Monte Carlo simulation.
        output_path: Path to save chart (optional).
        metric: Metric to plot.
        bins: Number of histogram bins.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = mc_df[metric]
    
    # Plot histogram
    n, bins_arr, patches = ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
    
    # Add statistics lines
    mean_val = data.mean()
    p5 = data.quantile(0.05)
    p95 = data.quantile(0.95)
    
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val*100:.2f}%" if "IRR" in metric else f"Mean: ${mean_val/1e6:.1f}M")
    ax.axvline(p5, color="orange", linestyle=":", linewidth=2, label=f"P5: {p5*100:.2f}%" if "IRR" in metric else f"P5: ${p5/1e6:.1f}M")
    ax.axvline(p95, color="green", linestyle=":", linewidth=2, label=f"P95: {p95*100:.2f}%" if "IRR" in metric else f"P95: ${p95/1e6:.1f}M")
    
    if "IRR" in metric:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))
    
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo Distribution - {metric} (n={len(data)})")
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cash_flow_waterfall(
    yearly_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    years_to_show: int = 10,
) -> None:
    """Generate cash flow waterfall chart.
    
    Args:
        yearly_df: DataFrame with yearly financial results.
        output_path: Path to save chart (optional).
        years_to_show: Number of years to display.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    df = yearly_df.head(years_to_show)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df["Revenue_USD"] / 1e6, width, label="Revenue", color="green", alpha=0.7)
    ax.bar(x, -df["OPEX_USD"] / 1e6, width, label="OPEX", color="red", alpha=0.7)
    ax.bar(x + width, df["EBITDA_USD"] / 1e6, width, label="EBITDA", color="blue", alpha=0.7)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("USD (Millions)")
    ax.set_title("Annual Cash Flow Components")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Y{i+1}" for i in range(len(df))])
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved waterfall chart to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_irr_probability_curve(
    mc_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    hurdle_rates: List[float] = None,
) -> None:
    """Generate IRR probability (S-curve) chart.
    
    Args:
        mc_df: DataFrame from Monte Carlo simulation.
        output_path: Path to save chart (optional).
        hurdle_rates: List of hurdle rates to mark.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return
    
    if hurdle_rates is None:
        hurdle_rates = [0.05, 0.08, 0.10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort IRR values and calculate cumulative probability
    irr_sorted = np.sort(mc_df["Project_IRR"])
    prob = np.arange(1, len(irr_sorted) + 1) / len(irr_sorted)
    
    ax.plot(irr_sorted, prob, linewidth=2, color="blue", label="Project IRR")
    
    # Mark hurdle rates
    for hurdle in hurdle_rates:
        prob_exceed = (mc_df["Project_IRR"] > hurdle).mean()
        ax.axvline(hurdle, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            f"{hurdle*100:.0f}%: {prob_exceed*100:.1f}% prob",
            xy=(hurdle, 0.5),
            xytext=(hurdle + 0.01, 0.5),
            fontsize=9,
        )
    
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    ax.set_xlabel("Project IRR")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("IRR Probability Distribution (S-Curve)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved S-curve to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_charts(
    sensitivity_path: Path = None,
    monte_carlo_path: Path = None,
    output_dir: Path = None,
) -> None:
    """Generate all visualization charts.
    
    Args:
        sensitivity_path: Path to sensitivity results CSV.
        monte_carlo_path: Path to Monte Carlo results CSV.
        output_dir: Directory to save charts.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sensitivity_path and sensitivity_path.exists():
        sens_df = pd.read_csv(sensitivity_path)
        plot_tornado_chart(sens_df, output_dir / "tornado_irr.png", "Project_IRR")
        plot_tornado_chart(sens_df, output_dir / "tornado_npv.png", "NPV_USD")
    
    if monte_carlo_path and monte_carlo_path.exists():
        mc_df = pd.read_csv(monte_carlo_path)
        plot_monte_carlo_histogram(mc_df, output_dir / "mc_irr_hist.png", "Project_IRR")
        plot_monte_carlo_histogram(mc_df, output_dir / "mc_npv_hist.png", "NPV_USD")
        plot_irr_probability_curve(mc_df, output_dir / "irr_scurve.png")
    
    print(f"\nAll charts saved to: {output_dir}")


def main():
    """Main entry point."""
    base_path = Path(__file__).parent
    
    sensitivity_path = base_path / "sensitivity_results.csv"
    monte_carlo_path = base_path / "monte_carlo_results.csv"
    output_dir = base_path / "charts"
    
    generate_all_charts(sensitivity_path, monte_carlo_path, output_dir)


if __name__ == "__main__":
    main()
