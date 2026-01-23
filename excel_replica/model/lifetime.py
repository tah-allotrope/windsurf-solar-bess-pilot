"""Lifetime (multi-year) expansion logic.

Applies PV and BESS degradation over 25-year project life.
Battery augmentation occurs in Year 11 and Year 22.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DegradationSchedule:
    """Degradation factors by year."""
    pv_factor: np.ndarray  # Cumulative PV capacity factor (1.0 = 100%)
    bess_factor: np.ndarray  # Cumulative BESS capacity factor with augmentation


@dataclass
class LifetimeResults:
    """Results from lifetime simulation."""
    yearly: pd.DataFrame
    totals: Dict[str, float] = field(default_factory=dict)


def load_degradation_from_excel(file_path: Path) -> DegradationSchedule:
    """Load degradation schedule from Loss sheet."""
    df = pd.read_excel(file_path, sheet_name="Loss", engine="openpyxl", header=None)

    # Find the data rows (Year 1-25)
    # Column structure: Year, Battery Loss, Battery, PV Loss, PV, Battery wt Replacement
    years = 25
    pv_factor = np.ones(years)
    bess_factor = np.ones(years)

    # Parse the Loss sheet - data starts at row 1 (0-indexed after header)
    for i in range(years):
        row_idx = i + 2  # Skip header rows
        if row_idx < len(df):
            # PV factor is in column 5 (index 4)
            pv_val = df.iloc[row_idx, 4]
            if pd.notna(pv_val) and isinstance(pv_val, (int, float)):
                pv_factor[i] = float(pv_val)

            # BESS factor with replacement is in column 6 (index 5)
            bess_val = df.iloc[row_idx, 5]
            if pd.notna(bess_val) and isinstance(bess_val, (int, float)):
                bess_factor[i] = float(bess_val)

    return DegradationSchedule(pv_factor=pv_factor, bess_factor=bess_factor)


def simulate_lifetime(
    year1_outputs: Dict[str, float],
    degradation: DegradationSchedule,
    project_years: int = 25,
) -> LifetimeResults:
    """Apply degradation across project life.

    Args:
        year1_outputs: Dict with solar_gen_mwh, discharge_mwh, power_surplus_mwh, etc.
        degradation: DegradationSchedule with PV and BESS factors.
        project_years: Number of years (default 25).

    Returns:
        LifetimeResults with yearly DataFrame and totals.
    """
    years = list(range(1, project_years + 1))

    # Year 1 baseline values
    solar_y1 = year1_outputs.get("solar_gen_mwh", 0.0)
    discharge_y1 = year1_outputs.get("discharge_mwh", 0.0)
    surplus_y1 = year1_outputs.get("power_surplus_mwh", 0.0)
    direct_pv_y1 = year1_outputs.get("direct_pv_mwh", 0.0)
    charge_y1 = year1_outputs.get("charge_mwh", 0.0)

    # Apply degradation factors
    solar_gen = solar_y1 * degradation.pv_factor
    discharge = discharge_y1 * degradation.bess_factor
    surplus = surplus_y1 * degradation.pv_factor
    direct_pv = direct_pv_y1 * degradation.pv_factor
    charge = charge_y1 * degradation.bess_factor

    yearly = pd.DataFrame({
        "Year": years,
        "PV_Factor": degradation.pv_factor,
        "BESS_Factor": degradation.bess_factor,
        "SolarGen_MWh": solar_gen,
        "DirectPV_MWh": direct_pv,
        "Charge_MWh": charge,
        "Discharge_MWh": discharge,
        "Surplus_MWh": surplus,
    })

    totals = {
        "total_solar_gen_mwh": float(np.sum(solar_gen)),
        "total_discharge_mwh": float(np.sum(discharge)),
        "total_surplus_mwh": float(np.sum(surplus)),
        "total_direct_pv_mwh": float(np.sum(direct_pv)),
        "total_charge_mwh": float(np.sum(charge)),
    }

    return LifetimeResults(yearly=yearly, totals=totals)
