"""Calc sheet replication engine.

Replicates the hourly Calc sheet formulas from the Excel model.
Key columns:
- SolarGen_kW, Load_kW, DirectPVConsumption_kW
- ChargeEnergy_kWh, DischargeEnergy_kWh, SOC_kWh
- PowerSurplus_kW, GridLoad_kW
- TOU cost columns (Ca_peak, Ca_normal, Ca_offpeak)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CalcConfig:
    """Configuration for Calc engine from Assumption sheet."""
    step_hours: float = 1.0
    bess_capacity_kwh: float = 0.0
    bess_power_kw: float = 0.0
    bess_efficiency: float = 0.90
    min_soc_kwh: float = 0.0
    ca_peak: float = 0.0
    ca_normal: float = 0.0
    ca_offpeak: float = 0.0


@dataclass
class CalcResults:
    """Results from Calc engine."""
    hourly: pd.DataFrame
    outputs: Dict[str, float] = field(default_factory=dict)


def _tou_tariff(period_flag: str, cfg: CalcConfig) -> float:
    """Return tariff rate based on TOU period flag (P/N/O)."""
    if period_flag == "P":
        return cfg.ca_peak
    elif period_flag == "O":
        return cfg.ca_offpeak
    return cfg.ca_normal


def run_calc(
    datetime_series: pd.Series,
    solar_kw: np.ndarray,
    load_kw: np.ndarray,
    period_flags: np.ndarray,
    allow_discharge: np.ndarray,
    cfg: CalcConfig,
) -> CalcResults:
    """Run the hourly Calc formulas and return results.

    Args:
        datetime_series: Hourly datetime index.
        solar_kw: Hourly solar generation (kW).
        load_kw: Hourly load demand (kW).
        period_flags: TOU period flags ('P', 'N', 'O').
        allow_discharge: Boolean array for discharge permission.
        cfg: CalcConfig with BESS and tariff parameters.

    Returns:
        CalcResults with hourly DataFrame and aggregated outputs.
    """
    hours = len(load_kw)

    # Output arrays
    direct_pv_kw = np.zeros(hours)
    charge_kwh = np.zeros(hours)
    discharge_kwh = np.zeros(hours)
    soc_kwh = np.zeros(hours)
    power_surplus_kw = np.zeros(hours)
    grid_load_kw = np.zeros(hours)
    tou_cost = np.zeros(hours)

    current_soc = 0.0

    for h in range(hours):
        # Direct PV consumption = min(solar, load)
        direct_pv_kw[h] = min(solar_kw[h], load_kw[h])

        # Excess solar available for charging
        excess_solar = max(solar_kw[h] - load_kw[h], 0.0)

        # Charging logic
        headroom = cfg.bess_capacity_kwh - current_soc
        max_charge = min(cfg.bess_power_kw * cfg.step_hours, headroom / cfg.bess_efficiency)
        actual_charge = min(excess_solar * cfg.step_hours, max_charge)
        charge_kwh[h] = actual_charge
        current_soc += actual_charge * cfg.bess_efficiency

        # Discharging logic
        net_load = load_kw[h] - solar_kw[h]
        if allow_discharge[h] and net_load > 0 and current_soc > cfg.min_soc_kwh:
            # Excel uses SOC * eff for available, min_soc is only a threshold check
            available_discharge = current_soc * cfg.bess_efficiency
            max_discharge = min(cfg.bess_power_kw * cfg.step_hours, available_discharge)
            actual_discharge = min(net_load * cfg.step_hours, max_discharge)
            discharge_kwh[h] = actual_discharge
            current_soc -= actual_discharge / cfg.bess_efficiency

        current_soc = max(0.0, min(current_soc, cfg.bess_capacity_kwh))
        soc_kwh[h] = current_soc

        # Power surplus (grid export)
        consumed = direct_pv_kw[h] + charge_kwh[h] / cfg.step_hours
        power_surplus_kw[h] = max(solar_kw[h] - consumed, 0.0)

        # Grid load (import)
        supplied = direct_pv_kw[h] + discharge_kwh[h] / cfg.step_hours
        grid_load_kw[h] = max(load_kw[h] - supplied, 0.0)

        # TOU cost
        tariff = _tou_tariff(period_flags[h], cfg)
        tou_cost[h] = grid_load_kw[h] * cfg.step_hours * tariff

    # Build hourly DataFrame
    hourly = pd.DataFrame({
        "DateTime": datetime_series,
        "SolarGen_kW": solar_kw,
        "Load_kW": load_kw,
        "DirectPVConsumption_kW": direct_pv_kw,
        "ChargeEnergy_kWh": charge_kwh,
        "DischargeEnergy_kWh": discharge_kwh,
        "SOC_kWh": soc_kwh,
        "PowerSurplus_kW": power_surplus_kw,
        "GridLoad_kW": grid_load_kw,
        "TimePeriodFlag": period_flags,
        "TOUCost": tou_cost,
    })

    # Aggregated outputs (MWh)
    outputs = {
        "solar_gen_mwh": np.sum(solar_kw) * cfg.step_hours / 1000,
        "direct_pv_mwh": np.sum(direct_pv_kw) * cfg.step_hours / 1000,
        "charge_mwh": np.sum(charge_kwh) / 1000,
        "discharge_mwh": np.sum(discharge_kwh) / 1000,
        "power_surplus_mwh": np.sum(power_surplus_kw) * cfg.step_hours / 1000,
        "grid_load_mwh": np.sum(grid_load_kw) * cfg.step_hours / 1000,
        "total_tou_cost": np.sum(tou_cost),
    }

    return CalcResults(hourly=hourly, outputs=outputs)
