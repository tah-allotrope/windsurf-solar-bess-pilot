"""DPPA (Direct Power Purchase Agreement) pricing logic.

Vietnam DPPA specifics:
- FMP: Spot/Market Price (Floating Market Price)
- CFMP: Ceiling/Retail Price (Consumer pays based on this)
- CfD: Contract for Difference settlement on FMP
- Strike Price: Fixed price in CfD contract
- PCL: Power Cost Limit
- CDPPAdv: DPPA Advance rate
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DPPAConfig:
    """DPPA pricing configuration."""
    strike_price_vnd: float = 1800.0  # VND/kWh
    pcl_vnd: float = 163.2            # Power Cost Limit VND/kWh
    cdppa_adv: float = 360.14         # DPPA Advance VND/kWh
    k_factor: float = 1.02            # Adjustment factor
    kpp_22kv: float = 1.02726         # Voltage level factor 22kV
    kpp_110kv: float = 1.00855        # Voltage level factor 110kV
    delta: float = 1.0                # Time step factor
    exchange_rate: float = 26000.0    # VND/USD


@dataclass
class DPPAResults:
    """Results from DPPA calculation."""
    hourly: pd.DataFrame
    totals: Dict[str, float]


def calculate_dppa_hourly(
    datetime_series: pd.Series,
    generation_kw: np.ndarray,
    consumption_kw: np.ndarray,
    period_flags: np.ndarray,
    cfg: DPPAConfig,
    voltage_level_kv: int = 22,
) -> DPPAResults:
    """Calculate hourly DPPA settlement.

    DPPA Formula (from Excel):
    - Q_delivered = MIN(generation, contracted_capacity)
    - FMP_revenue = Q_delivered * FMP_price
    - CfD_settlement = Q_delivered * (Strike_Price - FMP_price)
    - Consumer_payment = Q_delivered * CFMP_price
    - Net_revenue = FMP_revenue + CfD_settlement

    Args:
        datetime_series: Hourly datetime index.
        generation_kw: Hourly generation (kW).
        consumption_kw: Hourly consumption/contracted (kW).
        period_flags: TOU period flags ('P', 'N', 'O').
        cfg: DPPAConfig with pricing parameters.
        voltage_level_kv: Connection voltage (22 or 110 kV).

    Returns:
        DPPAResults with hourly DataFrame and totals.
    """
    hours = len(generation_kw)

    # Select Kpp based on voltage level
    kpp = cfg.kpp_22kv if voltage_level_kv == 22 else cfg.kpp_110kv

    # Output arrays
    q_delivered = np.zeros(hours)
    fmp_price = np.zeros(hours)
    cfmp_price = np.zeros(hours)
    fmp_revenue = np.zeros(hours)
    cfd_settlement = np.zeros(hours)
    consumer_payment = np.zeros(hours)
    net_revenue = np.zeros(hours)

    for h in range(hours):
        # Delivered quantity = min(generation, consumption)
        q_delivered[h] = min(generation_kw[h], consumption_kw[h])

        # FMP price varies by TOU period (simplified)
        # In reality, FMP is market-based; here we use PCL as proxy
        fmp_price[h] = cfg.pcl_vnd

        # CFMP price (retail tariff) - would come from Ca_peak/normal/offpeak
        # For DPPA, consumer pays based on CFMP
        cfmp_price[h] = cfg.cdppa_adv

        # FMP revenue (what generator receives from market)
        fmp_revenue[h] = q_delivered[h] * cfg.pcl_vnd / 1000  # VND thousands

        # CfD settlement (difference between strike and FMP)
        cfd_settlement[h] = q_delivered[h] * (cfg.strike_price_vnd - cfg.pcl_vnd) / 1000

        # Consumer payment (based on CFMP)
        consumer_payment[h] = q_delivered[h] * cfg.cdppa_adv * cfg.delta * kpp / 1000

        # Net revenue to generator
        net_revenue[h] = fmp_revenue[h] + cfd_settlement[h]

    # Build hourly DataFrame
    hourly = pd.DataFrame({
        "DateTime": datetime_series,
        "Generation_kW": generation_kw,
        "Consumption_kW": consumption_kw,
        "Q_Delivered_kW": q_delivered,
        "FMP_Price_VND": fmp_price,
        "FMP_Revenue_VND": fmp_revenue,
        "CfD_Settlement_VND": cfd_settlement,
        "Consumer_Payment_VND": consumer_payment,
        "Net_Revenue_VND": net_revenue,
    })

    # Totals (convert to USD)
    totals = {
        "total_delivered_mwh": float(np.sum(q_delivered)) / 1000,
        "total_fmp_revenue_usd": float(np.sum(fmp_revenue)) * 1000 / cfg.exchange_rate,
        "total_cfd_settlement_usd": float(np.sum(cfd_settlement)) * 1000 / cfg.exchange_rate,
        "total_consumer_payment_usd": float(np.sum(consumer_payment)) * 1000 / cfg.exchange_rate,
        "total_net_revenue_usd": float(np.sum(net_revenue)) * 1000 / cfg.exchange_rate,
    }

    return DPPAResults(hourly=hourly, totals=totals)


def load_dppa_config_from_excel(file_path) -> DPPAConfig:
    """Load DPPA configuration from Assumption sheet."""
    import pandas as pd
    from pathlib import Path

    df = pd.read_excel(file_path, sheet_name="Assumption", engine="openpyxl", header=None)

    def get_val(row: int, col: int, default: float) -> float:
        val = df.iloc[row, col]
        if pd.notna(val) and isinstance(val, (int, float)):
            return float(val)
        return default

    return DPPAConfig(
        strike_price_vnd=get_val(38, 16, 1800.0),
        pcl_vnd=get_val(49, 16, 163.2),
        cdppa_adv=get_val(50, 16, 360.14),
        exchange_rate=get_val(8, 10, 26000.0),
    )
