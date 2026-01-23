"""Financial calculator for cash flow, tax, and IRR/NPV.

Vietnam-specific:
- CIT: 20% standard
- Tax Holiday: 0% (4 years), 5% (9 years), 10% (2 years), then 20%
- MRA (Maintenance Reserve Account) for BESS augmentation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FinancialConfig:
    """Financial model configuration."""
    # CAPEX
    land_cost_usd: float = 0.0
    bop_cost_usd: float = 0.0
    pv_cost_usd: float = 0.0
    bess_cost_usd: float = 0.0

    # OPEX (annual, Year 1)
    om_pv_usd: float = 0.0
    om_bess_usd: float = 0.0
    insurance_pv_usd: float = 0.0
    insurance_bess_usd: float = 0.0
    other_opex_usd: float = 0.0
    land_lease_usd: float = 0.0

    # Escalation rates
    opex_escalation: float = 0.04
    price_escalation: float = 0.05

    # Debt
    leverage_ratio: float = 0.7
    debt_tenor_years: int = 10
    interest_rate: float = 0.08
    target_dscr: float = 1.3

    # Tax
    cit_rate: float = 0.20
    depreciation_years: int = 20

    # Project
    project_years: int = 25
    discount_rate: float = 0.10


@dataclass
class TaxHoliday:
    """Vietnam tax holiday schedule."""
    exempt_years: int = 4      # 0% tax
    reduced_years_5pct: int = 9  # 5% tax
    reduced_years_10pct: int = 2  # 10% tax
    standard_rate: float = 0.20

    def get_rate(self, year: int) -> float:
        """Get applicable tax rate for a given year."""
        if year <= self.exempt_years:
            return 0.0
        elif year <= self.exempt_years + self.reduced_years_5pct:
            return 0.05
        elif year <= self.exempt_years + self.reduced_years_5pct + self.reduced_years_10pct:
            return 0.10
        return self.standard_rate


@dataclass
class MRASchedule:
    """Maintenance Reserve Account buildup for BESS augmentation."""
    augmentation_years: List[int] = field(default_factory=lambda: [11, 22])
    buildup_schedule: List[float] = field(default_factory=lambda: [0.10, 0.30, 0.30, 0.30])

    def get_annual_contribution(self, year: int, augmentation_cost: float) -> float:
        """Calculate MRA contribution for a given year."""
        for aug_year in self.augmentation_years:
            buildup_start = aug_year - len(self.buildup_schedule)
            if buildup_start < year <= aug_year:
                idx = year - buildup_start - 1
                if 0 <= idx < len(self.buildup_schedule):
                    return augmentation_cost * self.buildup_schedule[idx]
        return 0.0


@dataclass
class FinancialResults:
    """Results from financial model."""
    yearly: pd.DataFrame
    project_irr: float
    equity_irr: float
    npv: float
    payback_years: float


def calculate_irr(cash_flows: np.ndarray) -> float:
    """Calculate IRR using numpy_financial or fallback."""
    try:
        import numpy_financial as npf
        return float(npf.irr(cash_flows))
    except ImportError:
        # Fallback: use scipy or simple Newton-Raphson
        from scipy.optimize import brentq

        def npv_func(r):
            years = np.arange(len(cash_flows))
            return np.sum(cash_flows / (1 + r) ** years)

        try:
            return float(brentq(npv_func, -0.99, 10.0))
        except (ValueError, RuntimeError):
            return 0.0


def calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    """Calculate NPV."""
    years = np.arange(len(cash_flows))
    return float(np.sum(cash_flows / (1 + discount_rate) ** years))


def calculate_payback(cumulative_cash_flows: np.ndarray) -> float:
    """Calculate payback period in years."""
    positive_idx = np.where(cumulative_cash_flows >= 0)[0]
    if len(positive_idx) == 0:
        return float(len(cumulative_cash_flows))
    return float(positive_idx[0])


def run_financial_model(
    lifetime_results: pd.DataFrame,
    revenue_per_mwh: float,
    cfg: FinancialConfig,
    tax_holiday: Optional[TaxHoliday] = None,
    mra: Optional[MRASchedule] = None,
) -> FinancialResults:
    """Run financial model and calculate IRR/NPV.

    Args:
        lifetime_results: DataFrame with yearly energy outputs (SolarGen_MWh, etc.)
        revenue_per_mwh: Revenue rate per MWh (USD)
        cfg: FinancialConfig with CAPEX, OPEX, debt parameters
        tax_holiday: Optional TaxHoliday schedule
        mra: Optional MRASchedule for BESS augmentation

    Returns:
        FinancialResults with yearly cash flows and IRR/NPV.
    """
    if tax_holiday is None:
        tax_holiday = TaxHoliday()
    if mra is None:
        mra = MRASchedule()

    years = cfg.project_years
    yearly_data = []

    # Initial CAPEX (Year 0)
    total_capex = cfg.land_cost_usd + cfg.bop_cost_usd + cfg.pv_cost_usd + cfg.bess_cost_usd
    debt_amount = total_capex * cfg.leverage_ratio
    equity_amount = total_capex - debt_amount

    # Depreciation per year
    annual_depreciation = total_capex / cfg.depreciation_years

    # BESS augmentation cost (assume same as initial BESS cost)
    augmentation_cost = cfg.bess_cost_usd

    for year in range(1, years + 1):
        # Revenue (from lifetime results)
        if year <= len(lifetime_results):
            solar_mwh = lifetime_results.iloc[year - 1]["SolarGen_MWh"]
        else:
            solar_mwh = 0.0

        # Apply price escalation
        escalation_factor = (1 + cfg.price_escalation) ** (year - 1)
        revenue = solar_mwh * revenue_per_mwh * escalation_factor

        # OPEX with escalation
        opex_factor = (1 + cfg.opex_escalation) ** (year - 1)
        total_opex = (
            cfg.om_pv_usd + cfg.om_bess_usd +
            cfg.insurance_pv_usd + cfg.insurance_bess_usd +
            cfg.other_opex_usd + cfg.land_lease_usd
        ) * opex_factor

        # MRA contribution
        mra_contribution = mra.get_annual_contribution(year, augmentation_cost)

        # EBITDA
        ebitda = revenue - total_opex - mra_contribution

        # Depreciation
        depreciation = annual_depreciation if year <= cfg.depreciation_years else 0.0

        # EBIT
        ebit = ebitda - depreciation

        # Tax
        tax_rate = tax_holiday.get_rate(year)
        tax = max(ebit * tax_rate, 0.0)

        # Net income
        net_income = ebit - tax

        # Debt service (simplified: equal principal + interest)
        if year <= cfg.debt_tenor_years:
            principal_payment = debt_amount / cfg.debt_tenor_years
            interest_payment = (debt_amount - principal_payment * (year - 1)) * cfg.interest_rate
            debt_service = principal_payment + interest_payment
        else:
            debt_service = 0.0

        # Free cash flow to equity
        fcfe = net_income + depreciation - debt_service

        yearly_data.append({
            "Year": year,
            "Revenue_USD": revenue,
            "OPEX_USD": total_opex,
            "MRA_USD": mra_contribution,
            "EBITDA_USD": ebitda,
            "Depreciation_USD": depreciation,
            "EBIT_USD": ebit,
            "Tax_Rate": tax_rate,
            "Tax_USD": tax,
            "Net_Income_USD": net_income,
            "Debt_Service_USD": debt_service,
            "FCFE_USD": fcfe,
        })

    yearly_df = pd.DataFrame(yearly_data)

    # Project cash flows (unlevered)
    project_cf = np.zeros(years + 1)
    project_cf[0] = -total_capex
    project_cf[1:] = yearly_df["EBITDA_USD"].values - yearly_df["Tax_USD"].values

    # Equity cash flows (levered)
    equity_cf = np.zeros(years + 1)
    equity_cf[0] = -equity_amount
    equity_cf[1:] = yearly_df["FCFE_USD"].values

    # Calculate metrics
    project_irr = calculate_irr(project_cf)
    equity_irr = calculate_irr(equity_cf)
    npv = calculate_npv(project_cf, cfg.discount_rate)

    cumulative_fcfe = np.cumsum(equity_cf)
    payback = calculate_payback(cumulative_fcfe)

    return FinancialResults(
        yearly=yearly_df,
        project_irr=project_irr,
        equity_irr=equity_irr,
        npv=npv,
        payback_years=payback,
    )
