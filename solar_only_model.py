import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SystemParameters:
    """System specifications and assumptions from Excel model - SOLAR ONLY"""
    # Solar System
    solar_capacity_kwp: float = 40360.0  # 40.36 MW
    global_horizontal_irradiation: float = 2200.3632608000044  # kWh/m2 p.a.
    performance_ratio: float = 0.8085913562510872
    annual_energy_yield: float = 71808298.62859994  # kWh p.a.
    
    # Financial Parameters
    equity_contribution_m: float = 24.92820311  # Million USD
    leverage_ratio: float = 0.49653412  # Debt/CAPEX
    debt_tenor_years: int = 10
    project_life_years: int = 25
    
    # Degradation factors (from Loss sheet)
    pv_degradation: List[float] = None
    
    def __post_init__(self):
        # Initialize degradation factors if not provided
        if self.pv_degradation is None:
            # From Excel Loss sheet: Year 2: 0.98, Year 3: 0.9745, etc.
            self.pv_degradation = [1.0, 0.98, 0.9745, 0.969, 0.9635, 0.958, 0.9525, 0.947, 0.9415, 0.936]

class SolarGeneration:
    """Solar PV generation calculations"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def calculate_solar_generation(self, irradiance_data: pd.Series) -> pd.Series:
        """Calculate solar generation from irradiance data"""
        # Convert irradiance to power generation
        # Using the performance ratio and system capacity
        solar_gen_kw = irradiance_data * self.params.solar_capacity_kwp * self.params.performance_ratio / 1000
        return solar_gen_kw
    
    def calculate_annual_generation(self, hourly_generation: pd.Series) -> float:
        """Calculate total annual generation in MWh"""
        return hourly_generation.sum() / 1000  # Convert kW to MWh

class FinancialCalculator:
    """Financial calculations for the solar project"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def calculate_capex(self) -> Dict[str, float]:
        """Calculate capital expenditure breakdown"""
        # Estimate CAPEX based on equity and leverage ratio
        total_capex = self.params.equity_contribution_m / (1 - self.params.leverage_ratio)
        
        return {
            'total_capex_m': total_capex,
            'equity_m': self.params.equity_contribution_m,
            'debt_m': total_capex * self.params.leverage_ratio
        }
    
    def calculate_energy_costs(self, 
                              baseline_load_kw: pd.Series,
                              final_load_kw: pd.Series,
                              fmp_prices: pd.Series) -> Dict[str, float]:
        """Calculate energy costs using FMP prices"""
        
        # Calculate costs using FMP prices
        baseline_cost = (baseline_load_kw * fmp_prices).sum()
        actual_cost = (final_load_kw * fmp_prices).sum()
        
        return {
            'baseline_cost_m': baseline_cost / 1e6,
            'actual_cost_m': actual_cost / 1e6,
            'savings_m': (baseline_cost - actual_cost) / 1e6
        }
    
    def calculate_annual_costs(self, 
                             solar_gen_mwh: float,
                             grid_purchase_mwh: float) -> Dict[str, float]:
        """Calculate annual operating costs"""
        
        # O&M costs (typically 1-2% of CAPEX for solar)
        capex = self.calculate_capex()['total_capex_m']
        solar_om = capex * 0.015  # 1.5% of CAPEX
        
        return {
            'solar_om_m': solar_om,
            'total_om_m': solar_om
        }
    
    def calculate_project_cashflows(self, 
                                 annual_results: List[Dict]) -> pd.DataFrame:
        """Calculate project cashflows over lifetime"""
        
        years = range(1, self.params.project_life_years + 1)
        cashflows = []
        
        capex = self.calculate_capex()
        
        for year, results in zip(years, annual_results):
            if year == 1:
                # Initial investment (negative cashflow)
                investment = -capex['total_capex_m']
            else:
                investment = 0
            
            # Annual costs
            costs = self.calculate_annual_costs(
                results['solar_gen_mwh'],
                results['grid_purchase_mwh']
            )
            
            # Revenue (energy savings)
            revenue = results['annual_savings_m']
            
            # Debt service (simplified)
            if year <= self.params.debt_tenor_years:
                debt_service = capex['debt_m'] * 0.08  # 8% interest rate
            else:
                debt_service = 0
            
            # Net cashflow
            net_cf = investment + revenue - costs['total_om_m'] - debt_service
            
            cashflows.append({
                'year': year,
                'investment': investment,
                'revenue': revenue,
                'om_costs': costs['total_om_m'],
                'debt_service': debt_service,
                'net_cashflow': net_cf,
                'cumulative_cf': sum([cf['net_cashflow'] for cf in cashflows] + [net_cf])
            })
        
        return pd.DataFrame(cashflows)
    
    def calculate_financial_metrics(self, cashflows: pd.DataFrame) -> Dict[str, float]:
        """Calculate NPV, IRR, payback period"""
        
        cf_series = cashflows['net_cashflow'].values
        
        # NPV (assuming 8% discount rate)
        discount_rate = 0.08
        npv = sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cf_series))
        
        # IRR
        try:
            irr = np.irr(cf_series)
        except:
            irr = 0.0
        
        # Payback period
        cumulative = 0
        payback_years = 0
        for i, cf in enumerate(cf_series):
            cumulative += cf
            if cumulative > 0:
                payback_years = i + (cumulative - cf) / abs(cf)
                break
        
        return {
            'npv_m': npv,
            'irr': irr,
            'payback_years': payback_years
        }

class SolarOnlyModel:
    """Solar-only financial model"""
    
    def __init__(self, params: SystemParameters = None):
        self.params = params or SystemParameters()
        self.solar = SolarGeneration(self.params)
        self.finance = FinancialCalculator(self.params)
        
    def run_simulation(self, 
                      hourly_data: pd.DataFrame) -> Dict:
        """
        Run complete solar-only simulation
        hourly_data should contain: DateTime, Load_kW, Irradiation_W/m2, FMP, CFMP
        """
        
        # Calculate solar generation
        solar_gen_kw = self.solar.calculate_solar_generation(hourly_data['Irradiation_W/m2'])
        
        # Calculate final grid load (solar reduces grid purchase)
        net_load_after_solar = hourly_data['Load_kW'] - solar_gen_kw
        final_grid_load = net_load_after_solar.clip(lower=0)  # Can't be negative
        
        # Calculate energy costs
        costs = self.finance.calculate_energy_costs(
            hourly_data['Load_kW'],
            final_grid_load,
            hourly_data['FMP']
        )
        
        # Compile results
        results = {
            'solar_gen_mwh': solar_gen_kw.sum() / 1000,
            'grid_purchase_mwh': final_grid_load.sum() / 1000,
            'baseline_grid_cost_m': costs['baseline_cost_m'],
            'actual_grid_cost_m': costs['actual_cost_m'],
            'annual_savings_m': costs['savings_m'],
            'hourly_results': pd.DataFrame({
                'DateTime': hourly_data['DateTime'],
                'Load_kW': hourly_data['Load_kW'],
                'SolarGen_kW': solar_gen_kw,
                'GridLoad_kW': final_grid_load
            })
        }
        
        return results
    
    def run_lifetime_analysis(self, hourly_data: pd.DataFrame) -> Dict:
        """Run 25-year lifetime analysis with degradation"""
        
        annual_results = []
        
        for year in range(1, self.params.project_life_years + 1):
            # Apply degradation factors from Excel Loss sheet
            if year <= len(self.params.pv_degradation):
                solar_degradation = self.params.pv_degradation[year - 1]
            else:
                # Continue with 0.55% annual degradation (matching Excel pattern)
                last_factor = self.params.pv_degradation[-1]
                additional_years = year - len(self.params.pv_degradation)
                solar_degradation = last_factor * (0.9945 ** additional_years)  # 0.55% annual degradation
            
            # Adjust system parameters for this year
            year_params = SystemParameters(
                solar_capacity_kwp=self.params.solar_capacity_kwp * solar_degradation,
                performance_ratio=self.params.performance_ratio,
                equity_contribution_m=self.params.equity_contribution_m,
                leverage_ratio=self.params.leverage_ratio,
                debt_tenor_years=self.params.debt_tenor_years,
                project_life_years=self.params.project_life_years
            )
            
            # Create year-specific model
            year_model = SolarOnlyModel(year_params)
            
            # Run simulation for this year
            year_results = year_model.run_simulation(hourly_data)
            year_results['year'] = year
            annual_results.append(year_results)
        
        # Calculate financial metrics
        cashflows = self.finance.calculate_project_cashflows(annual_results)
        financial_metrics = self.finance.calculate_financial_metrics(cashflows)
        
        return {
            'annual_results': annual_results,
            'cashflows': cashflows,
            'financial_metrics': financial_metrics,
            'summary': {
                'total_solar_gen_mwh': sum(r['solar_gen_mwh'] for r in annual_results),
                'total_savings_m': sum(r['annual_savings_m'] for r in annual_results)
            }
        }

if __name__ == "__main__":
    print("Solar-Only Financial Model")
    print("="*50)
    
    # Example usage
    model = SolarOnlyModel()
    print(f"Solar Capacity: {model.params.solar_capacity_kwp/1000:.2f} MW")
    print(f"Project Life: {model.params.project_life_years} years")
    print(f"Equity: ${model.params.equity_contribution_m:.2f}M")
    print(f"Leverage Ratio: {model.params.leverage_ratio:.2%}")
