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
    """System specifications and assumptions from Excel model"""
    # Solar System
    solar_capacity_kwp: float = 40360.0  # 40.36 MW
    global_horizontal_irradiation: float = 2200.3632608000044  # kWh/m2 p.a.
    performance_ratio: float = 0.8085913562510872
    annual_energy_yield: float = 71808298.62859994  # kWh p.a.
    
    # Battery System
    bess_capacity_kwh: float = 56100.0  # 56.1 MWh
    grid_capacity_kw: float = 28050.0  # 28.05 MW
    battery_efficiency: float = 0.9745  # Round-trip efficiency
    
    # Financial Parameters
    equity_contribution_m: float = 24.92820311  # Million USD
    leverage_ratio: float = 0.49653412  # Debt/CAPEX
    debt_tenor_years: int = 10
    project_life_years: int = 25
    
    # Tariff Parameters (from Other Input sheet)
    retail_tariff_110kv_2comp: Dict[str, float] = None
    retail_tariff_22kv_2comp: Dict[str, float] = None
    retail_tariff_110kv_1comp: Dict[str, float] = None
    
    def __post_init__(self):
        # Initialize tariff structures if not provided
        if self.retail_tariff_110kv_2comp is None:
            self.retail_tariff_110kv_2comp = {
                'cp_demand': 209459,  # Demand charge
                'ca_normal': 1253     # Energy charge
            }
        if self.retail_tariff_22kv_2comp is None:
            self.retail_tariff_22kv_2comp = {
                'cp_demand': 235414,
                'ca_normal': 1275
            }
        if self.retail_tariff_110kv_1comp is None:
            self.retail_tariff_110kv_1comp = {
                'cp_demand': 0,
                'ca_normal': 1811
            }

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

class BatteryStorage:
    """Battery Energy Storage System operations"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.soc_history = []
        
    def simulate_battery_operation(self, 
                                  excess_solar_kw: pd.Series,
                                  net_load_kw: pd.Series,
                                  demand_target_kw: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Simulate battery charge/discharge operations
        Returns: (discharge_power_kw, charge_power_kw, soc_kwh)
        """
        hours = len(excess_solar_kw)
        discharge_power = np.zeros(hours)
        charge_power = np.zeros(hours)
        soc = np.zeros(hours)
        
        current_soc = 0.0  # Start with empty battery
        
        for hour in range(hours):
            # Determine if we should charge or discharge
            if excess_solar_kw.iloc[hour] > 0 and current_soc < self.params.bess_capacity_kwh:
                # Charge with excess solar
                available_charge = min(excess_solar_kw.iloc[hour], 
                                     self.params.grid_capacity_kw,
                                     (self.params.bess_capacity_kwh - current_soc))
                charge_power[hour] = available_charge
                current_soc += available_charge * self.params.battery_efficiency
                
            elif net_load_kw.iloc[hour] > demand_target_kw and current_soc > 0:
                # Discharge to meet demand target
                needed_discharge = min(net_load_kw.iloc[hour] - demand_target_kw,
                                     self.params.grid_capacity_kw,
                                     current_soc)
                discharge_power[hour] = needed_discharge
                current_soc -= needed_discharge
            
            soc[hour] = current_soc
        
        return pd.Series(discharge_power), pd.Series(charge_power), pd.Series(soc)

class FinancialCalculator:
    """Financial calculations for the solar + BESS project"""
    
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
    
    def calculate_annual_costs(self, 
                             solar_gen_mwh: float,
                             grid_purchase_mwh: float,
                             battery_discharge_mwh: float) -> Dict[str, float]:
        """Calculate annual operating costs and revenues"""
        
        # Grid electricity costs (using retail tariff)
        energy_cost = grid_purchase_mwh * self.params.retail_tariff_110kv_2comp['ca_normal']
        
        # O&M costs (typically 1-2% of CAPEX for solar, higher for battery)
        capex = self.calculate_capex()['total_capex_m']
        solar_om = capex * 0.015 * 0.7  # 1.5% of CAPEX, 70% solar portion
        battery_om = capex * 0.025 * 0.3  # 2.5% of CAPEX, 30% battery portion
        
        return {
            'energy_cost_m': energy_cost / 1e6,
            'solar_om_m': solar_om,
            'battery_om_m': battery_om,
            'total_om_m': solar_om + battery_om
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
                results['grid_purchase_mwh'],
                results['battery_discharge_mwh']
            )
            
            # Revenue (energy savings from solar + battery)
            # Assuming avoided grid costs
            baseline_cost = results['baseline_grid_cost_m']
            actual_cost = costs['energy_cost_m']
            revenue = baseline_cost - actual_cost
            
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

class SolarBESSModel:
    """Main Solar + BESS financial model"""
    
    def __init__(self, params: SystemParameters = None):
        self.params = params or SystemParameters()
        self.solar = SolarGeneration(self.params)
        self.battery = BatteryStorage(self.params)
        self.finance = FinancialCalculator(self.params)
        
    def run_simulation(self, 
                      hourly_data: pd.DataFrame,
                      demand_target_kw: float = 17360.925651931142) -> Dict:
        """
        Run complete simulation
        hourly_data should contain: DateTime, Load_kW, Irradiation_W/m2, FMP, CFMP
        """
        
        # Calculate solar generation
        solar_gen_kw = self.solar.calculate_solar_generation(hourly_data['Irradiation_W/m2'])
        
        # Calculate net load after solar
        net_load_after_solar = hourly_data['Load_kW'] - solar_gen_kw
        net_load_after_solar = net_load_after_solar.clip(lower=0)  # Can't be negative
        
        # Calculate excess solar
        excess_solar = solar_gen_kw.clip(upper=hourly_data['Load_kW'])
        
        # Simulate battery operation
        discharge_kw, charge_kw, soc_kwh = self.battery.simulate_battery_operation(
            excess_solar, net_load_after_solar, demand_target_kw
        )
        
        # Calculate final grid load
        final_grid_load = net_load_after_solar - discharge_kw + charge_kw
        final_grid_load = final_grid_load.clip(lower=0)
        
        # Calculate energy costs
        baseline_grid_cost = (hourly_data['Load_kW'] * hourly_data['FMP']).sum() / 1e6  # Million USD
        actual_grid_cost = (final_grid_load * hourly_data['FMP']).sum() / 1e6
        
        # Compile results
        results = {
            'solar_gen_mwh': solar_gen_kw.sum() / 1000,
            'battery_discharge_mwh': discharge_kw.sum() / 1000,
            'grid_purchase_mwh': final_grid_load.sum() / 1000,
            'baseline_grid_cost_m': baseline_grid_cost,
            'actual_grid_cost_m': actual_grid_cost,
            'annual_savings_m': baseline_grid_cost - actual_grid_cost,
            'hourly_results': pd.DataFrame({
                'DateTime': hourly_data['DateTime'],
                'Load_kW': hourly_data['Load_kW'],
                'SolarGen_kW': solar_gen_kw,
                'BatteryDischarge_kW': discharge_kw,
                'BatteryCharge_kW': charge_kw,
                'SoC_kWh': soc_kwh,
                'GridLoad_kW': final_grid_load
            })
        }
        
        return results
    
    def run_lifetime_analysis(self, hourly_data: pd.DataFrame) -> Dict:
        """Run 25-year lifetime analysis with degradation"""
        
        annual_results = []
        
        for year in range(1, self.params.project_life_years + 1):
            # Apply degradation factors
            solar_degradation = 0.9745 ** (year - 1)  # From Loss sheet
            battery_degradation = 0.9745 ** (year - 1)
            
            # Adjust system parameters for this year
            year_params = SystemParameters(
                solar_capacity_kwp=self.params.solar_capacity_kwp * solar_degradation,
                bess_capacity_kwh=self.params.bess_capacity_kwh * battery_degradation,
                performance_ratio=self.params.performance_ratio,
                equity_contribution_m=self.params.equity_contribution_m,
                leverage_ratio=self.params.leverage_ratio,
                debt_tenor_years=self.params.debt_tenor_years,
                project_life_years=self.params.project_life_years
            )
            
            # Create year-specific model
            year_model = SolarBESSModel(year_params)
            
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
                'total_battery_mwh': sum(r['battery_discharge_mwh'] for r in annual_results),
                'total_savings_m': sum(r['annual_savings_m'] for r in annual_results)
            }
        }

if __name__ == "__main__":
    print("Solar + BESS Financial Model")
    print("="*50)
    
    # Example usage
    model = SolarBESSModel()
    print(f"Solar Capacity: {model.params.solar_capacity_kwp/1000:.2f} MW")
    print(f"BESS Capacity: {model.params.bess_capacity_kwh/1000:.2f} MWh")
    print(f"Project Life: {model.params.project_life_years} years")
    print(f"Equity: ${model.params.equity_contribution_m:.2f}M")
    print(f"Leverage Ratio: {model.params.leverage_ratio:.2%}")
