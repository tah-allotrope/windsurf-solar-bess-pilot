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
    grid_capacity_kw: float = 20000.0  # 20 MW (from Excel analysis)
    battery_efficiency: float = 0.9745  # Round-trip efficiency
    
    # Financial Parameters
    equity_contribution_m: float = 24.92820311  # Million USD
    leverage_ratio: float = 0.49653412  # Debt/CAPEX
    debt_tenor_years: int = 10
    project_life_years: int = 25
    
    # Demand Target (from Helper sheet)
    demand_target_kw: float = 17360.925651931142
    
    # Degradation factors (from Loss sheet)
    battery_degradation: List[float] = None
    pv_degradation: List[float] = None
    
    def __post_init__(self):
        # Initialize degradation factors if not provided
        if self.battery_degradation is None:
            # From Excel Loss sheet: Year 2: 0.9745, Year 3: 0.9375, etc.
            self.battery_degradation = [1.0, 0.9745, 0.9375, 0.9157, 0.89505, 0.87435, 0.85365, 0.83295, 0.81225, 0.79155]
        if self.pv_degradation is None:
            # From Excel Loss sheet
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

class BatteryStorage:
    """Battery Energy Storage System operations - EXACT Excel logic implementation"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
    def simulate_battery_operation_exact_excel(self, 
                                            load_kw: pd.Series,
                                            solar_gen_kw: pd.Series,
                                            demand_target_kw: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Simulate battery operation EXACTLY matching Excel logic
        Returns: (discharge_power_kw, charge_energy_kwh, soc_kwh, discharge_flag)
        """
        hours = len(load_kw)
        discharge_power = np.zeros(hours)
        charge_energy = np.zeros(hours)  # Energy charged in kWh
        soc = np.zeros(hours)
        discharge_flag = np.zeros(hours)
        
        current_soc = 0.0  # Start with empty battery
        
        for hour in range(hours):
            # Calculate net load after solar (can be negative)
            net_load_after_solar = load_kw.iloc[hour] - solar_gen_kw.iloc[hour]
            
            # EXCEL LOGIC: Calculate excess solar available for charging
            # From Excel analysis: ExcessSolarAvailable = min(SolarGen, Load)
            if solar_gen_kw.iloc[hour] > 0 and load_kw.iloc[hour] > 0:
                excess_solar_available = min(solar_gen_kw.iloc[hour], load_kw.iloc[hour])
            else:
                excess_solar_available = 0.0
            
            # Calculate headroom and charge limit (Excel style)
            headroom = self.params.bess_capacity_kwh - current_soc
            charge_limit_kw = min(self.params.grid_capacity_kw, headroom)
            
            # EXCEL LOGIC: Determine discharge condition flag
            # From Excel analysis: Discharge when Load > Demand Target AND SoC > 0
            if load_kw.iloc[hour] > demand_target_kw and current_soc > 0:
                discharge_flag[hour] = 1
            else:
                discharge_flag[hour] = 0
            
            # EXCEL LOGIC: Charging (only with excess solar, no grid charging)
            if excess_solar_available > 0 and headroom > 0:
                # Charge amount = min(Excess Solar, Charge Limit, Headroom)
                available_charge_kw = min(excess_solar_available, charge_limit_kw)
                charge_energy[hour] = available_charge_kw  # Convert to kWh (1 hour period)
                current_soc += charge_energy[hour] * self.params.battery_efficiency
            
            # EXCEL LOGIC: Discharging
            if discharge_flag[hour] == 1 and current_soc > 0:
                # Discharge to meet demand target
                needed_discharge_kw = min(net_load_after_solar - demand_target_kw, 
                                        self.params.grid_capacity_kw,
                                        current_soc)
                discharge_power[hour] = needed_discharge_kw
                current_soc -= needed_discharge_kw
            
            # Ensure SoC stays within bounds
            current_soc = max(0, min(current_soc, self.params.bess_capacity_kwh))
            soc[hour] = current_soc
        
        return pd.Series(discharge_power), pd.Series(charge_energy), pd.Series(soc), pd.Series(discharge_flag)

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
    
    def calculate_energy_costs_exact_excel(self, 
                                          baseline_load_kw: pd.Series,
                                          final_load_kw: pd.Series,
                                          fmp_prices: pd.Series) -> Dict[str, float]:
        """Calculate energy costs exactly matching Excel (using FMP prices)"""
        
        # Calculate costs using FMP prices (as shown in Excel analysis)
        baseline_cost = (baseline_load_kw * fmp_prices).sum()
        actual_cost = (final_load_kw * fmp_prices).sum()
        
        return {
            'baseline_cost_m': baseline_cost / 1e6,
            'actual_cost_m': actual_cost / 1e6,
            'savings_m': (baseline_cost - actual_cost) / 1e6
        }
    
    def calculate_annual_costs(self, 
                             solar_gen_mwh: float,
                             grid_purchase_mwh: float,
                             battery_discharge_mwh: float) -> Dict[str, float]:
        """Calculate annual operating costs and revenues"""
        
        # O&M costs (typically 1-2% of CAPEX for solar, higher for battery)
        capex = self.calculate_capex()['total_capex_m']
        solar_om = capex * 0.015 * 0.7  # 1.5% of CAPEX, 70% solar portion
        battery_om = capex * 0.025 * 0.3  # 2.5% of CAPEX, 30% battery portion
        
        return {
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

class SolarBESSModel:
    """Main Solar + BESS financial model - EXACT Excel replication"""
    
    def __init__(self, params: SystemParameters = None):
        self.params = params or SystemParameters()
        self.solar = SolarGeneration(self.params)
        self.battery = BatteryStorage(self.params)
        self.finance = FinancialCalculator(self.params)
        
    def run_simulation_exact_excel(self, 
                                  hourly_data: pd.DataFrame) -> Dict:
        """
        Run complete simulation EXACTLY matching Excel logic
        hourly_data should contain: DateTime, Load_kW, Irradiation_W/m2, FMP, CFMP
        """
        
        # Calculate solar generation
        solar_gen_kw = self.solar.calculate_solar_generation(hourly_data['Irradiation_W/m2'])
        
        # Simulate battery operation using EXACT Excel logic
        discharge_kw, charge_kwh, soc_kwh, discharge_flag = self.battery.simulate_battery_operation_exact_excel(
            hourly_data['Load_kW'], 
            solar_gen_kw, 
            self.params.demand_target_kw
        )
        
        # Calculate final grid load (Excel style)
        net_load_after_solar = hourly_data['Load_kW'] - solar_gen_kw
        final_grid_load = net_load_after_solar - discharge_kw
        final_grid_load = final_grid_load.clip(lower=0)  # Can't be negative
        
        # Calculate energy costs using Excel method (FMP prices)
        costs = self.finance.calculate_energy_costs_exact_excel(
            hourly_data['Load_kW'],
            final_grid_load,
            hourly_data['FMP']
        )
        
        # Compile results
        results = {
            'solar_gen_mwh': solar_gen_kw.sum() / 1000,
            'battery_discharge_mwh': discharge_kw.sum() / 1000,
            'battery_charge_mwh': charge_kwh.sum() / 1000,
            'grid_purchase_mwh': final_grid_load.sum() / 1000,
            'baseline_grid_cost_m': costs['baseline_cost_m'],
            'actual_grid_cost_m': costs['actual_cost_m'],
            'annual_savings_m': costs['savings_m'],
            'hourly_results': pd.DataFrame({
                'DateTime': hourly_data['DateTime'],
                'Load_kW': hourly_data['Load_kW'],
                'SolarGen_kW': solar_gen_kw,
                'BatteryDischarge_kW': discharge_kw,
                'BatteryCharge_kWh': charge_kwh,
                'SoC_kWh': soc_kwh,
                'DischargeFlag': discharge_flag,
                'GridLoad_kW': final_grid_load
            })
        }
        
        return results
    
    def run_lifetime_analysis_exact_excel(self, hourly_data: pd.DataFrame) -> Dict:
        """Run 25-year lifetime analysis with Excel degradation factors"""
        
        annual_results = []
        
        for year in range(1, self.params.project_life_years + 1):
            # Apply degradation factors from Excel Loss sheet
            if year <= len(self.params.pv_degradation):
                solar_degradation = self.params.pv_degradation[year - 1]
            else:
                # Continue with last known degradation rate
                solar_degradation = self.params.pv_degradation[-1]
                
            if year <= len(self.params.battery_degradation):
                battery_degradation = self.params.battery_degradation[year - 1]
            else:
                battery_degradation = self.params.battery_degradation[-1]
            
            # Adjust system parameters for this year
            year_params = SystemParameters(
                solar_capacity_kwp=self.params.solar_capacity_kwp * solar_degradation,
                bess_capacity_kwh=self.params.bess_capacity_kwh * battery_degradation,
                grid_capacity_kw=self.params.grid_capacity_kw,
                performance_ratio=self.params.performance_ratio,
                equity_contribution_m=self.params.equity_contribution_m,
                leverage_ratio=self.params.leverage_ratio,
                debt_tenor_years=self.params.debt_tenor_years,
                project_life_years=self.params.project_life_years,
                demand_target_kw=self.params.demand_target_kw
            )
            
            # Create year-specific model
            year_model = SolarBESSModel(year_params)
            
            # Run simulation for this year
            year_results = year_model.run_simulation_exact_excel(hourly_data)
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
    print("FINAL Solar + BESS Financial Model (Exact Excel Replication)")
    print("="*70)
    
    # Example usage
    model = SolarBESSModel()
    print(f"Solar Capacity: {model.params.solar_capacity_kwp/1000:.2f} MW")
    print(f"BESS Capacity: {model.params.bess_capacity_kwh/1000:.2f} MWh")
    print(f"Grid Capacity: {model.params.grid_capacity_kw/1000:.2f} MW")
    print(f"Project Life: {model.params.project_life_years} years")
    print(f"Equity: ${model.params.equity_contribution_m:.2f}M")
    print(f"Leverage Ratio: {model.params.leverage_ratio:.2%}")
    print(f"Demand Target: {model.params.demand_target_kw:.0f} kW")
    print(f"Battery Efficiency: {model.params.battery_efficiency:.2%}")
