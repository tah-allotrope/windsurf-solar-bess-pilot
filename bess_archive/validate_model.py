import pandas as pd
import numpy as np
from solar_bess_model import SolarBESSModel, SystemParameters
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_excel_data():
    """Load data from Excel file for validation"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    # Load hourly data
    hourly_data = pd.read_excel(file_path, sheet_name='Data Input')
    calc_data = pd.read_excel(file_path, sheet_name='Calc')
    
    # Load financial data
    financial_data = pd.read_excel(file_path, sheet_name='Financial', header=None)
    lifetime_data = pd.read_excel(file_path, sheet_name='Lifetime')
    
    return hourly_data, calc_data, financial_data, lifetime_data

def extract_excel_results(calc_data, financial_data, lifetime_data):
    """Extract key results from Excel for comparison"""
    
    # Extract hourly results from Calc sheet
    excel_hourly = {
        'solar_gen_mwh': calc_data['SolarGen_kW'].sum() / 1000,
        'battery_discharge_mwh': calc_data['DischargePower_kW'].sum() / 1000,
        'grid_purchase_mwh': calc_data['GridLoadAfterSolar+BESS_kW'].sum() / 1000,
        'baseline_cost_m': calc_data['BAU_GridEnergyExpense'].sum() / 1e6,
        'actual_cost_m': calc_data['RE_GridEnergyExpense'].sum() / 1e6,
        'annual_savings_m': (calc_data['BAU_GridEnergyExpense'] - calc_data['RE_GridEnergyExpense']).sum() / 1e6
    }
    
    # Extract lifetime projections
    excel_lifetime = {
        'total_solar_gen_mwh': lifetime_data.loc[lifetime_data['Year'] == 'SolarGen_MWh'].iloc[:, 1:].sum().sum(),
        'total_battery_mwh': lifetime_data.loc[lifetime_data['Year'] == 'BESSToLoad_MWh'].iloc[:, 1:].sum().sum(),
        'total_grid_mwh': lifetime_data.loc[lifetime_data['Year'] == 'GridEnergyUse_MWh'].iloc[:, 1:].sum().sum()
    }
    
    print("Excel Results Summary:")
    print(f"Solar Generation: {excel_hourly['solar_gen_mwh']:.2f} MWh")
    print(f"Battery Discharge: {excel_hourly['battery_discharge_mwh']:.2f} MWh")
    print(f"Grid Purchase: {excel_hourly['grid_purchase_mwh']:.2f} MWh")
    print(f"Baseline Cost: ${excel_hourly['baseline_cost_m']:.2f}M")
    print(f"Actual Cost: ${excel_hourly['actual_cost_m']:.2f}M")
    print(f"Annual Savings: ${excel_hourly['annual_savings_m']:.2f}M")
    
    return excel_hourly, excel_lifetime

def run_python_model(hourly_data):
    """Run Python model and get results"""
    
    # Create model with exact parameters from Excel
    params = SystemParameters(
        solar_capacity_kwp=40360.0,
        bess_capacity_kwh=56100.0,
        grid_capacity_kw=28050.0,
        performance_ratio=0.8085913562510872,
        equity_contribution_m=24.92820311,
        leverage_ratio=0.49653412,
        debt_tenor_years=10,
        project_life_years=25
    )
    
    model = SolarBESSModel(params)
    
    # Run single year simulation
    single_year_results = model.run_simulation(hourly_data)
    
    # Run lifetime analysis
    lifetime_results = model.run_lifetime_analysis(hourly_data)
    
    print("Python Results Summary:")
    print(f"Solar Generation: {single_year_results['solar_gen_mwh']:.2f} MWh")
    print(f"Battery Discharge: {single_year_results['battery_discharge_mwh']:.2f} MWh")
    print(f"Grid Purchase: {single_year_results['grid_purchase_mwh']:.2f} MWh")
    print(f"Baseline Cost: ${single_year_results['baseline_grid_cost_m']:.2f}M")
    print(f"Actual Cost: ${single_year_results['actual_grid_cost_m']:.2f}M")
    print(f"Annual Savings: ${single_year_results['annual_savings_m']:.2f}M")
    
    return single_year_results, lifetime_results

def compare_results(excel_hourly, python_hourly, excel_lifetime, python_lifetime):
    """Compare Excel vs Python results"""
    
    print("MODEL VALIDATION RESULTS")
    print("="*60)
    
    print("\nSINGLE YEAR COMPARISON:")
    print("-" * 40)
    
    metrics = {
        'Solar Generation (MWh)': ('solar_gen_mwh', excel_hourly['solar_gen_mwh'], python_hourly['solar_gen_mwh']),
        'Battery Discharge (MWh)': ('battery_discharge_mwh', excel_hourly['battery_discharge_mwh'], python_hourly['battery_discharge_mwh']),
        'Grid Purchase (MWh)': ('grid_purchase_mwh', excel_hourly['grid_purchase_mwh'], python_hourly['grid_purchase_mwh']),
        'Baseline Cost ($M)': ('baseline_grid_cost_m', excel_hourly['baseline_cost_m'], python_hourly['baseline_grid_cost_m']),
        'Actual Cost ($M)': ('actual_grid_cost_m', excel_hourly['actual_cost_m'], python_hourly['actual_grid_cost_m']),
        'Annual Savings ($M)': ('annual_savings_m', excel_hourly['annual_savings_m'], python_hourly['annual_savings_m'])
    }
    
    for metric_name, (key, excel_val, python_val) in metrics.items():
        diff = python_val - excel_val
        pct_diff = (diff / excel_val * 100) if excel_val != 0 else 0
        status = "✅" if abs(pct_diff) < 5 else "❌" if abs(pct_diff) > 10 else "⚠️"
        
        print(f"{metric_name}:")
        print(f"  Excel: {excel_val:,.2f}")
        print(f"  Python: {python_val:,.2f}")
        print(f"  Difference: {diff:+,.2f} ({pct_diff:+.1f}%) {status}")
        print()
    
    print("\nLIFETIME COMPARISON:")
    print("-" * 40)
    
    lifetime_metrics = {
        'Total Solar Generation (MWh)': ('total_solar_gen_mwh', excel_lifetime['total_solar_gen_mwh'], python_lifetime['summary']['total_solar_gen_mwh']),
        'Total Battery Discharge (MWh)': ('total_battery_mwh', excel_lifetime['total_battery_mwh'], python_lifetime['summary']['total_battery_mwh']),
    }
    
    for metric_name, (key, excel_val, python_val) in lifetime_metrics.items():
        diff = python_val - excel_val
        pct_diff = (diff / excel_val * 100) if excel_val != 0 else 0
        status = "✅" if abs(pct_diff) < 5 else "❌" if abs(pct_diff) > 10 else "⚠️"
        
        print(f"{metric_name}:")
        print(f"  Excel: {excel_val:,.0f}")
        print(f"  Python: {python_val:,.0f}")
        print(f"  Difference: {diff:+,.0f} ({pct_diff:+.1f}%) {status}")
        print()
    
    # Financial metrics comparison
    print("\nFINANCIAL METRICS:")
    print("-" * 40)
    
    python_financial = python_lifetime['financial_metrics']
    print(f"NPV: ${python_financial['npv_m']:.2f}M")
    print(f"IRR: {python_financial['irr']:.1%}")
    print(f"Payback: {python_financial['payback_years']:.1f} years")

def create_validation_plots(python_hourly, calc_data):
    """Create comparison plots"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Solar Generation', 'Battery SoC', 'Grid Load', 'Daily Energy Balance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Solar Generation comparison
    fig.add_trace(
        go.Scatter(x=calc_data['DateTime'], y=calc_data['SolarGen_kW'], 
                  name='Excel Solar', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=python_hourly['hourly_results']['DateTime'], 
                  y=python_hourly['hourly_results']['SolarGen_kW'],
                  name='Python Solar', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Battery SoC
    fig.add_trace(
        go.Scatter(x=calc_data['DateTime'], y=calc_data['SoC_kWh'], 
                  name='Excel SoC', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=python_hourly['hourly_results']['DateTime'], 
                  y=python_hourly['hourly_results']['SoC_kWh'],
                  name='Python SoC', line=dict(color='orange', dash='dash')),
        row=1, col=2
    )
    
    # Grid Load
    fig.add_trace(
        go.Scatter(x=calc_data['DateTime'], y=calc_data['GridLoadAfterSolar+BESS_kW'], 
                  name='Excel Grid Load', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=python_hourly['hourly_results']['DateTime'], 
                  y=python_hourly['hourly_results']['GridLoad_kW'],
                  name='Python Grid Load', line=dict(color='brown', dash='dash')),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Excel vs Python Model Comparison',
        height=600,
        showlegend=True
    )
    
    fig.show()

def main():
    """Main validation function"""
    
    print("Loading Excel data...")
    hourly_data, calc_data, financial_data, lifetime_data = load_excel_data()
    
    print("Extracting Excel results...")
    excel_hourly, excel_lifetime = extract_excel_results(calc_data, financial_data, lifetime_data)
    
    print("Running Python model...")
    python_hourly, python_lifetime = run_python_model(hourly_data)
    
    print("Comparing results...")
    compare_results(excel_hourly, python_hourly, excel_lifetime, python_lifetime)
    
    print("Creating validation plots...")
    create_validation_plots(python_hourly, calc_data)
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
