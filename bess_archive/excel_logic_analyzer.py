import pandas as pd
import numpy as np

def analyze_excel_battery_logic():
    """Analyze the exact battery dispatch logic from Excel"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    # Load the Calc sheet to understand the logic
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    data_df = pd.read_excel(file_path, sheet_name='Data Input')
    
    print("=== EXCEL BATTERY DISPATCH LOGIC ANALYSIS ===")
    
    # Key columns to analyze
    key_cols = [
        'DateTime', 'Load_kW', 'Irradiation_W/m2', 'SolarGen_kW',
        'PVActive2BESS_kW', 'NetLoadAfterSolar_kW', 'ExcessSolarAvailable_kW',
        'SoC_kWh', 'Delta', 'Headroom_kWh', 'ChargeLimit_kWh',
        'PVCharged_kWh', 'GridCharged_kWh', 'TotalBESSCharged_kWh',
        'DischargeConditionFlag', 'DischargePower_kW', 'DischargeEnergy_kWh',
        'GridLoadAfterSolar+BESS_kW'
    ]
    
    # Analyze first few hours to understand the logic
    sample_hours = calc_df.head(24)  # First day
    
    print("\nFirst 24 hours analysis:")
    for i, row in sample_hours.iterrows():
        if i < 24:  # Only show first day
            print(f"\nHour {i}: {row['DateTime']}")
            print(f"  Load: {row['Load_kW']:.1f} kW")
            print(f"  Solar: {row['SolarGen_kW']:.1f} kW")
            print(f"  Net Load After Solar: {row['NetLoadAfterSolar_kW']:.1f} kW")
            print(f"  Excess Solar: {row['ExcessSolarAvailable_kW']:.1f} kW")
            print(f"  SoC: {row['SoC_kWh']:.1f} kWh")
            print(f"  Headroom: {row['Headroom_kWh']:.1f} kWh")
            print(f"  Charge Limit: {row['ChargeLimit_kWh']:.1f} kWh")
            print(f"  PV Charged: {row['PVCharged_kWh']:.1f} kWh")
            print(f"  Discharge Flag: {row['DischargeConditionFlag']}")
            print(f"  Discharge Power: {row['DischargePower_kW']:.1f} kW")
            print(f"  Final Grid Load: {row['GridLoadAfterSolar+BESS_kW']:.1f} kW")
    
    # Analyze charge/discharge conditions
    print("\n=== CHARGE/DISCHARGE CONDITIONS ===")
    
    # When does battery charge?
    charge_conditions = calc_df[calc_df['PVCharged_kWh'] > 0].head(10)
    print("\nBattery charging conditions (first 10 instances):")
    for i, row in charge_conditions.iterrows():
        print(f"Hour {i}: Solar={row['SolarGen_kW']:.1f}, Load={row['Load_kW']:.1f}, "
              f"Excess={row['ExcessSolarAvailable_kW']:.1f}, SoC={row['SoC_kWh']:.1f}")
    
    # When does battery discharge?
    discharge_conditions = calc_df[calc_df['DischargePower_kW'] > 0].head(10)
    print("\nBattery discharging conditions (first 10 instances):")
    for i, row in discharge_conditions.iterrows():
        print(f"Hour {i}: Load={row['Load_kW']:.1f}, Solar={row['SolarGen_kW']:.1f}, "
              f"NetLoad={row['NetLoadAfterSolar_kW']:.1f}, SoC={row['SoC_kWh']:.1f}, "
              f"Discharge={row['DischargePower_kW']:.1f}")
    
    # Analyze SoC patterns
    print(f"\n=== SOC ANALYSIS ===")
    print(f"Min SoC: {calc_df['SoC_kWh'].min():.1f} kWh")
    print(f"Max SoC: {calc_df['SoC_kWh'].max():.1f} kWh")
    print(f"Avg SoC: {calc_df['SoC_kWh'].mean():.1f} kWh")
    
    # Check demand target logic
    if 'DemandTarget_kW' in calc_df.columns:
        demand_target = calc_df['DemandTarget_kW'].iloc[0]
        print(f"\nDemand Target: {demand_target:.1f} kW")
        
        # Check if discharge happens when load > target
        high_load_discharge = calc_df[
            (calc_df['Load_kW'] > demand_target) & 
            (calc_df['DischargePower_kW'] > 0)
        ]
        print(f"Hours with high load and discharge: {len(high_load_discharge)}")
    
    return calc_df, data_df

def analyze_tariff_structure():
    """Analyze the exact tariff calculation from Excel"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    
    print("\n=== TARIFF STRUCTURE ANALYSIS ===")
    
    # Analyze cost calculations
    print("Cost components analysis:")
    print(f"BAU Grid Energy Expense - Total: ${calc_df['BAU_GridEnergyExpense'].sum():,.0f}")
    print(f"RE Grid Energy Expense - Total: ${calc_df['RE_GridEnergyExpense'].sum():,.0f}")
    
    # Check if costs are calculated using FMP or CFMP
    if 'FMP' in calc_df.columns and 'CFMP' in calc_df.columns:
        print(f"\nFMP range: ${calc_df['FMP'].min():.2f} - ${calc_df['FMP'].max():.2f}")
        print(f"CFMP range: ${calc_df['CFMP'].min():.2f} - ${calc_df['CFMP'].max():.2f}")
        
        # Calculate what costs would be with FMP
        fmp_bau_cost = (calc_df['Load_kW'] * calc_df['FMP']).sum()
        fmp_re_cost = (calc_df['GridLoadAfterSolar+BESS_kW'] * calc_df['FMP']).sum()
        
        print(f"\nCalculated with FMP:")
        print(f"BAU cost: ${fmp_bau_cost:,.0f}")
        print(f"RE cost: ${fmp_re_cost:,.0f}")
        
        # Check if this matches Excel
        excel_bau = calc_df['BAU_GridEnergyExpense'].sum()
        excel_re = calc_df['RE_GridEnergyExpense'].sum()
        
        print(f"\nDifference from Excel:")
        print(f"BAU diff: ${abs(fmp_bau_cost - excel_bau):,.0f}")
        print(f"RE diff: ${abs(fmp_re_cost - excel_re):,.0f}")
    
    return calc_df

def analyze_degradation_factors():
    """Analyze degradation factors from Loss sheet"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    loss_df = pd.read_excel(file_path, sheet_name='Loss')
    
    print("\n=== DEGRADATION FACTORS ANALYSIS ===")
    print("Loss sheet structure:")
    print(loss_df.head(10).to_string())
    
    # Extract degradation factors
    if 'Year' in loss_df.columns:
        years = loss_df['Year'].values
        battery_loss = loss_df['Battery'].values if 'Battery' in loss_df.columns else None
        pv_loss = loss_df['PV'].values if 'PV' in loss_df.columns else None
        
        print(f"\nDegradation by year:")
        for i, year in enumerate(years):
            if i < len(battery_loss) and i < len(pv_loss):
                print(f"Year {year}: Battery={battery_loss[i]:.4f}, PV={pv_loss[i]:.4f}")
    
    return loss_df

if __name__ == "__main__":
    print("Analyzing Excel logic for Python model refinement...")
    
    # Analyze battery dispatch logic
    calc_df, data_df = analyze_excel_battery_logic()
    
    # Analyze tariff structure
    calc_df = analyze_tariff_structure()
    
    # Analyze degradation factors
    loss_df = analyze_degradation_factors()
