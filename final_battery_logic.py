import pandas as pd
import numpy as np

def analyze_exact_excel_battery_conditions():
    """Deep dive into Excel battery conditions to understand exact logic"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    
    print("=== EXACT EXCEL BATTERY LOGIC ANALYSIS ===")
    
    # Find the exact conditions for charging and discharging
    print("\n1. BATTERY CHARGING CONDITIONS:")
    
    # Look at hours where battery charges (PVCharged_kWh > 0)
    charge_hours = calc_df[calc_df['PVCharged_kWh'] > 0].head(20)
    
    print("First 20 charging hours with full context:")
    for i, row in charge_hours.iterrows():
        print(f"\nHour {i}: {row['DateTime']}")
        print(f"  Load: {row['Load_kW']:.1f} kW")
        print(f"  Solar: {row['SolarGen_kW']:.1f} kW")
        print(f"  Net Load After Solar: {row['NetLoadAfterSolar_kW']:.1f} kW")
        print(f"  Excess Solar Available: {row['ExcessSolarAvailable_kW']:.1f} kW")
        print(f"  SoC Before: {row['SoC_kWh']:.1f} kWh")
        print(f"  Headroom: {row['Headroom_kWh']:.1f} kWh")
        print(f"  Charge Limit: {row['ChargeLimit_kWh']:.1f} kWh")
        print(f"  PV Charged: {row['PVCharged_kWh']:.1f} kWh")
        print(f"  Grid Charged: {row['GridCharged_kWh']:.1f} kWh")
        print(f"  Total BESS Charged: {row['TotalBESSCharged_kWh']:.1f} kWh")
    
    print("\n2. BATTERY DISCHARGING CONDITIONS:")
    
    # Look at hours where battery discharges (DischargePower_kW > 0)
    discharge_hours = calc_df[calc_df['DischargePower_kW'] > 0].head(20)
    
    print("First 20 discharging hours with full context:")
    for i, row in discharge_hours.iterrows():
        print(f"\nHour {i}: {row['DateTime']}")
        print(f"  Load: {row['Load_kW']:.1f} kW")
        print(f"  Solar: {row['SolarGen_kW']:.1f} kW")
        print(f"  Net Load After Solar: {row['NetLoadAfterSolar_kW']:.1f} kW")
        print(f"  SoC Before: {row['SoC_kWh']:.1f} kWh")
        print(f"  Discharge Flag: {row['DischargeConditionFlag']}")
        print(f"  Discharge Power: {row['DischargePower_kW']:.1f} kW")
        print(f"  Discharge Energy: {row['DischargeEnergy_kWh']:.1f} kWh")
        print(f"  Final Grid Load: {row['GridLoadAfterSolar+BESS_kW']:.1f} kW")
    
    print("\n3. DEMAND TARGET ANALYSIS:")
    
    # Analyze demand target logic
    demand_target = calc_df['DemandTarget_kW'].iloc[0]
    print(f"Demand Target: {demand_target:.1f} kW")
    
    # Check correlation between load > demand_target and discharge
    high_load_hours = calc_df[calc_df['Load_kW'] > demand_target]
    discharge_hours = calc_df[calc_df['DischargePower_kW'] > 0]
    
    print(f"Hours with load > demand_target: {len(high_load_hours)}")
    print(f"Hours with discharge: {len(discharge_hours)}")
    print(f"Hours with both: {len(calc_df[(calc_df['Load_kW'] > demand_target) & (calc_df['DischargePower_kW'] > 0)])}")
    
    # Check if discharge only happens when SoC > 0
    discharge_with_soc = calc_df[(calc_df['DischargePower_kW'] > 0) & (calc_df['SoC_kWh'] > 0)]
    print(f"Discharge hours with SoC > 0: {len(discharge_with_soc)}")
    
    print("\n4. EXCESS SOLAR CALCULATION:")
    
    # Understand how excess solar is calculated
    print("Analyzing excess solar calculation...")
    
    # Look at relationship between SolarGen, Load, and ExcessSolarAvailable
    sample_hours = calc_df.head(48)  # First 2 days
    
    for i, row in sample_hours.iterrows():
        if row['SolarGen_kW'] > 0:  # Only look at daylight hours
            calculated_excess = min(row['SolarGen_kW'], row['Load_kW'])
            actual_excess = row['ExcessSolarAvailable_kW']
            
            if abs(calculated_excess - actual_excess) > 1:  # If difference is significant
                print(f"Hour {i}: Solar={row['SolarGen_kW']:.1f}, Load={row['Load_kW']:.1f}")
                print(f"  Calculated excess: {calculated_excess:.1f}")
                print(f"  Actual excess: {actual_excess:.1f}")
                print(f"  Difference: {abs(calculated_excess - actual_excess):.1f}")
    
    print("\n5. CHARGE LIMIT ANALYSIS:")
    
    # Understand charge limit logic
    print("Charge limit analysis...")
    charge_limit_values = calc_df['ChargeLimit_kWh'].unique()
    print(f"Unique charge limit values: {sorted(charge_limit_values)}")
    
    # Check relationship between headroom and charge limit
    print("Headroom vs Charge Limit analysis:")
    for i, row in calc_df.head(20).iterrows():
        if row['PVCharged_kWh'] > 0:
            print(f"Hour {i}: Headroom={row['Headroom_kWh']:.1f}, ChargeLimit={row['ChargeLimit_kWh']:.1f}, PVCharged={row['PVCharged_kWh']:.1f}")
    
    return calc_df

def create_exact_battery_algorithm():
    """Create the exact battery algorithm based on Excel analysis"""
    
    print("\n=== EXACT BATTERY ALGORITHM ===")
    print("Based on Excel analysis, the battery algorithm works as follows:")
    
    print("\n1. CHARGING LOGIC:")
    print("   - Battery charges when there is excess solar available")
    print("   - Excess solar = min(Solar Generation, Load)")
    print("   - Charge amount = min(Excess Solar, Charge Limit, Headroom)")
    print("   - Charge Limit appears to be fixed at 20,000 kW")
    print("   - Headroom = BESS Capacity - Current SoC")
    print("   - Grid charging appears to be 0 (no grid charging in Excel)")
    
    print("\n2. DISCHARGING LOGIC:")
    print("   - Battery discharges based on DischargeConditionFlag")
    print("   - Flag appears to be set when Load > Demand Target AND SoC > 0")
    print("   - Discharge power = min(Net Load - Demand Target, Grid Capacity, SoC)")
    print("   - Grid Capacity appears to be 20,000 kW (same as charge limit)")
    
    print("\n3. STATE OF CHARGE (SoC) LOGIC:")
    print("   - SoC increases by: Charged Energy × Battery Efficiency")
    print("   - SoC decreases by: Discharged Energy")
    print("   - Battery Efficiency = 0.9745 (from Loss sheet)")
    print("   - SoC is bounded between 0 and 56,100 kWh")
    
    print("\n4. DEMAND TARGET:")
    print("   - Fixed at 17,360.93 kW (from Helper sheet)")
    print("   - Used to determine when battery should discharge")
    
    print("\n5. CONSTRAINTS:")
    print("   - Max charge power: 20,000 kW")
    print("   - Max discharge power: 20,000 kW") 
    print("   - Battery capacity: 56,100 kWh")
    print("   - Round-trip efficiency: 97.45%")

if __name__ == "__main__":
    print("Analyzing exact Excel battery logic...")
    
    calc_df = analyze_exact_excel_battery_conditions()
    create_exact_battery_algorithm()
    
    print("\nAnalysis complete! Use this logic to refine the Python model.")
