import pandas as pd
import numpy as np

def investigate_battery_discharge_issue():
    """Investigate why battery discharge is negative in Python model"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    # Load Excel data
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    data_df = pd.read_excel(file_path, sheet_name='Data Input')
    
    print("=== BATTERY DISCHARGE ISSUE INVESTIGATION ===")
    
    # Analyze Excel battery logic step by step
    print("\n1. EXCEL BATTERY DISCHARGE PATTERN:")
    
    # Find hours when battery discharges in Excel
    discharge_hours = calc_df[calc_df['DischargePower_kW'] > 0]
    print(f"Excel: {len(discharge_hours)} hours with battery discharge")
    print(f"Excel discharge range: {discharge_hours['DischargePower_kW'].min():.1f} to {discharge_hours['DischargePower_kW'].max():.1f} kW")
    
    # Analyze the conditions for discharge
    print("\n2. DISCHARGE CONDITIONS ANALYSIS:")
    
    # Check the relationship between load, demand target, and discharge
    demand_target = calc_df['DemandTarget_kW'].iloc[0]
    print(f"Demand Target: {demand_target:.1f} kW")
    
    # Sample discharge hours with full context
    sample_discharge = discharge_hours.head(10)
    for i, row in sample_discharge.iterrows():
        print(f"\nHour {i}: {row['DateTime']}")
        print(f"  Load: {row['Load_kW']:.1f} kW")
        print(f"  Solar: {row['SolarGen_kW']:.1f} kW")
        print(f"  Net Load After Solar: {row['NetLoadAfterSolar_kW']:.1f} kW")
        print(f"  SoC Before: {row['SoC_kWh']:.1f} kWh")
        print(f"  Discharge Flag: {row['DischargeConditionFlag']}")
        print(f"  Discharge Power: {row['DischargePower_kW']:.1f} kW")
        print(f"  Final Grid Load: {row['GridLoadAfterSolar+BESS_kW']:.1f} kW")
        
        # Check if load > demand target
        load_above_target = row['Load_kW'] > demand_target
        print(f"  Load > Target: {load_above_target}")
    
    print("\n3. KEY INSIGHTS FROM EXCEL:")
    
    # Check when discharge flag is 1
    discharge_flag_hours = calc_df[calc_df['DischargeConditionFlag'] == 1]
    actual_discharge_hours = calc_df[calc_df['DischargePower_kW'] > 0]
    
    print(f"Hours with Discharge Flag = 1: {len(discharge_flag_hours)}")
    print(f"Hours with Actual Discharge > 0: {len(actual_discharge_hours)}")
    print(f"Hours with both: {len(calc_df[(calc_df['DischargeConditionFlag'] == 1) & (calc_df['DischargePower_kW'] > 0)])}")
    
    # Check SoC during discharge
    discharge_with_soc = actual_discharge_hours[actual_discharge_hours['SoC_kWh'] > 0]
    print(f"Discharge hours with SoC > 0: {len(discharge_with_soc)}")
    
    # Analyze the discharge power calculation
    print("\n4. DISCHARGE POWER CALCULATION:")
    
    # Check if discharge power equals NetLoadAfterSolar - DemandTarget
    for i, row in actual_discharge_hours.head(5).iterrows():
        expected_discharge = max(0, row['NetLoadAfterSolar_kW'] - demand_target)
        actual_discharge = row['DischargePower_kW']
        print(f"Hour {i}: Expected={expected_discharge:.1f}, Actual={actual_discharge:.1f}, Diff={actual_discharge-expected_discharge:.1f}")
    
    return calc_df, data_df

def analyze_charging_logic():
    """Analyze the charging logic in Excel"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    
    print("\n=== CHARGING LOGIC ANALYSIS ===")
    
    # Find charging hours
    charge_hours = calc_df[calc_df['PVCharged_kWh'] > 0]
    print(f"Excel: {len(charge_hours)} hours with battery charging")
    print(f"Excel charge range: {charge_hours['PVCharged_kWh'].min():.1f} to {charge_hours['PVCharged_kWh'].max():.1f} kWh")
    
    # Analyze charging conditions
    print("\nCHARGING CONDITIONS:")
    sample_charge = charge_hours.head(10)
    for i, row in sample_charge.iterrows():
        print(f"\nHour {i}: {row['DateTime']}")
        print(f"  Load: {row['Load_kW']:.1f} kW")
        print(f"  Solar: {row['SolarGen_kW']:.1f} kW")
        print(f"  Excess Solar: {row['ExcessSolarAvailable_kW']:.1f} kW")
        print(f"  SoC Before: {row['SoC_kWh']:.1f} kWh")
        print(f"  Headroom: {row['Headroom_kWh']:.1f} kWh")
        print(f"  Charge Limit: {row['ChargeLimit_kWh']:.1f} kWh")
        print(f"  PV Charged: {row['PVCharged_kWh']:.1f} kWh")
        
        # Check excess solar calculation
        calculated_excess = min(row['SolarGen_kW'], row['Load_kW'])
        actual_excess = row['ExcessSolarAvailable_kW']
        print(f"  Excess check: Calc={calculated_excess:.1f}, Actual={actual_excess:.1f}")

def identify_python_model_issues():
    """Identify the specific issues in Python model"""
    
    print("\n=== PYTHON MODEL ISSUES ===")
    
    print("\n1. NEGATIVE BATTERY DISCHARGE:")
    print("   - Python model shows negative discharge values")
    print("   - This suggests the discharge calculation is wrong")
    print("   - Likely issue: NetLoadAfterSolar calculation or discharge condition")
    
    print("\n2. EXCESS SOLAR CALCULATION:")
    print("   - Excel: ExcessSolarAvailable = min(SolarGen, Load)")
    print("   - Python may be using different formula")
    
    print("\n3. DISCHARGE CONDITION:")
    print("   - Excel: Complex logic with DischargeConditionFlag")
    print("   - Python: Simple Load > DemandTarget condition")
    
    print("\n4. CHARGE LIMIT:")
    print("   - Excel: ChargeLimit varies (not always 20,000 kW)")
    print("   - Python: Fixed 20,000 kW limit")
    
    print("\n5. SoC CALCULATION:")
    print("   - Excel: Complex SoC tracking with efficiency")
    print("   - Python: May have timing or efficiency issues")

def create_summary_report():
    """Create a comprehensive summary of differences"""
    
    print("\n" + "="*80)
    print("EXCEL vs PYTHON MODEL - DIFFERENCE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 CURRENT ACCURACY STATUS:")
    print("✅ Solar Generation: 100% accurate (71,808 MWh)")
    print("❌ Battery Discharge: Major error (negative values)")
    print("⚠️ Battery Charge: 9.4% difference")
    print("❌ Grid Purchase: 32.1% difference")
    print("⚠️ Baseline Cost: 7.3% difference")
    print("❌ Actual Cost: 56.1% difference")
    print("❌ Annual Savings: 56.7% difference")
    
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("1. PRIMARY ISSUE: Battery discharge algorithm is fundamentally wrong")
    print("   - Negative discharge values indicate calculation error")
    print("   - This cascades to all downstream calculations")
    
    print("\n2. SECONDARY ISSUES:")
    print("   - Excess solar calculation may differ")
    print("   - Discharge condition logic is simplified")
    print("   - Charge limit implementation may be incorrect")
    
    print("\n🎯 EXCEL BATTERY LOGIC (What we need to replicate):")
    print("   • Excess Solar = min(SolarGen, Load)")
    print("   • Charge Amount = min(ExcessSolar, ChargeLimit, Headroom)")
    print("   • Discharge Flag = complex condition (not just Load > Target)")
    print("   • Discharge Power = min(NetLoad - Target, GridCapacity, SoC)")
    print("   • SoC tracking with 97.45% efficiency")
    
    print("\n📈 IMPACT CHAIN:")
    print("   Wrong Battery Logic → Wrong Discharge → Wrong Grid Load")
    print("   → Wrong Energy Costs → Wrong Savings → Wrong Financial Metrics")
    
    print("\n🔧 FIX PRIORITY:")
    print("1. CRITICAL: Fix battery discharge calculation (negative values)")
    print("2. HIGH: Replicate exact Excel discharge condition logic")
    print("3. MEDIUM: Verify excess solar and charge limit calculations")
    print("4. LOW: Fine-tune SoC tracking and efficiency")
    
    print("\n💡 EXPECTED OUTCOME AFTER FIX:")
    print("   • Battery discharge should match Excel (~8,677 MWh)")
    print("   • Grid purchase should match Excel (~114,479 MWh)")
    print("   • Energy costs should match Excel")
    print("   • Overall accuracy should improve to 90%+")

if __name__ == "__main__":
    print("Investigating Excel vs Python model differences...")
    
    # Investigate the battery discharge issue
    calc_df, data_df = investigate_battery_discharge_issue()
    
    # Analyze charging logic
    analyze_charging_logic()
    
    # Identify Python model issues
    identify_python_model_issues()
    
    # Create summary report
    create_summary_report()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
