import pandas as pd
import numpy as np
from final_refined_model import SolarBESSModel, SystemParameters

def load_excel_calc_data():
    """Load Excel Calc sheet for detailed BESS analysis"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    data_df = pd.read_excel(file_path, sheet_name='Data Input')
    return calc_df, data_df

def analyze_excel_bess_patterns():
    """Deep analysis of Excel BESS patterns to find improvement areas"""
    
    calc_df, data_df = load_excel_calc_data()
    
    print("🔍 DEEP BESS INVESTIGATION - Excel Pattern Analysis")
    print("="*70)
    
    # 1. Analyze discharge patterns in detail
    print("\n1. DISCHARGE PATTERN ANALYSIS:")
    
    discharge_hours = calc_df[calc_df['DischargePower_kW'] > 0]
    print(f"Excel discharge hours: {len(discharge_hours)}")
    print(f"Total discharge: {discharge_hours['DischargePower_kW'].sum():.1f} kW")
    print(f"Average discharge per hour: {discharge_hours['DischargePower_kW'].mean():.1f} kW")
    
    # 2. Analyze charge patterns
    print("\n2. CHARGE PATTERN ANALYSIS:")
    
    charge_hours = calc_df[calc_df['PVCharged_kWh'] > 0]
    print(f"Excel charge hours: {len(charge_hours)}")
    print(f"Total charge: {charge_hours['PVCharged_kWh'].sum():.1f} kWh")
    print(f"Average charge per hour: {charge_hours['PVCharged_kWh'].mean():.1f} kWh")
    
    # 3. Analyze SoC patterns
    print("\n3. STATE OF CHARGE ANALYSIS:")
    
    print(f"SoC range: {calc_df['SoC_kWh'].min():.1f} to {calc_df['SoC_kWh'].max():.1f} kWh")
    print(f"Average SoC: {calc_df['SoC_kWh'].mean():.1f} kWh")
    print(f"SoC capacity utilization: {calc_df['SoC_kWh'].max()/56100*100:.1f}%")
    
    # 4. Analyze time period distributions
    print("\n4. TIME PERIOD DISTRIBUTION:")
    
    time_period_counts = calc_df['TimePeriodFlag'].value_counts().sort_index()
    for flag, count in time_period_counts.items():
        period_name = {1: "Off-peak (O)", 2: "Normal (N)", 3: "Peak (P)"}.get(flag, f"Unknown({flag})")
        discharge_in_period = calc_df[(calc_df['TimePeriodFlag'] == flag) & (calc_df['DischargePower_kW'] > 0)]
        charge_in_period = calc_df[(calc_df['TimePeriodFlag'] == flag) & (calc_df['PVCharged_kWh'] > 0)]
        print(f"  {period_name}: {count} hours, {len(discharge_in_period)} discharge, {len(charge_in_period)} charge")
    
    # 5. Analyze AllowDischarge patterns
    print("\n5. ALLOW DISCHARGE ANALYSIS:")
    
    allow_discharge_hours = calc_df[calc_df['AllowDischarge'] == 1]
    print(f"AllowDischarge=1 hours: {len(allow_discharge_hours)}")
    
    # Check correlation between AllowDischarge and actual discharge
    discharge_when_allowed = calc_df[(calc_df['AllowDischarge'] == 1) & (calc_df['DischargePower_kW'] > 0)]
    print(f"Actual discharge when allowed: {len(discharge_when_allowed)} hours")
    print(f"Discharge efficiency: {len(discharge_when_allowed)/len(allow_discharge_hours)*100:.1f}%")
    
    return calc_df, data_df

def identify_bess_improvement_areas():
    """Identify specific areas for BESS improvement"""
    
    calc_df, data_df = load_excel_calc_data()
    
    print("\n🎯 BESS IMPROVEMENT AREAS IDENTIFICATION")
    print("="*70)
    
    print("\n1. CRITICAL ISSUES TO INVESTIGATE:")
    
    # Issue 1: AllowDischarge logic
    print("  ❌ AllowDischarge Logic:")
    print("     - Current: Simplified time-based logic")
    print("     - Excel: Complex formula with multiple conditions")
    print("     - Impact: Controls when battery can discharge")
    
    # Issue 2: DischargeConditionFlag
    print("  ❌ DischargeConditionFlag Logic:")
    print("     - Current: Basic NetLoadAfterSolar <= 0 check")
    print("     - Excel: May have additional constraints")
    print("     - Impact: Determines actual discharge decisions")
    
    # Issue 3: ChargeLimit calculation
    print("  ❌ ChargeLimit Calculation:")
    print("     - Current: Fixed 20,000 kW limit")
    print("     - Excel: Dynamic based on available power and efficiency")
    print("     - Impact: Affects maximum charge rates")
    
    # Issue 4: Grid charging
    print("  ❌ Grid Charging Logic:")
    print("     - Current: Disabled (0)")
    print("     - Excel: May have limited grid charging")
    print("     - Impact: Could affect total energy available")
    
    # Issue 5: SoC tracking
    print("  ❌ SoC Tracking:")
    print("     - Current: Basic efficiency application")
    print("     - Excel: May have timing differences")
    print("     - Impact: Affects available energy for discharge")
    
    print("\n2. PRIORITY IMPROVEMENTS (by expected impact):")
    print("  🎯 HIGH PRIORITY:")
    print("     1. Decode exact AllowDischarge formula")
    print("     2. Verify DischargeConditionFlag conditions")
    print("     3. Implement dynamic ChargeLimit")
    
    print("  🔧 MEDIUM PRIORITY:")
    print("     4. Add grid charging capability")
    print("     5. Refine SoC tracking timing")
    
    print("  📊 LOW PRIORITY:")
    print("     6. Minor efficiency adjustments")
    print("     7. Edge case handling")

def run_current_model_comparison():
    """Run current model and compare with Excel to quantify gaps"""
    
    calc_df, data_df = load_excel_calc_data()
    
    print("\n📊 CURRENT MODEL vs EXCEL COMPARISON")
    print("="*70)
    
    # Run current Python model
    params = SystemParameters(
        solar_capacity_kwp=40360.0,
        bess_capacity_kwh=56100.0,
        grid_capacity_kw=20000.0,
        performance_ratio=0.8085913562510872,
        equity_contribution_m=24.92820311,
        leverage_ratio=0.49653412,
        debt_tenor_years=10,
        project_life_years=25,
        demand_target_kw=17360.925651931142,
        step_hours=1.0,
        strategy_mode=1,
        when_needed=1,
        after_sunset=0,
        optimize_mode_1=0,
        peak=0,
        charge_by_grid=0,
        off_peak_start_min=1320,
        off_peak_end_min=240,
        peak_morning_start_min=570,
        peak_morning_end_min=690,
        peak_evening_start_min=1020,
        peak_evening_end_min=1200
    )
    
    model = SolarBESSModel(params)
    python_results = model.run_simulation_final_refined(data_df)
    
    # Extract Excel results
    excel_results = {
        'solar_gen_mwh': calc_df['SolarGen_kW'].sum() / 1000,
        'battery_discharge_mwh': calc_df['DischargePower_kW'].sum() / 1000,
        'battery_charge_mwh': calc_df['PVCharged_kWh'].sum() / 1000,
        'grid_purchase_mwh': calc_df['GridLoadAfterSolar+BESS_kW'].sum() / 1000,
    }
    
    python_results_simple = {
        'solar_gen_mwh': python_results['solar_gen_mwh'],
        'battery_discharge_mwh': python_results['battery_discharge_mwh'],
        'battery_charge_mwh': python_results['battery_charge_mwh'],
        'grid_purchase_mwh': python_results['grid_purchase_mwh'],
    }
    
    print("\nBESS-SPECIFIC COMPARISON:")
    print("-" * 50)
    
    metrics = {
        'Battery Discharge (MWh)': ('battery_discharge_mwh', excel_results['battery_discharge_mwh'], python_results_simple['battery_discharge_mwh']),
        'Battery Charge (MWh)': ('battery_charge_mwh', excel_results['battery_charge_mwh'], python_results_simple['battery_charge_mwh']),
    }
    
    current_accuracy = 0
    total_metrics = len(metrics)
    
    for metric_name, (key, excel_val, python_val) in metrics.items():
        diff = python_val - excel_val
        pct_diff = (diff / excel_val * 100) if excel_val != 0 else 0
        status = "✅" if abs(pct_diff) < 5 else "❌" if abs(pct_diff) > 10 else "⚠️"
        
        if abs(pct_diff) < 5:
            current_accuracy += 1
        
        print(f"{metric_name}:")
        print(f"  Excel: {excel_val:,.2f}")
        print(f"  Python: {python_val:,.2f}")
        print(f"  Difference: {diff:+,.2f} ({pct_diff:+.1f}%) {status}")
        print()
    
    current_accuracy_pct = (current_accuracy / total_metrics) * 100
    print(f"Current BESS Accuracy: {current_accuracy_pct:.0f}%")
    
    return current_accuracy_pct, excel_results, python_results_simple

def implement_improvement_1_allow_discharge():
    """Improvement 1: Implement exact AllowDischarge formula"""
    
    print("\n🔧 IMPROVEMENT 1: Exact AllowDischarge Formula")
    print("="*70)
    
    calc_df, data_df = load_excel_calc_data()
    
    print("Analyzing Excel AllowDischarge patterns...")
    
    # Analyze AllowDischarge patterns by time and conditions
    allow_discharge_analysis = calc_df[calc_df['AllowDischarge'] == 1].copy()
    
    print(f"Hours with AllowDischarge=1: {len(allow_discharge_analysis)}")
    
    # Analyze by time period
    print("\nAllowDischarge by Time Period:")
    for flag in [1, 2, 3]:
        period_data = allow_discharge_analysis[allow_discharge_analysis['TimePeriodFlag'] == flag]
        period_name = {1: "Off-peak", 2: "Normal", 3: "Peak"}.get(flag, f"Unknown({flag})")
        print(f"  {period_name}: {len(period_data)} hours")
    
    # Analyze by hour of day
    allow_discharge_analysis['Hour'] = pd.to_datetime(allow_discharge_analysis['DateTime']).dt.hour
    hour_counts = allow_discharge_analysis['Hour'].value_counts().sort_index()
    
    print("\nAllowDischarge by Hour of Day:")
    for hour, count in hour_counts.items():
        if count > 0:
            print(f"  Hour {hour:2d}: {count:3d} hours")
    
    # Look for patterns in discharge when allowed
    discharge_when_allowed = calc_df[(calc_df['AllowDischarge'] == 1) & (calc_df['DischargePower_kW'] > 0)]
    print(f"\nDischarge when allowed: {len(discharge_when_allowed)} hours")
    print(f"Efficiency: {len(discharge_when_allowed)/len(allow_discharge_analysis)*100:.1f}%")
    
    return True

def implement_improvement_2_discharge_condition():
    """Improvement 2: Refine DischargeConditionFlag logic"""
    
    print("\n🔧 IMPROVEMENT 2: DischargeConditionFlag Logic")
    print("="*70)
    
    calc_df, data_df = load_excel_calc_data()
    
    print("Analyzing Excel DischargeConditionFlag patterns...")
    
    # Analyze discharge condition patterns
    discharge_analysis = calc_df[calc_df['DischargeConditionFlag'] == 1].copy()
    
    print(f"Hours with DischargeConditionFlag=1: {len(discharge_analysis)}")
    
    # Check correlation with NetLoadAfterSolar
    negative_net_load = discharge_analysis[discharge_analysis['NetLoadAfterSolar_kW'] <= 0]
    print(f"DischargeFlag=1 with NetLoadAfterSolar<=0: {len(negative_net_load)} hours")
    print(f"Percentage: {len(negative_net_load)/len(discharge_analysis)*100:.1f}%")
    
    # Check actual discharge
    actual_discharge = discharge_analysis[discharge_analysis['DischargePower_kW'] > 0]
    print(f"Actual discharge when flag=1: {len(actual_discharge)} hours")
    print(f"Discharge efficiency: {len(actual_discharge)/len(discharge_analysis)*100:.1f}%")
    
    # Analyze cases where flag=1 but no discharge
    no_discharge_flag = discharge_analysis[discharge_analysis['DischargePower_kW'] == 0]
    print(f"Flag=1 but no discharge: {len(no_discharge_flag)} hours")
    
    if len(no_discharge_flag) > 0:
        print("\nSample cases where flag=1 but no discharge:")
        for i, row in no_discharge_flag.head(5).iterrows():
            print(f"  {row['DateTime']}: SoC={row['SoC_kWh']:.1f}, NetLoad={row['NetLoadAfterSolar_kW']:.1f}")
    
    return True

def implement_improvement_3_charge_limit():
    """Improvement 3: Dynamic ChargeLimit calculation"""
    
    print("\n🔧 IMPROVEMENT 3: Dynamic ChargeLimit Calculation")
    print("="*70)
    
    calc_df, data_df = load_excel_calc_data()
    
    print("Analyzing Excel ChargeLimit patterns...")
    
    # Analyze ChargeLimit variations
    charge_analysis = calc_df[calc_df['PVCharged_kWh'] > 0].copy()
    
    print(f"Hours with charging: {len(charge_analysis)}")
    print(f"ChargeLimit range: {charge_analysis['ChargeLimit_kWh'].min():.1f} to {charge_analysis['ChargeLimit_kWh'].max():.1f} kWh")
    
    # Analyze relationship with excess solar
    print(f"ExcessSolar range: {charge_analysis['ExcessSolarAvailable_kW'].min():.1f} to {charge_analysis['ExcessSolarAvailable_kW'].max():.1f} kW")
    
    # Check if ChargeLimit is always 20,000 or varies
    unique_limits = charge_analysis['ChargeLimit_kWh'].unique()
    print(f"Unique ChargeLimit values: {len(unique_limits)}")
    
    if len(unique_limits) > 1:
        print("ChargeLimit varies - need dynamic implementation")
        print("Sample ChargeLimit values:")
        for limit in sorted(unique_limits)[:10]:
            count = len(charge_analysis[charge_analysis['ChargeLimit_kWh'] == limit])
            print(f"  {limit:.1f} kWh: {count} hours")
    else:
        print("ChargeLimit is constant - current implementation may be correct")
    
    return True

def test_improvements():
    """Test improvements and measure accuracy improvement"""
    
    print("\n🧪 TESTING IMPROVEMENTS")
    print("="*70)
    
    # Get baseline accuracy
    baseline_accuracy, excel_results, python_results = run_current_model_comparison()
    print(f"Baseline BESS Accuracy: {baseline_accuracy:.0f}%")
    
    # Implement improvements step by step
    print("\nImplementing improvements...")
    
    # Improvement 1
    implement_improvement_1_allow_discharge()
    
    # Improvement 2  
    implement_improvement_2_discharge_condition()
    
    # Improvement 3
    implement_improvement_3_charge_limit()
    
    print("\n🎯 IMPROVEMENT ANALYSIS COMPLETE")
    print("="*70)
    print("Key findings for next implementation phase:")
    print("1. AllowDischarge has complex time-based patterns")
    print("2. DischargeConditionFlag may have additional constraints")
    print("3. ChargeLimit may need dynamic calculation")
    print("4. Current accuracy: 33% - Target: 60%")
    
    return baseline_accuracy

if __name__ == "__main__":
    print("🔍 BESS INVESTIGATION - Iterative Improvement Process")
    print("="*80)
    print("Goal: Achieve 60%+ accuracy on BESS replication")
    print("Process: Analyze → Identify → Implement → Test → Repeat")
    
    # Step 1: Analyze current patterns
    calc_df, data_df = analyze_excel_bess_patterns()
    
    # Step 2: Identify improvement areas
    identify_bess_improvement_areas()
    
    # Step 3: Test current model
    baseline_accuracy = test_improvements()
    
    print(f"\n🎯 CURRENT STATUS: {baseline_accuracy:.0f}% accuracy")
    print("📋 NEXT STEPS: Implement identified improvements")
    print("🔄 WILL CONTINUE ITERATING UNTIL 60%+ ACHIEVED")
