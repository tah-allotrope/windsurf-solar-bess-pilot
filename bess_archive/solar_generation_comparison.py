import pandas as pd
import numpy as np
from final_solar_bess_model import SolarBESSModel, SystemParameters

def extract_excel_solar_generation():
    """Extract solar generation data from Excel Lifetime sheet"""
    df = pd.read_excel('c:\\Users\\tukum\\CascadeProjects\\windsurf-solar-bess-pilot\\AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx', sheet_name='Lifetime')
    
    # Extract solar generation data (row 0, which appears to be SolarGen_MWh)
    solar_gen_row = df.iloc[0]  # First row (index 0)
    solar_gen_values = solar_gen_row[1:26]  # Columns 1-25 for years 1-25
    
    return solar_gen_values.values

def calculate_python_solar_generation():
    """Calculate solar generation using Python model"""
    # Create synthetic hourly data for one year (8760 hours)
    np.random.seed(42)  # For reproducible results
    
    # Generate hourly irradiance data (simplified pattern)
    hours = np.arange(8760)
    day_of_year = hours // 24
    hour_of_day = hours % 24
    
    # Simple irradiance model: higher in summer, during day
    seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak in summer
    daily_factor = np.maximum(0, np.cos(np.pi * (hour_of_day - 12) / 12))  # Daylight hours
    
    # Base irradiance with some randomness
    base_irradiance = 800 * seasonal_factor * daily_factor * (0.9 + 0.2 * np.random.random(8760))
    
    # Generate load profile (simplified)
    base_load = 15000 + 5000 * np.sin(2 * np.pi * hour_of_day / 24) + 2000 * np.random.random(8760)
    
    # Create hourly data DataFrame
    hourly_data = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=8760, freq='H'),
        'Load_kW': base_load,
        'Irradiation_W/m2': base_irradiance,
        'FMP': 0.15 + 0.05 * np.random.random(8760),  # Simplified pricing
        'CFMP': 0.12 + 0.03 * np.random.random(8760)
    })
    
    # Initialize model
    params = SystemParameters()
    model = SolarBESSModel(params)
    
    # Run lifetime analysis
    lifetime_results = model.run_lifetime_analysis_exact_excel(hourly_data)
    
    # Extract annual solar generation
    annual_solar_gen = [r['solar_gen_mwh'] for r in lifetime_results['annual_results']]
    
    return np.array(annual_solar_gen)

def main():
    print("Solar Generation Comparison: Excel vs Python Model")
    print("="*60)
    
    # Extract data from both sources
    excel_solar = extract_excel_solar_generation()
    python_solar = calculate_python_solar_generation()
    
    print(f"\nExcel Lifetime Sheet Solar Generation (MWh):")
    print(f"Year 1: {excel_solar[0]:.2f}")
    print(f"Year 25: {excel_solar[-1]:.2f}")
    print(f"Total 25 years: {excel_solar.sum():.2f}")
    print(f"Average annual: {excel_solar.mean():.2f}")
    
    print(f"\nPython Model Solar Generation (MWh):")
    print(f"Year 1: {python_solar[0]:.2f}")
    print(f"Year 25: {python_solar[-1]:.2f}")
    print(f"Total 25 years: {python_solar.sum():.2f}")
    print(f"Average annual: {python_solar.mean():.2f}")
    
    # Calculate differences
    diff = python_solar - excel_solar
    pct_diff = (diff / excel_solar) * 100
    
    print(f"\nComparison Analysis:")
    print(f"Year 1 difference: {diff[0]:.2f} MWh ({pct_diff[0]:.2f}%)")
    print(f"Year 25 difference: {diff[-1]:.2f} MWh ({pct_diff[-1]:.2f}%)")
    print(f"Total difference: {diff.sum():.2f} MWh ({pct_diff.mean():.2f}% avg)")
    
    # Detailed yearly comparison
    print(f"\nDetailed Year-by-Year Comparison:")
    print("Year | Excel (MWh) | Python (MWh) | Difference (MWh) | % Diff")
    print("-"*70)
    
    for year in range(25):
        print(f"{year+1:4d} | {excel_solar[year]:11.2f} | {python_solar[year]:12.2f} | {diff[year]:14.2f} | {pct_diff[year]:6.2f}")
    
    # Analysis of degradation patterns
    excel_deg = (1 - excel_solar[-1]/excel_solar[0]) * 100
    python_deg = (1 - python_solar[-1]/python_solar[0]) * 100
    
    print(f"\nDegradation Analysis:")
    print(f"Excel degradation (25 years): {excel_deg:.2f}%")
    print(f"Python degradation (25 years): {python_deg:.2f}%")
    print(f"Difference in degradation: {abs(excel_deg - python_deg):.2f}%")
    
    return {
        'excel_solar': excel_solar,
        'python_solar': python_solar,
        'differences': diff,
        'percent_differences': pct_diff
    }

if __name__ == "__main__":
    results = main()
