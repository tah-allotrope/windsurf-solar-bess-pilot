import pandas as pd
import numpy as np

def extract_assumptions():
    """Extract key assumptions from the Assumption sheet"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    # Read assumptions sheet
    df = pd.read_excel(file_path, sheet_name='Assumption', header=None)
    
    # Find rows with actual data (skip empty header rows)
    data_rows = []
    for i, row in df.iterrows():
        if not row.isna().all():
            data_rows.append(row.tolist())
    
    # Extract key parameters
    assumptions = {}
    
    # Look for common solar + storage parameters
    for row in data_rows:
        if len(row) > 1 and isinstance(row[0], str):
            param_name = str(row[0]).strip()
            if any(keyword in param_name.lower() for keyword in ['capacity', 'size', 'mw', 'mwh', 'cost', 'price', 'rate']):
                assumptions[param_name] = row[1] if len(row) > 1 else None
    
    return assumptions

def extract_data_structure():
    """Analyze the main data structure"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    # Read main calculation sheet
    calc_df = pd.read_excel(file_path, sheet_name='Calc')
    
    print("=== DATA STRUCTURE ANALYSIS ===")
    print(f"Calc sheet shape: {calc_df.shape}")
    print(f"Date range: {calc_df['DateTime'].min()} to {calc_df['DateTime'].max()}")
    print(f"Total hours: {len(calc_df)}")
    
    # Key columns analysis
    key_columns = [
        'DateTime', 'Load_kW', 'Irradiation_W/m2', 'SolarGen_kW',
        'SoC_kWh', 'DischargePower_kW', 'GridLoadAfterSolar+BESS_kW',
        'BAU_GridEnergyExpense', 'RE_GridEnergyExpense'
    ]
    
    print("\n=== KEY METRICS SUMMARY ===")
    for col in key_columns:
        if col in calc_df.columns:
            series = calc_df[col].dropna()
            if len(series) > 0:
                print(f"{col}:")
                if col == 'DateTime':
                    print(f"  Start: {series.min()}")
                    print(f"  End: {series.max()}")
                else:
                    print(f"  Min: {series.min():.2f}")
                    print(f"  Max: {series.max():.2f}")
                    print(f"  Mean: {series.mean():.2f}")
                    print(f"  Total: {series.sum():.2f}")
                print()
    
    return calc_df

def extract_financial_summary():
    """Extract financial summary from Financial sheet"""
    file_path = "AUDIT 20251201 40MW Solar ^M BESS Ecoplexus.xlsx"
    
    try:
        fin_df = pd.read_excel(file_path, sheet_name='Financial', header=None)
        print("=== FINANCIAL SHEET STRUCTURE ===")
        print(f"Financial sheet shape: {fin_df.shape}")
        
        # Look for key financial metrics
        financial_metrics = {}
        for i, row in fin_df.iterrows():
            if len(row) > 1 and isinstance(row[0], str):
                metric_name = str(row[0]).strip()
                if any(keyword in metric_name.lower() for keyword in ['npv', 'irr', 'payback', 'revenue', 'cost', 'saving']):
                    financial_metrics[metric_name] = row[1] if len(row) > 1 else None
        
        return financial_metrics
    except Exception as e:
        print(f"Error reading Financial sheet: {e}")
        return {}

if __name__ == "__main__":
    print("Extracting key inputs and structure from Excel model...")
    
    # Extract assumptions
    assumptions = extract_assumptions()
    print("=== KEY ASSUMPTIONS ===")
    for key, value in assumptions.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Extract data structure
    calc_data = extract_data_structure()
    
    print("\n" + "="*50 + "\n")
    
    # Extract financial summary
    financial_metrics = extract_financial_summary()
    print("=== FINANCIAL METRICS ===")
    for key, value in financial_metrics.items():
        print(f"{key}: {value}")
