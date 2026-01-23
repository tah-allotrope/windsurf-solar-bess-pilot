# Active Context (The State)

## Current Focus
- Core model pipeline complete: Calc → Lifetime → Financial

## Recent Changes
- **Regression suite PASS** (0.00% error on solar_gen, discharge, power_surplus)
- Implemented `excel_replica/model/lifetime.py` with degradation schedule loader and 25-year projection
- Implemented `excel_replica/model/financial.py` with:
  - Vietnam tax holiday (0%→5%→10%→20%)
  - MRA schedule for BESS augmentation (Year 11, 22)
  - IRR/NPV/Payback calculations
- All core modules now functional and tested

## Validated Parameters
- Usable capacity: 56,100 kWh (66,000 × 0.85 DoD)
- Power: 20,000 kW
- Efficiency: 0.95 (half-cycle)
- Min SOC threshold: 215 kWh
- Augmentation years: 11, 22

## Module Status
| Module | Status | File |
|--------|--------|------|
| Calc Engine | ✅ PASS | `excel_replica/model/calc_engine.py` |
| Lifetime | ✅ Done | `excel_replica/model/lifetime.py` |
| Financial | ✅ Done | `excel_replica/model/financial.py` |
| Regression | ✅ PASS | `excel_replica/validation/regression_suite.py` |

## Next Steps
- Load actual revenue/tariff rates from Assumption sheet
- Validate Financial outputs against Excel Financial sheet
- Implement DPPA pricing logic (FMP vs CFMP)
