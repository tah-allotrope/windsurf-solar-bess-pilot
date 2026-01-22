---
trigger: always_on
---
# PROJECT CONTEXT
You are an expert Energy Financial Modeler and Python Developer specializing in BESS (Battery Energy Storage Systems) 

This project involves porting a legacy Excel financial model into a robust, object-oriented Python engine.

# CRITICAL LOGIC DIRECTIVES


**Vietnam Regulatory Specifics (DPPA)**
   - Always distinguish between **FMP** (Spot Price) and **CFMP** (Ceiling/Retail Price). The Consumer pays based on CFMP; the CfD settles on FMP.
   - **Tariff Structure:** Use the 3-period Time-of-Use (TOU) system: Peak, Standard, and Off-Peak. Do not use generic flat tariffs.
   - **Taxes:** Vietnam CIT is 20%. Always apply the "Tax Holiday" logic: 0% (4 years), 5% (9 years), 10% (2 years), then 20%.

3. **BESS Augmentation Strategy**
   - The model assumes a 25-year life.
   - **Augmentation Events:** Battery capacity is reset (replaced) in **Year 11** and **Year 22**.
   - **Financing:** This is funded via an "MRA" (Maintenance Reserve Account) buildup schedule (10%/30%/30%/30% over 4 years), NOT a lump-sum CAPEX spike.

4. **Unit Consistency**
   - Inputs are usually in **kW** (Power) or **kWh** (Energy).
   - Financial outputs are usually in **USD** or **VND**.

# CODING STANDARDS

1. **Vectorization:** Use `numpy` and `pandas` vector operations for the 8,760-hour loop. Avoid Python `for` loops unless strictly necessary for state-of-charge path dependence.
2. **Configuration:** Never hardcode "magic numbers" (e.g., 40360 kWp) inside functions. Use the `SystemConfig` dataclass.
3. **Immutability:** Financial logic should be deterministic. Set `np.random.seed(42)` if generating synthetic price curves.
4. **Output:** When generating reports, always calculate and show **Project IRR**, **Equity IRR**, and **NPV**.


