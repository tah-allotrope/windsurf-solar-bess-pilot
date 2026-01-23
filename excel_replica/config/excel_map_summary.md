# Excel Formula Map Summary (Deep Scan)

Source: `excel_replica/config/excel_map.json`

## DPPA Sheet Formula Patterns (Top 12)
- **7,762** × `IF(Does_model_is_actived?, LET(XLOOKUP(... SimulationData[DateTime] ... CHOOSECOLS(... {4,5,6}), {0,0,0}))`
- **7,762** × `XLOOKUP($A#, Calc!$A$#:$A$####, Calc!$E$#:$E$####, 0, 0, 2)`
- **7,762** × `IF(Does_model_is_actived?=1, XLOOKUP($A#, Calc!$A$#:$A$####, Calc!$AB$#:$AB$####, 0, 0, 2), 0)`
- **7,762** × `F#*C#`
- **7,762** × `F#/(k_factor*Kpp)*Delta`
- **7,762** × `MIN(B#, H#)`
- **7,762** × `I#*D#*Kpp`
- **7,762** × `I#*PCL`
- **7,762** × `I#*CDPPAdv`
- **7,762** × `B#-I#`
- **7,762** × `M# * XLOOKUP(E#, RetailTariff[Voltage Level], CHOOSECOLS(RetailTariff[], IF(DPPA_Connection_Voltage_Level=##, #, #)))`
- **7,762** × `J#+K#+L#+N#`

---

## Calc Sheet (324,119 formulas)
- **7,762** × `I#*StepHours+Q#*Charge_discharge_efficiency^#`
- **7,762** × `F#-Q#/StepHours+V#`
- **7,762** × `$D# * StepHours * SWITCH($E#, "P", Ca_peak, "O", Ca_offpeak, "N", Ca_normal)`
- **7,762** × `$Y#*StepHours * SWITCH($E#, "P", Ca_peak, ...)`
- **7,762** × `$I#*StepHours * SWITCH($E#, "P", Ca_peak, ...)`
- **7,762** × `I#*StepHours+W#-S#`
- **7,762** × `AC#-AD#`
- **7,762** × `AG#-AE#`

## Lifetime Sheet (600 formulas)
- **25** × `Total_Factory_Load/####`
- **4** × `B#/B#`, `C#/C#`, etc. (ratio columns)

## Assumption Sheet (27 formulas)
- **2** × `IF(Does_BESS_System_include_?=1, ..., E#*E#)`
- **1** × `SUM(SimulationData[[#All],[Irradiation_W/m#]])/####`
- **1** × `SUM(SimulationData[[#All],[SimulationProfile_kW]])`
- **1** × `SWITCH(E#, 1,"Energy Arbitrage", 2,"PeakShaving", 3,"Hybrid", ...)`

## Measures Sheet (77 formulas)
- **4** × `G#+G#`
- **4** × `Financial!L#`
- **2** × `G#/G#`, `G#-G#`
- **1** × `SUM(Calc!$D:$D)`, `SUM(Calc!$F:$F)`, `SUM(Calc!$Z:$Z)`, `SUM(Calc!$I:$I)`

## Loss Sheet (72 formulas)
- **17** × `C#-B#` (degradation delta)
- **17** × `E#-D#`
- **17** × `IF(MOD(A#,$F$#)=#,$C$#,F#*(1-B#))` (augmentation reset)

## Financial Sheet (3,068 formulas)
- **55** × `SUM(K#:AJ#)` (row totals)
- **34** × `SUM(K###:AJ###)`
- **25** × `$G$###` (fixed references)
- **21** × `Assumption!K#`
- **9** × `Assumption!Q#`

## Output Sheet (57 formulas)
- **8** × `Assumption!K#`
- **8** × `K#/$K$#` (percentage of total)
- **4** × `E#/E#`, `Financial!G###`
- **3** × `Financial!K#/####^#`, `Financial!L#/####^#`

---

## Notes
- `#` indicates row numbers normalized for pattern matching.
- DPPA is largely hourly, formula-driven columns referencing **Calc** and **SimulationData**.
- Calc sheet is the core hourly engine; Financial aggregates to annual cash flows.
