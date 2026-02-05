# Table 2 (alternate): Key Model Parameters (units-focused, compact)

This is an alternate, manuscript-friendly version of Table 2 that emphasizes **biophysical interpretation and units**.

- Model folder: `RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET`
- Evolutionary parameter space (min/max ranges): `RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/evol_params.py`
- Network implementation (where parameters are applied): `RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/netParams.py`

> Notes on units: In this model, positions are specified in culture-scale coordinates (on the order of 0–4000 in X and 0–2100 in Y), which are most naturally interpreted as **µm**. As implemented, connection delays are computed as `delay = dist_3D / propVelocity`, so `propVelocity` must be in **µm/ms** when `dist_3D` is in µm.

| Parameter | Range / value |
|---|---:|
| Propagation velocity<br>— <small>Conduction velocity used to map distance → delay, with delay computed as `dist_3D / propVelocity` (µm/ms, assuming `dist_3D` in µm and delay in ms)</small> | 0.1 → 0.3 |
| Connection distance length constant<br>— <small>Length constant for distance-dependent connection probability in `exp(-dist_3D / probLengthConst)` (µm)</small> | 1 → 5000 |
| E→E connection probability scale<br>— <small>Baseline multiplicative scale for excitatory→excitatory connection probability (dimensionless)</small> | 0 → 1 |
| E→I connection probability scale<br>— <small>Baseline multiplicative scale for excitatory→inhibitory connection probability (dimensionless)</small> | 0 → 1 |
| I→E connection probability scale<br>— <small>Baseline multiplicative scale for inhibitory→excitatory connection probability (dimensionless)</small> | 0 → 1 |
| I→I connection probability scale<br>— <small>Baseline multiplicative scale for inhibitory→inhibitory connection probability (dimensionless)</small> | 0 → 1 |
| E→E synaptic weight<br>— <small>Synaptic weight scalar for excitatory→excitatory connections (NetPyNE/NEURON model weight units)</small> | 0 → 1000 |
| E→I synaptic weight<br>— <small>Synaptic weight scalar for excitatory→inhibitory connections (NetPyNE/NEURON model weight units)</small> | 0 → 1000 |
| I→E synaptic weight<br>— <small>Synaptic weight scalar for inhibitory→excitatory connections (NetPyNE/NEURON model weight units)</small> | 0 → 1000 |
| I→I synaptic weight<br>— <small>Synaptic weight scalar for inhibitory→inhibitory connections (NetPyNE/NEURON model weight units)</small> | 0 → 1000 |
| Excitatory synapse rise time constant<br>— <small>Rise time constant of excitatory `Exp2Syn` (ms)</small> | 0.1 → 100 |
| Excitatory synapse decay time constant<br>— <small>Decay time constant of excitatory `Exp2Syn` (ms)</small> | 0.1 → 500 |
| Inhibitory synapse rise time constant<br>— <small>Rise time constant of inhibitory `Exp2Syn` (ms)</small> | 0.1 → 100 |
| Inhibitory synapse decay time constant<br>— <small>Decay time constant of inhibitory `Exp2Syn` (ms)</small> | 0.1 → 1000 |
| Excitatory reversal potential (fixed)<br>— <small>Reversal potential of excitatory synapses (mV)</small> | 0 |
| Inhibitory reversal potential (fixed)<br>— <small>Reversal potential of inhibitory synapses (mV)</small> | -75 |
| E-cell Na⁺ conductance density, mean<br>— <small>Mean maximal sodium conductance density in soma HH mechanism for excitatory cells (S·cm⁻²)</small> | 0 → 12 |
| E-cell Na⁺ conductance density, std<br>— <small>Std of maximal sodium conductance density for excitatory cells (S·cm⁻²)</small> | 0 → 4 |
| E-cell K⁺ conductance density, mean<br>— <small>Mean maximal potassium conductance density in soma HH mechanism for excitatory cells (S·cm⁻²)</small> | 0 → 4 |
| E-cell K⁺ conductance density, std<br>— <small>Std of maximal potassium conductance density for excitatory cells (S·cm⁻²)</small> | 0 → 1 |
| I-cell Na⁺ conductance density, mean<br>— <small>Mean maximal sodium conductance density in soma HH mechanism for inhibitory cells (S·cm⁻²)</small> | 0 → 10 |
| I-cell Na⁺ conductance density, std<br>— <small>Std of maximal sodium conductance density for inhibitory cells (S·cm⁻²)</small> | 0 → 3 |
| I-cell K⁺ conductance density, mean<br>— <small>Mean maximal potassium conductance density in soma HH mechanism for inhibitory cells (S·cm⁻²)</small> | 0 → 5 |
| I-cell K⁺ conductance density, std<br>— <small>Std of maximal potassium conductance density for inhibitory cells (S·cm⁻²)</small> | 0 → 2 |
| Leak conductance density (fixed)<br>— <small>Leak conductance density in soma HH mechanism (S·cm⁻²)</small> | 0.003 |
| Leak reversal potential (fixed)<br>— <small>Leak reversal potential in soma HH mechanism (mV)</small> | -70 |
| E-cell soma length, mean / std<br>— <small>Soma length for excitatory cells (µm); mean and heterogeneity</small> | 50 → 1000 / 0 → 150 |
| E-cell soma diameter, mean / std<br>— <small>Soma diameter for excitatory cells (µm); mean and heterogeneity</small> | 5 → 30 / 0 → 12 |
| E-cell axial resistance, mean / std<br>— <small>Axial resistance for excitatory cells (Ω·cm); mean and heterogeneity</small> | 70 → 200 / 0 → 50 |
| I-cell soma length, mean / std<br>— <small>Soma length for inhibitory cells (µm); mean and heterogeneity</small> | 50 → 500 / 0 → 100 |
| I-cell soma diameter, mean / std<br>— <small>Soma diameter for inhibitory cells (µm); mean and heterogeneity</small> | 4 → 15 / 0 → 6 |
| I-cell axial resistance, mean / std<br>— <small>Axial resistance for inhibitory cells (Ω·cm); mean and heterogeneity</small> | 80 → 200 / 0 → 40 |
