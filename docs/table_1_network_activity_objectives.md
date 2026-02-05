# Table 1: Network Activity Objectives Used for Model Calibration

This table summarizes the *objective terms* (fitness components) currently used to calibrate the **DIV21_FxHET** network model:

- Model folder: `RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET`
- Fitness function: `RBS_network_models/fitnessFunc.py` → `fitnessFunc_v3()`
- Fitness schema (objective selection + bounds/weights): `RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/fitness_schema/schema_2.py`

**How targets are set:** for any metric listed below, the *target value* is the corresponding metric computed from the Aim 1 HD-MEA recording (loaded from a `metrics.npy` file referenced by the run scripts, e.g. `refit_batch.py`). The schema provides the scoring bounds (`min_val`, `max_val`), and `fitnessFunc_v3()` scores deviation from the experimental target using an asymmetric parabolic curve.

> Note: Classic network-burst descriptors such as inter-burst interval (IBI) and burst duration exist in the metric structure, but are currently *disabled* (`include: False`) in this schema for DIV21_FxHET.

| Objective term (schema path) | What it measures (Aim 1 network metric) | Experimental target source | Scoring bounds in schema (`min_val`→`max_val`) | Weight | Notes |
|---|---|---|---:|---:|---|
| `spiking_data.i_frs` | Mean inhibitory-unit firing rate (spikes/s) across units labeled `I` | From Aim 1 `metrics.npy` (key matches this path) | 0.25 → 22 | 1.0 | Included as a population-level excitability constraint |
| `spiking_data.e_frs` | Mean excitatory-unit firing rate (spikes/s) across units labeled `E` | From Aim 1 `metrics.npy` | 0.15 → 3.5 | 1.0 | Helps anchor baseline E activity magnitude |
| `spiking_data.num_e_firing` | Number of excitatory units that fired (count of `E` units with ≥1 spike) | From Aim 1 `metrics.npy` | 99 → 203 | 1.0 | Schema comment notes “current target=203” |
| `spiking_data.num_i_firing` | Number of inhibitory units that fired (count of `I` units with ≥1 spike) | From Aim 1 `metrics.npy` | 40 → 98 | 1.0 | Schema comment notes “current target=98” |
| `spiking_data.EI_fr_ratios` | Ratio of E/I firing rates (dimensionless) | From Aim 1 `metrics.npy` | 0.001 → 0.5 | 1.0 | Schema comment notes “current target≈0.1674” |
| `spiking_data.EI_spike_ratios` | Ratio of total E spikes / total I spikes (dimensionless) | From Aim 1 `metrics.npy` | 0.001 → 2.0 | 1.0 | Useful when matching overall E vs I contribution |
| `mega_bursting_data.baseline` | Baseline level of the convolved “mega-burst” population signal (units depend on convolution normalization) | From Aim 1 `metrics.npy` | 2.0 → 4.0 | 1.0 | Schema comment notes target “around 2.5” |
| `mega_bursting_data.burst_metrics.burst_rate` | Mega-burst frequency (bursts/s) detected on the population signal | From Aim 1 `metrics.npy` | 0.02 → 1.0 | 1.0 | Schema comment notes “current target≈0.1933” |
| `mega_bursting_data.burst_metrics.burst_amp` | Mega-burst amplitude on the population signal (convolved units) | From Aim 1 `metrics.npy` | 0.25 → 11.0 | 1.0 | Schema comment includes mean/min/max (approx 4.64 / 1.78 / 9.52) |

## Metrics present but disabled in DIV21_FxHET `schema_2`

The following are common Aim 1 network metrics, but are currently turned off (`include: False`) for this model’s calibration schema:

- `mega_bursting_data.burst_metrics.ibi` (inter-burst interval)
- `mega_bursting_data.burst_metrics.burst_duration`
- `mega_bursting_data.burst_metrics.num_units_per_burst`
- `mega_bursting_data.burst_metrics.in_burst_fr`
- `spiking_data.frs` (overall/combined firing rate)
- `spiking_data.isi`, `spiking_data.e_isi`, `spiking_data.i_isi`

If you want Table 1 to include *burst duration, IBI, synchronization index,* etc. as explicit calibration objectives, the next step would be to enable them in the schema and (if needed) confirm the corresponding keys exist in the `metrics.npy` structure produced by the Aim 1 pipeline.
