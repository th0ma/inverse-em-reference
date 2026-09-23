# Phase 2: measurement noise and clean-training normalization

Phase 2 provides two scientific facilities only: frozen raw-complex measurement noise and task-specific normalization fitted from clean analytical training observations. It performs no model training, sealed evaluation, robustness evaluation, or Phase-3 work.

## Raw-complex measurement noise

Analytical observations begin as 30-point `complex128` fields `E_z` and `H_phi`. For each field family `F` independently,

```text
P_F = mean(|F|^2)
sigma_F = sqrt(P_F * 10^(-gamma/10) / 2)
F_noisy = F + sigma_F * (xi_R + i*xi_I)
```

E and H have separate powers and scales. E-real, E-imaginary, H-real, and H-imaginary use independent standard-normal streams. Noise is applied before channelization and before normalization.

Training uses one sequential `PCG64DXSM(augmentation_seed)` stream. Each training event draws one continuous `Uniform[30,40)` SNR followed by four 30-value standard-normal blocks in E-real, E-imaginary, H-real, H-imaginary order.

Robustness uses SNRs `(40,30,20,15,10)` dB and ten realizations per SNR. Each component is keyed by `(master_seed, sample_index, realization_index, component)` through `SeedSequence` and PCG64DXSM. For a fixed sample and realization, the same standardized directions are reused at every SNR; only the scale changes.

## Observations

Each source is evaluated with the pinned analytical PhysicsTM. For multisource configurations, complex E and H are summed before conversion to a float64 `(4,30)` array ordered:

1. `Re(E_z)`
2. `Im(E_z)`
3. `Re(H_phi)`
4. `Im(H_phi)`

The inverse grid is endpoint-excluded: `2*pi*arange(30,dtype=float64)/30`.

## Clean-training normalization

`fit_clean_training_normalizer(task)` is the sole public production-fitting workflow. It accepts only the task and internally resolves the committed Phase-1 TRAINING identity:

- classifier: all 70,000 observations, balanced as 14,000 for each source count 1 through 5;
- S1: the frozen 70,000-observation S1 training population;
- S2: the frozen 70,000-observation S2 training population;
- S3: all eight 70,000-observation curriculum training stages.

Validation, sealed, robustness, reference-fixture, surrogate, and caller-supplied populations are not production fitting inputs. Fitting invokes no measurement-noise routine or measurement-noise RNG.

Four channel means and population standard deviations (`ddof=0`) are accumulated over all observations and all 30 angles. S3 uses one count-weighted streaming statistic over all eight stages; stage means or standard deviations are never averaged.

The workflow returns an ordinary `FrozenChannelNormalizer` and a descriptive `NormalizationReceipt`. The receipt records the executed population definitions, RNG identities, PhysicsTM pin, grid, channel order, statistics, configuration, and software revision. Receipt and normalizer hashes support reproducibility and accidental-change detection; they are not authentication and confer no authority.
