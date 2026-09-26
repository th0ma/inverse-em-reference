# Noise protocol

[Phase 2](phase2.md) documents the implemented raw-complex noise law, training stream, paired-SNR addressing, and clean-training normalization. The historical task-specific RNG adapters in [S1](phase6.md), [S2](phase7.md), and [S3](phase8.md) retain their distinct addressing and field-SNR conventions; they must not be harmonized into a single historical stream. [Phase 4](phase4.md) describes the classifier data path and its historical-reproduction boundary. [Phase 9](phase9.md) documents bounded robustness infrastructure, not authorization for a scientific robustness run.

## Historical Phase-0 foundation

The following phase-time statement is retained verbatim; it does not describe the current implementation status.

The governing contract requires raw-complex measurement-noise addition before channel conversion and normalization. Training augmentation and paired-SNR robustness generation are distinct operations. They belong to Phase 2 and are not implemented in Phase 0.
