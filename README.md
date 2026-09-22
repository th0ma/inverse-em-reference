# Inverse EM Reference

This repository is the clean reference implementation for the final inverse electromagnetic source methodology. Phase 0 contains infrastructure only: typed configuration, deterministic canonical hashing, provenance schemas, artifact registration, scientific lifecycle validation, authorization receipts, a safe CLI skeleton, and tests.

No PhysicsTM source, population generator, noise generator, normalizer, neural model, loss, training loop, scientific inference, sealed evaluation, robustness evaluation, checkpoint, dataset, or prediction artifact is included in Phase 0.

## Governing documents

See `docs/governance.md`. The frozen Scientific & Software Specification v1.0 has highest authority, as clarified by the Blueprint Resolution Addendum v1.0.

## Safe Phase-0 commands

```text
inverse-em version
inverse-em config validate FILE
inverse-em config digest FILE
inverse-em manifest validate FILE
inverse-em registry validate FILE
inverse-em state validate FROM TO
```

Future scientific command groups exist only to fail closed with `Not implemented in Phase 0`.

The exact supported Phase-0 schema versions and canonical identity rules are documented in `docs/reproducibility.md`. Scientific configuration paths are portable logical paths; host-specific paths are provenance only.
