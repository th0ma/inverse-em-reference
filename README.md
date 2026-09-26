# Inverse EM Reference

Reference source code and regression infrastructure accompanying research on inverse electromagnetic source classification and localization.

## Status

Phases 0–11 constitute the published research-code baseline. Phase 11 was closed **with documented limitations**. This baseline supports the research methods, implementation provenance, interfaces, regression behavior, and bounded reproducibility evidence; it is not proof that every reported scientific result has been independently reproduced from scratch.

## Scope

The repository contains:

- Typed configuration, canonical hashing, provenance, artifact registration, and scientific lifecycle contracts.
- Analytical PhysicsTM forward modeling and deterministic population generation.
- Raw-complex measurement noise and clean-training-only normalization.
- Frozen electromagnetic surrogate interfaces, source-count classification, and one-, two-, and three-source localization components, including supervised and field-consistency losses.
- Training and checkpoint interfaces, sealed-evaluation and robustness infrastructure, bounded artifact compatibility, and regression/reproducibility checks.

Scientific components are implemented as Python modules. The current CLI provides configuration and provenance utilities; it is **not** an end-to-end production training or evaluation workflow. See the [phase documentation](docs/) for component-specific contracts and execution boundaries.

## Scientific and reproducibility boundary

Publication of this source-code and regression baseline does **not**, by itself, establish:

- Reproduction of historical production-training trajectories.
- Equivalence to production checkpoints.
- Bitwise scientific equivalence across runtimes.
- Reproduction of sealed-test or robustness results.
- Equivalence of production artifact migrations.
- New scientific performance results.

Regression fixtures and bounded execution evidence support methods and provenance, not blanket scientific equivalence. In particular, [Phase 10](docs/phase10.md) verifies synthetic artifact compatibility, and [Phase 11](docs/phase11.md) records a bounded verification procedure rather than a universal execution sandbox.

## Repository structure

- [src/inverse_em/](src/inverse_em/): configuration/provenance, physics, populations, observations, noise, normalization, surrogates, models, classifier/localization, losses, metrics, training, evaluation, and artifact-integration modules.
- [tests/](tests/): unit, numerical, invariant, integration, state, smoke, and reference regression tests, including bounded fixtures and historical closure checks.
- [docs/](docs/): governance, reproducibility contracts, phase history, and accepted limitations.

## Installation / environment

Python 3.11 or newer is required by the package metadata. From the repository root, install the package in editable mode:

```sh
python -m pip install -e .
```

For testing with neural and YAML support:

```sh
python -m pip install -e ".[test,phase3,yaml]"
```

The `phase3` extra supplies the PyTorch dependency used by neural components. Consult [pyproject.toml](pyproject.toml) for authoritative dependencies, optional extras, and the CLI entry point; the package version is defined in [_version.py](src/inverse_em/_version.py). The Python minimum is not a guarantee that every platform supports every verification mechanism: protected-artifact operations include Windows-specific requirements described in [Phase 10](docs/phase10.md).

## Verification / tests

After installation, ordinary Phase 0–10 repository regression tests can be invoked with:

```sh
python -m pytest tests --ignore=tests/phase11
```

This explicitly excludes the historical Phase-11 closure/accounting harness. Tests use bounded/reference computations and temporary artifacts; passing them does not reproduce a production scientific study.

The documented final **pre-publication staged-snapshot** result was **1,146 passed, zero failures, zero skips**. This is historical publication evidence, **not** a post-publication closure-suite pass on the committed HEAD. The Phase-11 harness pins the preceding Phase-10 baseline and validates a staged candidate; it is not a general post-publication test command. Its accounting, execution boundary, and lifecycle limitations are documented in [Phase 11](docs/phase11.md).

## Documentation

- [Governance](docs/governance.md): authority and scientific-contract hierarchy.
- [Reproducibility](docs/reproducibility.md): configuration, identity, provenance, and lifecycle rules.
- [Phase history](docs/): component design and implementation records.
- [Artifact integration and compatibility](docs/phase10.md): synthetic compatibility and protected-result lifecycle.
- [Final regression and reproducibility](docs/phase11.md): historical closure evidence and accepted publication limitations.

Phase documents retain development-era statements and verification conditions. The final freeze/publication record in Phase 11 describes the closed baseline; earlier candidate-status text is historical. Some governing documents and historical verification receipts are external and are not bundled in this repository.

## Known limitations

- Execution/interception and accounting guarantees apply to the documented bounded surface. They are not a universal hostile-code or native-code sandbox.
- Artifact compatibility evidence is bounded and synthetic; it does not establish acceptance or equivalence of historical production artifacts.
- Protected-result leases remain effective while OPEN and require explicit caller completion through `close()` or context-manager exit. They do not promise perpetual immutability after closure.
- Some protected-artifact guarantees depend on Windows locking and filesystem capabilities; unsupported environments fail closed rather than receive equivalent guarantees.
- Historical verification manifests and receipts are external. A supplied manifest digest provides consistency checking against that supplied value, not independent immutable authority; this self-attestation limitation was accepted at publication.
- The historical Phase-11 closure harness expects the earlier baseline HEAD and staged candidate. Its baseline pin makes it unsuitable as an unmodified post-publication closure check.

These are documented limits of the published baseline, not claims of additional verification or unresolved publication blockers.

## Citation

Use [CITATION.cff](CITATION.cff) for the authoritative software citation: Thomas D. Papadopoulos, *Inverse Electromagnetic Source Classification and Localization Reference Implementation*. The file supplies the software version and license; it does not currently supply a paper DOI or journal reference.

## License

The repository is distributed under the [MIT License](LICENSE). The vendored PhysicsTM code also carries its own [MIT license notice](src/inverse_em/physics/vendor/LICENSE), which must be preserved.
