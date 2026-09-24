"""Phase-9 frozen descriptors; production plans contain no execution handles."""
from dataclasses import dataclass, asdict
from types import MappingProxyType

from inverse_em.config.phase2 import Phase2ScientificConfig, TASK_NOISE_SEEDS
from inverse_em.provenance.canonical import canonical_sha256


@dataclass(frozen=True)
class EvaluationConfig:
    schema: str = "evaluation/1.0"
    studies: tuple[str, ...] = ("classifier", "s1", "s2", "s3")
    snrs: tuple[int, ...] = (40, 30, 20, 15, 10)
    realizations: tuple[int, ...] = tuple(range(10))
    components: tuple[str, ...] = ("E_real", "E_imag", "H_real", "H_imag")
    address: str = "SeedSequence((master_seed,sample_index,realization_index,component))/PCG64DXSM"
    row_index: str = "zero_based_immutable_population_position"
    aggregation_ddof: int = 1
    recovery: str = "fail_closed_no_automatic_retry_or_resume"
    persistence: str = "verified_primary_before_metrics_before_transition"
    numerical_environment: str = "CPU/float64"

    def __post_init__(self):
        def exact(value, expected):
            return (type(value) is type(expected) and
                    (len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
                     if isinstance(expected, tuple) else value == expected))
        for name, field in self.__dataclass_fields__.items():
            if not exact(getattr(self, name), field.default):
                raise ValueError(f"Frozen evaluation field changed: {name}")

    @property
    def sha256(self):
        return canonical_sha256({"evaluation": asdict(self), "phase2": Phase2ScientificConfig().sha256})


def production_plan():
    return MappingProxyType({
        "schema": "phase9-production-description/1.0", "execution_available": False,
        "configuration_sha256": EvaluationConfig().sha256,
        "studies": EvaluationConfig().studies,
        "robustness_seeds": tuple((x.task.value, x.robustness_seed) for x in TASK_NOISE_SEEDS),
        "snrs": EvaluationConfig().snrs, "realizations": tuple(range(10)),
        "clean_traversals": 1, "robustness_traversals": 50, "total_traversals": 51,
    })
