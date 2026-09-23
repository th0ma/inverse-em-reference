from __future__ import annotations

from dataclasses import asdict, dataclass

from inverse_em.errors import SchemaValidationError
from inverse_em.provenance.canonical import canonical_sha256


CLASSIFIER_COUNTS = (1, 2, 3, 4, 5)
CLASSIFIER_INITIALIZATION_SEED = 20261903
CLASSIFIER_MINIBATCH_SEED = 20261904
CLASSIFIER_AUGMENTATION_SEED = 20261905
CLASSIFIER_SEALED_SEED = 20261906
CLASSIFIER_ROBUSTNESS_SEED = 20261907
DROPOUT_DERIVATION_STRING = "inverse-em-reference|phase4|classifier|dropout|20261903"
DROPOUT_DERIVATION_SHA256 = "735aed26a077452634215c1c065599430027ee99ba84c9aeb15ecbd2c04cdfa0"
DROPOUT_SEED = 2757741876565858931
HISTORICAL_INITIAL_STATE_SHA256 = "e97893358d9633d3d693b56c15bdef5edea56ee8d68a06861ef144a322ade925"


@dataclass(frozen=True)
class ClassifierPopulationConfig:
    training_per_class: int = 14_000
    validation_per_class: int = 3_000
    sealed_per_class: int = 3_000
    training_seed: int = 20261901
    validation_seed: int = 20261902
    sealed_seed: int = CLASSIFIER_SEALED_SEED
    robustness_seed: int = CLASSIFIER_ROBUSTNESS_SEED

    def __post_init__(self):
        expected = (14_000, 3_000, 3_000, 20261901, 20261902, 20261906, 20261907)
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Classifier population configuration differs from frozen Phase-4 contract")


@dataclass(frozen=True)
class ClassifierModelConfig:
    input_channels: int = 4
    angle_count: int = 30
    hidden_channels: int = 64
    residual_blocks: int = 4
    kernel_size: int = 3
    negative_slope: float = 0.01
    dropout_probability: float = 0.1
    class_count: int = 5
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self):
        expected = (4, 30, 64, 4, 3, 0.01, 0.1, 5, "float64", "cpu")
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Classifier model configuration differs from frozen Phase-4 contract")


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    initialization_seed: int = CLASSIFIER_INITIALIZATION_SEED
    minibatch_seed: int = CLASSIFIER_MINIBATCH_SEED
    augmentation_seed: int = CLASSIFIER_AUGMENTATION_SEED
    dropout_seed: int = DROPOUT_SEED
    batch_size: int = 128
    maximum_epochs: int = 200
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    scheduler_threshold: float = 1e-4
    scheduler_threshold_mode: str = "abs"
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 1e-6
    scheduler_epsilon: float = 1e-8
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-4

    def __post_init__(self):
        expected = (20261903, 20261904, 20261905, DROPOUT_SEED, 128, 200, 1e-3,
                    (0.9, 0.999), 1e-8, 0.0, False, "max", 0.5, 8, 1e-4, "abs",
                    0, 1e-6, 1e-8, 20, 1e-4)
        if tuple(asdict(self).values()) != expected:
            raise SchemaValidationError("Classifier training configuration differs from frozen Phase-4 contract")


@dataclass(frozen=True)
class ClassifierScientificConfig:
    population: ClassifierPopulationConfig = ClassifierPopulationConfig()
    model: ClassifierModelConfig = ClassifierModelConfig()
    training: ClassifierTrainingConfig = ClassifierTrainingConfig()
    objective: str = "unweighted_mean_cross_entropy"
    label_smoothing: float = 0.0
    checkpoint_metric: str = "clean_validation_macro_f1"
    checkpoint_tie_policy: str = "earliest_epoch_strict_numerical_increase"

    def __post_init__(self):
        if (self.objective, self.label_smoothing, self.checkpoint_metric, self.checkpoint_tie_policy) != (
            "unweighted_mean_cross_entropy", 0.0, "clean_validation_macro_f1",
            "earliest_epoch_strict_numerical_increase"):
            raise SchemaValidationError("Classifier scientific configuration differs from frozen Phase-4 contract")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))
