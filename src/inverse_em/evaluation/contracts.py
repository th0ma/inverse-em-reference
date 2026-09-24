"""Small, immutable fixture contracts. No population or model loader is exposed."""
from dataclasses import dataclass, fields
import re
import numpy as np

from inverse_em.config.evaluation import EvaluationConfig
from inverse_em.config.phase2 import Phase2ScientificConfig
from inverse_em.provenance.canonical import canonical_sha256


def digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA-256")
    return value


def label(value):
    if type(value) is not str or re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value) is None:
        raise ValueError("Invalid portable identity")
    return value


def fixture_seed(seed):
    # Deliberately small namespace, disjoint from all frozen production seeds.
    if type(seed) is not int or not 0 <= seed < 10000:
        raise PermissionError("Only nonproduction fixture seeds 0..9999 are executable")
    return seed


@dataclass(frozen=True)
class Bindings:
    study: str
    task: str
    configuration: str
    checkpoint: str
    normalizer: str
    population: str
    row_order: str
    clean_fields: str
    repository: str
    environment: str
    role: str = "REFERENCE_FIXTURE"
    checkpoint_role: str = "BEST_VALIDATION"

    def __post_init__(self):
        label(self.study)
        if self.role != "REFERENCE_FIXTURE" or self.checkpoint_role != "BEST_VALIDATION":
            raise PermissionError("Fixture BEST bindings only; no historical/production loading")
        if self.task not in EvaluationConfig().studies:
            raise ValueError("Unknown independent task")
        for name in ("configuration", "checkpoint", "normalizer", "population", "row_order", "clean_fields"):
            digest(getattr(self, name))
        if self.configuration != EvaluationConfig().sha256:
            raise ValueError("Evaluation configuration mismatch")
        if not self.repository or not self.environment.startswith("CPU/float64;"):
            raise ValueError("Repository and CPU/float64 runtime identity required")

    @property
    def sha256(self):
        return canonical_sha256(self)


@dataclass(frozen=True)
class Condition:
    snr: int
    realization: int
    seed: int
    row_order: str
    clean_fields: str
    environment: str
    phase2: str
    numpy_version: str
    address: str = EvaluationConfig().address
    components: tuple[str, ...] = EvaluationConfig().components
    scaling: str = "raw-complex-gaussian-v1"

    def __post_init__(self):
        if type(self.snr) is not int or self.snr not in EvaluationConfig().snrs:
            raise ValueError("Invalid SNR")
        if type(self.realization) is not int or self.realization not in range(10):
            raise ValueError("Invalid realization index")
        fixture_seed(self.seed)
        digest(self.row_order)
        digest(self.clean_fields)
        if (self.phase2 != Phase2ScientificConfig().sha256 or self.numpy_version != np.__version__
                or self.address != EvaluationConfig().address or self.components != EvaluationConfig().components
                or self.scaling != "raw-complex-gaussian-v1" or not self.environment.startswith("CPU/float64;")):
            raise ValueError("Noise convention/runtime mismatch")

    @property
    def key(self):
        return f"r{self.realization:02d}-snr{self.snr}"


def row_identity(ids):
    return canonical_sha256(tuple(ids))


def array(value, dtype, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"Expected {dtype} array {shape}")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite numeric array")
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(shape)


@dataclass(frozen=True)
class PredictionBundle:
    bindings: Bindings
    attempt: str
    ids: tuple[str, ...]
    rows: np.ndarray
    arrays: tuple[tuple[str, np.ndarray], ...]
    active_count: int
    condition: Condition | None = None
    class_order: tuple[int, ...] = (1, 2, 3, 4, 5)
    units: str = "rho:R=1;phi:radians"
    slot_convention: str = "fixed_slots_unsorted_predictions"

    def __post_init__(self):
        if type(self.bindings) is not Bindings:
            raise TypeError("Typed bindings required")
        label(self.attempt)
        if type(self.ids) is not tuple or not 1 <= len(self.ids) <= 16:
            raise PermissionError("Bounded fixture population: 1..16 rows")
        if any(type(x) is not str or not x.strip() or "\x00" in x for x in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Invalid/duplicate configuration identifiers")
        if row_identity(self.ids) != self.bindings.row_order:
            raise ValueError("Population row identity mismatch")
        n = len(self.ids)
        object.__setattr__(self, "rows", array(self.rows, "int64", (n,)))
        if not np.array_equal(self.rows, np.arange(n, dtype=np.int64)):
            raise ValueError("Missing, duplicate or reordered row coverage")
        if self.class_order != (1, 2, 3, 4, 5) or self.units != "rho:R=1;phi:radians" or self.slot_convention != "fixed_slots_unsorted_predictions":
            raise ValueError("Output convention mismatch")
        if type(self.arrays) is not tuple or len(dict(self.arrays)) != len(self.arrays):
            raise ValueError("Duplicate/invalid array keys")
        values = dict(self.arrays)
        if self.bindings.task == "classifier":
            if type(self.active_count) is not int or self.active_count != 0 or set(values) != {"truth", "logits", "predicted"}:
                raise ValueError("Classifier schema mismatch")
            converted = {k: array(v, "float64" if k == "logits" else "int64", (n, 5) if k == "logits" else (n,)) for k, v in values.items()}
            if np.any((converted["truth"] < 1) | (converted["truth"] > 5)) or not np.array_equal(converted["predicted"], np.argmax(converted["logits"], axis=1) + 1):
                raise ValueError("Class labels or first-argmax predictions inconsistent")
        else:
            s = int(self.bindings.task[1])
            if type(self.active_count) is not int or self.active_count != s or set(values) != {"target_rho", "target_phi", "rho", "phi", "cos_like", "sin_like"}:
                raise ValueError("Localization active-slot schema mismatch")
            converted = {k: array(v, "float64", (n, s)) for k, v in values.items()}
            # Use the closed decoder, including its atan2 signed-zero semantics.
            import torch
            from inverse_em.localization.decoding import decode_angle
            phi = decode_angle(torch.from_numpy(converted["cos_like"].copy()), torch.from_numpy(converted["sin_like"].copy())).numpy()
            if not np.array_equal(phi, converted["phi"]):
                raise ValueError("Decoded angle inconsistent with active angular outputs")
        object.__setattr__(self, "arrays", tuple(sorted(converted.items())))
        if self.condition is not None:
            if type(self.condition) is not Condition or any(getattr(self.condition, x) != getattr(self.bindings, x) for x in ("row_order", "clean_fields", "environment")):
                raise ValueError("Condition/bundle bindings mismatch")

    def to_mapping(self):
        from dataclasses import asdict
        return {"schema": "primary-predictions/1.0", "bindings": asdict(self.bindings), "attempt": self.attempt,
                "ids": list(self.ids), "rows": self.rows.tolist(), "active_count": self.active_count,
                "condition": asdict(self.condition) if self.condition else None,
                "class_order": list(self.class_order), "units": self.units, "slot_convention": self.slot_convention,
                "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "values": v.tolist()} for k, v in self.arrays}}

    @classmethod
    def from_mapping(cls, data):
        expected = {f.name for f in fields(cls)} | {"schema"}
        if set(data) != expected or data["schema"] != "primary-predictions/1.0":
            raise ValueError("Unknown bundle schema/fields")
        condition = data["condition"]
        if condition is not None:
            condition = dict(condition)
            condition["components"] = tuple(condition["components"])
            condition = Condition(**condition)
        arrays = []
        for name, item in data["arrays"].items():
            if set(item) != {"dtype", "shape", "values"} or item["dtype"] not in ("float64", "int64"):
                raise ValueError("Invalid stored numeric schema")
            if item["dtype"] == "int64" and any(type(x) is not int for x in item["values"]):
                raise ValueError("Stored integer labels cannot be coerced")
            value = np.asarray(item["values"], dtype=item["dtype"])
            if list(value.shape) != item["shape"]:
                raise ValueError("Stored shape mismatch")
            arrays.append((name, value))
        if any(type(x) is not int for x in data["rows"]):
            raise ValueError("Stored row indices cannot be coerced")
        return cls(Bindings(**data["bindings"]), data["attempt"], tuple(data["ids"]),
                   np.asarray(data["rows"], dtype=np.int64), tuple(arrays), data["active_count"], condition,
                   tuple(data["class_order"]), data["units"], data["slot_convention"])
