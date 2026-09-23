import copy
import numpy as np
import pytest
import torch

from inverse_em.classifier import (BoundedClassifierTrainer, ClassifierCheckpointContext, load_best_for_inference, load_classifier_checkpoint, make_initialized_classifier,
    restore_resumable_checkpoint, save_best_checkpoint, save_resumable_checkpoint)
from inverse_em.config.phase2 import InverseTask
from inverse_em.normalization import FrozenChannelNormalizer
from inverse_em.observations import ComplexFields


def fixture():
    fields = tuple(ComplexFields(np.full(30, i + 1j), np.full(30, i + 2j)) for i in range(5))
    return fields, torch.arange(5, dtype=torch.int64), FrozenChannelNormalizer(InverseTask.CLASSIFIER, np.zeros(4), np.ones(4), 1, 30)


def metadata(): return {"repository_revision":"fixture", "normalizer_sha256":"n", "training_population_identity":"TRAINING:fixture", "validation_population_identity":"VALIDATION:fixture", "physics_identity":"pinned-PhysicsTM", "software":{"torch":torch.__version__}}
def context(): return ClassifierCheckpointContext("n", "pinned-PhysicsTM", "TRAINING:fixture", "VALIDATION:fixture")


def nested_equal(left, right):
    if isinstance(left, torch.Tensor): return torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_best_checkpoint_strict_reload_and_compatibility(tmp_path):
    fields, labels, norm = fixture(); trainer = BoundedClassifierTrainer(make_initialized_classifier())
    trainer.run(fields, labels, fields, labels, norm, 1); path = tmp_path / "best.pt"
    save_best_checkpoint(path, trainer, metadata()); restored = make_initialized_classifier()
    load_best_for_inference(path, restored, context())
    assert all(torch.equal(restored.state_dict()[key], trainer.best_model_state[key]) for key in restored.state_dict())
    with pytest.raises(ValueError): load_best_for_inference(path, restored, ClassifierCheckpointContext("wrong", "pinned-PhysicsTM", "TRAINING:fixture", "VALIDATION:fixture"))
    with pytest.raises(TypeError): load_classifier_checkpoint(path, None)


def test_exact_interrupted_resume(tmp_path):
    fields, labels, norm = fixture(); uninterrupted = BoundedClassifierTrainer(make_initialized_classifier())
    uninterrupted.run(fields, labels, fields, labels, norm, 2)
    partial = BoundedClassifierTrainer(make_initialized_classifier()); partial.run(fields, labels, fields, labels, norm, 1)
    path = tmp_path / "resume.pt"; save_resumable_checkpoint(path, partial, metadata())
    resumed = BoundedClassifierTrainer(make_initialized_classifier()); restore_resumable_checkpoint(path, resumed, context())
    resumed.run(fields, labels, fields, labels, norm, 1)
    assert nested_equal(uninterrupted.state_dict(), resumed.state_dict())


@pytest.mark.parametrize("field,value", [
    ("class_count", 4), ("dtype", "float32"), ("input_shape", (4, 29)),
    ("normalizer_sha256", "wrong"), ("physics_identity", "wrong"),
    ("training_population_identity", "TRAINING:wrong"), ("validation_population_identity", "VALIDATION:wrong"),
    ("rng_contract", "wrong"),
])
def test_public_load_rejects_each_material_mismatch(tmp_path, field, value):
    fields, labels, norm = fixture(); trainer = BoundedClassifierTrainer(make_initialized_classifier())
    trainer.run(fields, labels, fields, labels, norm, 1); valid = tmp_path / "valid.pt"
    save_best_checkpoint(valid, trainer, metadata()); payload = torch.load(valid, map_location="cpu", weights_only=True)
    payload[field] = value; altered = tmp_path / f"altered-{field}.pt"; torch.save(payload, altered)
    with pytest.raises(ValueError): load_classifier_checkpoint(altered, context())
