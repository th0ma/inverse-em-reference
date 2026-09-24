"""Bounded fixture simulation. Production evaluation always rejects first."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib

from inverse_em.config.evaluation import EvaluationConfig
from inverse_em.config.phase2 import Phase2ScientificConfig
from inverse_em.evaluation.contracts import Bindings, Condition, PredictionBundle, fixture_seed, label
from inverse_em.evaluation.artifacts import FixtureStore, immutable_write
from inverse_em.evaluation.reporting import bundle_metrics, aggregate_realizations
from inverse_em.provenance.evaluation import FixtureAuthorization, Accounting
from inverse_em.scientific.state import ScientificState


def protected_evaluation(*args, **kwargs):
    raise PermissionError("Phase-9 production SEALED/ROBUSTNESS execution is unavailable")


def load_historical_checkpoint(*args, **kwargs):
    raise PermissionError("Historical checkpoint loading/migration is unavailable")


def fixture_noise(fields, seed, sample_index, realization_index):
    fixture_seed(seed)
    if type(sample_index) is not int or not 0 <= sample_index < 16:
        raise PermissionError("Fixture row index out of bounds")
    if type(realization_index) is not int or realization_index not in range(10):
        raise ValueError("Realization must be 0..9")
    from inverse_em.noise.api import draw_standardized_directions, scale_paired_robustness, add_scaled_noise
    directions = draw_standardized_directions(seed, sample_index, realization_index)
    return directions, tuple(add_scaled_noise(fields, noise) for noise in scale_paired_robustness(fields, directions))


def now():
    return datetime.now(timezone.utc).isoformat()


class FixtureEngine:
    """Not a production runner: at most 16 predeclared rows, trusted test producer.

    Failed engines/store claims cannot be resumed. No checkpoint, population,
    analytical solver, normalizer fitter or model constructor is available here.
    """
    def __init__(self, root, bindings, state=ScientificState.FROZEN, predecessor="0" * 64):
        if type(bindings) is not Bindings or type(state) is not ScientificState:
            raise TypeError("Typed fixture bindings and state required")
        from inverse_em.evaluation.contracts import digest
        digest(predecessor)
        self.store = FixtureStore(root)
        self.bindings = bindings
        self.state = state
        self.predecessor = predecessor

    def run(self, authorization, attempt, producer, *, robustness_seed=123, fault=None):
        if self.state is ScientificState.CLOSED:
            raise PermissionError("CLOSED forbids fresh inference")
        if type(authorization) is not FixtureAuthorization or authorization.bindings != self.bindings:
            raise PermissionError("Authorization binding mismatch")
        if authorization.before is not self.state or authorization.predecessor != self.predecessor:
            raise PermissionError("Authorization predecessor mismatch")
        label(attempt)
        fixture_seed(robustness_seed)
        import numpy as np
        conditions = [None] if authorization.after is ScientificState.SEALED_CLEAN else [
            Condition(snr, r, robustness_seed, self.bindings.row_order, self.bindings.clean_fields,
                      self.bindings.environment, Phase2ScientificConfig().sha256, np.__version__)
            for r in range(10) for snr in EvaluationConfig().snrs]
        accounting = Accounting()
        record = {"schema": "fixture-attempt/1.0", "attempt": attempt, "authorization": asdict(authorization),
                  "authorization_sha256": authorization.sha256, "bindings": asdict(self.bindings),
                  "before": self.state.value, "after": authorization.after.value, "started": now(),
                  "accounting": accounting.snapshot()}
        # Enums serialized explicitly; no custom/default serializer masks mistakes.
        record["authorization"]["before"] = authorization.before.value
        record["authorization"]["after"] = authorization.after.value
        directory = self.store.start(attempt, authorization, record)
        phase = "before_inference"
        records, manifests = [], []

        def boundary(name):
            nonlocal phase
            phase = name
            if fault is not None:
                fault(name)

        def batch_completed(rows):
            if type(rows) is not int or not 1 <= rows <= 16:
                raise ValueError("Invalid fixture batch accounting")
            accounting.forward_calls += 1
            accounting.rows_processed += rows
            immutable_write(directory / f"batch-{accounting.forward_calls:04d}.json",
                            {"at": now(), "accounting": accounting.snapshot()})

        try:
            for condition in conditions:
                key = condition.key if condition else "clean"
                boundary("before_inference")
                phase = "during_inference"
                accounting.exact_execution_count_known = False
                previous_rows = accounting.rows_processed
                bundle = producer(condition, batch_completed)
                if type(bundle) is not PredictionBundle:
                    raise TypeError("Fixture producer must return a complete typed bundle")
                if bundle.bindings != self.bindings or bundle.attempt != attempt or bundle.condition != condition:
                    raise ValueError("Inference bundle identity mismatch")
                if accounting.rows_processed - previous_rows != len(bundle.ids):
                    raise ValueError("Inference row accounting mismatch")
                # Bind the validated output before any post-producer hook or staging.
                from inverse_em.provenance.canonical import canonical_sha256
                producer_digest = canonical_sha256(bundle.to_mapping())
                immutable_write(directory / (key + ".PRODUCER.json"), {
                    "schema": "validated-producer-output/1.0", "attempt": attempt,
                    "bindings": self.bindings.sha256, "authorization": authorization.sha256,
                    "content_sha256": producer_digest})
                accounting.exact_execution_count_known = True
                accounting.inference_complete_traversals += 1
                boundary("after_inference")
                boundary("during_primary_staging")
                manifest = self.store.stage_primary(directory, key, bundle)
                self.store.publish_primary(directory, key, manifest)
                manifests.append(manifest)
                boundary("after_primary_persistence")
                boundary("during_reopen_verification")
                persisted, verified = self.store.reopen(directory, key, self.bindings, attempt)
                if canonical_sha256(persisted.to_mapping()) != producer_digest:
                    raise ValueError("Reopened output differs from producer receipt")
                if persisted.condition != condition:
                    raise ValueError("Persisted condition mismatch")
                accounting.verified_primary_bundles += 1
                boundary("during_metrics")
                metrics = bundle_metrics(persisted)
                records.append({"complete": True, "bindings": self.bindings.sha256,
                    "snr": condition.snr if condition else None,
                    "realization": condition.realization if condition else None,
                    "seed": condition.seed if condition else None, "primary": verified, "metrics": metrics})
            report = {"schema": "fixture-evaluation-report/1.0", "attempt": attempt,
                      "bindings": asdict(self.bindings), "records": records}
            if conditions != [None]:
                report["aggregate"] = {str(snr): aggregate_realizations([r for r in records if r["snr"] == snr]) for snr in EvaluationConfig().snrs}
            boundary("during_report_persistence")
            self.store.publish(directory, "REPORT", report)
            report_hash = hashlib.sha256((directory / "REPORT.json").read_bytes()).hexdigest()
            boundary("before_final_transition_commit")
            committed = accounting.snapshot()
            committed["officially_committed_evaluations"] = len(conditions)
            success = {"schema": "fixture-success/1.0", "attempt": attempt,
                       "study": self.bindings.study,
                       "authorization_sha256": authorization.sha256, "predecessor": self.predecessor,
                       "bindings": self.bindings.sha256, "before": self.state.value,
                       "after": authorization.after.value, "report_sha256": report_hash,
                       "completed": now(), "accounting": committed}
            self.store.publish(directory, "SUCCESS", success)
        except BaseException as error:
            try:
                if not (directory / "INCOMPLETE.json").exists():
                    immutable_write(directory / "INCOMPLETE.json", {"schema": "fixture-incomplete/1.0", "attempt": attempt,
                        "authorization_sha256": authorization.sha256, "bindings": self.bindings.sha256,
                        "predecessor": self.predecessor, "phase": phase, "error_type": type(error).__name__,
                        "at": now(), "accounting": accounting.snapshot(), "available_primary_manifests": manifests})
            finally:
                self.store.relinquish()
            raise
        # The durable success marker is the commit point; state is only its view.
        try:
            self.state = authorization.after
            from inverse_em.provenance.canonical import canonical_sha256
            self.predecessor = canonical_sha256(success)
            self.store.release_after_success(self.bindings.study)
            return self.store.official(attempt)
        finally:
            self.store.relinquish()
