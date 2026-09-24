"""Fixture-only immutable store. A durable commit marker publishes a bundle."""
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from inverse_em.evaluation.contracts import PredictionBundle, label
from inverse_em.provenance.canonical import canonical_sha256


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def immutable_write(path, value):
    """Create once, flush before publication; failed writes are never overwritten."""
    payload = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


class FixtureStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        # Explicit creation, not import-time I/O. No production artifact path API.
        self.root.mkdir(parents=True, exist_ok=True)
        self._lease = None
        self._owner = None

    def directory(self, attempt):
        return self.root / label(attempt)

    def acquire(self, study):
        """Process-lifetime OS lock, also excluding reentrant/thread retry calls.

        The lock file is never unlinked; process death releases the OS lock,
        but the durable claim still requires explicit reconciliation.
        """
        if self._lease is not None:
            raise PermissionError("An attempt is actively owned by this store")
        stream = (self.root / ("ownership-" + label(study) + ".lock")).open("a+b")
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise PermissionError("Study has an active owner") from error
        self._lease = stream

    def relinquish(self):
        stream, self._lease = self._lease, None
        self._owner = None
        if stream is not None:
            stream.close()

    def validate_predecessor(self, authorization):
        successes = []
        for marker in self.root.glob("*/SUCCESS.json"):
            # Read STARTED to select the study; never trust SUCCESS for routing.
            started, _ = self.validated_start(marker.parent.name)
            if started["bindings"]["study"] != authorization.bindings.study:
                continue
            item = self.official(marker.parent.name)
            if item["bindings"] != authorization.bindings.sha256:
                raise PermissionError("Frozen study dependency identities changed")
            successes.append(item)
        if authorization.predecessor != "0" * 64:
            # A declared receipt must resolve positively, even with no markers.
            # Only fully verified evidence from this study can satisfy it.
            declared = [x for x in successes if canonical_sha256(x) == authorization.predecessor]
            if len(declared) != 1 or declared[0]["after"] != authorization.before.value:
                raise PermissionError("Declared predecessor SUCCESS evidence missing or incompatible")
        if successes:
            predecessors = [x["predecessor"] for x in successes]
            if len(set(predecessors)) != len(predecessors):
                raise PermissionError("Multiple successors for one predecessor")
            tips = [x for x in successes if canonical_sha256(x) not in predecessors]
            if len(tips) != 1 or canonical_sha256(tips[0]) != authorization.predecessor or tips[0]["after"] != authorization.before.value:
                raise PermissionError("Durable scientific predecessor mismatch")

    def start(self, attempt, authorization, record):
        # Read-only integrity checks precede claim/attempt mutations.
        self.validate_predecessor(authorization)
        self.acquire(authorization.bindings.study)
        try:
            self.validate_predecessor(authorization)
            for retry in self.root.glob("*/RETRY_AUTHORIZATION.json"):
                approved = json.loads(retry.read_bytes())
                if approved["study"] == authorization.bindings.study and not (self.root / ("authorization-" + approved["authorization_sha256"] + ".json")).exists():
                    if approved["authorization_sha256"] != authorization.sha256 or approved["new_attempt"] != attempt:
                        raise PermissionError("A different explicitly authorized attempt is pending")
            claim = self.root / ("claim-" + authorization.bindings.study)
            claim.mkdir()
            immutable_write(self.root / ("authorization-" + authorization.sha256 + ".json"), record)
            directory = self.directory(attempt)
            directory.mkdir()
            immutable_write(directory / "STARTED.json", record)
            ownership = {"attempt": attempt, "authorization": authorization.sha256,
                         "start_sha256": canonical_sha256(record), "study": authorization.bindings.study}
            immutable_write(directory / "OWNERSHIP.json", ownership)
            self._owner = (directory, authorization, ownership)
        except BaseException:
            # A failed registration remains fail-closed, even if no inference ran.
            self.relinquish()
            raise
        return directory

    def approve_new_attempt(self, previous, new_attempt, authorization):
        """Explicit fixture-only operator action, never invoked by run/recovery."""
        # A missing terminal record is insufficient: OS ownership must be free.
        self.acquire(authorization.bindings.study)
        try:
            return self._approve_new_attempt(previous, new_attempt, authorization)
        finally:
            self.relinquish()

    def _approve_new_attempt(self, previous, new_attempt, authorization):
        label(new_attempt)
        directory = self.directory(previous)
        started, _ = self.validated_start(previous)
        self.validate_predecessor(authorization)
        if self.official(previous) is not None or previous == new_attempt:
            raise PermissionError("Only distinct attempts after incomplete execution")
        if (started["authorization_sha256"] == authorization.sha256
                or started["authorization"]["identity"] == authorization.identity
                or started["bindings"] != __import__("dataclasses").asdict(authorization.bindings)
                or started["before"] != authorization.before.value
                or started["after"] != authorization.after.value
                or started["authorization"]["predecessor"] != authorization.predecessor):
            raise PermissionError("New explicit authorization/identity required")
        if self.directory(new_attempt).exists():
            raise FileExistsError("New attempt identity already used")
        if not (directory / "INCOMPLETE.json").exists():
            # Crash reconciliation records uncertainty; it never invents zero work.
            self.publish(directory, "INCOMPLETE", {"schema": "fixture-incomplete/1.0", "attempt": previous,
                "phase": "interruption_reconciliation", "exact_execution_count_known": False,
                "reason": "started_without_terminal_record", "authorization_sha256": started["authorization_sha256"]})
        self.publish(directory, "RETRY_AUTHORIZATION", {"study": authorization.bindings.study,
            "previous_attempt": previous, "new_attempt": new_attempt, "authorization_sha256": authorization.sha256,
            "quiescence_evidence": "exclusive_OS_ownership_acquired", "recorded_at": datetime.now(timezone.utc).isoformat()})
        self.release_after_success(authorization.bindings.study)

    def validated_start(self, attempt):
        from inverse_em.provenance.evaluation import FixtureAuthorization, Accounting
        from inverse_em.evaluation.contracts import Bindings
        from inverse_em.scientific.state import ScientificState
        data = json.loads((self.directory(attempt) / "STARTED.json").read_bytes())
        expected = {"schema", "attempt", "authorization", "authorization_sha256", "bindings", "before", "after", "started", "accounting"}
        if set(data) != expected or data["schema"] != "fixture-attempt/1.0" or data["attempt"] != attempt:
            raise ValueError("Invalid durable start schema")
        a = dict(data["authorization"])
        a["bindings"] = Bindings(**a["bindings"])
        a["before"], a["after"] = ScientificState(a["before"]), ScientificState(a["after"])
        authorization = FixtureAuthorization(**a)
        if (authorization.sha256 != data["authorization_sha256"] or asdict(authorization.bindings) != data["bindings"]
                or data["before"] != authorization.before.value or data["after"] != authorization.after.value
                or canonical_sha256(data["accounting"]) != canonical_sha256(Accounting().snapshot())):
            raise ValueError("Durable start/authorization mismatch")
        if datetime.fromisoformat(data["started"]).tzinfo is None:
            raise ValueError("Start time requires timezone")
        registered = json.loads((self.root / ("authorization-" + authorization.sha256 + ".json")).read_bytes())
        if canonical_sha256(registered) != canonical_sha256(data):
            raise ValueError("Registered authorization/start mismatch")
        ownership = json.loads((self.directory(attempt) / "OWNERSHIP.json").read_bytes())
        if ownership != {"attempt": attempt, "authorization": authorization.sha256,
                         "start_sha256": canonical_sha256(data), "study": authorization.bindings.study}:
            raise ValueError("Durable ownership/start mismatch")
        return data, authorization

    def assert_owner(self, directory):
        if self._lease is None or self._lease.closed or self._owner is None or self._owner[0] != directory:
            raise PermissionError("Writer no longer owns this attempt")
        _, authorization, ownership = self._owner
        if json.loads((directory / "OWNERSHIP.json").read_bytes()) != ownership:
            raise PermissionError("Durable ownership changed")
        started, recovered = self.validated_start(directory.name)
        if recovered != authorization or canonical_sha256(started) != ownership["start_sha256"]:
            raise PermissionError("Owner/start mismatch")
        if not (self.root / ("claim-" + authorization.bindings.study)).is_dir():
            raise PermissionError("Study claim no longer exists")
        if (directory / "INCOMPLETE.json").exists() or (directory / "RETRY_AUTHORIZATION.json").exists():
            raise PermissionError("Abandoned writer cannot publish")
        self.validate_predecessor(authorization)

    def stage_primary(self, directory, key, bundle):
        label(key)
        value = bundle.to_mapping()
        binding = json.loads((directory / (key + ".PRODUCER.json")).read_bytes())
        if canonical_sha256(value) != binding["content_sha256"]:
            raise ValueError("Primary differs from validated producer output")
        digest = immutable_write(directory / (key + ".primary.json"), value)
        return {"file": key + ".primary.json", "sha256": digest,
                "bytes": (directory / (key + ".primary.json")).stat().st_size,
                "metadata_sha256": canonical_sha256(value), "bindings": bundle.bindings.sha256}

    def publish_primary(self, directory, key, manifest):
        # Atomic create via hard link: readers never see a partially written marker.
        self.publish(directory, key + ".PRIMARY_COMMIT", manifest)

    def publish(self, directory, name, value):
        label(name.replace(".", "_"))
        if name == "SUCCESS":
            self.assert_owner(directory)
            self.verify_success(directory.name, value)
            immutable_write(directory / "TRANSITION_WITNESS.json", {"success_sha256": canonical_sha256(value)})
        stage = directory / (name + ".staging")
        immutable_write(stage, value)
        if name == "SUCCESS":
            self.assert_owner(directory)
        os.link(stage, directory / (name + ".json"))

    def reopen(self, directory, key, bindings, attempt):
        manifest = json.loads((directory / (key + ".PRIMARY_COMMIT.json")).read_bytes())
        if manifest["file"] != key + ".primary.json":
            raise ValueError("Invalid primary path")
        raw = (directory / manifest["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"] or len(raw) != manifest["bytes"]:
            raise ValueError("Primary hash/size mismatch")
        value = json.loads(raw)
        producer = json.loads((directory / (key + ".PRODUCER.json")).read_bytes())
        _, authorization = self.validated_start(attempt)
        if producer != {"schema": "validated-producer-output/1.0", "attempt": attempt,
                        "bindings": bindings.sha256, "authorization": authorization.sha256,
                        "content_sha256": canonical_sha256(value)}:
            raise ValueError("Persisted producer-output binding mismatch")
        if canonical_sha256(value) != manifest["metadata_sha256"]:
            raise ValueError("Primary metadata mismatch")
        bundle = PredictionBundle.from_mapping(value)
        if bundle.bindings != bindings or bundle.attempt != attempt or manifest["bindings"] != bindings.sha256:
            raise ValueError("Reopened identity mismatch")
        return bundle, manifest

    def official(self, attempt):
        """No success marker means no official result, regardless of orphan files."""
        directory = self.directory(attempt)
        marker = directory / "SUCCESS.json"
        if not marker.exists():
            return None
        value = json.loads(marker.read_bytes())
        witness = json.loads((directory / "TRANSITION_WITNESS.json").read_bytes())
        if witness != {"success_sha256": canonical_sha256(value)}:
            raise ValueError("Success receipt differs from transition witness")
        self.verify_success(attempt, value)
        return value

    def verify_success(self, attempt, value):
        directory = self.directory(attempt)
        started, authorization = self.validated_start(attempt)
        expected_keys = {"schema", "attempt", "study", "authorization_sha256", "predecessor", "bindings", "before", "after", "report_sha256", "completed", "accounting"}
        if set(value) != expected_keys:
            raise ValueError("Invalid success schema")
        required = {"schema": "fixture-success/1.0", "attempt": attempt, "study": authorization.bindings.study,
                    "authorization_sha256": authorization.sha256, "predecessor": authorization.predecessor,
                    "bindings": authorization.bindings.sha256, "before": authorization.before.value,
                    "after": authorization.after.value}
        if any(value[k] != v for k, v in required.items()):
            raise ValueError("Success receipt/start mismatch")
        completed = datetime.fromisoformat(value["completed"])
        if completed.tzinfo is None or completed < datetime.fromisoformat(started["started"]):
            raise ValueError("Invalid completion timestamp")
        raw = (directory / "REPORT.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != value["report_sha256"]:
            raise ValueError("Report identity mismatch")
        report = json.loads(raw)
        if report["attempt"] != attempt or canonical_sha256(report["bindings"]) != value["bindings"]:
            raise ValueError("Official report binding mismatch")
        from inverse_em.evaluation.contracts import Bindings
        bindings = Bindings(**report["bindings"])
        expected_conditions = {(None, None)} if value["after"] == "SEALED_CLEAN" else {(snr, r) for snr in (40, 30, 20, 15, 10) for r in range(10)}
        actual_conditions = [(r["snr"], r["realization"]) for r in report["records"]]
        if len(actual_conditions) != len(expected_conditions) or set(actual_conditions) != expected_conditions:
            raise ValueError("Incomplete/duplicate report conditions")
        total_rows = 0
        for record in report["records"]:
            key = "clean" if record["snr"] is None else f"r{record['realization']:02d}-snr{record['snr']}"
            bundle, manifest = self.reopen(directory, key, bindings, attempt)
            total_rows += len(bundle.ids)
            condition = bundle.condition
            if (record["complete"] is not True or record["bindings"] != bindings.sha256
                    or record["snr"] != (condition.snr if condition else None)
                    or record["realization"] != (condition.realization if condition else None)
                    or record["seed"] != (condition.seed if condition else None)):
                raise ValueError("Report condition/bundle mismatch")
            if manifest != record["primary"]:
                raise ValueError("Official primary identity mismatch")
        journals = sorted(directory.glob("batch-*.json"))
        rows = 0
        for index, path in enumerate(journals, 1):
            if path.name != f"batch-{index:04d}.json":
                raise ValueError("Incomplete batch journal")
            journal = json.loads(path.read_bytes())["accounting"]
            if type(journal["rows_processed"]) is not int or not 1 <= journal["rows_processed"] - rows <= 16 or journal["forward_calls"] != index:
                raise ValueError("Invalid batch accounting")
            rows = journal["rows_processed"]
        count = len(expected_conditions)
        expected_accounting = {"attempts_started": 1, "forward_calls": len(journals), "rows_processed": total_rows,
                               "inference_complete_traversals": count, "verified_primary_bundles": count,
                               "officially_committed_evaluations": count, "exact_execution_count_known": True}
        if rows != total_rows or canonical_sha256(value["accounting"]) != canonical_sha256(expected_accounting):
            raise ValueError("Official accounting not supported by durable evidence")

    def release_after_success(self, study):
        # Empty claim directory only. Historical authorization files remain forever.
        (self.root / ("claim-" + label(study))).rmdir()
