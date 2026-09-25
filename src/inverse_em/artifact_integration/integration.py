"""Transactional synthetic-byte authority. Intentionally no production import API."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .contracts import Context, Domain, EvidenceRegistry, Identity, Outcome, Use, identity, require
from .planning import Decision
from .validation import plain, validate


MAX_BYTES = 64 * 1024 * 1024


def _commit_read_lock(path):
    """Windows mandatory deny-write/deny-delete handle, held through commit.

    Advisory POSIX locks do not establish this guarantee against other writers;
    an unimplemented platform must fail closed, not silently weaken publication.
    """
    if os.name != "nt":
        raise PermissionError("Mandatory publication locking is implemented only for Windows")
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    # GENERIC_READ, FILE_SHARE_READ only: deny content writes and deletion of
    # this opened name. Each required pathname needs its own protection handle.
    handle = create(str(path), 0x80000000, 1, None, 3, 0x00200080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close(handle)
        raise
    try:
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, "Bounded bytes required")
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


@dataclass(frozen=True)
class Source:
    name: str
    identity: Identity
    size: int


@dataclass(frozen=True)
class AcceptedSynthetic:
    """Only validated serialized replacement state; no mutation/restoration methods."""
    document: bytes
    receipt: bytes


@dataclass(frozen=True)
class Result:
    decision: Decision
    accepted: AcceptedSynthetic | None


class ReleaseFailure(OSError):
    """Explicitly retry cleanup.close(); committed acceptance is not rollback."""
    def __init__(self, cleanup, errors):
        super().__init__("Accepted release failed" if cleanup.committed else "Uncommitted cleanup failed")
        self.cleanup = cleanup
        self.committed = cleanup.committed
        self.errors = tuple(errors)


class _PublicationResources:
    def __init__(self, temporary):
        self.handles = []
        self.paths = [Path(temporary)]
        self.committed = False
        self.released = False

    def close(self):
        if self.released:
            return
        errors = []
        for handle in reversed(self.handles):
            if not handle.closed:
                try:
                    handle.close()
                except BaseException as error:
                    errors.append(error)
        # A failed close may leave mandatory protection active. Preserve its
        # ownership for explicit retry, rather than relying on finalization.
        if all(handle.closed for handle in self.handles):
            for path in self.paths:
                try:
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    errors.append(error)
        elif not errors:
            errors.append(RuntimeError("Protection handle did not close"))
        if errors:
            raise ReleaseFailure(self, errors)
        self.released = True


class ProtectedResult:
    """Caller-owned protection: use with integrate(...) or explicitly close()."""
    def __init__(self, decision, accepted, resources):
        self._decision = decision
        self._accepted = accepted
        self._resources = resources
        self._state = "OPEN"

    @property
    def decision(self):
        return self._decision

    @property
    def state(self):
        return self._state

    @property
    def accepted(self):
        if self.state != "OPEN":
            raise RuntimeError("Usable state access requires an OPEN protected lease")
        return self._accepted

    def close(self):
        if self.state == "CLOSED":
            return
        try:
            self._resources.close()
        except BaseException:
            self._state = "RELEASE_FAILED"
            raise
        self._state = "CLOSED"

    def __enter__(self):
        if self.state != "OPEN":
            raise RuntimeError("Lease is not OPEN")
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _protected_handoff(resources, source_raw, envelope, destination):
    """Terminal checks after evidence publication; ownership crosses return.

    The staging writer is already flushed and closed. A separate accepted-path
    handle is required: deny-delete protection of one hard-link name must not be
    assumed to protect another name. Release only the staging handle and unlink
    its temporary name while source and accepted-path handles remain protected.
    """
    for handle, expected in zip(resources.handles, (source_raw, envelope, envelope)):
        handle.seek(0)
        require(handle.read(MAX_BYTES + 1) == expected, "Protected handoff bytes changed")
    require(destination.read_bytes() == envelope, "Protected destination changed")
    resources.handles[1].close()
    resources.paths[-1].unlink()
    require(destination.read_bytes() == envelope, "Protected cleanup changed destination")


class RejectedArtifact(ValueError):
    """Rejection carries a use-bound decision but never any usable state."""
    def __init__(self, decision):
        super().__init__(decision.reason)
        self.decision = decision


class SyntheticStore:
    """Owns a new temporary directory and only accepts sources it created.

    This is an accidental-misuse firewall, not a sandbox against hostile Python
    executing in the same interpreter. Arbitrary paths and pickle are unsupported.
    """
    def __init__(self, parent=None, *, evidence=None):
        require(evidence is None or type(evidence) is EvidenceRegistry, "Supported evidence registry required")
        self._evidence = EvidenceRegistry() if evidence is None else evidence
        self._root = Path(tempfile.mkdtemp(prefix="phase10-synthetic-", dir=parent))
        self._sources = {}
        self._contexts = {}

    @property
    def root(self):
        return self._root

    def create_source(self, name, document, context):
        require(type(context) is Context, "Mandatory context")
        digest = context.sha256
        require(type(name) is str and name.isascii() and name.replace("_", "").isalnum(), "Simple synthetic source name")
        require(type(document) is dict and document.get("scope") == "SYNTHETIC_ONLY", "Synthetic documents only")
        raw = encode(document)
        require(len(raw) <= MAX_BYTES, "Bounded synthetic artifact")
        source = Source(name, Identity(Domain.SOURCE, "synthetic-source-bytes/1", hashlib.sha256(raw).hexdigest()), len(raw))
        with (self._root / (name + ".source.json")).open("xb") as stream:
            stream.write(raw)
        self._sources[name] = source
        self._contexts[name] = digest
        return source

    def _read(self, source):
        require(type(source) is Source and self._sources.get(source.name) == source, "Unregistered source / external path")
        path = self._root / (source.name + ".source.json")
        require(not path.is_symlink(), "Source link forbidden")
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        require(len(raw) == source.size and hashlib.sha256(raw).hexdigest() == source.identity.digest, "Source mutation detected")
        return raw

    def integrate(self, source, context):
        require(type(context) is Context, "Mandatory context")
        context_digest = context.sha256
        require(type(source) is Source and self._contexts.get(source.name) == context_digest, "Unregistered or modified context")
        raw = self._read(source)
        document = decode(raw)
        try:
            outcome, reason = validate(document, context, self._evidence)
        except (ValueError, TypeError, KeyError) as error:
            evidence = identity(Domain.NATIVE, "synthetic-rejection-evidence/1", (source, context_digest, str(error)))
            decision = Decision(source.identity, context_digest, context.use, Outcome.INSUFFICIENT_EVIDENCE,
                                (evidence,), str(error))
            raise RejectedArtifact(decision) from error
        self._read(source)  # Detect replacement/mutation during validation.
        evidence = identity(Domain.NATIVE, "synthetic-validation-evidence/1", (source, context_digest, outcome, reason))
        decision = Decision(source.identity, context_digest, context.use, outcome, (evidence,), reason)
        if outcome is not Outcome.DIRECTLY_COMPATIBLE:
            return Result(decision, None)
        destination_identity = identity(Domain.NATIVE, "validated-synthetic-document/1", document)
        transaction_evidence = EvidenceRegistry()
        transaction_evidence._records = self._evidence._records.copy()
        receipt = {"scope": "SYNTHETIC_ONLY", "decision": plain(asdict(decision)), "source_size": source.size,
                   "destination": plain(asdict(destination_identity)),
                   "mapping": plain(asdict(transaction_evidence._publication(source.identity, destination_identity, context_digest, context.use)))}
        accepted = None if context.use is Use.PROVENANCE_ONLY else AcceptedSynthetic(raw, encode(receipt))
        envelope = encode({"document": document, "receipt": receipt})
        destination = self._root / (source.name + ".accepted.json")
        # Exclusive atomic link publishes document and receipt together, never overwrites.
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self._root)
        resources = _PublicationResources(temporary)
        previous_evidence = self._evidence._records.copy()
        try:
            try:
                writer = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                raise
            with writer as stream:
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            staged = Path(temporary).read_bytes()
            require(staged == envelope, "Destination byte verification")
            require(validate(decode(staged)["document"], context, self._evidence)[0] is Outcome.DIRECTLY_COMPATIBLE, "Destination validation")
            source_path = self._root / (source.name + ".source.json")
            resources.handles.append(_commit_read_lock(source_path))
            resources.handles.append(_commit_read_lock(temporary))
            require(resources.handles[0].read(MAX_BYTES + 1) == raw, "Final source mutation")
            require(resources.handles[1].read(MAX_BYTES + 1) == envelope, "Final destination mutation")
            self._read(source)
            os.link(temporary, destination)
            resources.paths.insert(0, destination)  # Roll back only our new link.
            resources.handles.append(_commit_read_lock(destination))
            require(os.path.samefile(temporary, destination) and destination.read_bytes() == envelope,
                    "Committed bytes differ from validated bytes")
            self._evidence._records.update(transaction_evidence._records)
            _protected_handoff(resources, raw, envelope, destination)
            # Construct the return value before committing ownership; no unlock,
            # fallible cleanup or finalizer is interposed before the return.
            result = ProtectedResult(decision, accepted, resources) if accepted is not None else Result(decision, None)
            resources.paths.clear()  # Protected cleanup removed the staging link.
            resources.committed = True  # Acceptance is now irrevocable.
        except BaseException:
            self._evidence._records = previous_evidence
            resources.close()  # Failure retains deterministic retry ownership.
            raise
        if accepted is None:
            # Provenance-only has no usable state and needs no caller lease.
            # A release error explicitly identifies committed acceptance.
            resources.close()
        return result


def load_production(*args, **kwargs):
    raise PermissionError("Production artifact acceptance is not authorized or implemented in Phase 10")
