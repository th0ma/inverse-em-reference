from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inverse_em import __version__
from inverse_em.config import load_config
from inverse_em.errors import InverseEMError, PhaseNotImplementedError, SchemaValidationError
from inverse_em.provenance.manifest import RunManifest
from inverse_em.provenance.registry import ArtifactRegistry
from inverse_em.scientific.state import ScientificState, validate_transition


FUTURE_GROUPS = ("physics", "data", "surrogate", "classifier", "localizer", "validation", "study", "sealed-clean", "robustness", "report", "release")


def _json(path: str):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError("Document root must be a mapping")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="inverse-em", description="Inverse-EM reference infrastructure (Phase 0)")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("version")
    config = commands.add_parser("config")
    config_sub = config.add_subparsers(dest="action", required=True)
    for action in ("validate", "digest"):
        item = config_sub.add_parser(action)
        item.add_argument("path")
    manifest = commands.add_parser("manifest")
    manifest_sub = manifest.add_subparsers(dest="action", required=True)
    item = manifest_sub.add_parser("validate"); item.add_argument("path")
    registry = commands.add_parser("registry")
    registry_sub = registry.add_subparsers(dest="action", required=True)
    item = registry_sub.add_parser("validate"); item.add_argument("path")
    state = commands.add_parser("state")
    state_sub = state.add_subparsers(dest="action", required=True)
    item = state_sub.add_parser("validate"); item.add_argument("before"); item.add_argument("after")
    for group in FUTURE_GROUPS:
        commands.add_parser(group, help="Reserved for a later authorized phase")
    return root


def run(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "version":
        print(__version__); return 0
    if args.command == "config":
        config = load_config(args.path)
        print(config.sha256 if args.action == "digest" else "VALID"); return 0
    if args.command == "manifest":
        RunManifest.from_mapping(_json(args.path)); print("VALID"); return 0
    if args.command == "registry":
        ArtifactRegistry.from_mapping(_json(args.path)); print("VALID"); return 0
    if args.command == "state":
        validate_transition(ScientificState(args.before), ScientificState(args.after)); print("VALID"); return 0
    if args.command in FUTURE_GROUPS:
        raise PhaseNotImplementedError("Not implemented in Phase 0")
    raise AssertionError("unreachable")


def main() -> None:
    try:
        raise SystemExit(run())
    except (InverseEMError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

