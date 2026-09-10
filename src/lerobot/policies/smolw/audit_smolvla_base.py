"""Compare a local, untrained SmolW base with its original SmolVLA artifact.

Reads one tensor pair at a time on CPU; no model, tokenizer or VidTwin loading.
Only artifact transfer is certified, not training settings or LIBERO success.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def compare_tensors(source: Path, target: Path, *, allow_extra: bool = False) -> list[str]:
    issues = []
    with (
        safe_open(source, framework="pt", device="cpu") as left,
        safe_open(target, framework="pt", device="cpu") as right,
    ):
        left_keys, right_keys = set(left.keys()), set(right.keys())
        issues.extend(f"missing: {key}" for key in sorted(left_keys - right_keys))
        if not allow_extra:
            issues.extend(f"extra: {key}" for key in sorted(right_keys - left_keys))
        for key in sorted(left_keys & right_keys):
            a, b = left.get_tensor(key), right.get_tensor(key)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                issues.append(f"different dtype, shape or values: {key}")
    return issues


def compare_processors(source: Path, target: Path, name: str) -> list[str]:
    left = json.loads((source / f"{name}.json").read_text())["steps"]
    right = json.loads((target / f"{name}.json").read_text())["steps"]
    # This is the one intentional legacy conversion step removed at A load.
    right = [step for step in right if step.get("registry_name") != "smolw_stationary_action_padding"]
    if len(left) != len(right):
        return [f"{name}: different step counts"]
    issues = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        a, b = dict(a), dict(b)
        a_state, b_state = a.pop("state_file", None), b.pop("state_file", None)
        if a != b:
            issues.append(f"{name}[{index}]: different step configuration")
        if bool(a_state) != bool(b_state):
            issues.append(f"{name}[{index}]: missing processor state")
        elif a_state:
            issues.extend(
                f"{name}[{index}]: {issue}" for issue in compare_tensors(source / a_state, target / b_state)
            )
    return issues


def audit(source: Path, target: Path) -> dict:
    source, target = source.expanduser().resolve(), target.expanduser().resolve()
    source_config = json.loads((source / "config.json").read_text())
    target_config = json.loads((target / "config.json").read_text())
    if source_config.get("type") != "smolvla" or target_config.get("type") != "smolw":
        raise ValueError("Expected an original SmolVLA source and a converted SmolW target.")
    issues = compare_tensors(source / "model.safetensors", target / "model.safetensors", allow_extra=True)
    for name in (POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME):
        issues.extend(compare_processors(source, target, name))
    return {
        "source": str(source),
        "target": str(target),
        "weights_and_processors_match": not issues,
        "issues": issues,
        "shared_config_differences_before_launch_overrides": {
            key: {"source": source_config[key], "target": target_config[key]}
            for key in sorted(source_config.keys() & target_config.keys())
            if source_config[key] != target_config[key]
        },
        "scope": "Artifact transfer only; compare effective training/eval configs separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source, args.target)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["weights_and_processors_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
