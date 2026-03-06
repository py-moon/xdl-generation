import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from verifier import verify


def _stable_accuracy(history: Sequence[float], window: int, tolerance: float) -> bool:
    if len(history) < window:
        return False
    window_values = history[-window:]
    return max(window_values) - min(window_values) <= tolerance


class ConstraintBootstrapTrainer:
    """Iteratively learns syntax/parameter constraints from verifier failures."""

    def __init__(
        self,
        generate_xdl_fn: Callable[[str, str], str],
        suggest_constraints_fn: Callable[[str], List[str]],
        verify_xdl_fn: Callable[[str], List[Dict]] = verify.verify_xdl,
    ) -> None:
        self.generate_xdl_fn = generate_xdl_fn
        self.suggest_constraints_fn = suggest_constraints_fn
        self.verify_xdl_fn = verify_xdl_fn

    def train(
        self,
        samples: Sequence[Dict[str, str]],
        rounds: int = 10,
        stable_window: int = 3,
        stable_tolerance: float = 0.01,
        min_rounds: int = 3,
    ) -> Dict:
        constraints: List[str] = []
        accuracy_history: List[float] = []
        round_logs: List[Dict] = []

        for round_idx in range(rounds):
            total = len(samples)
            num_correct = 0
            failures = []

            constraints_text = "\n".join(f"- {item}" for item in constraints)
            for sample in samples:
                description = sample["description"]
                generated_xdl = self.generate_xdl_fn(description, constraints_text)
                errors = self.verify_xdl_fn(generated_xdl)

                if not errors:
                    num_correct += 1
                    continue

                failures.append(
                    {
                        "description": description,
                        "generated_xdl": generated_xdl,
                        "errors": errors,
                    }
                )

            accuracy = num_correct / max(total, 1)
            accuracy_history.append(accuracy)

            new_constraints = []
            for failure in failures:
                prompt_payload = {
                    "task_description": failure["description"],
                    "invalid_xdl": failure["generated_xdl"],
                    "verifier_errors": failure["errors"],
                    "current_constraints": constraints,
                    "instruction": (
                        "Extract missing syntax and parameter-validity constraints as short rules. "
                        "Return only actionable constraints."
                    ),
                }
                candidate_constraints = self.suggest_constraints_fn(
                    json.dumps(prompt_payload, ensure_ascii=False)
                )
                for item in candidate_constraints:
                    stripped = item.strip()
                    if stripped and stripped not in constraints and stripped not in new_constraints:
                        new_constraints.append(stripped)

            constraints.extend(new_constraints)
            round_logs.append(
                {
                    "round": round_idx + 1,
                    "accuracy": accuracy,
                    "num_correct": num_correct,
                    "num_total": total,
                    "new_constraints": new_constraints,
                    "num_failures": len(failures),
                }
            )

            if round_idx + 1 >= min_rounds and _stable_accuracy(
                accuracy_history, stable_window, stable_tolerance
            ):
                break

        return {
            "constraints": constraints,
            "accuracy_history": accuracy_history,
            "round_logs": round_logs,
            "is_stable": _stable_accuracy(accuracy_history, stable_window, stable_tolerance),
        }


def load_training_samples(input_dir: str) -> List[Dict[str, str]]:
    samples = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f:
            samples.append({"id": filename, "description": f.read()})
    return samples
