import json
import os

GREEK_LETTERS = [
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
    'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
]

_PATH = os.path.join(os.path.dirname(__file__), "data", "progress.json")
_MASTERY = 5   # correct answers to consider a letter mastered


class ProgressTracker:
    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(_PATH):
            with open(_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {l: {"correct": 0, "attempts": 0} for l in GREEK_LETTERS}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def record(self, letter: str, correct: bool) -> None:
        entry = self._data.setdefault(letter, {"correct": 0, "attempts": 0})
        entry["attempts"] += 1
        if correct:
            entry["correct"] += 1
        self._save()

    def is_mastered(self, letter: str) -> bool:
        return self._data.get(letter, {}).get("correct", 0) >= _MASTERY

    def correct(self, letter: str) -> int:
        return self._data.get(letter, {}).get("correct", 0)

    def attempts(self, letter: str) -> int:
        return self._data.get(letter, {}).get("attempts", 0)

    def accuracy(self, letter: str) -> float:
        a = self.attempts(letter)
        return self.correct(letter) / a if a > 0 else 0.0

    def total_mastered(self) -> int:
        return sum(1 for l in GREEK_LETTERS if self.is_mastered(l))
