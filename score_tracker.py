import json
import os
from datetime import datetime

_PATH      = os.path.join(os.path.dirname(__file__), "data", "scores.json")
_MAX_SAVED = 5


class ScoreTracker:
    def __init__(self):
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(_PATH):
            with open(_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {"high_scores": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def add_score(self, score: int, letters_done: int, time_taken: float) -> int:
        """Save result; return rank (1-based). Returns 0 if not in top list."""
        entry = {
            "score":        score,
            "letters":      letters_done,
            "time":         round(time_taken, 1),
            "date":         datetime.now().strftime("%d/%m/%Y"),
        }
        scores = self._data["high_scores"]
        scores.append(entry)
        scores.sort(key=lambda x: (-x["score"], x["time"]))
        self._data["high_scores"] = scores[:_MAX_SAVED]
        self._save()
        try:
            return self._data["high_scores"].index(entry) + 1
        except ValueError:
            return 0

    def high_scores(self) -> list[dict]:
        return self._data["high_scores"]

    def best_score(self) -> int:
        scores = self._data["high_scores"]
        return scores[0]["score"] if scores else 0
