import re
from typing import List, Optional

try:
    from swift.plugin import ORM, orms
except ImportError:
    from swift.rewards.orm import ORM, orms


OPTION_PATTERN = re.compile(r"\(([A-J])\)\s*([^\n<]+)?", re.IGNORECASE)

# Heavy weights on the 3 weakest species to maximize learning signal
_REWARD_WEIGHTS = {
    "black carp":         3.0,   # was 1.1 — most critical to improve
    "mud carp":           3.0,   # was 2.0
    "chinese sucker":     2.5,   # was 1.0
    "schizothorax fish":  2.0,   # was 1.5
    "redeye barbel":      1.5,   # was 1.8
    "serrated barb":      1.0,   # was 3.0 (already good at 96.5%)
    "wuchang bream":      1.0,   # was 1.3 (already good at 96.1%)
    "chinese labeo":      1.0,   # was 1.1 (already good at 98.1%)
}


def _extract_option_label(text: str) -> Optional[str]:
    if not text:
        return None
    if isinstance(text, (list, tuple)):
        if not text:
            return None
        text = text[0]
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1)
    matches = OPTION_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1][0].upper()


def _gold_species(gold_text: str) -> Optional[str]:
    """Return lower-case species name from a gold answer string like '(E) mud carp'."""
    m = re.search(r"\(([A-J])\)\s*([^\n<(]+)", gold_text or "", re.IGNORECASE)
    return m.group(2).strip().lower() if m else None


class FishFinalAnswerAccuracy(ORM):

    def __call__(self, completions, ground_truth=None, solution=None, answer=None, **kwargs) -> List[float]:
        targets = ground_truth if ground_truth is not None else (solution if solution is not None else answer)
        if targets is None:
            raise ValueError("FishFinalAnswerAccuracy requires `ground_truth`, `solution`, or `answer`.")

        rewards = []
        for completion, gold in zip(completions, targets):
            pred_label = _extract_option_label(completion)
            gold_label = _extract_option_label(gold)
            base = 1.0 if pred_label is not None and pred_label == gold_label else 0.0
            weight = _REWARD_WEIGHTS.get(_gold_species(gold), 1.0)
            rewards.append(base * weight)
        return rewards


orms["fish_final_answer_accuracy"] = FishFinalAnswerAccuracy
