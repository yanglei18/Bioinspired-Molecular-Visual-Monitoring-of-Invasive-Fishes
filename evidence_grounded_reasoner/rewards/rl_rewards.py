import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

from openai import OpenAI
from swift.rewards import ORM, orms

logger = logging.getLogger(__name__)


class FactCheckingJudgeReward(ORM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.environ["JUDGE_API_KEY"]
        api_base = os.environ["JUDGE_API_BASE"]
        self.model = os.environ["JUDGE_MODEL"]
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.lambda_think = float(os.getenv("R_THINK_WEIGHT", "0.5"))
        self.lambda_rethink = float(os.getenv("R_RETHINK_WEIGHT", "0.5"))
        self.max_workers = int(os.getenv("JUDGE_MAX_WORKERS", "8"))
        self.max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "3"))

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return value if isinstance(value, str) else ""

    def _call_judge(self, prompt: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                response_text = response.choices[0].message.content or ""
                json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
                if not json_match:
                    logger.warning("Judge response did not contain JSON: %s", response_text[:200])
                    return {"R_think": 0.0, "R_rethink": 0.0}
                return json.loads(json_match.group(0))
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    logger.warning("Judge call failed after retries: %s", exc)
                    return {"R_think": 0.0, "R_rethink": 0.0}
                time.sleep(2 ** attempt)
        return {"R_think": 0.0, "R_rethink": 0.0}

    def _evaluate_single(self, completion: str, reference_reasoning: str) -> Tuple[float, float, float]:
        gen_think = self._extract_tag(completion, "think")
        gen_rethink = self._extract_tag(completion, "rethink")
        ref_think = self._extract_tag(reference_reasoning, "think")
        ref_rethink = self._extract_tag(reference_reasoning, "rethink")

        if not gen_think and not gen_rethink:
            return 0.0, 0.0, 0.0
        if not ref_think and not ref_rethink:
            return 0.0, 0.0, 0.0

        prompt = f"""You are an expert fact-checking judge evaluating a vision-language model's fish-identification reasoning against an expert-derived reference trajectory.

[Expert reference]
<ref_think>
{ref_think}
</ref_think>
<ref_rethink>
{ref_rethink}
</ref_rethink>

[Model generation]
<gen_think>
{gen_think}
</gen_think>
<gen_rethink>
{gen_rethink}
</gen_rethink>

Evaluate two reward components:
1. R_think: factual evidence alignment. Reward concrete diagnostic appearance and behavioral attributes that are supported by the reference evidence. Penalize unsupported attribute claims as hallucinations.
2. R_rethink: fact-based alternative exclusion. Reward exclusions of plausible look-alike species only when they are justified by observable counter-evidence from the reference. Penalize unsupported deductive leaps.

Hallucination, JustifiedExclusion, and unsupported deductive leaps are rubric criteria for judge-based scoring, not deterministic set-matching quantities.

Return only a JSON object with exactly these keys and float values between 0.0 and 1.0:
{{"R_think": 0.0, "R_rethink": 0.0}}
"""
        scores = self._call_judge(prompt)
        try:
            r_think = min(max(float(scores.get("R_think", 0.0)), 0.0), 1.0)
            r_rethink = min(max(float(scores.get("R_rethink", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            r_think, r_rethink = 0.0, 0.0
        total = self.lambda_think * r_think + self.lambda_rethink * r_rethink
        return total, r_think, r_rethink

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        reasoning_contents = kwargs.get("reasoning_content", [])
        solutions = kwargs.get("solution", [])
        references = []

        max_len = max(len(reasoning_contents), len(solutions))
        for i in range(max_len):
            reasoning = self._normalize_text(reasoning_contents[i] if i < len(reasoning_contents) else "")
            solution = self._normalize_text(solutions[i] if i < len(solutions) else "")
            references.append(reasoning or solution)

        if references and len(references) < len(completions):
            repeats = len(completions) // len(references)
            references = [ref for ref in references for _ in range(repeats)]
        references.extend([""] * (len(completions) - len(references)))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._evaluate_single, completions, references[:len(completions)]))

        rewards = []
        for idx, (total, r_think, r_rethink) in enumerate(results):
            logger.info("sample=%d R_total=%.4f R_think=%.4f R_rethink=%.4f", idx, total, r_think, r_rethink)
            rewards.append(float(total))
        return rewards


class AnswerAccuracyReward(ORM):
    @staticmethod
    def _extract_answer(text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
        answer = match.group(1) if match else text or ""
        answer = answer.strip().lower()
        answer = re.sub(r"^(answer is|answer:|the answer is)\s*", "", answer)
        return answer.strip(".。 ,\n\t")

    def __call__(self, completions: List[str], solution: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion, target in zip(completions, solution):
            target_text = str(target[0]) if isinstance(target, list) and target else str(target)
            pred_answer = self._extract_answer(completion)
            target_answer = self._extract_answer(target_text)
            rewards.append(1.0 if pred_answer and pred_answer == target_answer else 0.0)
        return rewards


orms["fact_checking_judge"] = FactCheckingJudgeReward
orms["answer_accuracy"] = AnswerAccuracyReward
