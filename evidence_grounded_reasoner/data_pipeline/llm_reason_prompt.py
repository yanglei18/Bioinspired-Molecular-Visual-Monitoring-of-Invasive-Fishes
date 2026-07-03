import json
from typing import List, Dict, Optional

class RethinkPrompt:
    """
    Build a STRICT Logic-Gated Rethink prompt.
    ENFORCES HARD BOUNDARIES:
    - Keep -> Evidence (Include even if imperfect)
    - Drop/Downgrade -> Uncertainty
    """

    def __init__(self):
        pass

    def system_prompt(self) -> str:
        return (
            "You are a Logic-Gated Reasoning Generator.\n"
            "Your Task: Construct a coherent argument based on the provided VALIDATED attributes.\n"
            "You must TRUST the validation decisions made in the previous step.\n\n"
            
            "[STRICT MAPPING RULES]\n"
            "1. **GROUP A (Evidence)**: ALL attributes with `decision: 'keep'`.\n"
            "   - CONSTRAINT: You MUST include EVERY 'keep' attribute in 'evidence_selection'.\n"
            "   - HANDLING FLAWS: If a 'keep' attribute has a note about minor deviations (e.g., 'faint spots' on an 'unspotted' fin), you MUST still use it as evidence. Explain that it matches the *primary* definition (shape/location) despite the minor variation.\n"
            "2. **GROUP B (Uncertainty)**: ALL attributes with `decision: 'drop'`, `decision: 'downgrade'`, or `status: 'not_visible'`.\n"
            "   - CONSTRAINT: Discuss these in 'uncertainty reasoning'.\n\n"
            
            "[Reasoning Procedure]\n"
            "Step 1. Evidence Selection\n"
            "- Extract ALL attributes where decision='keep'.\n"
            "Step 2. Supporting Reasoning\n"
            "- Synthesize the evidence. If an attribute is 'keep' but imperfect, argue FOR it (e.g., 'While A6 shows minor spotting, its shallow forked shape is a strong match...').\n"
            "Step 3. Exclusion / Counterfactual Reasoning\n"
            "- Use the 'Confusion Vocabulary' and your Step 1 Evidence to exclude alternatives.\n"
            "Step 4. Uncertainty Reasoning\n"
            "- Discuss ONLY the attributes from GROUP B.\n"
            "Step 5. Summary\n"
            "- Summarize using ALL codes from Step 1.\n\n"
            
            "[Output JSON Format]\n"
            "Return STRICT JSON with keys: evidence_selection, supporting_reasoning, exclusion_counterfactual, uncertainty reasoning, summary."
        )

    def rethink_prompt(
        self,
        validated_appearance: List[Dict],
        validated_behavior: List[Dict],
        confusions_vocab: List[str],
        task_spec: Dict,
    ) -> str:
        user_parts = [
            "[INPUT DATA]\n",
            "--- VALIDATED APPEARANCE (Trusted Decisions) ---",
            json.dumps(validated_appearance, indent=2, ensure_ascii=False),
            "\n--- VALIDATED BEHAVIOR (Trusted Decisions) ---",
            json.dumps(validated_behavior, indent=2, ensure_ascii=False),
            "\n--- CONFUSION VOCABULARY ---",
            "\n".join([f"- {c}" for c in confusions_vocab]),
            "\n--- TASK SPECIFICATION ---",
            json.dumps(task_spec, indent=2, ensure_ascii=False)
        ]

        user_parts.append(
            "\n[INSTRUCTION]\n"
            "Generate the JSON. Remember: If it is marked 'keep', it IS evidence. Do not move it to uncertainty."
        )

        return "\n".join(user_parts)