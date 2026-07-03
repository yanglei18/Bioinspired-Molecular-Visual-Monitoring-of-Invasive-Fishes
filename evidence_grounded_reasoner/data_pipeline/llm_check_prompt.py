import json
from typing import Dict, List, Optional

# --- 核心修改点 1: 增加完整性约束常量 ---
INTEGRITY_RULE = (
    "!!! CRITICAL INTEGRITY RULE !!!\n"
    "1. ONE-TO-ONE MAPPING: The output list MUST contain EXACTLY the same number of items as the input list.\n"
    "2. NO SILENT DELETION: If an attribute is irrelevant, invalid, or absent, you MUST output it with decision='drop'. DO NOT remove it from the list.\n"
    "3. NAME PRESERVATION: The 'name' field in the output MUST match the input exactly.\n"
    "Input Count == Output Count. This is a HARD CONSTRAINT."
)

# --- 保持不变: 宽松的评分标准 (80% 匹配原则) ---
RUBRIC_TEXT = (
    "Rubric (Score 0/1/2 per item - LENIENT/ROBUST EVALUATION):\n"
    "- evidence_text_quality:\n"
    "  2 = Clear visual cues.\n"
    "  1 = Vague cues.\n"
    "  0 = Irrelevant.\n"
    "- definition_alignment (KEY):\n"
    "  2 = Perfect match.\n"
    "  1 = SUBSTANTIAL MATCH (approx 80%+). Matches core features (shape, location, main color) but may have minor deviations (e.g., 'typically unspotted' but shows 'faint spots').\n"
    "  0 = MAJOR CONTRADICTION (e.g., wrong shape, wrong location).\n"
    "- cross_attribute_consistency:\n"
    "  2 = Consistent.\n"
    "  1 = Minor tension.\n"
    "  0 = Impossible conflict.\n"
    "reliability = (sum)/6.0."
)

# --- 保持不变: 输出 Schema ---
APPEARANCE_SCHEMA = {
  "validated_appearance": [
    {
      "name": "attribute key",
      "decision": "keep | drop | downgrade",
      "rubric scores": {
        "evidence text quality": 0,
        "definition alignment": 0,
        "cross attribute consistency": 0
      },
      "reliability": 0.0,
      "verbal confidence": "high",
      "reason": "Explain decision"
    }
  ],
  "quality_flags": [],
  "notes": "optional"
}

BEHAVIOR_SCHEMA = {
  "validated_behavior": [
    {
      "name": "attribute key",
      "decision": "keep | drop | downgrade",
      "rubric scores": {
        "evidence text quality": 0,
        "definition alignment": 0,
        "cross attribute consistency": 0
      },
      "reliability": 0.0,
      "verbal confidence": "high",
      "reason": "Explain decision"
    }
  ],
  "quality_flags": [],
  "notes": "optional"
}

def _json(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

class _BaseLLMValidator:
    def __init__(self, expert_definitions: Dict[str, str], rules: Optional[Dict] = None):
        self.E = expert_definitions
        self.rules = rules

class AppearanceLLMCheckPrompt(_BaseLLMValidator):
    def appearance_system(self) -> str:
        # --- 修改点 2: System Prompt 中强调不删除 ---
        return (
            "You are a ROBUST Evidence-Text Auditor.\n"
            "Goal: Validate if the attribute is 'Mainly Correct' (approx >80% match).\n"
            "Rules:\n"
            "1. Tolerate minor deviations IF the core definition is correct.\n"
            "2. Decision 'KEEP' means 'This is valid evidence'.\n"
            "3. [IMPORTANT] Never delete an attribute. Use 'drop' for rejections. Output size MUST match Input size.\n" 
            "4. Return STRICT JSON only."
        )

    def appearance_prompt(self, video_id: str, appearance_candidates: List[Dict]) -> str:
        # --- 修改点 3: 动态计算输入数量 ---
        input_count = len(appearance_candidates)
        
        header = "[Expert Knowledge]\n" + _json(self.E)
        header += "\n\n[Candidates]\n" + _json(appearance_candidates)
        header += "\n\n[ROBUST DECISION RULES]\n" + RUBRIC_TEXT
        
        parts = [header]
        # --- 修改点 4: 在 Prompt 中明确指出应输出的数量 ---
        parts.append(f"[Task Logic]\n"
                     f"You received {input_count} candidate attributes.\n" 
                     f"You MUST output exactly {input_count} validated items.\n\n"
                     "Evaluate each candidate:\n"
                     "1. Check Status:\n"
                     "   - If 'not_visible' -> Decision: DROP (or DOWNGRADE if unsure).\n"
                     "   - If 'absent' -> Decision: DROP.\n"
                     "2. If 'present', Apply the '80% RULE' (Score & Decide).\n"
                     "3. Provide reasoning.")
        
        # --- 修改点 5: 插入完整性规则 ---
        parts.append(f"[CONSTRAINT]\n{INTEGRITY_RULE}")
        
        parts.append("[Output Schema]\n" + _json(APPEARANCE_SCHEMA))
        return "\n\n".join(parts)

class BehaviorLLMCheckPrompt(_BaseLLMValidator):
    def behavior_system(self) -> str:
        # --- 修改点 6: System Prompt 中强调不删除 ---
        return (
            "You are a ROBUST Evidence-Text Auditor for Behavior.\n"
            "Goal: Validate if the behavior is substantially present.\n"
            "Rules: If the action matches, mark as KEEP. If not, mark as DROP.\n"
            "CRITICAL: Do NOT remove items. Input List Size MUST EQUAL Output List Size.\n"
            "Return STRICT JSON only."
        )

    def behavior_prompt(self, video_id: str, behavior_candidates: List[Dict]) -> str:
        # --- 修改点 7: 动态计算输入数量 ---
        input_count = len(behavior_candidates)
        
        header = "[Expert Knowledge]\n" + _json(self.E)
        header += "\n\n[Candidates]\n" + _json(behavior_candidates)
        header += "\n\n[ROBUST DECISION RULES]\n" + RUBRIC_TEXT
        
        parts = [header]
        # --- 修改点 8: 在 Prompt 中明确指出应输出的数量 ---
        parts.append(f"[Task Logic]\n"
                     f"You received {input_count} candidate behaviors.\n"
                     f"You MUST output exactly {input_count} validated items.\n\n"
                     "Evaluate each candidate:\n"
                     "1. Check Status:\n"
                     "   - If 'not_visible' -> DROP.\n"
                     "   - If 'absent' -> DROP.\n"
                     "2. If 'present', Apply '80% RULE'.\n"
                     "3. Provide reasoning.")
        
        # --- 修改点 9: 插入完整性规则 ---
        parts.append(f"[CONSTRAINT]\n{INTEGRITY_RULE}")
        
        parts.append("[Output Schema]\n" + _json(BEHAVIOR_SCHEMA))
        return "\n\n".join(parts)