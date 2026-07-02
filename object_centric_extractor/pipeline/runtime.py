"""Runtime construction for SAM2 and GroundingDINO models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_GROUNDING_MODEL_ID = "./checkpoints/grounding-dino-base"


@dataclass
class PipelineRuntime:
    device: str
    torch: Any
    video_predictor: Any
    image_predictor: Any
    processor: Any
    grounding_model: Any


def build_runtime(
    sam2_checkpoint: str = DEFAULT_SAM2_CHECKPOINT,
    model_cfg: str = DEFAULT_SAM2_MODEL_CFG,
    grounding_model_id: str = DEFAULT_GROUNDING_MODEL_ID,
) -> PipelineRuntime:
    import torch
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        grounding_model_id,
    ).to(device)

    return PipelineRuntime(
        device=device,
        torch=torch,
        video_predictor=video_predictor,
        image_predictor=image_predictor,
        processor=processor,
        grounding_model=grounding_model,
    )
