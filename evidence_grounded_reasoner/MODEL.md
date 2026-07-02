# FishVL-4B-Thinking

The released model. A full fine-tune of `Qwen/Qwen3-VL-4B-Thinking` for
fine-grained freshwater fish species classification from video, producing an
explicit `<think>/<rethink>/<answer>` reasoning trace.

> **Licensing.** The training/evaluation **code** in this repository is MIT-licensed.
> These **model weights** are a derivative of `Qwen/Qwen3-VL-4B-Thinking` and are
> therefore governed by the upstream **Qwen3-VL model license** — not MIT. Redistribution
> and use of the weights must comply with that license. The weights are hosted externally
> (not in git); download link to be added.

| Property | Value |
|----------|-------|
| Name | **FishVL-4B-Thinking** |
| Architecture | Qwen3-VL (4B), full fine-tune |
| Precision | bfloat16 (safetensors) |
| Output format | `<think>/<rethink>/<answer>` |
| Validation macro accuracy (`data/val.json`) | **98.10%** |
| Size | ~9.6 GB |

## Location

The weights ship in this repository:

```
FishVL-4B-Thinking/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
├── added_tokens.json / special_tokens_map.json
├── preprocessor_config.json / video_preprocessor_config.json
└── chat_template.jinja
```

## Usage

Load it like any Qwen3-VL model (transformers / ms-swift), or evaluate it
directly:

```bash
CHECKPOINT_PATH=FishVL-4B-Thinking bash eval/eval.sh
```

Use the system prompt and `<think>/<rethink>/<answer>` format shown in
`eval/README.md` for inference.
