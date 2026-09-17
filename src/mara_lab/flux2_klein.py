from __future__ import annotations

from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
FP8_TRANSFORMER_ID = "Photoroom/FLUX.2-klein-4b-fp8-diffusers"
FP8_TRANSFORMER_REVISION = "408c457f3589e17a1be1dae5bf0dcaf09cd4985f"
FP8_TRANSFORMER_SHA256 = "13b37a5ca5cd9cf190236e7e99a3f086cf24618682f74e27a6f00cb173c308c8"
FP8_CONFIG_SHA256 = "34ad7273b27bf5be235efc070ae0d08536dc99e439a7d519399527cb618b6d45"
TORCHAO_VERSION = "0.16.0"
TEXT_ENCODER_MAX_SHARD_BYTES = 512 * 1024**2

DEFAULT_MODEL_ROOT = Path(".cache/models/flux2-klein-4b")
DEFAULT_TEXT_ENCODER_ROOT = Path(".cache/models/flux2-klein-4b/text_encoder-resharded-512mib")
DEFAULT_FP8_MODEL_ROOT = Path(".cache/models/photoroom-flux2-klein-4b-fp8")
DEFAULT_FP8_TRANSFORMER_ROOT = DEFAULT_FP8_MODEL_ROOT / "transformer_fp8_static"

MODEL_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "vae/*",
)
FP8_FILES = (
    "transformer_fp8_static/config.json",
    "transformer_fp8_static/model_fp8_static.pt",
)
