from __future__ import annotations

import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
import warnings
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from importlib import metadata
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from PIL import Image

from mara_lab.artifacts import atomic_write_json, sha256_file, utc_now
from mara_lab.backends.reference import ReferenceImageBackend
from mara_lab.config import ResolvedReferenceExperimentConfig
from mara_lab.disk_offload import (
    add_cpu_values_to_store,
    alias_wrapped_base_weights,
    attach_disk_offload,
    offload_photomaker_lora,
    prepare_model_store,
    read_offload_index,
    validate_offload_store,
    write_state_store,
)
from mara_lab.domain import (
    BackendInfo,
    CompiledPrompt,
    GenerationResult,
    ReferenceGenerationRequest,
    ReferenceShot,
    RuntimeMetrics,
)
from mara_lab.errors import BackendUnavailableError, PromptTooLongError
from mara_lab.workflows.prepare_model import prepare_diffusers_model

_GIB = 1024**3
_OFFLOAD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _OffloadLayout:
    root: Path

    @property
    def ready_path(self) -> Path:
        return self.root / "ready.json"

    @property
    def lora_path(self) -> Path:
        return self.root / "photomaker-lora.fp16.safetensors"

    def component(self, name: str) -> Path:
        return self.root / name


def compile_photomaker_prompt(
    shot: ReferenceShot, trigger_word: str, template_version: str = "v1"
) -> str:
    if template_version == "v1":
        return (
            f"unretouched {shot.capture_style} {shot.framing} portrait photo of an adult "
            f"woman {trigger_word}, {shot.view}, {shot.expression}, wearing {shot.wardrobe}, "
            f"in {shot.setting}, lit by {shot.lighting}, {shot.camera_behavior}, natural skin "
            "texture, realistic exposure"
        )
    if template_version != "v2":
        raise BackendUnavailableError(f"unsupported PhotoMaker prompt template: {template_version}")
    return (
        f"unretouched {shot.capture_style} {shot.framing} photo of adult "
        f"woman {trigger_word}, {shot.view}, {shot.expression}, wearing {shot.wardrobe}, "
        f"{shot.setting}, {shot.lighting}, {shot.camera_behavior}, natural skin"
    )


def _model_revision(config: ResolvedReferenceExperimentConfig) -> str:
    return config.model.resolved_revision or config.model.revision


def _materialized_model_path(config: ResolvedReferenceExperimentConfig) -> Path:
    model_slug = "models--" + config.model.model_id.replace("/", "--")
    revision = _model_revision(config)
    return (
        config.model.materialized_cache_dir.resolve()
        / model_slug
        / "snapshots"
        / f"{revision}-fp16-{config.model.materialized_max_shard_mib}m"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackendUnavailableError(f"expected a JSON object in {path}")
    return value


def _installed_photomaker_revision() -> str | None:
    try:
        distribution = metadata.distribution("photomaker")
        direct_url_text = distribution.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if isinstance(commit, str):
        return commit

    source_url = direct_url.get("url")
    if not isinstance(source_url, str) or not source_url.startswith("file:"):
        return None
    source_path = unquote(urlparse(source_url).path)
    if os.name == "nt" and len(source_path) >= 3 and source_path[0] == "/":
        source_path = source_path[1:]
    try:
        result = subprocess.run(
            ["git", "-C", source_path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _load_cached_embedding(path: Path) -> Any | None:
    try:
        import numpy as np

        embedding = np.load(path, allow_pickle=False)
    except (ImportError, OSError, ValueError):
        return None
    if (
        embedding.shape != (1, 512)
        or embedding.dtype != np.float32
        or not np.isfinite(embedding).all()
    ):
        return None
    return embedding


def _extract_reference_embeddings(
    config: ResolvedReferenceExperimentConfig,
) -> tuple[Any, list[Path]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise BackendUnavailableError(
            "PhotoMaker reference dependencies are missing. Run `uv sync --extra reference`."
        ) from exc

    cache_root = (
        config.conditioner.insightface_cache_dir.resolve().parent
        / "embeddings"
        / config.conditioner.insightface_model
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    embeddings: list[Any] = []
    embedding_paths: list[Path] = []
    for image_path in config.reference_images:
        image_hash = sha256_file(image_path)
        output_path = cache_root / f"{image_hash}.npy"
        embedding = _load_cached_embedding(output_path)
        if embedding is None:
            environment = os.environ.copy()
            environment.setdefault(
                "MPLCONFIGDIR",
                str(config.conditioner.insightface_cache_dir.resolve().parent / "matplotlib"),
            )
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mara_lab.face_embedding_worker",
                        "--image",
                        str(image_path),
                        "--output",
                        str(output_path),
                        "--root",
                        str(config.conditioner.insightface_cache_dir.resolve()),
                        "--model",
                        config.conditioner.insightface_model,
                        "--provider",
                        config.conditioner.insightface_provider,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=environment,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise BackendUnavailableError(
                    f"cannot run the isolated face detector for {image_path}: {exc}"
                ) from exc
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()
                raise BackendUnavailableError(
                    f"face detector rejected {image_path}: {detail[-2000:]}"
                )
            embedding = _load_cached_embedding(output_path)
            if embedding is None:
                raise BackendUnavailableError(
                    f"face detector did not create a valid embedding for {image_path}"
                )
        embeddings.append(embedding)
        embedding_paths.append(output_path)
    return np.concatenate(embeddings, axis=0), embedding_paths


class PhotoMakerV2Backend(ReferenceImageBackend):
    def __init__(self, config: ResolvedReferenceExperimentConfig) -> None:
        self._config = config
        self._pipe: Any = None
        self._torch: Any = None
        self._id_embeds: Any = None
        self._reference_images: list[Image.Image] = []
        self._info: BackendInfo | None = None
        self._execution_mode: str | None = None

    def _ensure_materialized_model(self) -> tuple[Path, str]:
        model_path = _materialized_model_path(self._config)
        manifest_path = model_path / "mara-materialization.json"
        if not manifest_path.is_file():
            summary = prepare_diffusers_model(self._config.model)
            model_path = summary.output_dir
            manifest_path = summary.manifest_path
        manifest = _load_json(manifest_path)
        expected = {
            "model_id": self._config.model.model_id,
            "revision": _model_revision(self._config),
            "variant": "fp16",
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise BackendUnavailableError(
                f"materialized model manifest does not match the experiment: {mismatches}"
            )
        return model_path, sha256_file(manifest_path)

    def _download_conditioner(self) -> Path:
        try:
            import truststore
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise BackendUnavailableError(
                "PhotoMaker reference dependencies are missing. Run `uv sync --extra reference`."
            ) from exc

        truststore.inject_into_ssl()
        arguments = {
            "repo_id": self._config.conditioner.model_id,
            "filename": self._config.conditioner.weight_file,
            "revision": self._config.conditioner.revision,
            "cache_dir": self._config.conditioner.cache_dir.resolve(),
        }
        try:
            try:
                checkpoint = Path(hf_hub_download(**arguments, local_files_only=True))
            except Exception:
                checkpoint = Path(hf_hub_download(**arguments))
        except Exception as exc:
            raise BackendUnavailableError(f"cannot load PhotoMaker V2 weights: {exc}") from exc
        actual_hash = sha256_file(checkpoint)
        if actual_hash != self._config.conditioner.weight_sha256:
            raise BackendUnavailableError(
                "PhotoMaker V2 checkpoint hash does not match the conditioner profile"
            )
        return checkpoint

    def _select_execution_mode(self, torch: Any) -> str:
        configured = self._config.conditioner.execution_mode
        if configured != "auto":
            return configured
        try:
            import psutil

            available_ram = psutil.virtual_memory().available
        except ImportError:
            available_ram = 0
        total_vram = torch.cuda.get_device_properties(0).total_memory
        if total_vram <= 16 * _GIB or available_ram < 24 * _GIB:
            return "disk"
        return "direct"

    def _offload_layout(self) -> _OffloadLayout:
        key = (
            f"{_model_revision(self._config)[:12]}-"
            f"{self._config.conditioner.weight_sha256[:12]}-fp16"
        )
        return _OffloadLayout(
            self._config.conditioner.offload_cache_dir.resolve() / "photomaker-v2" / key
        )

    def _expected_store_identity(self, materialized_hash: str) -> dict[str, Any]:
        return {
            "schema_version": _OFFLOAD_SCHEMA_VERSION,
            "base_model_id": self._config.model.model_id,
            "base_revision": _model_revision(self._config),
            "materialized_manifest_sha256": materialized_hash,
            "conditioner_model_id": self._config.conditioner.model_id,
            "conditioner_revision": self._config.conditioner.revision,
            "conditioner_weight_sha256": self._config.conditioner.weight_sha256,
            "source_revision": self._config.conditioner.source_revision,
            "dtype": "float16",
        }

    def _validate_ready_store(
        self, layout: _OffloadLayout, materialized_hash: str
    ) -> dict[str, Any]:
        if not layout.ready_path.is_file():
            raise BackendUnavailableError(
                f"incomplete PhotoMaker disk store at {layout.root}; "
                "move that cache directory aside and retry"
            )
        manifest = _load_json(layout.ready_path)
        expected = self._expected_store_identity(materialized_hash)
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise BackendUnavailableError(
                f"PhotoMaker disk store identity mismatch at {layout.root}: {mismatches}"
            )
        if not layout.lora_path.is_file():
            raise BackendUnavailableError(f"PhotoMaker disk store lacks {layout.lora_path.name}")
        if sha256_file(layout.lora_path) != manifest.get("lora_sha256"):
            raise BackendUnavailableError("cached PhotoMaker LoRA hash does not match ready.json")
        components = manifest.get("components")
        if not isinstance(components, dict):
            raise BackendUnavailableError("PhotoMaker ready.json lacks component records")
        for name in ("unet", "vae", "text_encoder", "text_encoder_2", "id_encoder"):
            record = components.get(name)
            if not isinstance(record, dict) or not isinstance(record.get("tensor_count"), int):
                raise BackendUnavailableError(f"PhotoMaker ready.json lacks component {name}")
            validate_offload_store(layout.component(name), record["tensor_count"])
        return manifest

    def _make_disk_skeletons(self, model_root: Path) -> dict[str, Any]:
        import torch
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from photomaker.model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

        with init_empty_weights(include_buffers=False):
            unet = UNet2DConditionModel.from_config(
                UNet2DConditionModel.load_config(model_root / "unet")
            )
            vae = AutoencoderKL.from_config(AutoencoderKL.load_config(model_root / "vae"))
            text_encoder = CLIPTextModel(
                CLIPTextConfig.from_pretrained(model_root / "text_encoder", local_files_only=True)
            )
            text_encoder_2 = CLIPTextModelWithProjection(
                CLIPTextConfig.from_pretrained(model_root / "text_encoder_2", local_files_only=True)
            )
            with redirect_stdout(StringIO()):
                id_encoder = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        models = {
            "unet": unet,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "id_encoder": id_encoder,
        }
        for model in models.values():
            model.to(dtype=torch.float16)
        vae.config.force_upcast = False
        return models

    def _construct_disk_pipeline(self, model_root: Path, models: dict[str, Any]) -> Any:
        from diffusers import EulerDiscreteScheduler
        from photomaker import PhotoMakerStableDiffusionXLPipeline
        from transformers import CLIPImageProcessor, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_root / "tokenizer_2", local_files_only=True
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_root / "scheduler", local_files_only=True
        )
        pipe = PhotoMakerStableDiffusionXLPipeline(
            vae=models["vae"],
            text_encoder=models["text_encoder"],
            text_encoder_2=models["text_encoder_2"],
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=models["unet"],
            scheduler=scheduler,
            image_encoder=None,
            feature_extractor=None,
            force_zeros_for_empty_prompt=True,
            add_watermarker=False,
        )
        pipe.num_tokens = 2
        pipe.pm_version = self._config.conditioner.version
        pipe.trigger_word = self._config.conditioner.trigger_word
        pipe.id_image_processor = CLIPImageProcessor()
        pipe.id_encoder = models["id_encoder"]
        pipe.tokenizer.add_tokens([pipe.trigger_word], special_tokens=True)
        pipe.tokenizer_2.add_tokens([pipe.trigger_word], special_tokens=True)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _build_store(
        self,
        layout: _OffloadLayout,
        model_root: Path,
        checkpoint: Path,
        materialized_hash: str,
        models: dict[str, Any],
        pipe: Any,
    ) -> tuple[_OffloadLayout, dict[str, Any]]:
        import torch
        from safetensors.torch import save_file

        layout.root.parent.mkdir(parents=True, exist_ok=True)
        stage_layout = _OffloadLayout(
            layout.root.with_name(f".{layout.root.name}.building-{os.getpid()}-{uuid4().hex[:8]}")
        )
        stage_layout.root.mkdir(parents=False, exist_ok=False)
        logging.getLogger(__name__).info("building PhotoMaker disk store at %s", stage_layout.root)

        component_specs = {
            "unet": (
                model_root / "unet" / "diffusion_pytorch_model.safetensors.index.fp16.json",
                "conv_in",
            ),
            "vae": (
                model_root / "vae" / "diffusion_pytorch_model.safetensors.index.fp16.json",
                "encoder.conv_in",
            ),
            "text_encoder": (
                model_root / "text_encoder" / "model.safetensors.index.fp16.json",
                "text_model.embeddings.token_embedding",
            ),
            "text_encoder_2": (
                model_root / "text_encoder_2" / "model.safetensors.index.fp16.json",
                "text_model.embeddings.token_embedding",
            ),
        }
        for name, (checkpoint_index, anchor) in component_specs.items():
            prepare_model_store(
                models[name],
                checkpoint_index,
                stage_layout.component(name),
                anchor=anchor,
                dtype=torch.float16,
            )

        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        try:
            if list(state) != ["id_encoder", "lora_weights"]:
                raise BackendUnavailableError(
                    "PhotoMaker checkpoint must contain id_encoder and lora_weights"
                )
            write_state_store(
                state["id_encoder"], models["id_encoder"], stage_layout.component("id_encoder")
            )
            lora_state = {
                name: value.to(dtype=torch.float16).contiguous()
                for name, value in state["lora_weights"].items()
            }
            save_file(lora_state, stage_layout.lora_path)
            pipe.load_lora_weights(lora_state, adapter_name="photomaker")
            lora_count = offload_photomaker_lora(
                models["unet"],
                lora_state,
                stage_layout.component("unet"),
                dtype=torch.float16,
            )
            add_cpu_values_to_store(models["unet"], stage_layout.component("unet"))
            alias_count = alias_wrapped_base_weights(models["unet"], stage_layout.component("unet"))
            del lora_state
        finally:
            del state
            gc.collect()

        component_counts = {
            name: len(read_offload_index(stage_layout.component(name))) for name in models
        }
        manifest = {
            **self._expected_store_identity(materialized_hash),
            "created_at": utc_now(),
            "lora_file": stage_layout.lora_path.name,
            "lora_sha256": sha256_file(stage_layout.lora_path),
            "lora_tensor_count": lora_count,
            "base_weight_alias_count": alias_count,
            "components": {
                name: {"tensor_count": count} for name, count in component_counts.items()
            },
        }
        atomic_write_json(stage_layout.ready_path, manifest)
        try:
            os.replace(stage_layout.root, layout.root)
        except OSError as exc:
            raise BackendUnavailableError(
                f"cannot publish PhotoMaker disk store {layout.root}: {exc}"
            ) from exc
        return layout, manifest

    def _load_disk_pipeline(
        self, model_root: Path, checkpoint: Path, materialized_hash: str
    ) -> tuple[Any, dict[str, Any], Path]:
        import torch
        from safetensors.torch import load_file

        layout = self._offload_layout()
        exists = layout.root.exists()
        manifest = self._validate_ready_store(layout, materialized_hash) if exists else None
        models = self._make_disk_skeletons(model_root)
        pipe = self._construct_disk_pipeline(model_root, models)
        if manifest is None:
            layout, manifest = self._build_store(
                layout,
                model_root,
                checkpoint,
                materialized_hash,
                models,
                pipe,
            )
        else:
            lora_state = load_file(layout.lora_path, device="cpu")
            pipe.load_lora_weights(lora_state, adapter_name="photomaker")
            del lora_state
            gc.collect()

        device = torch.device("cuda")
        for name, model in models.items():
            attach_disk_offload(model, layout.component(name), device=device)
        return pipe, manifest, layout.root

    def _load_direct_pipeline(self, model_root: Path, checkpoint: Path) -> Any:
        import torch
        from diffusers import EulerDiscreteScheduler
        from photomaker import PhotoMakerStableDiffusionXLPipeline

        try:
            pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                model_root,
                variant="fp16",
                torch_dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map="cuda",
                add_watermarker=False,
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            try:
                pipe.load_photomaker_adapter(
                    state,
                    weight_name=self._config.conditioner.weight_file,
                    trigger_word=self._config.conditioner.trigger_word,
                    pm_version=self._config.conditioner.version,
                )
            finally:
                del state
                gc.collect()
            pipe.vae.config.force_upcast = False
            pipe.set_progress_bar_config(disable=True)
            return pipe
        except Exception as exc:
            raise BackendUnavailableError(f"cannot load PhotoMaker in direct mode: {exc}") from exc

    def prepare(self) -> BackendInfo:
        if self._info is not None:
            return self._info
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(self._config.conditioner.insightface_cache_dir.resolve().parent / "matplotlib"),
        )
        if self._config.compute.device != "cuda":
            raise BackendUnavailableError("PhotoMaker V2 currently requires device=cuda")
        if self._config.model.dtype != "float16":
            raise BackendUnavailableError("PhotoMaker V2 currently requires a float16 base profile")

        source_revision = _installed_photomaker_revision()
        if source_revision is None:
            raise BackendUnavailableError(
                "cannot verify the installed PhotoMaker source revision; "
                "run `uv sync --extra reference`"
            )
        if source_revision != self._config.conditioner.source_revision:
            raise BackendUnavailableError(
                f"installed PhotoMaker revision is {source_revision}, expected "
                f"{self._config.conditioner.source_revision}"
            )

        model_root, materialized_hash = self._ensure_materialized_model()
        checkpoint = self._download_conditioner()
        embeddings, embedding_paths = _extract_reference_embeddings(self._config)

        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                "PhotoMaker reference dependencies are missing. Run `uv sync --extra reference`."
            ) from exc
        if self._config.compute.require_cuda and not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA is required but PyTorch cannot see an NVIDIA GPU. Run `mara-lab doctor`."
            )

        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
        )
        logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("diffusers.loaders.lora_base").setLevel(logging.ERROR)
        logging.getLogger("diffusers.loaders.peft").setLevel(logging.ERROR)
        mode = self._select_execution_mode(torch)
        store_manifest: dict[str, Any] | None = None
        offload_root: Path | None = None
        if mode == "disk":
            pipe, store_manifest, offload_root = self._load_disk_pipeline(
                model_root, checkpoint, materialized_hash
            )
        else:
            pipe = self._load_direct_pipeline(model_root, checkpoint)

        self._pipe = pipe
        self._torch = torch
        self._execution_mode = mode
        self._id_embeds = torch.from_numpy(embeddings)
        self._reference_images = [
            Image.open(path).convert("RGB") for path in self._config.reference_images
        ]
        driver_version = "unknown"
        with suppress(OSError, subprocess.SubprocessError):
            driver_version = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        details: dict[str, Any] = {
            "base_model_id": self._config.model.model_id,
            "base_revision": _model_revision(self._config),
            "materialized_model": model_root.as_posix(),
            "materialized_manifest_sha256": materialized_hash,
            "conditioner_weight_file": self._config.conditioner.weight_file,
            "conditioner_weight_sha256": sha256_file(checkpoint),
            "conditioner_source_repository": self._config.conditioner.source_repository,
            "conditioner_source_revision": source_revision,
            "execution_mode": mode,
            "offload_root": offload_root.as_posix() if offload_root else None,
            "offload_manifest_sha256": (
                sha256_file(offload_root / "ready.json") if offload_root else None
            ),
            "offload_lora_sha256": store_manifest.get("lora_sha256") if store_manifest else None,
            "scheduler_class": type(pipe.scheduler).__name__,
            "reference_image_sha256": [sha256_file(path) for path in self._config.reference_images],
            "face_embedding_sha256": [sha256_file(path) for path in embedding_paths],
            "insightface_model": self._config.conditioner.insightface_model,
            "insightface_provider": self._config.conditioner.insightface_provider,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver_version,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / _GIB, 4),
        }
        self._info = BackendInfo(
            backend="photomaker_v2",
            model_id=self._config.conditioner.model_id,
            requested_revision=self._config.conditioner.revision,
            resolved_revision=self._config.conditioner.revision,
            device="cuda",
            dtype="float16",
            resident_allocated_vram_gib=round(torch.cuda.memory_allocated() / _GIB, 4),
            resident_reserved_vram_gib=round(torch.cuda.memory_reserved() / _GIB, 4),
            details=details,
        )
        return self._info

    def compile_prompt(self, request: ReferenceGenerationRequest) -> CompiledPrompt:
        return CompiledPrompt(
            positive=compile_photomaker_prompt(
                request.shot,
                self._config.conditioner.trigger_word,
                self._config.prompt_template_version,
            ),
            negative=self._config.negative_prompt,
        )

    def _validate_prompt(self, prompt: CompiledPrompt) -> None:
        if self._pipe is None:
            raise BackendUnavailableError("backend must be prepared before prompt validation")
        trigger = self._config.conditioner.trigger_word
        trigger_matches = re.findall(rf"(?<!\w){re.escape(trigger)}(?!\w)", prompt.positive)
        if len(trigger_matches) != 1:
            raise BackendUnavailableError("PhotoMaker prompt must contain exactly one trigger word")
        for tokenizer_name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self._pipe, tokenizer_name)
            positive_ids = tokenizer(
                prompt.positive,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
            expanded_length = len(positive_ids) + 2 * len(self._reference_images) - 1
            if expanded_length > tokenizer.model_max_length:
                raise PromptTooLongError(
                    f"PhotoMaker prompt expands to {expanded_length} tokens with {tokenizer_name}; "
                    f"maximum is {tokenizer.model_max_length}"
                )
            negative_ids = tokenizer(
                prompt.negative,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
            if len(negative_ids) > tokenizer.model_max_length:
                raise PromptTooLongError(
                    f"negative prompt uses {len(negative_ids)} tokens with {tokenizer_name}; "
                    f"maximum is {tokenizer.model_max_length}"
                )

    def generate(self, request: ReferenceGenerationRequest) -> GenerationResult:
        if self._pipe is None or self._torch is None or self._id_embeds is None:
            raise BackendUnavailableError("backend must be prepared before generation")
        prompt = self.compile_prompt(request)
        self._validate_prompt(prompt)
        torch = self._torch
        generator = torch.Generator(device="cuda").manual_seed(request.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self._pipe(
                    prompt=prompt.positive,
                    negative_prompt=prompt.negative,
                    input_id_images=self._reference_images,
                    id_embeds=self._id_embeds,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    start_merge_step=request.start_merge_step,
                    generator=generator,
                )
        except Exception as exc:
            raise BackendUnavailableError(f"PhotoMaker V2 generation failed: {exc}") from exc
        torch.cuda.synchronize()
        image = output.images[0].convert("RGB")
        del output
        if self._execution_mode == "disk":
            gc.collect()
        return GenerationResult(
            image=image,
            metrics=RuntimeMetrics(
                wall_seconds=round(time.perf_counter() - started, 4),
                max_allocated_vram_gib=round(torch.cuda.max_memory_allocated() / _GIB, 4),
                max_reserved_vram_gib=round(torch.cuda.max_memory_reserved() / _GIB, 4),
            ),
        )

    def close(self) -> None:
        for image in self._reference_images:
            image.close()
        self._reference_images.clear()
        self._id_embeds = None
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._execution_mode = None
