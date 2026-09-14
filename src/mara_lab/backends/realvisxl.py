from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

from mara_lab.artifacts import sha256_file
from mara_lab.backends.base import ImageBackend
from mara_lab.config import ComputeProfile, ModelProfile
from mara_lab.domain import BackendInfo, GenerationRequest, GenerationResult, RuntimeMetrics
from mara_lab.errors import BackendUnavailableError, PromptTooLongError


class RealVisXLBackend(ImageBackend):
    def __init__(
        self,
        model: ModelProfile,
        compute: ComputeProfile,
        *,
        adapter_path: Path | None = None,
        adapter_weight: float = 1.0,
    ) -> None:
        self._model = model
        self._compute = compute
        self._adapter_path = adapter_path.resolve() if adapter_path is not None else None
        self._adapter_weight = adapter_weight
        self._pipe: Any = None
        self._torch: Any = None
        self._info: BackendInfo | None = None
        self._prompt_cache: dict[tuple[str, str, bool], tuple[Any, Any, Any, Any]] = {}

    def prepare(self) -> BackendInfo:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            import truststore
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        except ImportError as exc:
            raise BackendUnavailableError(
                "GPU dependencies are missing. Run `uv sync --extra gpu`."
            ) from exc

        truststore.inject_into_ssl()
        logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message=r"`upcast_vae` is deprecated.*",
            category=FutureWarning,
            module=r"diffusers\.pipelines\.stable_diffusion_xl.*",
        )

        class StagedStableDiffusionXLPipeline(StableDiffusionXLPipeline):
            @property
            def device(self):
                return self.unet.device

            @property
            def _execution_device(self):
                return self.unet.device

        if self._compute.require_cuda and not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA is required but PyTorch cannot see an NVIDIA GPU. Run `mara-lab doctor`."
            )
        if self._compute.allow_cpu_offload:
            raise BackendUnavailableError("v0.1 deliberately disallows CPU model offload")
        if self._compute.device != "cuda":
            raise BackendUnavailableError("the RealVisXL v0.1 backend requires device=cuda")

        dtype = torch.bfloat16 if self._model.dtype == "bfloat16" else torch.float16
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise BackendUnavailableError("the configured GPU does not support bfloat16")

        requested_revision = self._model.resolved_revision or self._model.revision
        if not self._model.weight_file:
            raise BackendUnavailableError("RealVisXL requires an explicit single-file checkpoint")

        published_weight_sha256: str | None = None
        local_files_only = self._model.resolved_revision is not None
        if self._model.resolved_revision:
            resolved_revision = self._model.resolved_revision
        elif re.fullmatch(r"[0-9a-f]{40}", requested_revision):
            resolved_revision = requested_revision
        else:
            try:
                model_info = HfApi().model_info(
                    self._model.model_id,
                    revision=requested_revision,
                    files_metadata=True,
                )
                resolved_revision = model_info.sha
                for sibling in model_info.siblings or []:
                    if sibling.rfilename != self._model.weight_file:
                        continue
                    lfs = sibling.lfs
                    if isinstance(lfs, dict):
                        published_weight_sha256 = lfs.get("sha256")
                    elif lfs is not None:
                        published_weight_sha256 = getattr(lfs, "sha256", None)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"cannot resolve model revision for {self._model.model_id}: {exc}"
                ) from exc
            if not resolved_revision:
                raise BackendUnavailableError("Hugging Face returned an empty model revision")

        cache_dir = self._model.cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            config_snapshot = snapshot_download(
                self._model.model_id,
                revision=resolved_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=[
                    "model_index.json",
                    "scheduler/*",
                    "text_encoder/config.json",
                    "text_encoder_2/config.json",
                    "tokenizer/*",
                    "tokenizer_2/*",
                    "unet/config.json",
                    "vae/config.json",
                ],
            )
            checkpoint_path = hf_hub_download(
                self._model.model_id,
                filename=self._model.weight_file,
                revision=resolved_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            actual_weight_sha256 = sha256_file(Path(checkpoint_path))
            if (
                self._model.weight_sha256 is not None
                and actual_weight_sha256 != self._model.weight_sha256
            ):
                raise BackendUnavailableError(
                    "downloaded checkpoint hash does not match the model profile"
                )
            if (
                published_weight_sha256 is not None
                and actual_weight_sha256 != published_weight_sha256
            ):
                raise BackendUnavailableError(
                    "downloaded checkpoint hash does not match Hub metadata"
                )
            pipe = StagedStableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                config=config_snapshot,
                torch_dtype=dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )
            if self._model.vae_tiling:
                pipe.vae.enable_tiling()
            adapter_details: dict[str, Any] | None = None
            if self._adapter_path is not None:
                if not self._adapter_path.is_file():
                    raise BackendUnavailableError(
                        f"character adapter does not exist: {self._adapter_path}"
                    )
                if not 0 < self._adapter_weight <= 2:
                    raise BackendUnavailableError("adapter weight must be above 0 and at most 2")
                adapter_sha256 = sha256_file(self._adapter_path)
                pipe.load_lora_weights(
                    str(self._adapter_path.parent),
                    weight_name=self._adapter_path.name,
                    adapter_name="character",
                )
                pipe.set_adapters(["character"], adapter_weights=[self._adapter_weight])
                adapter_details = {
                    "path": self._adapter_path.as_posix(),
                    "sha256": adapter_sha256,
                    "weight": self._adapter_weight,
                }
            pipe.set_progress_bar_config(disable=True)
            pipe.to(self._compute.device)
        except Exception as exc:
            raise BackendUnavailableError(f"failed to load RealVisXL: {exc}") from exc

        self._pipe = pipe
        self._torch = torch
        gib = 1024**3
        driver_version = "unknown"
        with suppress(OSError, subprocess.SubprocessError):
            driver_version = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        details = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver_version,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / gib, 4),
            "scheduler_class": type(pipe.scheduler).__name__,
            "cpu_offload": False,
            "text_encoder_staging": self._compute.stage_text_encoders,
            "attention_backend": self._compute.attention_backend,
            "selected_weight_file": self._model.weight_file,
            "selected_weight_sha256": actual_weight_sha256,
            "character_adapter": adapter_details,
        }
        self._info = BackendInfo(
            backend="realvisxl",
            model_id=self._model.model_id,
            requested_revision=self._model.revision,
            resolved_revision=resolved_revision,
            device=self._compute.device,
            dtype=self._model.dtype,
            resident_allocated_vram_gib=round(torch.cuda.memory_allocated() / gib, 4),
            resident_reserved_vram_gib=round(torch.cuda.memory_reserved() / gib, 4),
            details=details,
        )
        return self._info

    def _validate_prompt_length(self, prompt: str, label: str) -> None:
        if self._pipe is None:
            raise BackendUnavailableError("backend must be prepared before prompt validation")
        for tokenizer_name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self._pipe, tokenizer_name)
            token_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
            if len(token_ids) > tokenizer.model_max_length:
                raise PromptTooLongError(
                    f"{label} prompt uses {len(token_ids)} tokens with {tokenizer_name}; "
                    f"maximum is {tokenizer.model_max_length}"
                )

    def _encode_prompt(self, request: GenerationRequest) -> tuple[Any, Any, Any, Any]:
        if self._pipe is None or self._torch is None:
            raise BackendUnavailableError("backend must be prepared before prompt encoding")

        do_classifier_free_guidance = request.guidance_scale > 1.0
        cache_key = (request.prompt, request.negative_prompt, do_classifier_free_guidance)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        self._validate_prompt_length(request.prompt, "positive")
        self._validate_prompt_length(request.negative_prompt, "negative")

        if self._compute.stage_text_encoders:
            self._pipe.text_encoder.to(self._compute.device)
            self._pipe.text_encoder_2.to(self._compute.device)
        with self._torch.inference_mode():
            embeddings = self._pipe.encode_prompt(
                prompt=request.prompt,
                device=self._compute.device,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=request.negative_prompt,
            )
        if self._compute.stage_text_encoders:
            self._pipe.text_encoder.to("cpu")
            self._pipe.text_encoder_2.to("cpu")
            self._torch.cuda.empty_cache()

        detached = tuple(
            embedding.detach() if embedding is not None else None for embedding in embeddings
        )
        self._prompt_cache[cache_key] = detached
        return detached

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._pipe is None or self._torch is None:
            raise BackendUnavailableError("backend must be prepared before generation")

        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(request)
        generator = torch.Generator(device=self._compute.device).manual_seed(request.seed)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self._pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )
        except Exception as exc:
            raise BackendUnavailableError(f"RealVisXL generation failed: {exc}") from exc
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        gib = 1024**3
        return GenerationResult(
            image=output.images[0].convert("RGB"),
            metrics=RuntimeMetrics(
                wall_seconds=round(wall_seconds, 4),
                max_allocated_vram_gib=round(torch.cuda.max_memory_allocated() / gib, 4),
                max_reserved_vram_gib=round(torch.cuda.max_memory_reserved() / gib, 4),
            ),
        )

    def close(self) -> None:
        self._prompt_cache.clear()
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
