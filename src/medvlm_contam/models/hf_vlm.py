"""HuggingFace ``AutoModelForVision2Seq`` scorer.

Works out-of-the-box with the open VLMs called out in the proposal —
Qwen2.5-VL-7B-Instruct, InternVL3-8B, LLaVA-OneVision-7B, LLaVA-Med v1.5,
CheXagent — provided the chat template is set correctly by the upstream
processor (which it is for all of the above as of transformers >= 4.45).

This file imports torch / transformers lazily so the rest of the package
remains importable in environments without a GPU stack (the numeric
detectors and benchmark loaders do not need torch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .base import ScoredAnswer, VLMScorer


class HFVisionScorer(VLMScorer):
    """Generic HF VLM scorer.

    Parameters
    ----------
    model_id:
        HF Hub id, e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``.
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    dtype:
        ``"bfloat16"``, ``"float16"``, or ``"float32"``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        import torch  # noqa: F401 — surfaces a friendly error early
        import transformers
        from transformers import AutoProcessor

        # ``AutoModelForVision2Seq`` was renamed to
        # ``AutoModelForImageTextToText`` in transformers 5.x. Prefer the
        # new name when available, fall back for older releases.
        AutoVLM = getattr(
            transformers,
            "AutoModelForImageTextToText",
            getattr(transformers, "AutoModelForVision2Seq", None),
        )
        if AutoVLM is None:
            raise ImportError(
                "transformers does not expose AutoModelForImageTextToText or "
                "AutoModelForVision2Seq; please upgrade transformers."
            )

        self.name = model_id
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(dtype)
        self._torch = __import__("torch")

        self.processor = self._load_processor(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = self._load_model(
            model_id,
            AutoVLM=AutoVLM,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    @staticmethod
    def _load_processor(model_id: str, *, trust_remote_code: bool):
        """Try ``AutoProcessor`` first, then fall back to manual assembly.

        Some older medical-VLM repos (notably ``microsoft/llava-med-v1.5``)
        ship a tokenizer + image processor but no ``processor_config.json``,
        so the generic ``AutoProcessor`` factory refuses to load them.
        """
        from transformers import AutoProcessor

        try:
            return AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        except (ValueError, OSError, KeyError):
            pass

        # Fallback: assemble a LLaVA-style processor from tokenizer + image
        # processor. Works for the LLaVA-Med family.
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            LlavaProcessor,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        # ``microsoft/llava-med-v1.5-mistral-7b`` ships no
        # ``preprocessor_config.json``, so ``AutoImageProcessor`` on the model
        # repo itself raises. In that case source the image processor from the
        # configured vision tower (``mm_vision_tower``), which for the
        # LLaVA-Med v1.5 family is ``openai/clip-vit-large-patch14-336``.
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        except (ValueError, OSError, KeyError):
            image_processor = self._image_processor_from_vision_tower(
                model_id, trust_remote_code=trust_remote_code
            )

        return LlavaProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )

    @staticmethod
    def _image_processor_from_vision_tower(
        model_id: str, *, trust_remote_code: bool
    ):
        """Load the CLIP image processor named by the model's vision tower.

        Legacy LLaVA checkpoints record their vision encoder in the config
        under ``mm_vision_tower`` (or ``vision_tower`` / ``vision_config``)
        rather than shipping a standalone ``preprocessor_config.json``.
        """
        from transformers import AutoConfig, CLIPImageProcessor

        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        vision_config = getattr(cfg, "vision_config", None)
        candidates = [
            getattr(cfg, "mm_vision_tower", None),
            getattr(cfg, "vision_tower", None),
            getattr(vision_config, "_name_or_path", None)
            if vision_config is not None
            else None,
            "openai/clip-vit-large-patch14-336",
        ]
        last_err: Optional[Exception] = None
        for tower in candidates:
            if not tower:
                continue
            try:
                return CLIPImageProcessor.from_pretrained(tower)
            except (ValueError, OSError, KeyError) as err:
                last_err = err
        raise RuntimeError(
            f"Could not load a CLIP image processor for {model_id!r} from any "
            f"of the configured vision towers {candidates!r}."
        ) from last_err

    def _load_model(self, model_id: str, *, AutoVLM, trust_remote_code: bool):
        """Load with ``AutoModelForImageTextToText``; fall back to ``AutoModel``.

        InternVL3 and CheXagent ship custom configs that are not registered
        with the image-text-to-text auto-mapping in transformers 5.x. They
        still expose a standard ``forward()`` returning logits via the
        remote-code path, so ``AutoModel`` works as a fallback.
        """
        from transformers import AutoModel, AutoModelForCausalLM

        kwargs = dict(
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        try:
            model = AutoVLM.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError):
            try:
                model = AutoModel.from_pretrained(model_id, **kwargs)
            except (ValueError, KeyError):
                # CheXagent etc. only register themselves under
                # AutoModelForCausalLM via the auto_map in config.json.
                model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model.to(self.device)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, dtype: str):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

    def _load_image(self, image_path: Optional[Path]):
        if image_path is None:
            return None
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def _normalize_pixel_values(self, inputs):
        """Squeeze any extraneous singleton ``num_images-per-example`` dim.

        Some processors in transformers >= 5.x wrap ``pixel_values`` as
        ``(B, num_images, C, H, W)`` even when each example has a single
        image (notably InstructBLIP / BLIP-2 family with newer processor
        configs). The underlying vision encoders unpack the tensor as
        ``batch, channels, height, width = pixel_values.shape`` and raise
        ``too many values to unpack (expected 4)``. Reshaping the 5-D
        ``(B, K, C, H, W)`` tensor to ``(B*K, C, H, W)`` is the
        documented HF workaround and is a no-op when K=1, which is the
        regime we use for per-example scoring.

        Most processors (Qwen2.5-VL, LLaVA-OneVision, BLIP-2 default
        path) already produce 4-D pixel_values, in which case this is a
        no-op.
        """
        pv = inputs.get("pixel_values", None)
        if pv is None or pv.dim() == 4:
            return inputs
        if pv.dim() == 5:
            b, k, c, h, w = pv.shape
            inputs["pixel_values"] = pv.reshape(b * k, c, h, w)
            return inputs
        # Unexpected shape — leave untouched so the underlying error is
        # surfaced rather than silently mangled.
        return inputs

    # ----------------------------------------------------------------- scoring

    def score(
        self,
        image_path: Optional[Path],
        prompt: str,
        answer: str,
        *,
        topk: int = 0,
    ) -> ScoredAnswer:
        # Model-family dispatch. InternVL3's ``AutoProcessor`` is just a
        # tokenizer (no image processing) and its model expects pixel_values
        # to be pre-tiled to 448x448 patches, with input_ids containing an
        # ``<IMG_CONTEXT>`` block sized by ``num_image_token * num_patches``.
        # That doesn't fit the standard processor(image=, text=) path, so
        # route it through a dedicated adapter.
        if "internvl" in self.model_id.lower():
            return self._score_internvl(image_path, prompt, answer, topk=topk)

        torch = self._torch
        image = self._load_image(image_path)

        # Build a chat-formatted prompt + assistant turn, then score the
        # answer tokens by feeding the full sequence and slicing logits at
        # the answer span. Some medical VLMs (e.g. CheXagent) ship a
        # processor without any chat_template — for them we fall back to
        # raw text concatenation. The same processor is used for both
        # prompt-only and full sequence, so the answer-span slicing
        # arithmetic still works regardless of which path is taken.
        messages_prompt = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"}, {"type": "text", "text": prompt}]
                    if image is not None
                    else [{"type": "text", "text": prompt}]
                ),
            },
        ]
        messages_full = messages_prompt + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, add_generation_prompt=True, tokenize=False
            )
            full_text = self.processor.apply_chat_template(
                messages_full, add_generation_prompt=False, tokenize=False
            )
        except (ValueError, AttributeError):
            # Processor has no chat_template (CheXagent, some medical VLMs).
            # Use a minimal generic prompt format; the answer-span slice
            # only depends on prompt_text being a prefix of full_text.
            prompt_text = f"USER: {prompt}\nASSISTANT: "
            full_text = prompt_text + answer

        proc_kwargs = {"text": [full_text], "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = self.processor(**proc_kwargs).to(self.device)
        inputs = self._normalize_pixel_values(inputs)

        # Compute prompt length by running the SAME processor on the prompt
        # alone (with the image). Using the bare tokenizer here is incorrect
        # for VLMs because the processor expands the <image> placeholder into
        # hundreds of image-patch tokens — the bare tokenizer doesn't, so
        # full_len - prompt_len would then include all those patch tokens
        # and mis-score the answer span.
        prompt_proc_kwargs = {"text": [prompt_text], "return_tensors": "pt"}
        if image is not None:
            prompt_proc_kwargs["images"] = [image]
        prompt_inputs = self.processor(**prompt_proc_kwargs)
        prompt_inputs = self._normalize_pixel_values(prompt_inputs)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        full_len = int(inputs["input_ids"].shape[1])
        answer_len = full_len - prompt_len
        if answer_len <= 0:
            # Chat templates can collapse — fall back to whitespace split.
            answer_len = max(1, len(self.processor.tokenizer.encode(answer)))
            prompt_len = full_len - answer_len

        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits  # (1, T, V) — predicts token t+1 from prefix [..t]

        # Position t predicts token at t+1. Answer tokens occupy
        # [prompt_len, full_len); their predicting logits are at
        # [prompt_len - 1, full_len - 1).
        pred_slice = logits[0, prompt_len - 1 : full_len - 1, :]
        target_ids = inputs["input_ids"][0, prompt_len:full_len]
        log_probs_full = torch.log_softmax(pred_slice.float(), dim=-1)
        token_logprobs = log_probs_full.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

        topk_logprobs = None
        if topk and topk > 0:
            topk_vals, _ = torch.topk(log_probs_full, k=topk, dim=-1)
            topk_logprobs = topk_vals.cpu().numpy()

        return ScoredAnswer(
            token_ids=target_ids.cpu().numpy(),
            token_logprobs=token_logprobs.cpu().numpy(),
            topk_logprobs=topk_logprobs,
        )

    def score_many(
        self,
        items: Sequence[tuple[Optional[Path], str, str]],
        *,
        topk: int = 0,
    ) -> Iterable[ScoredAnswer]:
        for image_path, prompt, answer in items:
            yield self.score(image_path, prompt, answer, topk=topk)

    # ------------------------------------------------------------- InternVL3

    # Standard InternVL preprocessing constants (image_mean/std are ImageNet
    # values, which matches the preprocessor_config.json shipped in the
    # OpenGVLab/InternVL3-* HF repos).
    _INTERNVL_IMAGE_MEAN = (0.485, 0.456, 0.406)
    _INTERNVL_IMAGE_STD = (0.229, 0.224, 0.225)
    _INTERNVL_SYSTEM_MESSAGE = (
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学"
        "及多家合作单位联合开发的多模态大语言模型。"
    )

    @staticmethod
    def _internvl_find_closest_ratio(aspect_ratio, target_ratios, width, height, image_size):
        # Picks the (i, j) tile grid whose aspect ratio is closest to the
        # source image's. Ties broken by larger total area coverage,
        # matching the reference implementation in InternVL's HF repo.
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for r in target_ratios:
            tr = r[0] / r[1]
            diff = abs(aspect_ratio - tr)
            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = r
            elif diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * r[0] * r[1]:
                    best_ratio = r
        return best_ratio

    def _internvl_dynamic_preprocess(
        self,
        image,
        *,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ):
        """Tile a PIL image into ``image_size``-square patches.

        Direct port of ``dynamic_preprocess`` from the InternVL HF repo.
        Returns a list of PIL.Image tiles, optionally with a thumbnail
        appended (the thumbnail is the full image resized to one tile).
        """
        from PIL import Image

        orig_w, orig_h = image.size
        aspect_ratio = orig_w / orig_h

        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda r: r[0] * r[1],
        )

        ratio = self._internvl_find_closest_ratio(
            aspect_ratio, target_ratios, orig_w, orig_h, image_size
        )
        target_w = image_size * ratio[0]
        target_h = image_size * ratio[1]
        blocks = ratio[0] * ratio[1]

        resized = image.resize((target_w, target_h), Image.BICUBIC)
        tiles = []
        for i in range(blocks):
            box = (
                (i % (target_w // image_size)) * image_size,
                (i // (target_w // image_size)) * image_size,
                ((i % (target_w // image_size)) + 1) * image_size,
                ((i // (target_w // image_size)) + 1) * image_size,
            )
            tiles.append(resized.crop(box))
        if use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((image_size, image_size), Image.BICUBIC))
        return tiles

    def _internvl_pixel_values(self, image):
        """Convert a PIL image into the (N, 3, 448, 448) tensor InternVL expects."""
        import torchvision.transforms as T

        tiles = self._internvl_dynamic_preprocess(image)
        transform = T.Compose(
            [
                T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
                T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=self._INTERNVL_IMAGE_MEAN, std=self._INTERNVL_IMAGE_STD
                ),
            ]
        )
        torch = self._torch
        pv = torch.stack([transform(t) for t in tiles])
        return pv.to(device=self.device, dtype=self.torch_dtype)

    def _internvl_tokenizer(self):
        # On InternVL3, AutoProcessor.from_pretrained returns the tokenizer
        # directly (the repo has no full processor). Be defensive in case
        # a future version wraps it.
        return getattr(self.processor, "tokenizer", self.processor)

    def _score_internvl(
        self,
        image_path: Optional[Path],
        prompt: str,
        answer: str,
        *,
        topk: int = 0,
    ) -> ScoredAnswer:
        torch = self._torch
        image = self._load_image(image_path)
        if image is None:
            raise ValueError("InternVL3 scoring requires an image.")

        pixel_values = self._internvl_pixel_values(image)
        num_patches = int(pixel_values.shape[0])
        num_image_token = int(getattr(self.model, "num_image_token", 256))

        IMG_START = "<img>"
        IMG_END = "</img>"
        IMG_CTX = "<IMG_CONTEXT>"
        image_token_block = (
            IMG_START + IMG_CTX * (num_image_token * num_patches) + IMG_END
        )

        # internvl2_5 conversation template (MPT separator style). System
        # message and role tags are copied verbatim from the repo's
        # conversation.py so token boundaries line up with what the model
        # was trained on.
        sysmsg = self._INTERNVL_SYSTEM_MESSAGE
        user_msg = f"{image_token_block}\n{prompt}"
        prefix = (
            f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prefix + answer + "<|im_end|>\n"

        tokenizer = self._internvl_tokenizer()
        prompt_ids = tokenizer(prefix, return_tensors="pt").input_ids
        full = tokenizer(full_text, return_tensors="pt")
        full_ids = full.input_ids.to(self.device)
        attention_mask = full.attention_mask.to(self.device)

        prompt_len = int(prompt_ids.shape[1])
        full_len = int(full_ids.shape[1])
        if full_len - prompt_len <= 0:
            answer_len = max(1, len(tokenizer.encode(answer)))
            prompt_len = full_len - answer_len

        image_flags = torch.ones(num_patches, dtype=torch.long, device=self.device)

        # InternVL3's forward expects ``img_context_token_id`` to be set on
        # the model so it knows which positions to splice visual features
        # into. The ``chat()`` helper sets this attribute lazily; do the
        # same here.
        self.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)

        with torch.no_grad():
            out = self.model(
                pixel_values=pixel_values,
                input_ids=full_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            )
        logits = out.logits

        pred_slice = logits[0, prompt_len - 1 : full_len - 1, :]
        target_ids = full_ids[0, prompt_len:full_len]
        log_probs_full = torch.log_softmax(pred_slice.float(), dim=-1)
        token_logprobs = log_probs_full.gather(
            1, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        topk_logprobs = None
        if topk and topk > 0:
            topk_vals, _ = torch.topk(log_probs_full, k=topk, dim=-1)
            topk_logprobs = topk_vals.cpu().numpy()

        return ScoredAnswer(
            token_ids=target_ids.cpu().numpy(),
            token_logprobs=token_logprobs.cpu().numpy(),
            topk_logprobs=topk_logprobs,
        )
