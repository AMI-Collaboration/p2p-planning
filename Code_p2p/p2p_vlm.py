from __future__ import annotations

import base64
import os
import threading
from typing import List, Tuple

from p2p_config import MAX_NEW_TOKENS


# ── 토큰 사용량 추적 ──────────────────────────────────────────────────────────
#
# _last_usage  : 마지막 단일 호출의 토큰
# _total_usage : 하나의 실험 동안 발생한 모든 VLM 호출의 누적 토큰
#
# p2p_tracker.py가 tracker.start()에서 0으로 초기화하고
# tracker.stop()에서 이 값을 읽는다.
#

_last_usage: dict = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
}

_usage_lock: threading.Lock = threading.Lock()

_total_usage: dict = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
}


# ── 백엔드 선택 ───────────────────────────────────────────────────────────────

VLM_BACKEND = os.environ.get("VLM_BACKEND", "qwen").lower()


# ──────────────────────────────────────────────────────────────────────────────
# Qwen 백엔드
# ──────────────────────────────────────────────────────────────────────────────

_qwen_model = None
_qwen_processor = None


def _load_qwen():
    global _qwen_model, _qwen_processor

    if _qwen_model is not None:
        return

    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    # ============================================================
    # 사용할 모델
    # ============================================================

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )

    _qwen_processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("model loaded:", MODEL_ID)


def _run_qwen(
    image_path: str,
    prompt: str,
    return_logprobs: bool = False,
) -> Tuple[str, List[float]]:

    import torch
    from PIL import Image, ImageOps

    _load_qwen()

    # ============================================================
    # 1. 이미지
    # ============================================================

    image = ImageOps.exif_transpose(
        Image.open(image_path)
    ).convert("RGB")


    # ============================================================
    # 2. 메시지 구성
    # ============================================================

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


    # ============================================================
    # 3. Chat template
    # ============================================================

    text_in = _qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    # ============================================================
    # 4. Processor
    # ============================================================

    inputs = _qwen_processor(
        text=[text_in],
        images=[image],
        return_tensors="pt",
    ).to(_qwen_model.device)


    # ============================================================
    # 5. Qwen inference
    # ============================================================

    with torch.no_grad():

        out = _qwen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
        )


    # ============================================================
    # 6. 실제 생성된 output token 분리
    # ============================================================

    input_token_count = int(
        inputs["input_ids"].numel()
    )

    gen_ids = out.sequences[
        :,
        inputs["input_ids"].shape[1]:
    ]

    output_token_count = int(
        gen_ids.numel()
    )


    # ============================================================
    # 7. ⭐ Qwen Token Usage 기록
    # ============================================================
    #
    # OpenAI API와 달리 HuggingFace 로컬 generate()는
    # response.usage.prompt_tokens 같은 값을 자동으로
    # 제공하지 않는다.
    #
    # 따라서 실제 input_ids / generated ids를 이용해서
    # 직접 token 수를 계산한다.
    #
    # input_tokens:
    #     processor가 생성한 텍스트 input token 수
    #
    # output_tokens:
    #     실제 새롭게 생성된 token 수
    #
    # 전체 실험에서는 여러 VLM 호출의 token을 누적한다.
    #

    with _usage_lock:

        _last_usage["prompt_tokens"] = input_token_count
        _last_usage["completion_tokens"] = output_token_count

        _total_usage["prompt_tokens"] += input_token_count
        _total_usage["completion_tokens"] += output_token_count


    # ============================================================
    # 8. 생성 텍스트
    # ============================================================

    text_out = _qwen_processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
    )[0].strip()


    # ============================================================
    # 9. Log probabilities
    # ============================================================

    log_probs: List[float] = []

    if return_logprobs and out.scores:

        for step_idx, step_scores in enumerate(out.scores):

            # 안전하게 실제 생성 길이까지만 처리
            if step_idx >= gen_ids.shape[1]:
                break

            token_id = gen_ids[
                0,
                step_idx
            ].item()

            log_probs.append(
                torch.log_softmax(
                    step_scores[0],
                    dim=-1
                )[token_id].item()
            )


    # ============================================================
    # 10. 디버깅용 출력
    # ============================================================
    #
    # 필요하면 이 print를 지워도 됨.
    #

    print(
        f"[Qwen Usage] "
        f"input={input_token_count:,} | "
        f"output={output_token_count:,} | "
        f"total={input_token_count + output_token_count:,}"
    )


    return text_out, log_probs


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI GPT-4o 백엔드
# ──────────────────────────────────────────────────────────────────────────────

_openai_client = None


def _load_openai():

    global _openai_client

    if _openai_client is not None:
        return

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  os.environ['OPENAI_API_KEY'] = 'sk-...' 로 설정하세요."
        )

    _openai_client = OpenAI(
        api_key=api_key
    )

    print("OpenAI client loaded: gpt-4o")


def _run_openai(
    image_path: str,
    prompt: str,
    return_logprobs: bool = False,
) -> Tuple[str, List[float]]:

    _load_openai()

    ext = image_path.rsplit(".", 1)[-1].lower()

    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(
        ext,
        "image/jpeg",
    )

    with open(image_path, "rb") as f:

        img_b64 = base64.b64encode(
            f.read()
        ).decode()


    kwargs = {
        "model": "gpt-4o",
        "max_tokens": MAX_NEW_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime};base64,{img_b64}"
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }


    if return_logprobs:

        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1


    response = _openai_client.chat.completions.create(
        **kwargs
    )


    # ============================================================
    # OpenAI token usage
    # ============================================================

    _last_usage["prompt_tokens"] = (
        response.usage.prompt_tokens
    )

    _last_usage["completion_tokens"] = (
        response.usage.completion_tokens
    )


    with _usage_lock:

        _total_usage["prompt_tokens"] += (
            response.usage.prompt_tokens
        )

        _total_usage["completion_tokens"] += (
            response.usage.completion_tokens
        )


    # ============================================================
    # 결과
    # ============================================================

    text_out = (
        response.choices[0]
        .message.content
        .strip()
    )

    log_probs = []

    if (
        return_logprobs
        and response.choices[0].logprobs
    ):

        log_probs = [
            t.logprob
            for t in response.choices[0]
            .logprobs.content
        ]


    return text_out, log_probs


# ──────────────────────────────────────────────────────────────────────────────
# 통합 인터페이스
# ──────────────────────────────────────────────────────────────────────────────

def run_vlm(
    image_path: str,
    prompt: str,
    return_logprobs: bool = False,
) -> Tuple[str, List[float]]:

    """
    VLM 추론 통합 인터페이스.

    VLM_BACKEND 환경변수에 따라
    Qwen 또는 GPT-4o를 사용한다.

    Args:
        image_path:
            입력 이미지 경로

        prompt:
            VLM prompt

        return_logprobs:
            True이면 token log-prob 리스트도 반환

    Returns:
        (생성 텍스트, log_probs)
    """

    backend = os.environ.get(
        "VLM_BACKEND",
        VLM_BACKEND,
    ).lower()


    if backend == "openai":

        return _run_openai(
            image_path,
            prompt,
            return_logprobs,
        )

    else:

        return _run_qwen(
            image_path,
            prompt,
            return_logprobs,
        )


def get_backend() -> str:

    """현재 사용 중인 백엔드 이름 반환."""

    return os.environ.get(
        "VLM_BACKEND",
        VLM_BACKEND,
    ).lower()
