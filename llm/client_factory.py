import torch

import config


def create_vlm_client(**kwargs):
    backend = config.LLM_BACKEND
    if backend == "local":
        from llm.qwen2_vl_client import Qwen2VLClient

        kwargs.setdefault("model_name", str(config.QWEN_VL_MODEL_NAME))
        kwargs.setdefault("torch_dtype", torch.bfloat16 if torch.cuda.is_available() else torch.float16)
        kwargs.setdefault("max_new_tokens", 1024)
        return Qwen2VLClient(**kwargs)

    if backend == "qwen_api":
        from llm.qwen_api_client import QwenAPIClient

        kwargs.pop("torch_dtype", None)
        kwargs.setdefault("model_name", config.QWEN_API_MODEL)
        kwargs.setdefault("max_new_tokens", config.QWEN_API_MAX_TOKENS)
        return QwenAPIClient(**kwargs)

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")
