import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import compute_summary_metrics, evaluate_sample, print_summary, save_json

try:
    from config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, CLIP_MODEL_NAME, QWEN_VL_MODEL_NAME
    from config import EVALUATION_JUDGE_MODE
except Exception:
    CHROMA_COLLECTION_NAME = "adas_manual_visual_index"
    CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
    CLIP_MODEL_NAME = "clip-ViT-B-32"
    QWEN_VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    EVALUATION_JUDGE_MODE = "local_llm"


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_pipeline() -> Dict[str, Any]:
    """Reuse the same core initialization as main.py's hybrid CLI path."""
    from agent.react_agent import EvidenceAwareVQAAgent
    from data_types import DocBlock
    from llm.client_factory import create_vlm_client
    from model.clip_encoder import CLIPEncoder
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.retriever import VisualRetriever

    qwen_client = create_vlm_client()
    clip_encoder = CLIPEncoder(model_name=str(CLIP_MODEL_NAME))
    visual_retriever = VisualRetriever(
        collection_name=str(CHROMA_COLLECTION_NAME),
        db_path=str(CHROMA_DB_DIR),
    )

    all_docs = visual_retriever.collection.get()
    docstore: Dict[str, DocBlock] = {}
    for doc_id, doc_content, doc_metadata in zip(
        all_docs.get("ids", []),
        all_docs.get("documents", []),
        all_docs.get("metadatas", []),
    ):
        docstore[doc_id] = DocBlock(
            id=doc_id,
            content=doc_content,
            block_type="image",
            metadata=doc_metadata or {},
        )

    hybrid_retriever = HybridRetriever(
        chroma_collection=visual_retriever.collection,
        docstore=docstore,
        clip_encoder=clip_encoder,
    )
    reranker = Reranker(model_name="/root/autodl-tmp/hf_cache/BAAI/bge-reranker-base/")
    agent = EvidenceAwareVQAAgent(qwen_client)

    return {
        "qwen_client": qwen_client,
        "clip_encoder": clip_encoder,
        "visual_retriever": visual_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "agent": agent,
    }


def doc_image_path(doc: Any) -> Optional[str]:
    if doc.block_type == "image" and doc.content:
        return doc.content
    for key in ("image_path", "page_image", "path", "source"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    return None


def docs_to_pages(docs: List[Any]) -> List[str]:
    pages = []
    for doc in docs:
        image_path = doc_image_path(doc)
        if image_path:
            pages.append(image_path)
        else:
            page = doc.metadata.get("page_num", doc.metadata.get("page"))
            pages.append(str(page) if page is not None else doc.id)
    return pages


def collect_text_fields(docs: List[Any]) -> Dict[str, str]:
    evidence_texts = []
    ocr_texts = []
    metadata_texts = []
    for doc in docs:
        if doc.content and doc.block_type != "image":
            evidence_texts.append(doc.content)
        if doc.metadata.get("ocr_text"):
            ocr_texts.append(str(doc.metadata["ocr_text"]))
        if doc.metadata:
            metadata_texts.append(json.dumps(doc.metadata, ensure_ascii=False))
    return {
        "evidence_text": "\n".join(evidence_texts),
        "ocr_text": "\n".join(ocr_texts),
        "metadata_text": "\n".join(metadata_texts),
    }


def get_peak_gpu_memory_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def extract_qwen_runtime_stats(qwen_client: Any) -> Dict[str, Optional[int]]:
    # Existing client currently prints these debug values but does not expose them as attributes.
    input_ids_length = getattr(qwen_client, "last_input_ids_length", None)
    visual_tokens = getattr(qwen_client, "last_visual_tokens", None)
    return {
        "input_ids_length": input_ids_length,
        "visual_tokens": visual_tokens,
    }


def run_one_sample(sample: Dict[str, Any], pipeline: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "gold_page": sample.get("gold_page"),
        "primary_gold_page": sample.get("primary_gold_page", sample.get("gold_page")),
        "gold_pages": sample.get("gold_pages") or [],
        "gold_answer": sample.get("gold_answer"),
        "acceptable_answers": sample.get("acceptable_answers") or [],
        "required_keypoints": sample.get("required_keypoints") or [],
        "optional_keypoints": sample.get("optional_keypoints") or [],
        "answer_type": sample.get("answer_type", sample.get("question_type", "unknown")),
        "retrieved_pages": [],
        "reranked_pages": [],
        "final_image_paths": [],
        "predicted_answer": None,
        "error": None,
        "retrieval_latency_sec": None,
        "rerank_latency_sec": None,
        "generation_latency_sec": None,
        "total_latency_sec": None,
        "peak_gpu_memory_gb": None,
        "input_ids_length": None,
        "visual_tokens": None,
        "evidence_text": "",
        "ocr_text": "",
        "metadata_text": "",
    }

    total_start = time.perf_counter()
    reset_peak_gpu_memory()

    try:
        retrieval_start = time.perf_counter()
        retrieved_docs = pipeline["hybrid_retriever"].retrieve(query=sample["question"], top_k=5)
        result["retrieval_latency_sec"] = time.perf_counter() - retrieval_start
        result["retrieved_pages"] = docs_to_pages(retrieved_docs)

        rerank_start = time.perf_counter()
        reranked_docs = pipeline["reranker"].rerank(query=sample["question"], docs=retrieved_docs, top_k=2)
        result["rerank_latency_sec"] = time.perf_counter() - rerank_start
        result["reranked_pages"] = docs_to_pages(reranked_docs)
        result["final_image_paths"] = [path for path in (doc_image_path(doc) for doc in reranked_docs) if path]
        result.update(collect_text_fields(reranked_docs))

        generation_start = time.perf_counter()
        agent_result = pipeline["agent"].run(query=sample["question"], docs=reranked_docs)
        result["generation_latency_sec"] = time.perf_counter() - generation_start
        result["predicted_answer"] = agent_result.get("answer") if isinstance(agent_result, dict) else str(agent_result)

        result.update(extract_qwen_runtime_stats(pipeline["qwen_client"]))
    except Exception as exc:
        result["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_latency_sec"] = time.perf_counter() - total_start
        result["peak_gpu_memory_gb"] = get_peak_gpu_memory_gb()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Visual-RAG Agent benchmark.")
    parser.add_argument("--benchmark", default="evaluation/benchmark_test.json")
    # parser.add_argument("--benchmark", default="evaluation/benchmark.json")
    parser.add_argument("--output_dir", default="evaluation/results")
    parser.add_argument(
        "--judge_mode",
        choices=["local_llm", "rule"],
        default=EVALUATION_JUDGE_MODE,
        help="Evaluation judge mode. Defaults to config.EVALUATION_JUDGE_MODE.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_benchmark(benchmark_path)
    pipeline = init_pipeline()

    results = []
    for sample in samples:
        sample_id = sample.get("id", len(results) + 1)
        print(f"[BENCHMARK] Running sample {sample_id}: {sample.get('question')}")
        result = run_one_sample(sample, pipeline)
        evaluate_sample(
            result,
            judge_mode=args.judge_mode,
            judge_client=pipeline.get("qwen_client"),
        )
        results.append(result)
        save_json(output_dir / f"result_{sample_id}.json", result)

    summary = compute_summary_metrics(results)
    save_json(output_dir / "summary.json", summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
