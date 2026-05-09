import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from statistics import mean
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional


PUNCT_RE = re.compile(r"[，。！？；：、,.!?:;\s\(\)（）\[\]【】<>《》\"'“”‘’]+")
COLOR_WORDS = [
    "黄色", "绿色", "棕色", "白色", "紫色", "灰色", "红色", "蓝色", "黑色", "橙色",
    "黄", "绿", "棕", "白", "紫", "灰", "红", "蓝", "黑", "橙",
]
STOP_WORDS = {
    "共有", "分别", "是什么", "什么", "多少", "一个", "一种", "可以", "需要", "应该",
    "进行", "包括", "以及", "和", "或", "的", "了", "为", "是", "在", "中",
}
FACT_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:根|个|条|路|页|V|A|mm|cm|m|%|号|pin|PIN)?"),
    re.compile(r"[A-Za-z]{2,}[-_/A-Za-z0-9.]*"),
    re.compile(r"第?\d+页"),
]

VALID_QA_LABELS = {"correct", "partial", "wrong"}


JUDGE_PROMPT_TEMPLATE = """你是一个答案覆盖度评估器。你的任务是判断 predicted_answer 是否覆盖 required_keypoints 中的核心信息。

输入包括：
- question
- gold_answer
- acceptable_answers
- required_keypoints
- optional_keypoints
- predicted_answer

[question]
{question}

[gold_answer]
{gold_answer}

[acceptable_answers]
{acceptable_answers}

[required_keypoints]
{required_keypoints}

[optional_keypoints]
{optional_keypoints}

[predicted_answer]
{predicted_answer}

判定规则：
1. 只判断 predicted_answer 是否覆盖 required_keypoints。
2. 语义等价即可，不要求完全同字。
3. 顺序不同不扣分。
4. 多写解释不扣分，除非与核心答案矛盾。
5. 不要因为措辞、编号、换行、标点不同而扣分。
6. 如果 required_keypoints 全部覆盖，qa_score 应为 1.0。
7. 如果只覆盖部分 required_keypoints，按覆盖比例给分。
8. 如果 predicted_answer 包含与 required_keypoints 明显矛盾的信息，可以降低分数。
9. 不要把 required_keypoints 中已经出现的词判为 unsupported claim。
10. 只输出 JSON，不要输出 markdown 代码块。

输出 JSON 格式：
{{
  "qa_label": "correct|partial|wrong",
  "qa_score": 0.0,
  "covered_keypoints": [],
  "missing_keypoints": [],
  "contradictions": [],
  "hallucination": false,
  "hallucination_score": 0.0,
  "reason": ""
}}

重要：
- 如果 predicted_answer 包含 “COM OUT1”、“NC YM1”、“NO”，而 required_keypoints 是 ["NO", "COM OUT1", "NC YM1"]，必须判为 correct。
- 如果 predicted_answer 只是顺序不同，必须判为 correct。
- 如果 predicted_answer 有额外解释但不矛盾，不要判 hallucination。
"""

RULE_FIRST_TYPES = {"entity_list", "number", "yes_no", "visual_ocr"}
RULE_THEN_LLM_TYPES = {
    "text_comprehension",
    "process_steps",
    "visual_relation",
    "spatial_structure",
    "table_understanding",
}
LLM_PREFERRED_TYPES = {"multi_hop_reasoning", "reasoning"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_text(text: str) -> str:
    """Normalize answer text for deterministic keypoint matching.

    The normalization is intentionally conservative: it keeps Chinese
    characters, letters, digits, and technical tokens such as COM OUT1,
    NC YM1, and DC 0-80V intact while making punctuation and spacing stable.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).upper()
    text = re.sub(r"[，。！？；：、｡。；：]", " ", text)
    text = re.sub(r"[“”‘’\"'`]", "", text)
    text = re.sub(r"[（）\[\]【】<>《》{}]", " ", text)
    text = re.sub(r"(?<=\d)\s*[VＶ]\b", "V", text)
    text = re.sub(r"(?<=\d)\s*[AＡ]\b", "A", text)
    text = re.sub(r"(?<=\d)\s*(米|M)\b", "M", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _loose_keypoint_text(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"[\s\-_:：()\[\]{}（）/\\]+", "", normalized)


def contains_keypoint(predicted_answer: str, keypoint: str) -> bool:
    """Return True if predicted_answer covers keypoint under strict or loose match."""
    pred = normalize_text(predicted_answer)
    key = normalize_text(keypoint)
    if not key:
        return False
    if key in pred:
        return True
    return _loose_keypoint_text(key) in _loose_keypoint_text(pred)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)] if str(value).strip() else []


def get_answer_type(item: Dict[str, Any]) -> str:
    return str(item.get("answer_type") or item.get("question_type") or "unknown").strip() or "unknown"


def rule_score_answer(predicted_answer: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Score an answer by required/optional keypoint coverage."""
    required_keypoints = _as_list(item.get("required_keypoints"))
    optional_keypoints = _as_list(item.get("optional_keypoints"))
    if not required_keypoints:
        return {
            "can_rule_judge": False,
            "reason": "no required_keypoints",
        }

    covered = [kp for kp in required_keypoints if contains_keypoint(predicted_answer, kp)]
    missing = [kp for kp in required_keypoints if kp not in covered]
    covered_optional = [kp for kp in optional_keypoints if contains_keypoint(predicted_answer, kp)]
    missing_optional = [kp for kp in optional_keypoints if kp not in covered_optional]

    required_score = len(covered) / len(required_keypoints)
    optional_score = 0.0
    if optional_keypoints:
        optional_score = 0.1 * len(covered_optional) / len(optional_keypoints)
    final_score = min(1.0, required_score + optional_score)

    if final_score >= 0.95:
        label = "correct"
    elif final_score <= 0.3:
        label = "wrong"
    else:
        label = "partial"

    return {
        "can_rule_judge": True,
        "qa_score": final_score,
        "covered_keypoints": covered,
        "missing_keypoints": missing,
        "covered_optional_keypoints": covered_optional,
        "missing_optional_keypoints": missing_optional,
        "judge_mode": "rule",
        "qa_label": label,
        "hallucination": False,
        "hallucination_score": 0.0,
        "reason": (
            f"required keypoint coverage {len(covered)}/{len(required_keypoints)}, "
            f"optional coverage {len(covered_optional)}/{len(optional_keypoints)}"
        ),
    }


def _acceptable_answer_match(predicted_answer: str, acceptable_answers: List[str]) -> Optional[str]:
    pred = normalize_text(predicted_answer)
    pred_loose = _loose_keypoint_text(predicted_answer)
    for answer in acceptable_answers:
        ans = normalize_text(answer)
        ans_loose = _loose_keypoint_text(answer)
        if ans and (ans == pred or ans in pred or pred in ans):
            return answer
        if ans_loose and (ans_loose == pred_loose or ans_loose in pred_loose or pred_loose in ans_loose):
            return answer
    return None


def _legacy_rule_as_answer_result(predicted_answer: str, item: Dict[str, Any], mode: str = "rule_no_llm") -> Dict[str, Any]:
    legacy = evaluate_with_rule({
        **item,
        "predicted_answer": predicted_answer,
        "gold_answer": item.get("gold_answer") or "",
    })
    return {
        **legacy,
        "judge_mode": mode,
        "covered_keypoints": legacy.get("rule_details", {}).get("qa_keyword_hits", []),
        "missing_keypoints": legacy.get("missing_points", []),
        "covered_optional_keypoints": [],
        "missing_optional_keypoints": [],
    }


def _normalize_llm_answer_result(result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_judge_result(result, raw_output=result.get("raw_judge_output", ""), judge_mode="llm")
    normalized["covered_keypoints"] = result.get("covered_keypoints", [])
    normalized["missing_keypoints"] = result.get("missing_keypoints", result.get("missing_points", []))
    normalized["covered_optional_keypoints"] = result.get("covered_optional_keypoints", [])
    normalized["missing_optional_keypoints"] = result.get("missing_optional_keypoints", [])
    normalized["contradictions"] = result.get("contradictions", [])
    return normalized


def evaluate_answer(
    predicted_answer: str,
    item: Dict[str, Any],
    llm_judge_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rule-first answer evaluation with optional LLM semantic fallback."""
    predicted_answer = predicted_answer or ""
    answer_type = get_answer_type(item)
    rule_result = None

    if _as_list(item.get("required_keypoints")):
        rule_result = rule_score_answer(predicted_answer, item)
        score = float(rule_result.get("qa_score", 0.0))
        if answer_type in RULE_FIRST_TYPES:
            if score >= 0.8 or score <= 0.3:
                return rule_result
        elif answer_type in RULE_THEN_LLM_TYPES:
            if score >= 0.95 or score <= 0.3:
                return rule_result
        elif answer_type in LLM_PREFERRED_TYPES:
            if score >= 0.95:
                return rule_result
        else:
            if score >= 0.95 or score <= 0.3:
                return rule_result
    else:
        acceptable_answers = _as_list(item.get("acceptable_answers"))
        if not acceptable_answers and item.get("gold_answer"):
            acceptable_answers = [str(item.get("gold_answer"))]
        matched = _acceptable_answer_match(predicted_answer, acceptable_answers)
        if matched:
            return {
                "judge_mode": "acceptable_answer",
                "qa_label": "correct",
                "qa_score": 1.0,
                "covered_keypoints": [matched],
                "missing_keypoints": [],
                "covered_optional_keypoints": [],
                "missing_optional_keypoints": [],
                "hallucination": False,
                "hallucination_score": 0.0,
                "reason": "matched acceptable_answers by normalized containment",
            }

    if llm_judge_fn is None:
        if rule_result and rule_result.get("can_rule_judge"):
            return {**rule_result, "judge_mode": "rule_no_llm"}
        return _legacy_rule_as_answer_result(predicted_answer, item, mode="rule_no_llm")

    try:
        raw_llm_result = llm_judge_fn(predicted_answer, item)
        llm_result = _normalize_llm_answer_result(raw_llm_result)
    except Exception as exc:
        if rule_result and rule_result.get("can_rule_judge"):
            return {**rule_result, "judge_mode": "rule_no_llm", "judge_error": str(exc)}
        fallback = _legacy_rule_as_answer_result(predicted_answer, item, mode="rule_no_llm")
        fallback["judge_error"] = str(exc)
        return fallback

    if rule_result and rule_result.get("can_rule_judge"):
        if rule_result.get("qa_score", 0.0) >= 0.95 and llm_result.get("qa_label") == "wrong":
            return {
                **rule_result,
                "judge_mode": "rule",
                "rule_result": rule_result,
                "llm_result": llm_result,
                "judge_conflict": True,
                "reason": f"{rule_result.get('reason', '')}; kept high-confidence rule result over LLM conflict",
            }
        return {
            **llm_result,
            "judge_mode": "llm_fallback",
            "rule_result": rule_result,
            "llm_result": llm_result,
            "judge_conflict": False,
        }

    return {
        **llm_result,
        "judge_mode": "llm",
        "llm_result": llm_result,
        "judge_conflict": False,
    }


def normalize_page_name(path_or_name: str) -> str:
    """Normalize page filenames/paths to page_N for page-hit metrics."""
    if path_or_name is None:
        return ""
    text = str(path_or_name)
    match = re.search(r"page[_-](\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"page_{int(match.group(1))}"
    if re.fullmatch(r"\d+", text.strip()):
        return f"page_{int(text.strip())}"
    return os.path.splitext(os.path.basename(text))[0].lower()


def get_gold_pages(item: Dict[str, Any]) -> List[str]:
    """Return gold pages with backwards-compatible field precedence."""
    gold_pages = _as_list(item.get("gold_pages"))
    if gold_pages:
        return _dedupe(gold_pages)
    return _dedupe(_as_list(item.get("primary_gold_page")) or _as_list(item.get("gold_page")))


def page_hit(pred_pages: List[str], gold_pages: List[str]) -> bool:
    pred_set = {normalize_page_name(page) for page in _as_list(pred_pages)}
    gold_set = {normalize_page_name(page) for page in _as_list(gold_pages)}
    pred_set.discard("")
    gold_set.discard("")
    return bool(pred_set & gold_set)


def add_retrieval_page_metrics(result: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add multi-gold-page retrieval/rerank/final-context hit metrics in-place."""
    item = item or result
    gold_pages = get_gold_pages(item)
    retrieved_pages = _as_list(result.get("retrieved_pages"))
    reranked_pages = _as_list(result.get("reranked_pages"))
    final_pages = _as_list(result.get("final_image_paths"))

    result["gold_pages"] = gold_pages
    result["primary_gold_page"] = item.get("primary_gold_page", item.get("gold_page"))
    result["gold_pages_normalized"] = [normalize_page_name(page) for page in gold_pages]
    result["retrieved_pages_normalized"] = [normalize_page_name(page) for page in retrieved_pages]
    result["reranked_pages_normalized"] = [normalize_page_name(page) for page in reranked_pages]
    result["final_pages_normalized"] = [normalize_page_name(page) for page in final_pages]
    result["retrieval_top1_hit"] = page_hit(retrieved_pages[:1], gold_pages)
    result["retrieval_hit_at_k"] = page_hit(retrieved_pages, gold_pages)
    result["rerank_top1_hit"] = page_hit(reranked_pages[:1], gold_pages)
    result["rerank_hit_at_k"] = page_hit(reranked_pages, gold_pages)
    result["final_context_hit"] = page_hit(final_pages, gold_pages)
    return result


def extract_keywords(text: Optional[str]) -> List[str]:
    """Rule-based keyword extraction for small local benchmark answers."""
    if not text:
        return []

    keywords: List[str] = []
    for color in COLOR_WORDS:
        if color in text:
            keywords.append(color)

    for pattern in FACT_PATTERNS:
        keywords.extend(match.group(0).strip() for match in pattern.finditer(text))

    for part in PUNCT_RE.split(text):
        part = part.strip()
        if not part or part in STOP_WORDS:
            continue
        if len(part) <= 1 and not part.isdigit():
            continue
        if part.endswith("色") or re.search(r"\d|[A-Za-z]", part):
            keywords.append(part)

    return _dedupe(keywords)


def _contains_keyword(text: str, keyword: str) -> bool:
    compact_text = re.sub(r"\s+", "", text or "")
    compact_keyword = re.sub(r"\s+", "", keyword or "")
    return bool(compact_keyword and compact_keyword in compact_text)


def evaluate_qa(gold_answer: str, predicted_answer: str) -> Dict[str, Any]:
    keywords = extract_keywords(gold_answer)
    if not keywords:
        return {"qa_label": "wrong", "qa_hit_ratio": 0.0, "qa_keywords": []}

    hits = [kw for kw in keywords if _contains_keyword(predicted_answer, kw)]
    hit_ratio = len(hits) / len(keywords)
    if hit_ratio >= 0.8:
        label = "correct"
    elif hit_ratio >= 0.4:
        label = "partial"
    else:
        label = "wrong"

    return {
        "qa_label": label,
        "qa_hit_ratio": hit_ratio,
        "qa_keywords": keywords,
        "qa_keyword_hits": hits,
    }


def extract_factual_terms(text: Optional[str]) -> List[str]:
    if not text:
        return []
    terms: List[str] = []
    for color in COLOR_WORDS:
        if color in text:
            terms.append(color)
    for pattern in FACT_PATTERNS:
        terms.extend(match.group(0).strip() for match in pattern.finditer(text))
    return _dedupe(terms)


def build_evidence_corpus(result: Dict[str, Any]) -> str:
    evidence_parts = [
        result.get("gold_answer") or "",
        result.get("evidence_text") or "",
        result.get("ocr_text") or "",
        result.get("metadata_text") or "",
    ]
    return "\n".join(part for part in evidence_parts if part)


def evaluate_hallucination(result: Dict[str, Any]) -> Dict[str, Any]:
    predicted = result.get("predicted_answer") or ""
    corpus = build_evidence_corpus(result)
    factual_terms = extract_factual_terms(predicted)
    unsupported = [term for term in factual_terms if not _contains_keyword(corpus, term)]
    return {
        "hallucination_flag": bool(unsupported),
        "potential_hallucination_terms": unsupported,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_judge_result(parsed: Dict[str, Any], raw_output: str, judge_mode: str) -> Dict[str, Any]:
    qa_label = str(parsed.get("qa_label", "wrong")).strip().lower()
    if qa_label not in VALID_QA_LABELS:
        qa_label = "wrong"

    missing_points = parsed.get("missing_points", [])
    missing_keypoints = parsed.get("missing_keypoints", missing_points)
    covered_keypoints = parsed.get("covered_keypoints", [])
    unsupported_claims = parsed.get("unsupported_claims", [])

    return {
        "judge_mode": judge_mode,
        "qa_label": qa_label,
        "qa_score": _safe_float(parsed.get("qa_score"), 0.0),
        "hallucination": bool(parsed.get("hallucination", False)),
        "hallucination_score": _safe_float(parsed.get("hallucination_score"), 0.0),
        "missing_points": missing_points if isinstance(missing_points, list) else [],
        "covered_keypoints": covered_keypoints if isinstance(covered_keypoints, list) else [],
        "missing_keypoints": missing_keypoints if isinstance(missing_keypoints, list) else [],
        "covered_optional_keypoints": parsed.get("covered_optional_keypoints", []),
        "missing_optional_keypoints": parsed.get("missing_optional_keypoints", []),
        "contradictions": parsed.get("contradictions", []),
        "unsupported_claims": unsupported_claims if isinstance(unsupported_claims, list) else [],
        "reason": str(parsed.get("reason", "")),
        "raw_judge_output": raw_output,
    }


def parse_judge_json(raw_output: str) -> Dict[str, Any]:
    """Parse strict JSON from a local judge, tolerating markdown fences."""
    if not raw_output or not raw_output.strip():
        raise ValueError("empty judge output")

    text = raw_output.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not a JSON object")
    return parsed


def build_judge_prompt(result: Dict[str, Any]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=result.get("question") or "",
        gold_answer=result.get("gold_answer") or "",
        acceptable_answers=json.dumps(_as_list(result.get("acceptable_answers")), ensure_ascii=False),
        required_keypoints=json.dumps(_as_list(result.get("required_keypoints")), ensure_ascii=False),
        optional_keypoints=json.dumps(_as_list(result.get("optional_keypoints")), ensure_ascii=False),
        predicted_answer=result.get("predicted_answer") or "",
    )


def evaluate_with_rule(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback-only local rule evaluator."""
    qa_eval = evaluate_qa(result.get("gold_answer") or "", result.get("predicted_answer") or "")
    hallucination_eval = evaluate_hallucination(result)
    hallucination = bool(hallucination_eval["hallucination_flag"])
    qa_label = qa_eval["qa_label"]
    if qa_label == "correct":
        qa_score = max(0.8, qa_eval.get("qa_hit_ratio", 0.0))
    elif qa_label == "partial":
        qa_score = min(0.79, max(0.4, qa_eval.get("qa_hit_ratio", 0.0)))
    else:
        qa_score = min(0.39, qa_eval.get("qa_hit_ratio", 0.0))

    return {
        "judge_mode": "rule",
        "qa_label": qa_label,
        "qa_score": qa_score,
        "hallucination": hallucination,
        "hallucination_score": 1.0 if hallucination else 0.0,
        "missing_points": [kw for kw in qa_eval.get("qa_keywords", []) if kw not in qa_eval.get("qa_keyword_hits", [])],
        "unsupported_claims": hallucination_eval.get("potential_hallucination_terms", []),
        "reason": "rule-based fallback evaluator",
        "raw_judge_output": "",
        "rule_details": {
            **qa_eval,
            **hallucination_eval,
        },
    }


def evaluate_with_local_llm(result: Dict[str, Any], judge_client: Any) -> Dict[str, Any]:
    """Use the already-loaded local Qwen2-VL/text-capable client as a text-only judge."""
    if judge_client is None:
        raise ValueError("judge_client is required for local_llm judge")

    prompt = build_judge_prompt(result)
    raw_output = judge_client.generate_single_response(
        text=prompt,
        images=[],
        system_prompt=None,
    )
    parsed = parse_judge_json(raw_output)
    return _normalize_judge_result(parsed, raw_output=raw_output, judge_mode="local_llm")


def evaluate_sample(result: Dict[str, Any], judge_mode: str = "local_llm", judge_client: Any = None) -> Dict[str, Any]:
    add_retrieval_page_metrics(result)

    def llm_judge_fn(predicted_answer: str, item: Dict[str, Any]) -> Dict[str, Any]:
        judge_input = {**result, **item, "predicted_answer": predicted_answer}
        return evaluate_with_local_llm(judge_input, judge_client)

    if judge_mode == "rule":
        judge_result = evaluate_answer(
            result.get("predicted_answer") or "",
            result,
            llm_judge_fn=None,
        )
        result["judge_result"] = judge_result
        result["judge_fallback"] = False
        result["judge_error"] = None
        result.update({k: v for k, v in judge_result.items() if k != "raw_judge_output"})
        return result

    if judge_mode != "local_llm":
        raise ValueError(f"Unsupported judge_mode: {judge_mode}")

    try:
        judge_result = evaluate_answer(
            result.get("predicted_answer") or "",
            result,
            llm_judge_fn=llm_judge_fn,
        )
        result["judge_result"] = judge_result
        result["judge_fallback"] = False
        result["judge_error"] = judge_result.get("judge_error")
    except Exception as exc:
        result["judge_fallback"] = True
        result["judge_error"] = str(exc)
        fallback = evaluate_answer(result.get("predicted_answer") or "", result, llm_judge_fn=None)
        fallback["fallback_from"] = "local_llm"
        result["judge_result"] = fallback

    result.update({k: v for k, v in result["judge_result"].items() if k != "raw_judge_output"})
    return result


def _first_basename(paths: Any) -> Optional[str]:
    if not isinstance(paths, list) or not paths:
        return None
    first = paths[0]
    if first is None:
        return None
    return os.path.basename(str(first))


def _avg(results: List[Dict[str, Any]], field: str) -> Optional[float]:
    values = [r.get(field) for r in results if isinstance(r.get(field), (int, float))]
    return mean(values) if values else None


def compute_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total_samples": 0,
            "top1_retrieval_accuracy": 0.0,
            "retrieval_top1_hit_rate": 0.0,
            "retrieval_hit_at_k_rate": 0.0,
            "rerank_top1_hit_rate": 0.0,
            "rerank_hit_at_k_rate": 0.0,
            "final_context_hit_rate": 0.0,
            "qa_accuracy": 0.0,
            "answer_exact_accuracy": 0.0,
            "weighted_qa_accuracy": 0.0,
            "answer_partial_rate": 0.0,
            "answer_wrong_rate": 0.0,
            "average_qa_score": 0.0,
            "hallucination_rate": 0.0,
            "qa_correct_count": 0,
            "qa_partial_count": 0,
            "qa_wrong_count": 0,
            "judge_mode_counts": {},
            "judge_conflict_count": 0,
            "accuracy_by_answer_type": {},
            "average_score_by_answer_type": {},
        }

    qa_correct = 0
    qa_partial = 0
    qa_wrong = 0
    hallucination_count = 0
    qa_scores = []
    judge_modes = Counter()
    judge_conflict_count = 0
    by_type_total = Counter()
    by_type_correct = Counter()
    by_type_scores = defaultdict(list)

    for result in results:
        add_retrieval_page_metrics(result)
        result["top1_retrieval_correct"] = bool(result.get("retrieval_top1_hit"))

        judge_result = result.get("judge_result") or evaluate_with_rule(result)
        result["judge_result"] = judge_result
        result.update({k: v for k, v in judge_result.items() if k != "raw_judge_output"})

        answer_type = get_answer_type(result)
        by_type_total[answer_type] += 1
        qa_label = judge_result.get("qa_label")
        qa_score = _safe_float(judge_result.get("qa_score"), 0.0)
        qa_scores.append(qa_score)
        by_type_scores[answer_type].append(qa_score)
        judge_modes[str(judge_result.get("judge_mode", "unknown"))] += 1
        if bool(judge_result.get("judge_conflict", result.get("judge_conflict", False))):
            judge_conflict_count += 1

        if qa_label == "correct":
            qa_correct += 1
            by_type_correct[answer_type] += 1
        elif qa_label == "partial":
            qa_partial += 1
        else:
            qa_wrong += 1

        if bool(judge_result.get("hallucination", False)):
            hallucination_count += 1

    def rate(field: str) -> float:
        return sum(1 for result in results if bool(result.get(field))) / total

    accuracy_by_answer_type = {
        key: by_type_correct[key] / by_type_total[key]
        for key in sorted(by_type_total)
    }
    average_score_by_answer_type = {
        key: mean(values) if values else 0.0
        for key, values in sorted(by_type_scores.items())
    }
    weighted_qa_accuracy = (qa_correct + 0.5 * qa_partial) / total

    return {
        "total_samples": total,
        "top1_retrieval_accuracy": rate("retrieval_top1_hit"),
        "retrieval_top1_hit_rate": rate("retrieval_top1_hit"),
        "retrieval_hit_at_k_rate": rate("retrieval_hit_at_k"),
        "rerank_top1_hit_rate": rate("rerank_top1_hit"),
        "rerank_hit_at_k_rate": rate("rerank_hit_at_k"),
        "final_context_hit_rate": rate("final_context_hit"),
        "qa_accuracy": weighted_qa_accuracy,
        "answer_exact_accuracy": qa_correct / total,
        "weighted_qa_accuracy": weighted_qa_accuracy,
        "answer_partial_rate": qa_partial / total,
        "answer_wrong_rate": qa_wrong / total,
        "average_qa_score": mean(qa_scores) if qa_scores else 0.0,
        "hallucination_rate": hallucination_count / total,
        "qa_correct_count": qa_correct,
        "qa_partial_count": qa_partial,
        "qa_wrong_count": qa_wrong,
        "judge_mode_counts": dict(judge_modes),
        "judge_conflict_count": judge_conflict_count,
        "accuracy_by_answer_type": accuracy_by_answer_type,
        "average_score_by_answer_type": average_score_by_answer_type,
        "avg_retrieval_latency_sec": _avg(results, "retrieval_latency_sec"),
        "avg_rerank_latency_sec": _avg(results, "rerank_latency_sec"),
        "avg_generation_latency_sec": _avg(results, "generation_latency_sec"),
        "avg_total_latency_sec": _avg(results, "total_latency_sec"),
        "avg_peak_gpu_memory_gb": _avg(results, "peak_gpu_memory_gb"),
        "avg_input_ids_length": _avg(results, "input_ids_length"),
        "avg_visual_tokens": _avg(results, "visual_tokens"),
        "results": results,
    }


def evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible summary entry point for saved result files."""
    return compute_summary_metrics(results)


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    result_files = sorted(results_dir.glob("result_*.json"))
    return [load_json(path) for path in result_files]


def print_summary(summary: Dict[str, Any]) -> None:
    def pct(value: Optional[float]) -> str:
        return f"{(value or 0.0) * 100:.2f}%"

    def num(value: Optional[float], digits: int = 2) -> str:
        return "null" if value is None else f"{value:.{digits}f}"

    def integer(value: Optional[float]) -> str:
        return "null" if value is None else str(int(round(value)))

    print("=" * 50)
    print("Benchmark Summary")
    print("=" * 50)
    print(f"Total Samples: {summary.get('total_samples', 0)}")
    print()
    print(f"Top-1 Retrieval Accuracy: {pct(summary.get('top1_retrieval_accuracy'))}")
    print(f"Retrieval Hit@K: {pct(summary.get('retrieval_hit_at_k_rate'))}")
    print(f"Rerank Top-1 Hit: {pct(summary.get('rerank_top1_hit_rate'))}")
    print(f"Rerank Hit@K: {pct(summary.get('rerank_hit_at_k_rate'))}")
    print(f"Final Context Hit: {pct(summary.get('final_context_hit_rate'))}")
    print()
    print(f"Answer Exact Accuracy: {pct(summary.get('answer_exact_accuracy'))}")
    print(f"Weighted QA Accuracy: {pct(summary.get('weighted_qa_accuracy', summary.get('qa_accuracy')))}")
    print(f"Average QA Score: {num(summary.get('average_qa_score'), 3)}")
    print(f"Hallucination Rate: {pct(summary.get('hallucination_rate'))}")
    print()
    print(f"QA Correct: {summary.get('qa_correct_count', 0)}")
    print(f"QA Partial: {summary.get('qa_partial_count', 0)}")
    print(f"QA Wrong: {summary.get('qa_wrong_count', 0)}")
    print(f"Judge Modes: {summary.get('judge_mode_counts', {})}")
    print(f"Judge Conflicts: {summary.get('judge_conflict_count', 0)}")
    print()
    print(f"Avg Retrieval Latency: {num(summary.get('avg_retrieval_latency_sec'))} s")
    print(f"Avg Rerank Latency: {num(summary.get('avg_rerank_latency_sec'))} s")
    print(f"Avg Generation Latency: {num(summary.get('avg_generation_latency_sec'))} s")
    print(f"Avg Total Latency: {num(summary.get('avg_total_latency_sec'))} s")
    print()
    print(f"Avg Peak GPU Memory: {num(summary.get('avg_peak_gpu_memory_gb'))} GB")
    print(f"Avg Input IDs Length: {integer(summary.get('avg_input_ids_length'))}")
    print(f"Avg Visual Tokens: {integer(summary.get('avg_visual_tokens'))}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark results.")
    parser.add_argument("--results_dir", default="evaluation/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    summary = compute_summary_metrics(results)
    save_json(results_dir / "summary.json", summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
