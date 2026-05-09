import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import evaluate_answer, page_hit


def test_keypoint_rule_correct():
    item = {
        "question": "继电器信号包含哪三个接口标识？",
        "gold_answer": "继电器信号包含NO、COM OUT1和NC YM1。",
        "required_keypoints": ["NO", "COM OUT1", "NC YM1"],
        "answer_type": "entity_list",
    }

    predicted_answer = """
根据提供的图片和文档内容，继电器信号包含以下三个接口标识：
1. COM OUT1 - 这是继电器的公共端
2. NC YM1 - 这是继电器的常闭触点
3. NO - 这是继电器的常开触点
"""

    result = evaluate_answer(predicted_answer, item)
    assert result["qa_label"] == "correct"
    assert result["qa_score"] >= 0.95
    assert result["judge_mode"] == "rule"
    assert result["missing_keypoints"] == []


def test_page_hit_multi_gold_page():
    gold_pages = ["adas_manual_page_10.png"]
    retrieved_pages = [
        "data/page_images/ADAS--20241202-v3.3_page_16.png",
        "data/page_images/ADAS--20241202-v3.3_page_10.png",
    ]
    assert page_hit(retrieved_pages, gold_pages) is True


if __name__ == "__main__":
    test_keypoint_rule_correct()
    test_page_hit_multi_gold_page()
    print("evaluation rule/keypoint tests passed")
