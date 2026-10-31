
import json
from pathlib import Path

GROUND_TRUTH_PATH = Path("data/evaluation/ground_truth.json")

NEW_CHUNK_MAPPING = {
    "Q001": [0],
    "Q002": [0],
    "Q003": [1],
    "Q004": [1],
    "Q005": [1],
    "Q006": [25],
    "Q007": [27],
    "Q008": [28],
    "Q009": [23, 24],
    "Q010": [32],
    "Q011": [8, 9],
    "Q012": [10],
    "Q013": [12],
    "Q014": [13],
    "Q015": [14],
    "Q016": [55, 56],
    "Q017": [60],
    "Q018": [69],
    "Q019": [67],
    "Q020": [39],
    "Q021": [41, 42],
    "Q022": [43],
    "Q023": [47, 48],
    "Q024": [49, 50],
    "Q025": [81],
    "Q026": [85],
    "Q027": [87],
    "Q028": [79],
}


def main():
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    for item in dataset:
        question_id = item["id"]

        if question_id not in NEW_CHUNK_MAPPING:
            continue

        chunk_ids = NEW_CHUNK_MAPPING[question_id]

        item["relevant_chunk_ids"] = chunk_ids
        item["relevance_grades"] = {
            str(chunk_id): 3
            for chunk_id in chunk_ids
        }

    GROUND_TRUTH_PATH.write_text(
        json.dumps(dataset, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Ground truth updated successfully!")
    print("Answerable questions remapped:", len(NEW_CHUNK_MAPPING))


if __name__ == "__main__":
    main()
