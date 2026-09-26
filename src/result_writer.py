import csv
import json
from pathlib import Path


def save_result(result: dict, result_dir: str, filename: str = "results"):
    """保存实验结果，并自动追加到 CSV 表格。"""

    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. 保存 JSON
    # =========================
    json_file = result_path / f"{filename}.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # =========================
    # 2. 追加到 CSV
    # =========================
    csv_file = result_path / f"{filename}.csv"

    file_exists = csv_file.exists()

    with open(
        csv_file,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        # 第一次写入时创建表头
        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print(f"实验结果已保存：{json_file}")
    print(f"实验结果表已更新：{csv_file}")