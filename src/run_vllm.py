# ============================================================
# run_vllm.py — vLLM 推理 + 结构化 JSON 输出
# 环境：Kaggle (kaggle-vllm) / 本地 4090 (标准 vllm)
# ============================================================
import os
import sys
import json
import subprocess
from pathlib import Path

# ---- 1. 环境准备（必须在 import vllm 之前）----
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Kaggle 专用：加 kaggle-vllm runtime 到搜索路径（本地无此目录会自动跳过）
for p in ("/kaggle/working/vllm-staged", "/kaggle/working/vllm-runtime-overlay"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Kaggle 专用：bootstrap + 激活 env（本地直接跳过）
def _kaggle_env():
    if not os.path.isdir("/kaggle/working"):
        return
    subprocess.run(["kaggle-vllm", "bootstrap", "--strict"], check=True)
    result = subprocess.run(["kaggle-vllm", "env"],
                            capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v.strip("'\"")

_kaggle_env()

# ---- 2. import（环境就绪后再导）----
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
try:
    from kaggle_vllm import KaggleLLM as LLM   # Kaggle
except ImportError:
    from vllm import LLM                        # 本地 4090
import torch

# ---- 3. 工程工具（复用你已有的模块）----
from config import load_config
from seed import set_seed
from logger import setup_logger
from result_writer import save_result

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "adjustment_set_schema.json"

PROMPT = """你是一个因果推断助手。
现在要估计处理变量 T 对结果变量 Y 的总因果效应。
已知因果关系：
X → T
X → Y
T → M
M → Y
T → C
Y → C
其中：
- X 是 T 和 Y 的共同原因
- M 是 T 影响 Y 的中介变量
- C 是 T 和 Y 的碰撞点
请判断每个变量的因果角色，以及在估计 T 对 Y 的总因果效应时是否应该调整。
严格按照提供的 JSON Schema 输出。
不要输出 JSON 以外的任何文字。"""

def main():
    config = load_config()
    set_seed(config["seed"])
    logger = setup_logger(config["logging"]["log_dir"], config["experiment"]["name"])

    logger.info(f"GPU 数量: {torch.cuda.device_count()}")

    # 加载 schema（从文件）
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    # 创建模型（模型名从配置读）
    llm = LLM(
        model=config["model"]["name"],
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    # 结构化 JSON 输出
    structured_outputs = StructuredOutputsParams(json=schema)
    sampling_params = SamplingParams(
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_new_tokens"],
        structured_outputs=structured_outputs,
    )

    logger.info("开始推理")
    outputs = llm.generate([PROMPT], sampling_params=sampling_params)
    raw_output = outputs[0].outputs[0].text
    logger.info(f"模型输出: {raw_output}")

    # 解析 JSON 并落盘
    try:
        parsed = json.loads(raw_output)
        json_valid = True
    except json.JSONDecodeError:
        parsed = {}
        json_valid = False
        logger.error("模型输出不是合法 JSON")

    save_result(
        result={
            "model": config["model"]["name"],
            "seed": config["seed"],
            "json_valid": json_valid,
            "output": raw_output,
        },
        result_dir=config["results"]["result_dir"],
    )
    print("\n===== 模型输出 =====")
    print(raw_output)

if __name__ == "__main__":
    main()
