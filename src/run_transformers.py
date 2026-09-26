# ============================================================
# run_transformers.py — Transformers 推理 + 结构化 JSON 输出
# 环境：Kaggle / 本地 4090（直接用 transformers，无需 kaggle-vllm）
# ============================================================
import json
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

# 工程工具（复用已有模块）
from config import load_config
from seed import set_seed
from logger import setup_logger
from result_writer import save_result

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "adjustment_set_schema.json"


def build_prompt(schema_text: str) -> str:
    return f"""你是一个因果推断助手。
现在需要分析一个简单的因果推断问题。
已知：
- treatment: T
- outcome: Y
- X1 是 T 和 Y 的共同原因，因此 X1 是混杂变量。
- X2 是 T 影响 Y 的中介变量。
请判断 X1 和 X2 的变量角色，并确定最终调整集。
请严格按照下面的 JSON Schema 输出。
重要要求：
1. 只能输出 JSON。
2. 不要输出 Markdown。
3. 不要输出 ```json。
4. 所有必填字段都必须存在。
5. role 必须使用 Schema 中允许的值。
6. action 必须使用 Schema 中允许的值。
JSON Schema：
{schema_text}"""


def main():
    config = load_config()
    set_seed(config["seed"])
    logger = setup_logger(config["logging"]["log_dir"], config["experiment"]["name"])

    model_name = config["model"]["name"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    temperature = config["generation"]["temperature"]

    # 1. 加载 schema（从工程文件读，不再写死）
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

    # 2. 加载 tokenizer 与模型
    logger.info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    logger.info(f"模型 device_map: {model.hf_device_map}")

    # 3. 构造 prompt 与输入
    prompt = build_prompt(schema_text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4. 生成（temperature=0 用贪心，符合确定性实验规范）
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f"模型输出: {response}")

    # 5. 解析 + JSON Schema 验证
    try:
        from jsonschema import validate
        result = json.loads(response)
        validate(instance=result, schema=schema)
        json_valid = True
        logger.info("JSON Schema 验证通过")
    except Exception as e:
        result = {}
        json_valid = False
        logger.error(f"JSON 验证失败: {e}")

    # 6. 结果落盘
    save_result(
        result={
            "engine": "transformers",
            "model": model_name,
            "seed": config["seed"],
            "json_valid": json_valid,
            "output": response,
        },
        result_dir=config["results"]["result_dir"],
    )
    print("\n===== 模型输出 =====")
    print(response)


if __name__ == "__main__":
    main()
