from pathlib import Path

import yaml

from seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    config = load_config()

    print("实验配置读取成功！")
    print(f"随机种子: {config['seed']}")
    print(f"模型: {config['model']['name']}")
    print(f"Temperature: {config['generation']['temperature']}")
    print(f"最大生成长度: {config['generation']['max_new_tokens']}")

    # 根据配置文件中的 seed 固定随机种子
    set_seed(config["seed"])

    print("随机种子设置成功！")