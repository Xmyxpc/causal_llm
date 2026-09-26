import logging
from pathlib import Path


def setup_logger(log_dir: str, experiment_name: str):
    """创建实验日志记录器。"""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"{experiment_name}.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler
    if logger.handlers:
        return logger

    # 保存到文件
    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8"
    )

    # 同时输出到终端
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger