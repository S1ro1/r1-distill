from dataclasses import dataclass
from typing import Any
from yaml import safe_load


@dataclass
class EvalConfig:
    tasks: dict[str, dict[str, Any]]
    batch_size: int | None = 1


@dataclass
class TrainConfig:
    datasets: list[dict[str, Any]]
    batch_size: int = 1
    epochs: int = 1
    lr: float = 0.00001
    max_seq_length: int = 512
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01


@dataclass
class ScriptConfig:
    project_name: str = "r1-distill"
    model_name: str = "Qwen-1.5B"
    run_name: str | None = None
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    student_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    logging_steps: int = 10
    save_steps: int = 5000

    eval_config: EvalConfig = None
    train_config: TrainConfig = None
    run_initial_eval: bool = True
    run_final_eval: bool = True

    teacher_in_8bit: str | None = None


def load_config(config_path: str) -> ScriptConfig:
    with open(config_path, "r") as f:
        config = safe_load(f)

    base_config = ScriptConfig(**config)

    base_config.eval_config = EvalConfig(**config.get("eval_config", {}))
    base_config.train_config = TrainConfig(**config.get("train_config", {}))

    return base_config
