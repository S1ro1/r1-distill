from dataclasses import dataclass
from yaml import safe_load


@dataclass
class EvalConfig:
    datasets: list[str]

    push_to_hub: bool = False

    batch_size: int | None = 1
    max_samples: int | None = 1


@dataclass
class ScriptConfig:
    project_name: str = "r1-distill"
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    student_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    epochs: int = 1

    lr: float = 1e-5
    batch_size: int = 16
    max_seq_length: int = 512
    max_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100

    eval_config: EvalConfig = None
    run_initial_eval: bool = True


def load_config(config_path: str) -> ScriptConfig:
    with open(config_path, "r") as f:
        config = safe_load(f)

    base_config = ScriptConfig(**config)

    base_config.eval_config = EvalConfig(**config.get("eval_config", {}))

    return base_config
