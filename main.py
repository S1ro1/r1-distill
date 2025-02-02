import wandb
import sys

from src.config import ScriptConfig, load_config
from src.bench import run_eval

from transformers import AutoModelForCausalLM

from src.redistill_trainer import train_redistill


def setup_env(config: ScriptConfig):
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model, device_map="cuda"
    )
    teacher_model = None

    if config.teacher_model is not None:
        quantization_config = None
        if config.teacher_in_8bit == "hqq":
            from transformers import HqqConfig

            quantization_config = HqqConfig(
                load_in_8bit=True,
            )
        elif config.teacher_in_8bit == "bnb":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            quantization_config=quantization_config,
            device_map="cuda",
        )

    return student_model, teacher_model


def main(config_path: str):
    config = load_config(config_path)
    wandb.init(project="r1-distill", name=config.run_name)
    wandb.save(config_path)

    student_model, teacher_model = setup_env(config)

    if config.run_initial_eval:
        results = run_eval(student_model, config)
        wandb.log(results)

    wandb.config.update(student_model.config.to_dict())

    final_model = train_redistill(student_model, teacher_model, config)

    if config.run_final_eval:
        results = run_eval(final_model, config, is_final_eval=True)
        wandb.log(results)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    main(sys.argv[1])
