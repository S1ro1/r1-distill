import wandb
import sys

from src.config import ScriptConfig, load_config
from src.bench import run_eval

from transformers import AutoModelForCausalLM
from torch import optim


def setup_env(config: ScriptConfig):
    student_model = AutoModelForCausalLM.from_pretrained(config.student_model)
    if config.teacher_model is not None:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
        )
    else:
        teacher_model = None

    optimizer = optim.AdamW(student_model.parameters(), lr=config.lr)

    return student_model, teacher_model, optimizer


def main(config_path: str):
    config = load_config(config_path)
    wandb.init(project="r1-distill", name=config.run_name)
    wandb.save(config_path)

    student_model, _, _ = setup_env(config)
    student_model.to("cuda")

    if config.run_initial_eval:
        results, model_config = run_eval(student_model, config)

    wandb.log(results)

    from pprint import pprint

    pprint(results)

    try:
        wandb.config.update(model_config)
    except Exception as e:
        print(f"Error updating wandb config: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    main(sys.argv[1])
