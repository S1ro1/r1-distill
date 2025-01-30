import wandb
import sys

from src.config import ScriptConfig, load_config
from src.bench import run_eval

from transformers import AutoModelForCausalLM
from torch import optim



def setup_env(config: ScriptConfig):
    student_model = AutoModelForCausalLM.from_pretrained(config.student_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model)

    optimizer = optim.AdamW(student_model.parameters(), lr=config.lr)

    return student_model, teacher_model, optimizer


def main(config_path: str):
    wandb.init(project="r1-distill")
    config = load_config(config_path)
    student_model, _, _ = setup_env(config)

    run_eval(student_model, config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    main(sys.argv[1])
