from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from typing import Any
from transformers import AutoModelForCausalLM
from src.config import ScriptConfig
from src.lm_wrapper import CustomHFML
from collections import defaultdict


def clean_metrics(results: dict) -> dict:
    cleaned = {}
    for key, value in results.items():
        clean_key = key.split(",")[0]
        if "." in clean_key:
            parts = clean_key.split(".")
            current = cleaned
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            cleaned[clean_key] = value
    cleaned.pop("alias")
    return cleaned


def clean_config(config: dict) -> dict:
    return {
        k: v
        for k, v in config.items()
        if v is not None and k != "batch_sizes"  # TODO: wtf?
    }


def run_single_task(
    model: CustomHFML,
    task_name: str,
    task_config: dict[str, Any],
    task_manager: TaskManager,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = simple_evaluate(
        model=model,
        tasks=[task_name],
        task_manager=task_manager,
        device="cuda",
        cache_requests=True,
        **task_config,
    )

    task_results = results["results"][task_name]
    cleaned_results = clean_metrics(task_results)

    return cleaned_results, clean_config(results["config"])


def run_eval(
    model: AutoModelForCausalLM,
    config: ScriptConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert model.device.type == "cuda", "Model must be on GPU, is on {}".format(
        model.device.type
    )

    model = CustomHFML(
        pretrained=model, batch_size=config.eval_config.batch_size, device="cuda"
    )

    task_manager = TaskManager()
    results = defaultdict(lambda: defaultdict(dict))

    for task_name, task_config in config.eval_config.tasks.items():
        task_results, model_config = run_single_task(
            model, task_name, task_config, task_manager
        )

        for metric_name, value in task_results.items():
            if metric_name:
                results[metric_name][task_name] = value

    model_config["model_dtype"] = str(model_config["model_dtype"])
    return results, model_config
