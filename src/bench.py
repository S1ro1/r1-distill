from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from typing import Any
from transformers import AutoModelForCausalLM
from src.config import ScriptConfig
from src.lm_wrapper import CustomHFML
from src.cleaners import clean_dispatch
import wandb


_metric_lookup = {
    "leaderboard_ifeval": "inst_strict_acc",
}


def run_single_task(
    model: CustomHFML,
    task_name: str,
    task_config: dict[str, Any],
    task_manager: TaskManager,
) -> dict[str, Any]:
    results = simple_evaluate(
        model=model,
        tasks=[task_name],
        task_manager=task_manager,
        device="cuda",
        cache_requests=True,
        use_cache="cache.db",
        **task_config,
    )

    task_metrics = clean_dispatch[task_name](results["results"])

    return task_metrics


def run_eval(
    model: AutoModelForCausalLM,
    config: ScriptConfig,
) -> dict[str, Any]:
    assert model.device.type == "cuda", "Model must be on GPU, is on {}".format(
        model.device.type
    )

    model = CustomHFML(
        pretrained=model, batch_size=config.eval_config.batch_size, device="cuda"
    )

    task_manager = TaskManager()
    results = {}

    main_task_accuracies = {}

    for task_name, task_config in config.eval_config.tasks.items():
        task_results = run_single_task(model, task_name, task_config, task_manager)
        results.update(task_results)

        if task_name in clean_dispatch:
            for key, value in task_results.items():
                if key == task_name:
                    main_task_accuracies[task_name.replace("leaderboard_", "")] = value[
                        _metric_lookup.get(task_name, "acc")
                    ]

    wandb.log(
        {
            "task_accuracies": wandb.plot.bar(
                wandb.Table(
                    data=[[k, v] for k, v in main_task_accuracies.items()],
                    columns=["task", "accuracy"],
                ),
                "task",
                "accuracy",
                title="Main Task Accuracies",
            )
        }
    )

    return results
