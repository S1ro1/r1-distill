from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from typing import Any
from transformers import AutoModelForCausalLM
from src.config import ScriptConfig
from src.lm_wrapper import CustomHFML
from collections import defaultdict
from functools import partial


def clean_ifeval(results: dict[str, Any]) -> dict[str, Any]:
    ret = {"leaderboard_ifeval": {}}
    for k, v in results["leaderboard_ifeval"].items():
        if k == "alias":
            continue
        if "stderr" in k:
            continue
        unparsed_metric_name = k.strip(",none")
        parts = unparsed_metric_name.split("_")

        metric_name = f"{parts[0]}_{parts[2]}_{parts[3]}"

        ret["leaderboard_ifeval"][metric_name] = v
    return ret


def clean_task(
    results: dict[str, Any], group_name: str | None = None
) -> dict[str, Any]:
    if group_name:
        results.pop(group_name)

    ret = defaultdict(dict)
    total_acc = 0.0

    for sub_task, sub_task_results in results.items():
        acc = (
            sub_task_results["acc_norm,none"]
            if group_name
            else sub_task_results["acc,none"]
        )
        ret[sub_task]["acc"] = acc
        total_acc += acc

    if group_name:
        ret[group_name]["acc"] = total_acc / len(results)
    return ret


_clean_dispatch = {
    "leaderboard_musr": partial(clean_task, group_name="leaderboard_musr"),
    "leaderboard_bbh": partial(clean_task, group_name="leaderboard_bbh"),
    "leaderboard_mmlu_pro": partial(clean_task, group_name=None),
    "leaderboard_gpqa": partial(clean_task, group_name="leaderboard_gpqa"),
    "leaderboard_ifeval": clean_ifeval,
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
        use_cache="cache.db",
        **task_config,
    )

    task_metrics = _clean_dispatch[task_name](results["results"])

    return task_metrics, None


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
    results = {}

    for task_name, task_config in config.eval_config.tasks.items():
        task_results, model_config = run_single_task(
            model, task_name, task_config, task_manager
        )
        results.update(task_results)

    return results, model_config
