from typing import Any
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


clean_dispatch = {
    "leaderboard_musr": partial(clean_task, group_name="leaderboard_musr"),
    "leaderboard_bbh": partial(clean_task, group_name="leaderboard_bbh"),
    "leaderboard_mmlu_pro": partial(clean_task, group_name=None),
    "leaderboard_gpqa": partial(clean_task, group_name="leaderboard_gpqa"),
    "leaderboard_ifeval": clean_ifeval,
}
