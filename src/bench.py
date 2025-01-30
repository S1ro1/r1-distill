from .config import ScriptConfig

from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.utils.utils import EnvConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


def run_eval(model, config: ScriptConfig):
    eval_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=config.eval_config.push_to_hub,
    )

    pipeline_params = PipelineParameters(
        override_batch_size=config.eval_config.batch_size,
        max_samples=config.eval_config.max_samples,
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
    )

    pipeline = Pipeline(
        tasks=config.eval_config.datasets[0],  # TODO: more tasks
        pipeline_parameters=pipeline_params,
        evaluation_tracker=eval_tracker,
        model=model,
    )

    pipeline.evaluate()
    pipeline.show_results()
