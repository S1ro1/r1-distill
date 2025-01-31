from .config import ScriptConfig

import os

from lighteval.pipeline import (
    Accelerator,
    Pipeline,
    PipelineParameters,
    ParallelismManager,
)
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.utils.utils import EnvConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


def run_eval(config: ScriptConfig):
    hf_home = os.getenv("HF_HOME")
    env_config = EnvConfig(cache_dir=hf_home)

    model_config = TransformersModelConfig(
        accelerator=Accelerator(log_with=["wandb"]),
        pretrained=config.student_model,
        dtype="bfloat16",
        use_chat_template=True,
    )

    eval_tracker = EvaluationTracker(
        output_dir="results/",
        save_details=True,
        push_to_hub=config.eval_config.push_to_hub,
        hub_results_org="s1ro1",
    )

    pipeline_params = PipelineParameters(
        override_batch_size=config.eval_config.batch_size,
        # max_samples=config.eval_config.max_samples,
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=env_config,
    )

    pipeline = Pipeline(
        tasks=config.eval_config.datasets[0],  # TODO: more tasks
        pipeline_parameters=pipeline_params,
        evaluation_tracker=eval_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()
