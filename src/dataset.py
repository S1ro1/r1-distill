from typing import Any
from datasets import load_dataset, Dataset
import datasets
from transformers import AutoTokenizer

from src.config import TrainConfig


def prepare_evol_instruct_code_80k_v1(
    dataset: Dataset, tokenizer: AutoTokenizer
) -> Dataset:
    _prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

    def _format_prompt(instruction: str) -> str:
        return _prompt.format(instruction=instruction)

    def _tokenize(examples):
        return tokenizer(
            _format_prompt(examples["instruction"]),
            padding="max_length",
            truncation=True,
        )

    return dataset.map(_tokenize)


def prepare_meta_math_qa(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    _prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )

    def _format_prompt(instruction: str) -> str:
        return _prompt.format(instruction=instruction)

    def _tokenize(examples):
        return tokenizer(
            _format_prompt(examples["query"]), padding="max_length", truncation=True
        )

    return dataset.map(_tokenize)


def prepare_orca_math(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    _prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )  # TODO: Find the correct prompt

    def _format_prompt(instruction: str) -> str:
        return _prompt.format(instruction=instruction)

    def _tokenize(examples):
        return tokenizer(
            _format_prompt(examples["question"]), padding="max_length", truncation=True
        )

    return dataset.map(_tokenize)


_dataset_preparation_dispatch = {
    "nickrosh/Evol-Instruct-Code-80k-v1": prepare_evol_instruct_code_80k_v1,
    "meta-math/MetaMathQA": prepare_meta_math_qa,
    "microsoft/orca-math-word-problems-200k": prepare_orca_math,
}


class DatasetManager:
    def __init__(
        self, config: TrainConfig, tokenizer: AutoTokenizer, split: str = "train"
    ):
        self.datasets: dict[str, Dataset] = {}
        self.tokenizer = tokenizer
        self._load_datasets(config.datasets, split)

    def _load_datasets(self, dataset_configs: list[dict[str, Any]], split: str):
        for dataset_config in dataset_configs:
            dataset = load_dataset(dataset_config["name"], split=split)

            if dataset_config["num_samples"]:
                dataset = dataset.shuffle().select(range(dataset_config["num_samples"]))

            if dataset_config["name"] in _dataset_preparation_dispatch:
                dataset = _dataset_preparation_dispatch[dataset_config["name"]](
                    dataset, self.tokenizer
                )
            else:
                raise ValueError(
                    f"No preparation function found for dataset {dataset_config['name']}"
                )

            friendly_name = dataset_config["name"].split("/")[-1]
            self.datasets[friendly_name] = dataset

    def get_dataset(self, name: str) -> Dataset:
        """Get a specific dataset by name"""
        return self.datasets[name]

    @property
    def length(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def get_interleaved_dataset(self) -> Dataset:
        """Get all datasets"""
        return datasets.interleave_datasets(list(self.datasets.values()))

    def get_dataset_names(self) -> list[str]:
        """Get list of available dataset names"""
        return list(self.datasets.keys())
