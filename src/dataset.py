from typing import Any
from datasets import load_dataset, IterableDataset

from src.config import TrainConfig


class DatasetManager:
    def __init__(self, config: TrainConfig, split: str = "train"):
        self.datasets: dict[str, IterableDataset] = {}
        self._load_datasets(config.datasets, split)

    def _load_datasets(self, dataset_configs: list[dict[str, Any]], split: str):
        for dataset_config in dataset_configs:
            dataset = load_dataset(dataset_config["name"], streaming=True, split=split)

            if dataset_config["num_samples"]:
                dataset = dataset.take(dataset_config["num_samples"])

            friendly_name = dataset_config["name"].split("/")[-1]
            self.datasets[friendly_name] = dataset

    def get_dataset(self, name: str) -> IterableDataset:
        """Get a specific dataset by name"""
        return self.datasets[name]

    def get_all_datasets(self) -> dict[str, IterableDataset]:
        """Get all datasets"""
        return self.datasets[list(self.datasets.keys())[0]]  # TODO: fix later

    def get_dataset_names(self) -> list[str]:
        """Get list of available dataset names"""
        return list(self.datasets.keys())
