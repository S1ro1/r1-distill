from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from typing_extensions import override

from src.utils import inherit_signature_from


class CustomHFML(HFLM):
    @override
    @inherit_signature_from(HFLM.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float]]:
        return super().loglikelihood(requests, disable_tqdm=False)
