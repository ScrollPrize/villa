from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    image: str
    description: str
    default_args: Dict[str, str]


# Initial registry; extend as more models are supported.
_SUPPORTED_MODELS: Dict[str, ModelInfo] = {
    "timesformer_gp": ModelInfo(
        name="timesformer_gp",
        # TODO: Put a proper image here
        image="",
        description="""
            Grand-Prize TimeSformer-based ink detection model. 
            See https://github.com/ScrollPrize/villa/tree/main/ink-detection for details.
        """,
        default_args={
            "tile_size": "64",
            "stride": "32",
            "batch_size": "64",
            "workers": "4",
        },
    ),
}


def list_models() -> List[ModelInfo]:
    return list(_SUPPORTED_MODELS.values())


def get_model(name: str) -> Optional[ModelInfo]:
    return _SUPPORTED_MODELS.get(name)
