"""
Training module for Vesuvius models.
"""

from .train import BaseTrainer, main
from .self_supervised_trainer import SelfSupervisedTrainer

__all__ = [
    'BaseTrainer',
    'SelfSupervisedTrainer',
    'main',
]