# deepgen/models/__init__.py
from .vae import VAEVampPrior
from .gan import Generator, ProjectionDiscriminator
from .classifier import LeNet5

__all__ = ["VAEVampPrior", "Generator", "ProjectionDiscriminator", "LeNet5"]
