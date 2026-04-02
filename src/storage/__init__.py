"""Storage backends for experiment results."""

from .mongodb import MongoDBStorage

__all__ = ["MongoDBStorage"]
