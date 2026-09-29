from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError
