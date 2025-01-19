from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class VectorDatabaseHandler(ABC):
    """
    designed to serve as an abstract base class for handling vector database operations,
    i.e. to define a blueprint for other classes to implement.
    """
    @abstractmethod
    def connect(self) -> 'VectorDatabaseHandler':
        """Establish database connection"""
        pass

    @abstractmethod
    def insert_embeddings_batch(self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: torch.Tensor
    ) -> None:
        """
        Insert batch of embeddings into database.

        Args:
            filename: Source document name
            chunks: List of dictionaries containing text and metadata
            embeddings: Tensor of embeddings
        """
        pass

    @abstractmethod
    def close_connection(self) -> None:
        """Close database connection"""
        pass

    def __enter__(self):
        """Support context manager protocol"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context"""
        self.close_connection()


class VectorRetriever(ABC):
    @abstractmethod
    def search(self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents with text and metadata
        """
        pass