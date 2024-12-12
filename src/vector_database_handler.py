from abc import ABC, abstractmethod
import numpy as np

class VectorDatabaseHandler(ABC):
    @abstractmethod
    def connect(self):
        """Establish database connection"""
        pass

    @abstractmethod
    def insert_embedding(self, document_id: str, text_chunk: str, embedding: np.ndarray):
        """Insert embedding into the database"""
        pass

    @abstractmethod
    def close_connection(self):
        """Close database connection"""
        pass

    def __enter__(self):
        """Support context manager protocol"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context"""
        self.close_connection()
