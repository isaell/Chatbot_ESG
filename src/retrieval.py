
import logging
from typing import List, Dict, Any
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from Qdrant_handler import QdrantHandler, QdrantRetriever
from MongoDB_handler import MongoDBHandler, MongoDBRetriever


class RetrievalService:
    def __init__(self, database_type: str, **db_params):
        """
        Initialize retrieval service with specified database type and parameters.

        Args:
            database_type: 'MongoDB' or 'Qdrant'
            **db_params: Database-specific parameters
                MongoDB: db_name, collection_name
                Qdrant: collection_name
        """
        self.logger = logging.getLogger(__name__)
        self.database_type = database_type
        self.db_params = db_params

        # Initialize embedding model (ESG-BERT)
        self.embed_model = HuggingFaceEmbedding(
            model_name = "nbroad/ESG-BERT",
            normalize = True  # Normalize embeddings for better similarity search
            #model_kwargs = {"device": "cuda"},  # Use "cpu" if no GPU available
        )
        Settings.embed_model = self.embed_model

        # Initialize database handler and retriever
        self.db_handler = None
        self.retriever = None
        self.connect_to_database()


    def connect_to_database(self):
        """Connect to the specified database and initialize the retriever."""
        if self.database_type == 'MongoDB':
            self.db_handler = MongoDBHandler(**self.db_params)
            self.db_handler.connect()
            self.retriever = MongoDBRetriever(self.db_handler.collection)
        elif self.database_type == 'Qdrant':
            self.db_handler = QdrantHandler(**self.db_params)
            self.db_handler.connect()
            self.retriever = QdrantRetriever(self.db_handler.client, self.db_params['collection_name'])
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of matching documents with text, page number, and certainty
        """
        try:
            results = self.retriever.search(query, top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []

    def handle_user_query(self):
        """Interactive function to handle user queries"""
        try:
            while True:
                query = input("\nEnter your query (or 'quit' to exit): ")
                if query.lower() == 'quit':
                    break

                results = self.search(query)

                print(f"\nTop {len(results)} results for query: '{query}'")
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Page: {result.get('page_number', 'N/A')}")
                    print(f"Certainty: {result.get('certainty', result.get('score', 0)):.2f}")
                    print(f"Text: {result['text'][:200]}...")


        except KeyboardInterrupt:
            print("\nExiting search...")
        finally:
            self.close()

    def close(self):
        """Close database connection"""
        if self.db_handler:
            self.db_handler.close_connection()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self  # Return the instance itself

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        self.close()  # Ensure the connection is closed
