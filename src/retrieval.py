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

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name = "nithinreddyy/finetuned-esg",  # sentence-transformers/all-mpnet-base-v2
            normalize = True  # Normalize embeddings
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


    def search(self, query: str, filename: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            filename: Optional filename to filter results
            top_k: Number of results to return

        Returns:
            List of matching documents with text, page number, and certainty
        """
        try:
            results = self.retriever.search(query, filename=filename, top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []

    def handle_user_query(self):
        """Interactive CLI for querying documents"""
        print("\nESG Document Query Interface")
        print("Enter your questions about ESG topics or type 'quit' to exit")
        print("To search in a specific file, use: @filename your question")

        try:
            while True:
                query = input("\nQuery> ").strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break

                if not query:
                    continue

                # Check for optional filename as input
                filename = None
                if query.startswith('@'):
                    parts = query.split(' ', 1)
                    if len(parts) > 1:
                        filename = parts[0][1:]  # Remove @ symbol
                        query = parts[1]

                results = self.search(query, filename=filename)

                if not results:
                    print("No relevant results found.")
                    continue

                print("\nRelevant passages found:")
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] Score: {result.get('score', 0):.2f}")
                    print(f"Source: {result['filename']}, Page: {result.get('page_number', 'N/A')}")
                    print(f"Text: {result['text'].strip()}")

        except KeyboardInterrupt:
            print("\nSearch terminated.")

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
