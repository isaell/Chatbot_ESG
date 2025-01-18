import torch
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from utils import safe_env_get
from vector_database_handler import VectorDatabaseHandler, VectorRetriever

# from importlib.metadata import version
# print(f"Pymongo version (for MongoDB): {version('pymongo')}")

class MongoDBHandler(VectorDatabaseHandler):
    """
    Initialize MongoDB connection parameters.

    :param db_name: Name of the database to connect to
    :param collection_name: Name of the collection to use
    :param uri: Optional MongoDB connection URI (overrides environment variable)
    """

    def __init__(self,
                db_name: str = None,
                collection_name: str = None,
                uri: Optional[str] = None):

        self.db_name = db_name
        self.collection_name = collection_name
        self.uri = uri or safe_env_get('MONGODB_URI')

        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None

        self.logger = logging.getLogger(__name__)

    def connect(self) -> 'MongoDBHandler':
        try:
            if not self.uri:
                raise ValueError('MongoDB URI not provided')

            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

            print(f'Connected to MongoDB database: {self.db_name}')
            return self

        except Exception as e:
            self.logger.error(f'Failed to connect to MongoDB: {str(e)}')
            raise

    def insert_embedding(self,
            document_id: str,
            text_chunk: str,
            embedding: np.ndarray):
            """
            Insert embeddings into MongoDB.

            Args:
                document_id (str):
                text_chunk (str):
                embeddings (array):
            """
            self.collection.insert_one({
                'document_id': document_id,
                'text_chunk': text_chunk,
                'embedding': embedding.tolist()
            })


    def insert_embeddings_batch(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: torch.Tensor
    ) -> None:
        """
        Insert batch of embeddings into MongoDB.

        Args:
            filename (str): Source filename
            chunks (list): List of chunk dictionaries containing text and metadata
            embeddings (torch.Tensor): Tensor of embeddings
        """
        try:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

                try:
                    #Note that Atlas vector search is used. Need to initialize first, or run:
                    self.collection.create_index([('embedding', 'vector')])
                except:
                    pass

            documents = [
                {
                    'filename': filename,
                    'chunk_id': i,
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {}),
                    'embedding': embedding.tolist()
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # insert_many internally optimizes the insertion process(native batch insertion)
            # Qdrant requires more specific handling
            self.collection.insert_many(documents)
            self.logger.info(f"Inserted {len(documents)} embeddings for {filename}")

        except Exception as e:
            self.logger.error(f"Failed to insert embeddings: {str(e)}")
            raise

    def close_connection(self) -> None:
        if self.client:
            self.client.close()
            print("MongoDB connection closed")
            self.client = None


class MongoDBRetriever(VectorRetriever):
    def __init__(self,
        collection,
        model_name: str = 'nbroad/ESG-BERT'):

        from ingestion import ParseChunkEmbed
        self.embedder = ParseChunkEmbed() # Create instance here
        self.collection = collection
        self.logger = logging.getLogger(__name__)

    def search(self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:

        try:
            # Generate query embedding
            query_embedding = self.embedder.compute_embeddings([query])[0].tolist()

            # MongoDB vector search
            results = self.collection.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": top_k,
                        "index": "vector_index",
                    }
                }
            ])

            return [
                {
                    'text': doc['text'],
                    'filename': doc['filename'],
                    'metadata': doc.get('metadata', {}),
                    'score': doc.get('score', 0.0)
                }
                for doc in results
            ]

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []


    # def retrieve_data(self, query: str, top_k: int = 5):
    #     self.connect()  # Re-establish connection
    #     try:
    #         # Perform your retrieval logic here
    #         results = self.search(query, top_k)
    #         return results
    #     finally:
    #         self.close_connection()  # Ensure the connection is closed afterward