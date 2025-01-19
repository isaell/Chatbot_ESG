import torch
import logging
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
    Initialize MongoDB connection parameters
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

            # Ensure vector search index exists
            #Need to initialize first, or run:
            try:
                self.collection.create_index([
                    ("embedding", "vector", {"numDimensions": 768, "similarity": "cosine"})
                ])
                self.logger.info("Vector search index created")
            except:
                pass

            self.logger.info(f'Connected to MongoDB database: {self.db_name}')
            return self

        except Exception as e:
            self.logger.error(f'Failed to connect to MongoDB: {str(e)}')
            raise


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

            documents = [
                {
                    'filename': filename,
                    'chunk_id': i,
                    'text': chunk['text'],
                    'metadata': {
                        **chunk.get('metadata', {}),
                        'page_number': chunk.get('metadata', {}).get('page_number', 0)
                    },
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
            self.logger.info("MongoDB connection closed")
            self.client = None


class MongoDBRetriever(VectorRetriever):
    def __init__(self,
        collection,
        #model_name: str = 'sentence-transformers/all-mpnet-base-v2'
        model_name: str = 'nithinreddyy/finetuned-esg'
        ):

        from ingestion import ParseChunkEmbed
        self.embedder = ParseChunkEmbed() # Create instance here
        self.collection = collection
        self.logger = logging.getLogger(__name__)

    def search(self,
        query: str,
        filename: str = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            # Generate query embedding
            query_embedding = self.embedder.compute_embeddings([query])[0].tolist()

            # Build the aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": top_k,
                        "index": "vector_index",
                    }
                }
            ]

            # Add filename filter if specified
            if filename:
                pipeline.append({
                    "$match": {"filename": filename}
                })

            # MongoDB vector search
            results = self.collection.aggregate(pipeline)
            results_list = list(results)

            self.logger.info(f"Found {len(results_list)} results for query: {query}")
            if filename:
                self.logger.info(f"Filtered by filename: {filename}")

            return [
                {
                    'text': doc['text'],
                    'filename': doc['filename'],
                    'metadata': doc.get('metadata', {}),
                    'page_number': doc.get('metadata', {}).get('page_number', 0),
                    'score': doc.get('score', 0.0)
                }
                for doc in results_list
            ]

        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []
