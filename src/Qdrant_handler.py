import uuid
import torch
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from utils import safe_env_get
from vector_database_handler import VectorDatabaseHandler, VectorRetriever


class QdrantHandler(VectorDatabaseHandler):
    """
    Initializes Qdrant connection
    """
    def __init__(self,
                collection_name: str = None,
                url: Optional[str] = None,
                api_key: Optional[str] = None,
                batch_size: int = 100):

        self.url = url or safe_env_get("QDRANT_HOST")
        self.api_key = api_key or safe_env_get("QDRANT_KEY")
        self.collection_name = collection_name
        self.batch_size = batch_size  # Maximum points per batch
        self.client: Optional[QdrantClient] = None
        self.logger = logging.getLogger(__name__)


    def connect(self) -> 'QdrantHandler':
        try:
            self.client = QdrantClient(
                url = self.url,
                api_key = self.api_key
            )

            self.logger.info(f"Connected to Qdrant database: {self.collection_name}")
            return self

        except Exception as e:
            self.logger.error(f"Qdrant connection failed: {str(e)}")
            raise


    def _create_points(self,
        filename: str,
        chunks: List[Dict],
        embeddings: np.ndarray,
        start_idx: int
    ) -> List[PointStruct]:
        """
        creates points as a list
        """
        return [
            PointStruct(
                id=start_idx + i,
                vector=embedding.tolist(),
                payload={
                    'text': chunk['text'],
                    'chunk_id': start_idx + i,
                    'filename': filename,
                    'metadata': chunk.get('metadata', {})
                }
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]


    def insert_embeddings_batch(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: torch.Tensor
    ) -> None:
        """
        Insert batch of embeddings into Qdrant.
        """
        try:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            points = [
                PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        'filename': filename,
                        'text': chunk['text'],
                        'chunk_id': i,
                        'metadata': {
                            **chunk.get('metadata', {}),
                            'page_number': chunk.get('metadata', {}).get('page_number', 0)
                        }
                    }
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            # Process in batches for better memory management
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            self.logger.info(f"Inserted {len(points)} embeddings for {filename}")

        except Exception as e:
            self.logger.error(f"Failed to insert embeddings: {str(e)}")
            raise


    def close_connection(self):
        if self.client:
            self.client.close()
            self.logger.info("Qdrant connection closed")
            self.client = None


class QdrantRetriever(VectorRetriever):
    def __init__(self,
        client,
        collection_name: str,
        #model_name: str = 'sentence-transformers/all-mpnet-base-v2'
        model_name: str = 'nithinreddyy/finetuned-esg'
    ):

        self.client = client
        self.collection_name = collection_name
        self.embedder = self._get_embedder(model_name)
        self.logger = logging.getLogger(__name__)

    def _get_embedder(self, model_name):
        from ingestion import ParseChunkEmbed  # Move import here to avoid circular import
        return ParseChunkEmbed(model_name=model_name)

    def search(self,
        query: str,
        filename: str = None,  # Optional filename filter
        top_k: int = 3
        ) -> List[Dict[str, Any]]:
        """
        Search with optional filename filter
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.compute_embeddings([query])[0].numpy()

            # Base search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": top_k,
                "score_threshold": 0.7  # Adjust threshold as needed
            }

            # Add filename filter if specified
            if filename:
                search_params["query_filter"] = {
                    "must": [
                        {"key": "filename", "match": {"value": filename}}
                    ]
                }

            # Perform search
            results = self.client.search(**search_params)

            self.logger.info(f"Found {len(results)} results for query: {query}")
            if filename:
                self.logger.info(f"Filtered by filename: {filename}")

            return [
                {
                    'text': hit.payload['text'],
                    'filename': hit.payload['filename'],
                    'metadata': hit.payload.get('metadata', {}),
                    'page_number': hit.payload.get('metadata', {}).get('page_number', 0),
                    'score': hit.score
                }
                for hit in results
            ]
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []