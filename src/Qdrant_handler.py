import uuid
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from utils import safe_env_get
from vector_database_handler import VectorDatabaseHandler, VectorRetriever


# from importlib.metadata import version
# print(f"Qdrant client version: {version('qdrant_client')}")


class QdrantHandler(VectorDatabaseHandler):
    """
    Initializes Qdrant connection.

    :param collection_name: Name of the collection to create (if it doesn't exist) and use
    :param url: Optional connection URI (overrides environment variable)
    :param apr_key: Optional api key (overrides environment variable)
    :batch_size
    Note that batch size etc is optimized for local use on my machine
    """
    def __init__(self,
                collection_name: str = None,
                url: Optional[str] = None,
                api_key: Optional[str] = None,
                batch_size: int = 100):

        self.url = url or safe_env_get("QDRANT_HOST")  # , "localhost"
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

            print(f"Connected to Qdrant database: {self.collection_name}")
            return self

        except Exception as e:
            #raise ConnectionError(f"Failed to connect to Qdrant: {e}")
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


    def insert_embedding(self,
        document_id: str,
        text_chunk: str,
        embedding: np.ndarray):
        """
        Inserts embeddings. Generate a unique UUID for each point (needed by qdrant)
        """
        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name = self.collection_name,
            points=[{
                'id': point_id,  # Use a UUID instead of the document name
                'payload': {
                    'document_id': document_id,  # Store original document ID in payload
                    'text_chunk': text_chunk
                },
                'vector': embedding.tolist()
            }]
        )


    def insert_embeddings_batch(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        embeddings: torch.Tensor
    ) -> None:
        """
        Insert batch of embeddings into Qdrant.

        Args:
            filename (str): Source filename
            chunks (list): List of chunk dictionaries containing text and metadata
            embeddings (torch.Tensor): Tensor of embeddings
        """
        try:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                # Create collection if it doesn't exist
                try:
                    self.client.get_collection(self.collection_name)
                except:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        # 768 = embedding dimension returned by the model (ESG-BERT)
                        vectors_config=VectorParams(
                            size=embeddings.shape[1], # Vector size from embeddings
                            distance=Distance.COSINE
                        )
                    )

                # Process in batches
                total_points = len(chunks)
                for batch_start in tqdm(range(0, total_points, self.batch_size), desc='Inserting Embeddings'):
                    batch_end = min(batch_start + self.batch_size, total_points)

                    batch_chunks = chunks[batch_start:batch_end]
                    batch_embeddings = embeddings[batch_start:batch_end]

                    points = self._create_points(filename, batch_chunks, batch_embeddings, batch_start)

                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )

                    #self.logger.info(f"Inserted batch {batch_start//self.batch_size + 1} ({len(points)} points)")

        except Exception as e:
            self.logger.error(f"Failed to insert embeddings: {str(e)}")
            raise


    def close_connection(self):
        if self.client:
            self.client.close()
            print("Qdrant connection closed")
            self.client = None


class QdrantRetriever(VectorRetriever):
    def __init__(self,
        client,
        collection_name: str,
        model_name: str = 'nbroad/ESG-BERT'):

        self.client = client
        self.collection_name = collection_name
        self.embedder = self._get_embedder(model_name)
        self.logger = logging.getLogger(__name__)

    def _get_embedder(self, model_name):
        from ingestion import ParseChunkEmbed  # Move import here to avoid circular import
        return ParseChunkEmbed(model_name=model_name)

    def search(self,
        query: str,
        top_k: int = 3
        ) -> List[Dict[str, Any]]:

        # Generate query embedding
        query_embedding = self.embedder.compute_embeddings([query])[0].numpy()

        # Qdrant search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            {
                'text': hit.payload['text'],
                'filename': hit.payload['filename'],
                'metadata': hit.payload.get('metadata', {}),
                'score': hit.score
            }
            for hit in results
        ]