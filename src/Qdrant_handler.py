import uuid
import numpy as np
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from utils import safe_env_get
from vector_database_handler import VectorDatabaseHandler


class QdrantHandler(VectorDatabaseHandler):
    """
    Initializes Qdrant connection.
    
    :param collection_name: Name of the collection to create (if it doesn't exist) and use
    :param url: Optional connection URI (overrides environment variable)
    :param apr_key: Optional api key (overrides environment variable)
    """    
    def __init__(self, 
                 collection_name: str = None, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None):
        
        self.url = url or safe_env_get("QDRANT_HOST")  # , "localhost"
        self.api_key = api_key or safe_env_get("QDRANT_KEY")
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None

    def connect(self):
        try:
            self.client = QdrantClient(
                url = self.url, 
                api_key=self.api_key
            )

            # Create collection if not exists
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config = VectorParams(size=768, distance=Distance.COSINE)  #768
                )
            
            print(f"Connected to Qdrant database: {self.collection_name}")
            return self
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")


    def insert_embedding(self, document_id: str, text_chunk: str, embedding: np.ndarray):
        # Generate a unique UUID for each point (needed by qdrant)
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


    def close_connection(self):
        # if self.client:
        #     print("Qdrant connection closed")
        #     self.client = None
        if self.client:
            self.client.close()
            print("Qdrant connection closed")
            self.client = None
            self.collection = None