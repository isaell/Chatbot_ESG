import numpy as np
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from vector_database_handler import VectorDatabaseHandler
from utils import safe_env_get


class MongoDBHandler(VectorDatabaseHandler):
    """
    Initialize MongoDB connection parameters.
    
    :param db_name: Name of the database to connect to
    :param collection_name: Name of the collection to use
    :param uri: Optional MongoDB connection URI (overrides environment variable)

    Note that Atlas vector search is used. Need to initialize first, or run:
        #self.collection.create_index([('embedding', 'vector')])
    :Database name and collections name
    """    
    
    def __init__(self, 
                db_name: str = None, 
                collection_name: str = None, 
                uri: Optional[str] = None):
        
        self.db_name = db_name
        self.collection_name = collection_name
        self.uri = uri or safe_env_get("MONGODB_URI")
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None

    def connect(self):
        try:
            if not self.uri:
                raise ValueError("MongoDB URI not provided")
            
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            print(f"Connected to MongoDB database: {self.db_name}")
            return self
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def insert_embedding(self, document_id: str, text_chunk: str, embedding: np.ndarray):
        self.collection.insert_one({
            'document_id': document_id,
            'text_chunk': text_chunk,
            'embedding': embedding.tolist()
        })

    def close_connection(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")
            self.client = None
            self.db = None
            self.collection = None