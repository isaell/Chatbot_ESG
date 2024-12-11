import os
import logging
from typing import Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure basic logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def get_size_format(b, factor=1024, suffix="B") -> int:
    """
    Convert bytes to a human-readable format (e.g., KB, MB, GB).
    Base factor defaults to 1024 for conversion, which is standard for binary prefixes (e.g., 1 KB = 1024 bytes).
    No prefix (bytes), kilobytes (KB), megabytes (MB), gigabytes (GB), terabytes (TB), petabytes (PB), 
    exabytes (EB), and zettabytes (ZB)
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        # size 'b' less than the 'factor'
        if b < factor:
            # format size as a strig
            return f"{b:.2f}{unit}{suffix}"
        # else, 'b' is divided by the factor and moves to the next unit in the list
        b /= factor
    return f"{b:.2f}Y{suffix}"


def safe_env_get(key: str, default: Any = None, required: bool = False) -> str:
    """Safely retrieve environment variables with optional defaults."""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    return value



class MongoDBHandler:

    def __init__(self, 
                 db_name: str = None, 
                 collection_name: str = None, 
                 uri: Optional[str] = None):
        """
        Initialize MongoDB connection parameters.
        
        :param db_name: Name of the database to connect to
        :param collection_name: Name of the collection to use
        :param uri: Optional MongoDB connection URI (overrides environment variable)

        Initialize MongoDB connection.
        Note that Atlas vector search is used. Need to initialize first, or run:
         #self.collection.create_index([('embedding', 'vector')])
        :Database name and collections name
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.uri = uri or safe_env_get("MONGODB_URI")
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None

    def connect(self):
        """
        Establish MongoDB connection.
        :raises ConnectionError: If connection fails
        """
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
        

    def insert_embedding(self, document_id, text_chunk, embedding):
        self.collection.insert_one({
            'document_id': document_id,
            'text_chunk': text_chunk,
            'embedding': embedding.tolist()
        })


    def close_connection(self):
        """Close MongoDB connection if active"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")
            self.client = None
            self.db = None
            self.collection = None


    def __enter__(self):
        """Support context manager protocol"""
        return self.connect()


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context"""
        self.close_connection()