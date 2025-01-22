from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from pymongo import InsertOne
from typing import List, Dict, Any

class MongoDBHandler:
    """
    A class to handle MongoDB operations such as connecting, creating collections, 
    and inserting data into collections.
    """
    def __init__(self, db_user: str, db_pass: str, host: str, options: str):
        """
        Initialize the MongoDBHandler with a URI and database name.
        
        :param uri: MongoDB connection string
        :param db_name: Name of the database to connect to
        """
        self.uri = f"mongodb+srv://{db_user}:{db_pass}@{host}/?{options}"


    def connect_to_database(self, db_name: str):
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        self.db = self.client[db_name]

    def create_collection(self, collection_name: str):
        """
        Create a collection in the database if it doesn't already exist.
        
        :param collection_name: Name of the collection to create
        """
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
        
        return self.db[collection_name]

    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Insert documents into a collection.
        
        :param collection_name: Name of the collection to insert into
        :param documents: List of documents to insert
        """
        collection = self.db[collection_name]
        collection.insert_many(documents)

    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()





