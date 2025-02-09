from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from typing import List, Dict, List, Optional, Any

import boto3

def get_secret_from_ssm(secret_name: str) -> Optional[str]:
    """
    Retrieves a secret value from AWS Systems Manager (SSM) Parameter Store.

    Params:
    ---------
    secret_name : str
        The name of the secret parameter in AWS SSM.

    Returns:
    ------
    Optional[str]
        The decrypted secret value if found, otherwise None.
    """
    ssm = boto3.client('ssm', region_name='eu-west-1') 
    try:
        response = ssm.get_parameter(
            Name=secret_name,
            WithDecryption=True  # Decrypt the AWS Parameter SecureString
        )
        return response['Parameter']['Value']
    
    except Exception as e:
        print(f"Error retrieving secret: {str(e)}")
        return None

def load_aws_secrets(secret_names_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Loads multiple AWS secrets from SSM Parameter Store.

    Params:
    ---------
    secret_names_list : List[str]
        A list of secret names to retrieve from AWS SSM.

    Returns:
    ------
    Dict[str, Optional[str]]
        A dictionary mapping secret names to their retrieved values (or None if retrieval fails).
    """
    return {secret_name: get_secret_from_ssm(secret_name) for secret_name in secret_names_list}



class MongoDBHandler:
    """
    A class to handle MongoDB operations such as connecting, creating collections, 
    and inserting data into collections.
    """

    def __init__(self, db_user: str, db_pass: str, host: str, options: str = "retryWrites=true&w=majority"):
        """
        Initialize the MongoDBHandler with a URI and database name.

        Parameters:
        -----------
        db_user : str
            Username for MongoDB authentication.
        db_pass : str
            Password for MongoDB authentication.
        host : str
            MongoDB host URL.
        options : str, optional
            Additional connection options (default is "retryWrites=true&w=majority").
        """
        self.uri = f"mongodb+srv://{db_user}:{db_pass}@{host}/?{options}"

    def connect_to_database(self, db_name: str) -> None:
        """
        Connects to the specified MongoDB database.

        Parameters:
        -----------
        db_name : str
            The name of the database to connect to.

        Returns:
        --------
        None
        """
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            self.db = self.client[db_name]
        except Exception as e:
            print(e)

    def check_create_collection(self, collection_name: str) -> Any:
        """
        Checks if a collection exists; if not, creates it.

        Parameters:
        -----------
        collection_name : str
            The name of the collection to create.

        Returns:
        --------
        Any
            The collection object if successful, otherwise None.
        """
        try:
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
                print(f"Collection {collection_name} created.")
            else:
                print("Collection already exists.")
            return self.db[collection_name]
        except Exception as e:
            print("You did not connect to a database yet.", {e})
            return None

    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> Any:
        """
        Inserts multiple documents into a MongoDB collection.

        Parameters:
        -----------
        collection_name : str
            The name of the collection to insert into.
        documents : List[Dict[str, Any]]
            A list of documents to insert.

        Returns:
        --------
        Any
            The result of the insert_many operation.
        """
        collection = self.db[collection_name]
        return collection.insert_many(documents)

    def load_to_mongodb(self, database: str, collection_name: str, documents: List[Dict]) -> None:
        """
        Loads a documents dictionary into a MongoDB collection.

        Parameters:
        -----------
        database : str
            The name of the MongoDB database.
        collection_name : str
            The name of the collection where the documents will be inserted.
        documents : List[Dict]
            The list of dictionaries containing the data to be inserted.
        """

        self.connect_to_database(database)
        self.check_create_collection(collection_name)
        self.insert_documents(collection_name=collection_name, documents=documents)
