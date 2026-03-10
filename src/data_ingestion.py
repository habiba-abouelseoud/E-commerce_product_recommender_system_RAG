from langchain_astradb import AstraDBVectorStore # vectore db used
#from langchain_huggingface import HuggingFaceEndpointEmbeddings # embedding model
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_converter import DataConv # data converter from csv to docs for vect
from src.config import Config

class DataIngestor:
    def __init__(self):
        # converting text to embedding
        self.embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        # setting the vectore db 
        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name= 'ecommerce_project_database',
            api_endpoint= Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
            )
        # This function loads the dataset and inserts it into the vector database.

    # This function loads the dataset and inserts it into the vector database.
    def ingest(self, loading_existing=True):

        # if DB already exists just return it
        if loading_existing:
            return self.vstore

        docs = DataConv('Data/flipkart_product_review.csv')

        # add docs to vector db
        self.vstore.add_documents(docs)

        return self.vstore



    # def ingest(self,loading_existing=True):
    #         # instead of repeating the same steps again if we added smth we just updating the new one 
    #         if loading_existing==True:
    #             return self.vstore
            
    #         docs = DataConv('Data/flipkart_product_review.csv')
    #     # Add the documents to the vector database.
    #     # Internally this will:
    #     # 1. Convert each document into an embedding vector
    #     # 2. Store the vector + metadata in Astra DB
    #         self.vstore.add_documents(docs)

    #         return self.vstore