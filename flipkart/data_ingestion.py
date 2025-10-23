from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config

class DataIngestor:
    """
    Manages the connection to and data ingestion into an AstraDB vector store.

    This class initializes the connection to AstraDB and the embedding model
    using configuration from the `Config` object. It provides a method to
    populate the database from a CSV file or to simply return the existing
    database connection.

    Attributes:
        embedding (HuggingFaceEndpointEmbeddings): The model used to create
            text embeddings (vectors).
        vstore (AstraDBVectorStore): The client object representing the
            connection to the AstraDB vector store.
    """
    def __init__(self):
        """
        Initializes the DataIngestor.

        Sets up the embedding model and the AstraDBVectorStore connection
        using credentials and model names specified in the `Config` module.
        """
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, load_existing=True):
        """
        Ingests data into the vector store or returns the existing connection.

        This method controls the data loading process. By default, it assumes
        the data is already in the database and just returns the connection.
        If `load_existing` is set to False, it will read the CSV, convert
        it to Documents, and add them to the vector store.

        Args:
            load_existing (bool, optional): If True (default), returns the
                initialized vstore without adding documents. If False,
                performs a full data ingestion from the CSV file.

        Returns:
            AstraDBVectorStore: The configured and (if `load_existing=False`)
                                populated vector store instance, ready to be
                                used as a retriever.
        """
        if load_existing==True:
            # Assumes data is already in the database
            return self.vstore
        
        # Performs a fresh ingestion
        docs = DataConverter("data/flipkart_product_review.csv").convert()

        self.vstore.add_documents(docs)

        return self.vstore