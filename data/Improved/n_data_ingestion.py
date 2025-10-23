from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter  # Using the improved one
from flipkart.config import Config
import logging

# Set up logging for feedback
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class DataIngestor:
    """
    Manages the connection to and data ingestion into an AstraDB vector store.
    """
    def __init__(self):
        """
        Initializes the embedding model and the AstraDBVectorStore connection.
        """
        log.info("Initializing embedding model...")
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        log.info("Connecting to AstraDB...")
        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )
        log.info("AstraDB connection established.")

    def get_vector_store(self) -> AstraDBVectorStore:
        """
        Returns the initialized vector store instance.
        """
        return self.vstore

    def ingest(self, file_path: str, batch_size: int = 20):
        """
        Performs a fresh data ingestion from a CSV file into the vector store.

        It uses the DataConverter to load documents from the file and then
        adds them to AstraDB in batches.

        Args:
            file_path (str): The path to the CSV file to ingest.
            batch_size (int, optional): The number of documents to add
                                         in a single batch. Defaults to 20.
        
        Returns:
            AstraDBVectorStore: The populated vector store.
        """
        log.info(f"Starting data ingestion from {file_path}...")
        
        # 1. Convert data
        # We can now specify which columns to use if they are different
        converter = DataConverter(file_path, content_col="review", metadata_col="product_title")
        docs = converter.convert()
        
        if not docs:
            log.warning("No documents found. Ingestion skipped.")
            return self.vstore

        log.info(f"Converted {len(docs)} documents. Adding to AstraDB in batches of {batch_size}...")

        # 2. Add documents in batches
        # The AstraDB client handles batching automatically if you pass the parameter.
        self.vstore.add_documents(docs, batch_size=batch_size)

        log.info("Data ingestion complete.")
        return self.vstore

# How you would use this:
#
# ingestor = DataIngestor()
#
# # To get the store (if data is already loaded):
# vstore = ingestor.get_vector_store()
#
# # To run a fresh ingestion:
# vstore = ingestor.ingest(file_path="data/flipkart_product_review.csv")