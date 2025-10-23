import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    """
    Converts product review data from a CSV file into a list of LangChain Documents.

    This class reads a CSV file, isolates the "product_title" and "review" columns,
    and transforms each row into a Document object. The review text is used as the
    main `page_content`, and the product title is stored in the `metadata` under
    the key "product_name". This format is ideal for loading into a vector store.

    Attributes:
        file_path (str): The path to the source CSV file.
    """
    def __init__(self,file_path:str):
        """
        Initializes the DataConverter.

        Args:
            file_path (str): The file path to the CSV file that needs to be processed.
        """
        self.file_path = file_path

    def convert(self):
        """
        Performs the conversion from CSV to a list of LangChain Documents.

        Reads the CSV file specified during initialization, processes the
        "product_title" and "review" columns, and creates a Document for each row.

        Returns:
            list[Document]: A list of Document objects ready for embedding
                            or storage.
        """
        df = pd.read_csv(self.file_path)[["product_title","review"]]  

        docs = [
            Document(page_content=row['review'] , metadata = {"product_name" : row["product_title"]})
            for _, row in df.iterrows()
        ]

        return docs