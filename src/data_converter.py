import pandas as pd  # panadas
from langchain_core.documents import Document 
#'Data/flipkart_product_review.csv'

"""
Class for storing a piece of text and associated metadata.

!!! note

Document is for retrieval workflows, not chat I/O. For sending text to an LLM in a conversation, use message types from langchain.messages.

Example:

        from langchain_core.documents import Document

        document = Document(
            page_content="Hello, world!", metadata={"source": "https://example.com"}
        )
"""

"""
class dataconv is used to convert the csv into chuncks to be later ingested in vector 
usinging init to automatically run and self to spcify the current object data like file path
and convert func is reading the csv then converting it to df and selecting the related cols needed
the using Documents func from langchain and storing it in docs variable

"""
class DataConv:

    def __init__(self,file_path:str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[['product_title','review']]

        docs = [Document(page_content=row['review'], metadata={'product_name': row['product_title']})
                for _,row in df.iterrows()
        ]

        return docs
