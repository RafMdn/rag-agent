from pydantic import BaseModel
from langchain_core.documents import Document

# class GuestDocument(BaseModel):
#     page_content : str
#     metadata : dict

class GuestDataset(BaseModel):
    guests_docs : list[Document]
