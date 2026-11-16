from langchain_community.retrievers import BM25Retriever
from schema import GuestDataset
from langchain_core.tools import tool


with open("../data/gotham_guests.json", "r", encoding="utf-8") as f:
    guest_data = GuestDataset.model_validate_json(f.read())

documents = guest_data.guests_docs

bm25_retriever= BM25Retriever.from_documents(documents= documents)

@tool
def extract_text(query : str) -> str:
    """
    Search the guest database for information about gala invitees.

    Use this tool whenever the user asks about:
    - A guest's name, background, or relationship.
    - A guest's email or other stored information.
    - Finding guests matching a particular description.

    Do not invent guest information. Retrieve it from this tool.
    """
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])

    return "No mathching guests information found"


# if __name__ == '__main__' : 
#     query = "how is marie's research going?"
#     print(extract_text(query= query))

