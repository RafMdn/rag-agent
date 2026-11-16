import datasets
from langchain_core.documents import Document
from src.schema import  GuestDataset
import json
from dotenv import load_dotenv

load_dotenv()

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees" , split="train")

documents = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}",
        ]),
        metadata = {"name" : guest["name"]}
    )
    for guest in guest_dataset
]

guest_data = GuestDataset(guests_docs= documents)

with open("data/guests.json", "w", encoding="utf-8") as f:
    f.write(guest_data.model_dump_json(indent=4))
