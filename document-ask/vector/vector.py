from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from core.embeddings import embeddings


VECTOR_STORE_DIR = "vector_store"


class VectorStoreService:

    def vectorize_chunks(self, chunks: List[Document]) -> Chroma:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR
        )

        return vector_store