from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class RetrieverService:

    def similarity_search(
            self,
            vector_store: Chroma,
            query: str,
            k: int = 3
    ) -> List[Document]:

        return vector_store.similarity_search(
            query=query,
            k=k
        )

    def get_retriever(
            self,
            vector_store: Chroma,
            k: int = 3
    ):
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k
            }
        )
