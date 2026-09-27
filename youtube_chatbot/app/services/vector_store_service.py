import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.llm import embedding_model
from app.dto.vectorstore import VectorizeResponse
from app.exceptions.vectorstore import VectorStoreException

VECTOR_STORE_DIR = "vector_store"


class VectorStoreService:

    def vectorize_chunks(self, video_id: str, chunks: List[str]) -> VectorizeResponse:
        documents = [
            Document(page_content=chunk, metadata={"video_id": video_id, "chunk_index": index})
            for index, chunk in enumerate(chunks)
        ]

        try:
            vector_store = FAISS.from_documents(documents, embedding_model)
        except Exception as exc:
            raise VectorStoreException(
                message=f"Failed to embed and store chunks for video '{video_id}'"
            ) from exc

        store_path = os.path.join(VECTOR_STORE_DIR, video_id)
        vector_store.save_local(store_path)

        return VectorizeResponse(
            video_id=video_id,
            chunk_count=len(chunks),
            vector_store_path=store_path,
        )
