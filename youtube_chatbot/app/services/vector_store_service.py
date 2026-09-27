import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.llm import embedding_model
from app.dto.vectorstore import RetrievedChunk, RetrieveResponse, VectorizeResponse
from app.exceptions.vectorstore import VectorStoreException, VectorStoreNotFoundException

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

        store_path = self._store_path(video_id)
        vector_store.save_local(store_path)

        return VectorizeResponse(
            video_id=video_id,
            chunk_count=len(chunks),
            vector_store_path=store_path,
        )

    def retrieve(self, video_id: str, query: str, top_k: int) -> RetrieveResponse:
        documents = self.search_documents(video_id=video_id, query=query, top_k=top_k)

        results = [
            RetrievedChunk(
                chunk_index=document.metadata.get("chunk_index", -1),
                text=document.page_content,
            )
            for document in documents
        ]

        return RetrieveResponse(
            video_id=video_id,
            query=query,
            results=results,
        )

    def search_documents(self, video_id: str, query: str, top_k: int) -> List[Document]:
        store_path = self._store_path(video_id)

        if not os.path.isdir(store_path):
            raise VectorStoreNotFoundException(
                message=f"No vector store found for video '{video_id}'. "
                "Run /video/vectorize for this video first."
            )

        try:
            vector_store = FAISS.load_local(
                store_path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": max(20, top_k * 4),
                    "lambda_mult": 0.5,
                },
            )
            return retriever.invoke(query)
        except VectorStoreNotFoundException:
            raise
        except Exception as exc:
            raise VectorStoreException(
                message=f"Failed to retrieve chunks for video '{video_id}'"
            ) from exc

    def _store_path(self, video_id: str) -> str:
        return os.path.join(VECTOR_STORE_DIR, video_id)
