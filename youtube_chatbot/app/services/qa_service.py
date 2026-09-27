from app.chains.qa import qa_chain
from app.dto.qa import AskResponse
from app.exceptions.qa import AnswerGenerationException
from app.services.vector_store_service import VectorStoreService

DEFAULT_TOP_K = 4


class QAService:

    def __init__(self, vector_store_service: VectorStoreService):
        self._vector_store_service = vector_store_service

    def ask(self, video_id: str, query: str) -> AskResponse:
        documents = self._vector_store_service.search_documents(
            video_id=video_id, query=query, top_k=DEFAULT_TOP_K
        )
        context = "\n\n".join(document.page_content for document in documents)

        try:
            answer = qa_chain.invoke({"context": context, "question": query})
        except Exception as exc:
            raise AnswerGenerationException(
                message=f"Failed to generate an answer for video '{video_id}'"
            ) from exc

        return AskResponse(
            video_id=video_id,
            query=query,
            answer=answer,
        )
