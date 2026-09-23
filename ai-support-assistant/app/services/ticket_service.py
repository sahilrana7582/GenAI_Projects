from app.chains.classifier import classifier_chain
from app.exceptions.ticket import AppException
from app.models.ticket import TicketCategory


class TicketService:

    categories = ", ".join(category.value for category in TicketCategory)

    async def classify_ticket(self, ticket: str):
        try:
            return await classifier_chain.ainvoke({
                "ticket": ticket,
                "categories": self.categories,
            })
        except Exception as exc:
            raise AppException(
                message="Failed to classify support ticket",
                status=500,
            ) from exc