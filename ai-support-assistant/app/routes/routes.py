from fastapi import APIRouter

from app.dto.ticket import TicketRequest
from app.models.ticket import TicketClassification
from app.services.ticket_service import TicketService


router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)

ticket_service = TicketService()


@router.post("/classify", response_model=TicketClassification)
async def classify_ticket(request: TicketRequest):

    classification = await ticket_service.classify_ticket(
        request.ticket
    )

    return classification