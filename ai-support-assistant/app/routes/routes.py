from fastapi import APIRouter

from app.chains.classifier import TicketClassification
from app.dto.ticket import TicketRequest
from app.services.ticket_service import TicketService


router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)

ticket_service = TicketService()


@router.post("/analyze", response_model=TicketClassification)
async def analyze_ticket(request: TicketRequest):

    classification = await ticket_service.classify_ticket(
        request.ticket
    )

    return classification