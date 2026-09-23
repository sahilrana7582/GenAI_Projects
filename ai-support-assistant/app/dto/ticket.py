from pydantic import BaseModel, Field
from app.models.ticket import (
    TicketCategory, TicketPriority, TicketSentiment
)

class TicketRequest(BaseModel):
    ticket: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer support ticket"
    )

class TicketResponse(BaseModel):

    category: TicketCategory
    summary: str
    sentiment: TicketSentiment
    priority: TicketPriority
    suggested_action: str    