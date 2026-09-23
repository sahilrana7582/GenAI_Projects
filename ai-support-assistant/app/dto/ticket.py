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

class TicketAnalysisResponse(BaseModel):
    category: TicketCategory
    summary: str = Field(description="The summary of the overall processing.")
    priority: TicketPriority
    sentiment: TicketSentiment
    suggested_action: str = Field(description="The suggested best course of action for the user to resolve the issue he is facing")