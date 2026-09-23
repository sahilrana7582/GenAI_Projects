from pydantic import BaseModel, Field

from app.core.llm import model
from app.models.ticket import TicketCategory
from app.prompts.classifier import classifier_prompt

class TicketClassification(BaseModel):
    category: TicketCategory = Field(
        description="The category of the support ticket"
    )


classifier_chain = (
    classifier_prompt
    | model.with_structured_output(TicketClassification)
)