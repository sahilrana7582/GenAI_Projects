from app.core.llm import model
from app.models.ticket import TicketClassification
from app.prompts.classifier import classifier_prompt

classifier_chain = (
    classifier_prompt
    | model.with_structured_output(TicketClassification)
)