from app.core.llm import model
from app.models.ticket import TicketCategory, TicketClassification
from app.models.enums import enum_values
from app.prompts.classifier import classifier_prompt

classifier_chain = (
    classifier_prompt.partial(categories=enum_values(TicketCategory))
    | model.with_structured_output(TicketClassification, method="function_calling")
)