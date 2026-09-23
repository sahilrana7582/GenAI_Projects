from app.core.llm import model
from app.chains.support import (
    account_chain,
    technical_chain,
    general_chain,
    billing_chain
)
from app.models.ticket import TicketCategory
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from app.chains.classifier import classifier_chain



category_router = RunnableBranch(
    (
        lambda x: x["category"] == TicketCategory.BILLING,
        billing_chain,
    ),
    (
        lambda x: x["category"] == TicketCategory.ACCOUNT,
        account_chain,
    ),
    (
        lambda x: x["category"] == TicketCategory.TECHNICAL,
        technical_chain,
    ),
    general_chain,
)

classify_and_route = (
    RunnablePassthrough.assign(
        classification=classifier_chain
    )
    | RunnablePassthrough.assign(
        category=lambda x: x["classification"].category
    )
    | category_router
)