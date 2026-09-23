from pydantic import BaseModel, Field

from app.models.enums import (
    AccountIssueType,
    BillingIssueType,
    CustomerIntent,
    GeneralIssueType,
    PaymentStatus,
    TechnicalIssueType,
)


class BaseTicketAnalysis(BaseModel):
    customer_intent: CustomerIntent = Field(
        description="What the customer ultimately wants done about their issue"
    )


class BillingAnalysis(BaseTicketAnalysis):
    issue_type: BillingIssueType = Field(
        description="The specific kind of billing problem described in the ticket"
    )
    payment_status: PaymentStatus = Field(
        description="The current state of the payment involved, if mentioned or implied"
    )


class AccountAnalysis(BaseTicketAnalysis):
    issue_type: AccountIssueType = Field(
        description="The specific kind of account problem described in the ticket"
    )


class TechnicalAnalysis(BaseTicketAnalysis):
    issue_type: TechnicalIssueType = Field(
        description="The specific kind of technical problem described in the ticket"
    )
    affected_feature: str = Field(
        description="The specific product feature or component the customer says is affected"
    )


class GeneralAnalysis(BaseTicketAnalysis):
    issue_type: GeneralIssueType = Field(
        description="The specific kind of general inquiry described in the ticket"
    )
