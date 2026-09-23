from app.core.llm import model
from app.models.analysis import BillingAnalysis
from app.models.enums import BillingIssueType, CustomerIntent, PaymentStatus, enum_values
from app.prompts.billing import billing_prompt

billing_chain = billing_prompt.partial(
    issue_types=enum_values(BillingIssueType),
    payment_statuses=enum_values(PaymentStatus),
    intents=enum_values(CustomerIntent),
) | model.with_structured_output(BillingAnalysis)
