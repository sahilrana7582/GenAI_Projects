from langchain_core.prompts import PromptTemplate


billing_prompt = PromptTemplate(
    input_variables=["ticket", "issue_types", "payment_statuses", "intents"],
    template="""
You are a billing support specialist. The ticket below has already been
classified as a billing issue - your job is to extract structured detail
from it, not to re-classify it.

Extract the following:

1. Issue Type - the specific kind of billing problem.
   Allowed values: {issue_types}

2. Payment Status - the current state of the payment involved.
   Allowed values: {payment_statuses}

3. Customer Intent - what the customer ultimately wants done about it.
   Allowed values: {intents}

Rules:
1. Choose exactly one value per field, using only the allowed values listed above.
2. Base your answer only on information stated or clearly implied in the ticket.
3. Do not invent details that are not present in the ticket.
4. If the ticket raises multiple issues, focus on the primary billing problem.
5. If the payment status is not mentioned and cannot be reasonably inferred, use "unknown".

Customer Ticket:
{ticket}
""",
)
