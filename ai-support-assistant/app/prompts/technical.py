from langchain_core.prompts import PromptTemplate


technical_prompt = PromptTemplate(
    input_variables=["ticket", "issue_types", "intents"],
    template="""
You are a technical support specialist. The ticket below has already been
classified as a technical issue - your job is to extract structured detail
from it, not to re-classify it.

Extract the following:

1. Issue Type - the specific kind of technical problem.
   Allowed values: {issue_types}

2. Affected Feature - the specific product feature or component the customer
   says is affected (for example: "checkout page", "mobile app login",
   "billing API"). If the ticket does not name one, use "unspecified".

3. Customer Intent - what the customer ultimately wants done about it.
   Allowed values: {intents}

Rules:
1. For Issue Type and Customer Intent, choose exactly one value using only the allowed values listed above. If no value is a perfect match, choose the closest one - do not invent a new value.
2. Base your answer only on information stated or clearly implied in the ticket.
3. Do not invent details that are not present in the ticket.
4. If the ticket raises multiple issues, focus on the primary technical problem.

Customer Ticket:
{ticket}
""",
)
