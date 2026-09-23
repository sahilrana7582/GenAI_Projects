from langchain_core.prompts import PromptTemplate


account_prompt = PromptTemplate(
    input_variables=["ticket", "issue_types", "intents"],
    template="""
You are an account support specialist. The ticket below has already been
classified as an account issue - your job is to extract structured detail
from it, not to re-classify it.

Extract the following:

1. Issue Type - the specific kind of account problem.
   Allowed values: {issue_types}

2. Customer Intent - what the customer ultimately wants done about it.
   Allowed values: {intents}

Rules:
1. Choose exactly one value per field, using only the allowed values listed above. If no value is a perfect match, choose the closest one - do not invent a new value.
2. Base your answer only on information stated or clearly implied in the ticket.
3. Do not invent details that are not present in the ticket.
4. If the ticket raises multiple issues, focus on the primary account problem.

Customer Ticket:
{ticket}
""",
)
