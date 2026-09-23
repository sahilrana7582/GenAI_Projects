from langchain_core.prompts import PromptTemplate


general_prompt = PromptTemplate(
    input_variables=["ticket", "issue_types", "intents"],
    template="""
You are a customer support specialist handling a ticket that does not fit
neatly into billing, account, or technical issues.

Extract the following:

1. Issue Type - the general nature of the ticket.
   Allowed values: {issue_types}

2. Customer Intent - what the customer ultimately wants done about it.
   Allowed values: {intents}

Rules:
1. Choose exactly one value per field, using only the allowed values listed above.
2. Base your answer only on information stated or clearly implied in the ticket.
3. Do not invent details that are not present in the ticket.
4. If the ticket raises multiple issues, focus on the primary one.
5. If nothing listed fits well, use "other" for Issue Type.

Customer Ticket:
{ticket}
""",
)
