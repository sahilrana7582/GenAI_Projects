from app.core.llm import model
from app.models.analysis import AccountAnalysis
from app.models.enums import AccountIssueType, CustomerIntent, enum_values
from app.prompts.account import account_prompt

account_chain = account_prompt.partial(
    issue_types=enum_values(AccountIssueType),
    intents=enum_values(CustomerIntent),
) | model.with_structured_output(AccountAnalysis, method="function_calling")
