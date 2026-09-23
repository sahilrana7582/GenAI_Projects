from app.core.llm import model
from app.models.analysis import GeneralAnalysis
from app.models.enums import CustomerIntent, GeneralIssueType, enum_values
from app.prompts.general import general_prompt

general_chain = general_prompt.partial(
    issue_types=enum_values(GeneralIssueType),
    intents=enum_values(CustomerIntent),
) | model.with_structured_output(GeneralAnalysis, method="function_calling")
