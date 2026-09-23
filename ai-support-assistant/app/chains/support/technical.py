from app.core.llm import model
from app.models.analysis import TechnicalAnalysis
from app.models.enums import CustomerIntent, TechnicalIssueType, enum_values
from app.prompts.technical import technical_prompt

technical_chain = technical_prompt.partial(
    issue_types=enum_values(TechnicalIssueType),
    intents=enum_values(CustomerIntent),
) | model.with_structured_output(TechnicalAnalysis)
