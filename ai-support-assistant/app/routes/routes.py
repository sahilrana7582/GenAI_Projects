from fastapi import APIRouter

from app.dto.ticket import TicketRequest
from app.models.ticket import TicketClassification, TicketCategory
from app.models.analysis import (
    BillingAnalysis,
    AccountAnalysis,
    GeneralAnalysis,
    TechnicalAnalysis,
)
from app.services.ticket_service import TicketService
from app.chains.classifier import classifier_chain
from app.chains.support.billing import billing_chain
from app.chains.support.account import account_chain
from app.chains.support.general import general_chain
from app.chains.support.technical import technical_chain
from app.chains.category_router import category_router, classify_and_route


router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)

ticket_service = TicketService()


@router.post("/classify", response_model=TicketClassification)
async def classify_ticket(request: TicketRequest):

    classification = await ticket_service.classify_ticket(
        request.ticket
    )

    res = billing_chain.invoke({
        "ticket": request.ticket
    })

    print(res)

    return classification


@router.post("/classify/test/classifier", response_model=TicketClassification)
async def classify_ticket_test_classifier(request: TicketRequest):

    res = classifier_chain.invoke({
        "ticket": request.ticket
    })


    return res


@router.post("/classify/test/billing", response_model=BillingAnalysis)
async def classify_ticket_test_billing(request: TicketRequest):

    res = billing_chain.invoke({
        "ticket": request.ticket
    })

    print(res, "<<<<<<<<<<Response From LLM")

    return res


@router.post("/classify/test/account", response_model=AccountAnalysis)
async def classify_ticket_test_account(request: TicketRequest):

    res = account_chain.invoke({
        "ticket": request.ticket
    })


    return res


@router.post("/classify/test/general", response_model=GeneralAnalysis)
async def classify_ticket_test_general(request: TicketRequest):

    res = general_chain.invoke({
        "ticket": request.ticket
    })


    return res


@router.post("/classify/test/technical", response_model=TechnicalAnalysis)
async def classify_ticket_test_technical(request: TicketRequest):

    res = technical_chain.invoke({
        "ticket": request.ticket
    })

    return res

@router.post("/classify/route/simple")
async def classify_ticket_test_route(request: TicketRequest):

    classification = classifier_chain.invoke({
        "ticket": request.ticket
    })

    result = category_router.invoke({
        "ticket": request.ticket,
        "category": classification.category,
    })

    return {
        "category": classification.category,
        "analysis": result,
    }

@router.post("/classify/route")
async def classify_ticket_test_route(request: TicketRequest):

    result = classify_and_route.invoke({
        "ticket": request.ticket
    })

    return result