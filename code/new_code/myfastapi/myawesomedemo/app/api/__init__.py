"""The main APIRouter is defined to include all the sub routers from each
module inside the API folder"""
from fastapi import APIRouter
from .base import base_router
# TODO: import your modules here.
from .model import base_router as model_router

api_router = APIRouter()
# TODO: include the routers from other modules
api_router.include_router(base_router, tags=["base"])
api_router.include_router(model_router, tags=["model"])
