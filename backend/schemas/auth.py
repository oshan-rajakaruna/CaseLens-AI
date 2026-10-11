"""Development-only login contracts without owner identity input."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DevelopmentLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=1024)


class DevelopmentLoginResponse(BaseModel):
    authenticated: Literal[True] = True
    csrf_token: str
