"""Exception hierarchy used across toolcaller."""

from __future__ import annotations

from typing import Any, Optional


class ToolcallerError(Exception):
    """Base class for all toolcaller errors."""

    #: HTTP-ish status that an OpenAI-compatible front-end should report.
    status: int = 500
    #: OpenAI ``error.type`` value.
    error_type: str = "server_error"

    def __init__(self, message: str, *, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_error_object(self) -> dict:
        obj = {
            "message": self.message,
            "type": self.error_type,
        }
        if self.detail is not None:
            obj["param"] = None
            obj["code"] = None
            obj["detail"] = self.detail
        return obj


class TemplateError(ToolcallerError):
    """The jinja template could not be loaded, parsed or rendered."""

    status = 400
    error_type = "invalid_request_error"


class ConfigError(ToolcallerError):
    """Invalid configuration / command line usage."""

    status = 400
    error_type = "invalid_request_error"


class RequestError(ToolcallerError):
    """The supplied OpenAI chat-completion request is malformed."""

    status = 400
    error_type = "invalid_request_error"


class ServerError(ToolcallerError):
    """The upstream completion server failed or was unreachable."""

    status = 502


class ParseError(ToolcallerError):
    """The model output could not be parsed, even in lenient mode."""

    status = 500


class UnsupportedFeature(ToolcallerError):
    """The request asks for something this backend cannot do (images, ...)."""

    status = 400
    error_type = "invalid_request_error"
