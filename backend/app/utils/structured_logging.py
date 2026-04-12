"""
Structured JSON logging with correlation IDs and context for production observability.

This module provides:
- Request correlation IDs for distributed tracing
- Structured JSON logs for machine parsing
- User context injection for multi-tenant debugging
- Performance metrics (execution time, token counts)
- Configurable log levels per component
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
component_path_var: ContextVar[list] = ContextVar("component_path", default_factory=list)


class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogComponent(Enum):
    """Component categories for log segregation."""
    API = "api"
    AGENT = "agent"
    LLM = "llm"
    SERVICE = "service"
    KNOWLEDGE = "knowledge"
    EMBEDDING = "embedding"
    WORKFLOW = "workflow"
    NOTIFICATION = "notification"
    STORE = "store"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"


class StructuredLogRecord:
    """Represents a structured log entry."""

    def __init__(
        self,
        timestamp: datetime,
        level: str,
        component: str,
        operation: str,
        message: str,
        request_id: str = "",
        user_id: str = "",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.level = level
        self.component = component
        self.operation = operation
        self.message = message
        self.request_id = request_id or request_id_var.get()
        self.user_id = user_id or user_id_var.get()
        self.duration_ms = duration_ms
        self.metadata = metadata or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "component": self.component,
            "operation": self.operation,
            "message": self.message,
            "request_id": self.request_id,
            "user_id": self.user_id,
        }

        if self.duration_ms is not None:
            result["duration_ms"] = round(self.duration_ms, 2)

        if self.metadata:
            result["metadata"] = self.metadata

        if self.error:
            result["error"] = self.error

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract structured data if present
        structured_data = getattr(record, "structured_data", None)

        if structured_data:
            return structured_data.to_json()

        # Fallback to standard format with context
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        structured_data = getattr(record, "structured_data", None)

        if structured_data:
            # Format structured data nicely
            color = self.COLORS.get(structured_data.level, "")
            reset = self.COLORS["RESET"]

            parts = [
                f"[{color}{structured_data.level}{reset}]",
                f"[{structured_data.component}]",
                f"{structured_data.operation}",
            ]

            if structured_data.request_id:
                parts.append(f"req={structured_data.request_id[:8]}")

            if structured_data.user_id:
                parts.append(f"user={structured_data.user_id[:12]}")

            parts.append(f"{structured_data.message}")

            if structured_data.duration_ms:
                parts.append(f"({structured_data.duration_ms}ms)")

            return " ".join(parts)

        # Standard format
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        return f"[{color}{record.levelname}{reset}] {record.name}: {record.getMessage()}"


class StructuredLogger:
    """Logger that produces structured, context-aware logs."""

    def __init__(self, name: str, component: LogComponent):
        self.name = name
        self.component = component
        self._logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        level_name: str,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[float] = None,
    ):
        """Create and emit a structured log entry."""
        # Build error info if present
        error_info = None
        if error:
            error_info = {
                "type": type(error).__name__,
                "message": str(error),
            }

        # Create structured record
        structured_data = StructuredLogRecord(
            timestamp=datetime.now(timezone.utc),
            level=level_name,
            component=self.component.value,
            operation=operation,
            message=message,
            request_id=request_id_var.get(),
            user_id=user_id_var.get(),
            duration_ms=duration_ms,
            metadata=metadata,
            error=error_info,
        )

        # Create standard log record with structured data attached
        record = self._logger.makeRecord(
            self.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None,
        )
        record.structured_data = structured_data

        self._logger.handle(record)

    def debug(
        self,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log debug message."""
        self._log(logging.DEBUG, "DEBUG", operation, message, metadata)

    def info(
        self,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log info message."""
        self._log(logging.INFO, "INFO", operation, message, metadata)

    def warning(
        self,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ):
        """Log warning message."""
        self._log(logging.WARNING, "WARNING", operation, message, metadata, error)

    def error(
        self,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ):
        """Log error message."""
        self._log(logging.ERROR, "ERROR", operation, message, metadata, error)

    def critical(
        self,
        operation: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ):
        """Log critical message."""
        self._log(logging.CRITICAL, "CRITICAL", operation, message, metadata, error)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log performance metric."""
        combined_metadata = {
            "metric_type": "performance",
            **(metadata or {}),
        }
        self._log(
            logging.INFO,
            "INFO",
            operation,
            message or f"Operation completed",
            combined_metadata,
            duration_ms=duration_ms,
        )


class RequestContext:
    """Context manager for request tracking."""

    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())[:16]
        self.user_id = user_id or ""
        self._request_token = None
        self._user_token = None

    def __enter__(self):
        """Enter context and set variables."""
        self._request_token = request_id_var.set(self.request_id)
        if self.user_id:
            self._user_token = user_id_var.set(self.user_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and reset variables."""
        if self._request_token:
            request_id_var.reset(self._request_token)
        if self._user_token:
            user_id_var.reset(self._user_token)


def timed(operation: str, component: Optional[LogComponent] = None):
    """Decorator to log function execution time."""
    def decorator(func: Callable) -> Callable:
        logger_name = func.__module__
        comp = component or _infer_component(func.__module__)
        logger = StructuredLogger(logger_name, comp)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.log_performance(operation, duration, f"{operation} completed")
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(operation, f"{operation} failed", {"error_type": type(e).__name__}, error=e, duration_ms=duration)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.log_performance(operation, duration, f"{operation} completed")
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(operation, f"{operation} failed", {"error_type": type(e).__name__}, error=e, duration_ms=duration)
                raise

        return async_wrapper if hasattr(func, "__code__") and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator


def _infer_component(module_name: str) -> LogComponent:
    """Infer log component from module name."""
    if ".api." in module_name or module_name.endswith("api"):
        return LogComponent.API
    elif ".agent." in module_name or ".agents." in module_name:
        return LogComponent.AGENT
    elif ".llm." in module_name:
        return LogComponent.LLM
    elif ".knowledge" in module_name:
        return LogComponent.KNOWLEDGE
    elif ".embedding" in module_name:
        return LogComponent.EMBEDDING
    elif ".workflow" in module_name:
        return LogComponent.WORKFLOW
    elif ".notification" in module_name:
        return LogComponent.NOTIFICATION
    elif ".store" in module_name or "_store" in module_name:
        return LogComponent.STORE
    else:
        return LogComponent.SERVICE


# Logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, component: Optional[LogComponent] = None) -> StructuredLogger:
    """Get or create a structured logger."""
    cache_key = f"{name}:{component.value if component else 'auto'}"

    if cache_key in _loggers:
        return _loggers[cache_key]

    if component is None:
        component = _infer_component(name)

    logger = StructuredLogger(name, component)
    _loggers[cache_key] = logger
    return logger


# Convenience functions for common components
def get_api_logger(name: str) -> StructuredLogger:
    """Get API component logger."""
    return get_logger(name, LogComponent.API)


def get_agent_logger(name: str) -> StructuredLogger:
    """Get Agent component logger."""
    return get_logger(name, LogComponent.AGENT)


def get_llm_logger(name: str) -> StructuredLogger:
    """Get LLM component logger."""
    return get_logger(name, LogComponent.LLM)


def get_service_logger(name: str) -> StructuredLogger:
    """Get Service component logger."""
    return get_logger(name, LogComponent.SERVICE)


def get_knowledge_logger(name: str) -> StructuredLogger:
    """Get Knowledge component logger."""
    return get_logger(name, LogComponent.KNOWLEDGE)


def get_embedding_logger(name: str) -> StructuredLogger:
    """Get Embedding component logger."""
    return get_logger(name, LogComponent.EMBEDDING)


def get_notification_logger(name: str) -> StructuredLogger:
    """Get Notification component logger."""
    return get_logger(name, LogComponent.NOTIFICATION)


def setup_structured_logging(
    level: Union[int, str] = logging.INFO,
    format_style: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """Setup structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: "json" for structured JSON, "pretty" for human-readable
        log_file: Optional file path for log output
        max_bytes: Max size for log rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose formatter
    if format_style == "json":
        formatter = JSONFormatter()
    else:
        formatter = PrettyFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Log startup
    startup_logger = get_logger("app.system", LogComponent.SYSTEM)
    startup_logger.info(
        "logging_initialized",
        f"Structured logging initialized",
        {"level": logging.getLevelName(level), "format": format_style, "file": log_file},
    )


def get_current_context() -> Dict[str, Any]:
    """Get current logging context for debugging."""
    return {
        "request_id": request_id_var.get(),
        "user_id": user_id_var.get(),
    }
