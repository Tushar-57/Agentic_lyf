"""Enhanced logging configuration with category-aware colored output."""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from colorama import Fore, Back, Style, init

try:
    import colorlog
except ImportError:  # pragma: no cover - handled gracefully in runtime envs
    colorlog = None

# Preserve ANSI sequences in non-TTY deployments (e.g., Render logs) for color visibility.
init(autoreset=True, strip=False, convert=False)


class LogCategory(Enum):
    """Categories for different types of logs."""
    AGENT = "AGENT"
    LLM = "LLM"
    API = "API"
    SERVICE = "SERVICE"
    WORKFLOW = "WORKFLOW"
    COMMUNICATION = "COMM"
    KNOWLEDGE = "KNOWLEDGE"
    FACTORY = "FACTORY"
    REGISTRY = "REGISTRY"
    EMBEDDING = "EMBEDDING"
    CONVERSATION = "CONVERSATION"
    SYSTEM = "SYSTEM"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and category support."""
    
    # Color mappings for log levels
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    # Color mappings for categories
    CATEGORY_COLORS = {
        LogCategory.AGENT: Fore.BLUE,
        LogCategory.LLM: Fore.MAGENTA,
        LogCategory.API: Fore.GREEN,
        LogCategory.SERVICE: Fore.CYAN,
        LogCategory.WORKFLOW: Fore.YELLOW,
        LogCategory.COMMUNICATION: Fore.WHITE,
        LogCategory.KNOWLEDGE: Fore.LIGHTBLUE_EX,
        LogCategory.FACTORY: Fore.LIGHTGREEN_EX,
        LogCategory.REGISTRY: Fore.LIGHTMAGENTA_EX,
        LogCategory.EMBEDDING: Fore.LIGHTCYAN_EX,
        LogCategory.CONVERSATION: Fore.LIGHTYELLOW_EX,
        LogCategory.SYSTEM: Fore.LIGHTWHITE_EX,
    }
    
    def __init__(self, include_timestamp: bool = True, include_category: bool = True):
        """
        Initialize the colored formatter.
        
        Args:
            include_timestamp: Whether to include timestamp in logs
            include_category: Whether to include category in logs
        """
        self.include_timestamp = include_timestamp
        self.include_category = include_category
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and category."""
        # Get level color
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        
        # Build the message parts
        parts = []
        
        # Timestamp
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            parts.append(f"{Fore.WHITE}[{timestamp}]{Style.RESET_ALL}")
        
        # Log level
        level_name = f"{level_color}[{record.levelname:^8}]{Style.RESET_ALL}"
        parts.append(level_name)
        
        # Category (if available)
        if self.include_category and hasattr(record, 'category'):
            category = record.category
            if isinstance(category, LogCategory):
                category_color = self.CATEGORY_COLORS.get(category, "")
                category_text = f"{category_color}[{category.value:^10}]{Style.RESET_ALL}"
            else:
                category_text = f"{Fore.WHITE}[{str(category):^10}]{Style.RESET_ALL}"
            parts.append(category_text)
        
        # Logger name (shortened)
        logger_name = record.name.split('.')[-1]  # Just the last part
        parts.append(f"{Fore.WHITE}{logger_name}:{Style.RESET_ALL}")
        
        # Message
        message = record.getMessage()
        parts.append(message)
        
        # Additional context (if available)
        context_parts = []
        if hasattr(record, 'agent_id'):
            context_parts.append(f"agent_id={record.agent_id}")
        if hasattr(record, 'execution_time'):
            context_parts.append(f"exec_time={record.execution_time}ms")
        if hasattr(record, 'request_id'):
            context_parts.append(f"req_id={record.request_id}")
        
        if context_parts:
            context_str = f"{Fore.LIGHTBLACK_EX}({', '.join(context_parts)}){Style.RESET_ALL}"
            parts.append(context_str)
        
        return " ".join(parts)


class ColorlogCategoryFormatter(logging.Formatter):
    """Category-aware formatter backed by colorlog when available."""

    CATEGORY_ANSI = {
        LogCategory.AGENT: "\033[34m",
        LogCategory.LLM: "\033[35m",
        LogCategory.API: "\033[32m",
        LogCategory.SERVICE: "\033[36m",
        LogCategory.WORKFLOW: "\033[33m",
        LogCategory.COMMUNICATION: "\033[37m",
        LogCategory.KNOWLEDGE: "\033[94m",
        LogCategory.FACTORY: "\033[92m",
        LogCategory.REGISTRY: "\033[95m",
        LogCategory.EMBEDDING: "\033[96m",
        LogCategory.CONVERSATION: "\033[93m",
        LogCategory.SYSTEM: "\033[97m",
    }

    def __init__(self, include_timestamp: bool = True, include_category: bool = True):
        self.include_timestamp = include_timestamp
        self.include_category = include_category

        fmt_parts = []
        if include_timestamp:
            fmt_parts.append("%(white)s[%(asctime)s]%(reset)s")
        fmt_parts.append("%(log_color)s[%(levelname)-8s]%(reset)s")
        if include_category:
            fmt_parts.append("%(category_color)s[%(category_display)-12s]%(reset)s")
        fmt_parts.append("%(white)s%(name)s:%(reset)s")
        fmt_parts.append("%(message)s")

        if colorlog is None:
            super().__init__(fmt=" ".join(fmt_parts), datefmt="%H:%M:%S")
            self._fallback = True
        else:
            self._formatter = colorlog.ColoredFormatter(
                fmt=" ".join(fmt_parts),
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red,bg_white",
                },
                reset=True,
            )
            self._fallback = False

    def format(self, record: logging.LogRecord) -> str:
        category_value = getattr(record, "category", LogCategory.SYSTEM)
        if isinstance(category_value, LogCategory):
            category = category_value
        else:
            normalized = str(category_value or "SYSTEM").strip().upper()
            category = LogCategory.__members__.get(normalized, LogCategory.SYSTEM)

        record.category_display = category.value
        record.category_color = self.CATEGORY_ANSI.get(category, "\033[97m")

        if self._fallback:
            return super().format(record)
        return self._formatter.format(record)


class CategoryLogger:
    """Logger wrapper that adds category information."""
    
    def __init__(self, logger: logging.Logger, category: LogCategory):
        """
        Initialize the category logger.
        
        Args:
            logger: The underlying logger instance
            category: The category for this logger
        """
        self.logger = logger
        self.category = category
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, message, *args, **kwargs)
    
    def _log(self, level: int, message: str, *args, **kwargs):
        """Internal logging method that adds category and context."""
        extra = {"category": self.category}
        
        # Add context from kwargs
        for key in ["agent_id", "execution_time", "request_id"]:
            if key in kwargs:
                extra[key] = kwargs.pop(key)

        self.logger.log(level, message, *args, extra=extra, **kwargs)


# Global logger registry
_loggers: Dict[str, CategoryLogger] = {}


def setup_logging(
    level: int = logging.INFO,
    include_timestamp: bool = True,
    include_category: bool = True,
    format_style: str = "colored"
) -> None:
    """
    Setup the global logging configuration.
    
    Args:
        level: The logging level
        include_timestamp: Whether to include timestamps
        include_category: Whether to include categories
        format_style: The format style ("colored" or "plain")
    """
    env_level = os.getenv("LOG_LEVEL", "").strip().upper()
    if env_level:
        level = getattr(logging, env_level, level)

    color_enabled_raw = os.getenv("LOG_COLOR_ENABLED", "true").strip().lower()
    color_enabled = color_enabled_raw in {"1", "true", "yes", "on"}

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set formatter based on style
    if format_style == "colored" and color_enabled:
        if colorlog is not None:
            formatter = ColorlogCategoryFormatter(
                include_timestamp=include_timestamp,
                include_category=include_category,
            )
        else:
            formatter = ColoredFormatter(
                include_timestamp=include_timestamp,
                include_category=include_category,
            )
    else:
        # Plain formatter
        format_str = ""
        if include_timestamp:
            format_str += "[%(asctime)s] "
        format_str += "[%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(format_str)
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_agent_logger(
    name: str,
    category: LogCategory = LogCategory.AGENT,
    level: Optional[int] = None
) -> CategoryLogger:
    """
    Get a category-specific logger for agents and components.
    
    Args:
        name: The name of the logger (usually module name)
        category: The category for this logger
        level: Optional specific level for this logger
    
    Returns:
        CategoryLogger instance with enhanced features
    """
    # Create cache key
    cache_key = f"{name}:{category.value}"
    
    # Return cached logger if exists
    if cache_key in _loggers:
        return _loggers[cache_key]
    
    # Create new logger
    logger = logging.getLogger(name)
    
    # Set specific level if provided
    if level is not None:
        logger.setLevel(level)
    
    # Create category logger
    category_logger = CategoryLogger(logger, category)
    
    # Cache and return
    _loggers[cache_key] = category_logger
    return category_logger


def get_logger_stats() -> Dict[str, Any]:
    """Get statistics about the current loggers."""
    return {
        "total_loggers": len(_loggers),
        "categories": list(set(logger.category.value for logger in _loggers.values())),
        "logger_names": list(_loggers.keys()),
    }


# Setup default logging on import
setup_logging()


# Convenience functions for different categories
def get_agent_logger_for_category(category: LogCategory, name: str = None) -> CategoryLogger:
    """Get a logger for a specific category."""
    if name is None:
        name = f"app.{category.value.lower()}"
    return get_agent_logger(name, category)


# Pre-configured loggers for common categories
def get_agent_category_logger(name: str = "app.agent") -> CategoryLogger:
    """Get an agent category logger."""
    return get_agent_logger(name, LogCategory.AGENT)


def get_llm_category_logger(name: str = "app.llm") -> CategoryLogger:
    """Get an LLM category logger."""
    return get_agent_logger(name, LogCategory.LLM)


def get_api_category_logger(name: str = "app.api") -> CategoryLogger:
    """Get an API category logger."""
    return get_agent_logger(name, LogCategory.API)


def get_service_category_logger(name: str = "app.service") -> CategoryLogger:
    """Get a service category logger."""
    return get_agent_logger(name, LogCategory.SERVICE)


def get_workflow_category_logger(name: str = "app.workflow") -> CategoryLogger:
    """Get a workflow category logger."""
    return get_agent_logger(name, LogCategory.WORKFLOW)


def get_communication_category_logger(name: str = "app.communication") -> CategoryLogger:
    """Get a communication category logger."""
    return get_agent_logger(name, LogCategory.COMMUNICATION)


def get_knowledge_category_logger(name: str = "app.knowledge") -> CategoryLogger:
    """Get a knowledge category logger."""
    return get_agent_logger(name, LogCategory.KNOWLEDGE)


def get_factory_category_logger(name: str = "app.factory") -> CategoryLogger:
    """Get a factory category logger."""
    return get_agent_logger(name, LogCategory.FACTORY)


def get_registry_category_logger(name: str = "app.registry") -> CategoryLogger:
    """Get a registry category logger."""
    return get_agent_logger(name, LogCategory.REGISTRY)


def get_system_category_logger(name: str = "app.system") -> CategoryLogger:
    """Get a system category logger."""
    return get_agent_logger(name, LogCategory.SYSTEM)


def get_embedding_category_logger(name: str = "app.embedding") -> CategoryLogger:
    """Get an embedding category logger."""
    return get_agent_logger(name, LogCategory.EMBEDDING)


def get_conversation_category_logger(name: str = "app.conversation") -> CategoryLogger:
    """Get a conversation category logger."""
    return get_agent_logger(name, LogCategory.CONVERSATION)


# Example usage and testing function
def test_logging_system():
    """Test the logging system with different categories and levels."""
    print("🎨 Testing Enhanced Logging System")
    print("=" * 50)
    
    # Test different categories
    agent_logger = get_agent_category_logger("test.agent")
    llm_logger = get_llm_category_logger("test.llm")
    api_logger = get_api_category_logger("test.api")
    service_logger = get_service_category_logger("test.service")
    
    # Test different log levels
    agent_logger.debug("This is a debug message from an agent")
    agent_logger.info("Agent initialized successfully", agent_id="agent_123")
    agent_logger.warning("Agent capacity nearly full")
    agent_logger.error("Agent execution failed", agent_id="agent_456", execution_time=1250)
    
    llm_logger.info("LLM provider switched to Ollama")
    llm_logger.warning("LLM response time exceeded threshold", execution_time=5000)
    
    api_logger.info("API endpoint called", request_id="req_789")
    api_logger.error("API rate limit exceeded")
    
    service_logger.info("Knowledge base updated with new entries")
    service_logger.debug("Vector index rebuilt")
    
    print("\n📊 Logger Statistics:")
    stats = get_logger_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_logging_system()