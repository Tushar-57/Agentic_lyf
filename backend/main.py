#Fast API main.py 

import os
import json
import asyncio
import logging
import re
import urllib.request
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.langgraph.workflow import AgentGraphWorkflow
from app.agents.registry import get_agent_registry
from app.api.knowledge import router as knowledge_router
from app.api.approval import router as approval_router
from app.agents.factory import initialize_agents, AgentFactory
from app.agents.base import AgentType
from app.llm.service import get_llm_service
from app.llm.base import LLMProviderType
from app.llm.config import LLMConfig
from app.llm.ollama_provider import OllamaProvider
from app.utils.logging import get_api_category_logger, get_conversation_category_logger
from app.services.config_storage import get_config_storage
from app.services.interaction_recorder import get_interaction_recorder
from app.services.knowledge_base import get_knowledge_base_service
from app.auth.user_context import get_current_user, resolve_request_user, set_request_user, reset_request_user
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file
load_dotenv()

# Use enhanced logging
logger = get_api_category_logger("main")
conversation_logger = get_conversation_category_logger("app.conversation.api")


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


API_CHAT_LOG_ENABLED = parse_bool_env("AI_CONVERSATION_LOG_ENABLED", True)
API_CHAT_LOG_FULL_TEXT = parse_bool_env("AI_CONVERSATION_LOG_FULL_TEXT", True)
API_CHAT_LOG_MAX_CHARS = parse_positive_int_env("AI_CONVERSATION_LOG_MAX_CHARS", 64000, 500)


def serialize_chat_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)

    if API_CHAT_LOG_FULL_TEXT or len(text) <= API_CHAT_LOG_MAX_CHARS:
        return text
    return f"{text[:API_CHAT_LOG_MAX_CHARS - 3]}..."


def bind_registry_knowledge_base_for_request(registry) -> str:
    """Ensure agents use the current request user's knowledge base instance."""
    resolved_user_id = get_current_user().storage_key
    request_scoped_kb = get_knowledge_base_service(resolved_user_id)

    rebound_count = 0
    for agent in registry.get_all_agents():
        if not hasattr(agent, "knowledge_base"):
            continue

        if getattr(agent, "knowledge_base", None) is request_scoped_kb:
            continue

        setattr(agent, "knowledge_base", request_scoped_kb)
        rebound_count += 1

    if rebound_count:
        logger.info(
            "Rebound %d agent knowledge_base references for user=%s",
            rebound_count,
            resolved_user_id,
        )

    return resolved_user_id

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Initialize agents (LLM service will be initialized on-demand)
    await initialize_agents()
    
    # Initialize the interaction recorder with required service
    
    try:
        knowledge_service = get_knowledge_base_service()
        llm_service = await get_llm_service()
        _ = get_interaction_recorder(knowledge_service, llm_service)
        logger.info("Successfully initialized interaction recorder")
    except Exception as e:
        logger.warning(f"Could not initialize interaction recorder: {e}")
    
    # Initialize the workflow with agents now loaded
    global _workflow, _graph
    try:
        _workflow = AgentGraphWorkflow()
        _graph = _workflow.get_compiled_graph()
        logger.info("Successfully initialized LangGraph workflow with agents")
    except Exception as e:
        logger.warning(f"Could not initialize workflow with agents: {e}")
    
    yield
    # (Optional) Add shutdown/cleanup logic here

app = FastAPI(
    title="AI Agent Ecosystem API",
    description="Backend API for the AI Agent Ecosystem",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def attach_user_context(request: Request, call_next):
    """Resolve and attach request user scope for per-user data partitioning."""
    resolved_user = resolve_request_user(request)
    context_token = set_request_user(resolved_user)

    try:
        response = await call_next(request)
    finally:
        reset_request_user(context_token)

    response.headers["X-Agentic-User"] = resolved_user.storage_key
    return response

def parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_origin(origin: str) -> str:
    return origin.rstrip("/")


def wildcard_origin_to_regex(origin_pattern: str) -> str:
    escaped = re.escape(normalize_origin(origin_pattern))
    wildcard_regex = escaped.replace("\\*", ".*")
    return f"^{wildcard_regex}$"


def build_cors_config(origins: List[str], origin_regex: Optional[str]) -> tuple[List[str], Optional[str]]:
    exact_origins: List[str] = []
    wildcard_origin_patterns: List[str] = []

    for origin in origins:
        normalized = normalize_origin(origin)
        if "*" in normalized:
            wildcard_origin_patterns.append(normalized)
        else:
            exact_origins.append(normalized)

    regex_parts: List[str] = []
    if origin_regex:
        cleaned_regex = origin_regex.strip()
        if cleaned_regex:
            regex_parts.append(f"(?:{cleaned_regex})")

    regex_parts.extend(wildcard_origin_to_regex(pattern) for pattern in wildcard_origin_patterns)

    # Preserve order while deduplicating exact origins.
    deduped_exact_origins = list(dict.fromkeys(exact_origins))
    merged_regex = "|".join(regex_parts) if regex_parts else None
    return deduped_exact_origins, merged_regex

cors_allowed_origins = parse_csv_env(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8088,https://agenticlyf.vercel.app"
)
raw_cors_allowed_origin_regex = os.getenv(
    "CORS_ALLOWED_ORIGIN_REGEX",
    r"^https://.*\.vercel\.app$|^https://.*\.netlify\.app$"
).strip() or None

cors_allowed_origins, cors_allowed_origin_regex = build_cors_config(
    cors_allowed_origins,
    raw_cors_allowed_origin_regex,
)

logger.info(
    f"Configured CORS origins={cors_allowed_origins} "
    f"origin_regex={cors_allowed_origin_regex}"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_origin_regex=cors_allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(knowledge_router)
app.include_router(approval_router, prefix="/api/approval", tags=["approval"])


class ChatRequest(BaseModel):
    message: str
    agent: str = "orchestrator"
    conversation_id: str

class ChatResponse(BaseModel):
    response: Any  # Accepts str, dict, list, etc.
    agent: str
    reasoning: Any = None  # Accepts any type
    timestamp: datetime

@app.get("/")
async def root():
    return {"message": "AI Agent Ecosystem API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/health")
async def api_health_check():
    """Simple API health check without LLM dependencies."""
    return {
        "status": "healthy",
        "api": "ready",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        registry = get_agent_registry()
        resolved_user_id = bind_registry_knowledge_base_for_request(registry)

        if API_CHAT_LOG_ENABLED:
            conversation_logger.info(
                "API_CHAT_REQUEST user=%s conversation_id=%s agent=%s message=%s",
                resolved_user_id,
                request.conversation_id,
                request.agent,
                serialize_chat_payload(request.message),
            )

        # Find orchestrator agent by type
        logger.info(f"{registry.get_agent_ids()}")
        orchestrators = registry.get_agents_by_type(AgentType.ORCHESTRATOR)
        orchestrator = orchestrators[0] if orchestrators else None
        if orchestrator is None:
            logging.error("Orchestrator agent not found in registry. Agent ecosystem may not be initialized.")
            return ChatResponse(
                response="I'm the orchestrator agent. I encountered an issue: Orchestrator agent not found.",
                agent="orchestrator",
                reasoning="Error: Orchestrator agent not found in registry.",
                timestamp=datetime.now()
            )
        state = {
            "user_input": request.message,
            "context": {},
            "conversation_id": request.conversation_id,
            "agent": orchestrator.agent_id
        }
        # Use LangGraph workflow for multi-agent orchestration
        graph_workflow = await get_workflow()
        result = await graph_workflow.run(state)
        logger.info(f"[DEBUG] workflow.run result type: {type(result)} value: {result}")
        response = None
        reasoning = None
        logger.info(f"[DEBUG] Step: result type: {type(result)} value: {result}")
        # Handle dict
        if isinstance(result, dict):
            logger.info(f"[DEBUG] Dict result keys: {list(result.keys())}")
            response = result.get("response")
            reasoning = result.get("reasoning")
        # Handle tuple
        elif isinstance(result, tuple):
            logger.info(f"[DEBUG] Tuple result length: {len(result)} value: {result}")
            if len(result) == 2:
                response, reasoning = result
            elif len(result) == 1:
                response = result[0]
                reasoning = None
            else:
                response = str(result)
                reasoning = None
        # Handle string (try to parse as JSON)
        elif isinstance(result, str):
            logger.info(f"[DEBUG] String result: {result}")
            try:
                parsed = json.loads(result)
                logger.info(f"[DEBUG] Parsed JSON from string result: {parsed}")
                response = parsed.get("response", str(parsed))
                reasoning = parsed.get("reasoning")
            except Exception as json_err:
                logger.info(f"[DEBUG] Could not parse string result as JSON: {json_err}")
                response = result
                reasoning = None
        else:
            logger.info(f"[DEBUG] Unexpected result type: {type(result)} value: {result}")
            response = str(result)
            reasoning = None

        if response is None or (isinstance(response, str) and not response.strip()):
            response = (
                "I could not generate a complete response right now. "
                "Please try again in a moment, or verify your AI provider settings."
            )

        if isinstance(response, str):
            lowered = response.lower()
            if (
                "llm_provider_unavailable" in lowered
                or "no healthy providers available" in lowered
                or "llm service not initialized" in lowered
            ):
                response = (
                    "I cannot reach an AI provider right now. "
                    "Please connect OpenAI in settings or ensure Ollama is running, then retry your message."
                )

        logger.info(f"[DEBUG] Final response type: {type(response)} value: {response}")
        logger.info(f"[DEBUG] Final reasoning type: {type(reasoning)} value: {reasoning}")

        if API_CHAT_LOG_ENABLED:
            conversation_logger.info(
                "API_CHAT_RESPONSE user=%s conversation_id=%s agent=%s response=%s reasoning=%s",
                resolved_user_id,
                request.conversation_id,
                state.get("agent", orchestrator.agent_id),
                serialize_chat_payload(response),
                serialize_chat_payload(reasoning),
            )

        return ChatResponse(
            response=response,
            agent=state.get("agent", orchestrator.agent_id),
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    except Exception as e:
        logging.error("Orchestrator Error: %s", e)
        error_text = str(e)
        normalized_error = error_text.lower()
        if (
            "llm_provider_unavailable" in normalized_error
            or "no healthy providers available" in normalized_error
            or "llm service not initialized" in normalized_error
        ):
            user_facing_message = (
                "I cannot reach an AI provider right now. "
                "Please connect OpenAI in settings or ensure Ollama is running, then retry your message."
            )
        else:
            user_facing_message = f"I'm the orchestrator agent. I encountered an issue: {error_text}."

        if API_CHAT_LOG_ENABLED:
            conversation_logger.error(
                "API_CHAT_ERROR user=%s conversation_id=%s agent=%s error=%s",
                get_current_user().storage_key,
                request.conversation_id,
                request.agent,
                serialize_chat_payload(error_text),
            )

        return ChatResponse(
            response=user_facing_message,
            agent="orchestrator",
            reasoning=f"Error: {error_text}",
            timestamp=datetime.now()
        )

@app.get("/api/agents/status")
async def get_agents_status():
    try:
        llm_service = await get_llm_service()
        health_status = await llm_service.health_check()
        current_provider = llm_service.get_current_provider()
        
        # Convert health status to the expected format
        formatted_health = {}
        for provider_type, health in health_status.items():
            provider_name = str(provider_type).lower().replace('llmprovidertype.', '')
            formatted_health[provider_name] = {
                "is_healthy": health.is_healthy,
                "model": health.model,
                "response_time_ms": health.response_time_ms,
                "error": health.error
            }
        
        return {
            "current_provider": str(current_provider).lower().replace('llmprovidertype.', ''),
            "health_status": formatted_health,
            "agents": {
                "orchestrator": {"status": "active", "description": "Main coordination agent"},
                "productivity": {"status": "active", "description": "Task and goal management"},
                "health": {"status": "active", "description": "Wellness and habits"},
                "finance": {"status": "active", "description": "Budget and expenses"},
                "scheduling": {"status": "active", "description": "Calendar management"},
                "journal": {"status": "active", "description": "Reflection and insights"}
            }
        }
    except Exception as e:
        print(f"Status check error: {e}")
        # Fallback with error indication
        return {
            "current_provider": "none",
            "health_status": {
                "openai": {"is_healthy": False, "model": "gpt-3.5-turbo", "response_time_ms": 0, "error": str(e)},
                "ollama": {"is_healthy": False, "model": "llama3.2:3b", "response_time_ms": 0, "error": str(e)}
            },
            "agents": {
                "orchestrator": {"status": "active", "description": "Main coordination agent"},
                "productivity": {"status": "active", "description": "Task and goal management"},
                "health": {"status": "active", "description": "Wellness and habits"},
                "finance": {"status": "active", "description": "Budget and expenses"},
                "scheduling": {"status": "active", "description": "Calendar management"},
                "journal": {"status": "active", "description": "Reflection and insights"}
            }
        }

class ProviderSwitchRequest(BaseModel):
    provider: str
    config: Optional[dict] = {}

@app.post("/api/llm/switch-provider")
async def switch_provider(request: ProviderSwitchRequest):
    try:
        # Validate provider type
        if request.provider not in ['openai', 'ollama']:
            raise HTTPException(status_code=400, detail="Invalid provider type")
        
        provider_type = LLMProviderType.OPENAI if request.provider == 'openai' else LLMProviderType.OLLAMA

        # Persist incoming provider-specific settings first.
        config_storage = get_config_storage()

        if provider_type == LLMProviderType.OPENAI:
            existing_openai = config_storage.get_openai_config() or {}
            merged_openai = dict(existing_openai)

            provided_model = str((request.config or {}).get("model", "")).strip()
            if provided_model:
                merged_openai["model"] = provided_model

            provided_embedding_model = str((request.config or {}).get("embedding_model", "")).strip()
            if provided_embedding_model:
                merged_openai["embedding_model"] = provided_embedding_model

            provided_base_url = str((request.config or {}).get("base_url", "")).strip()
            if provided_base_url:
                merged_openai["base_url"] = provided_base_url

            if merged_openai:
                config_storage.set_openai_config(merged_openai)

        elif provider_type == LLMProviderType.OLLAMA:
            existing_ollama = config_storage.get_ollama_config() or {}
            merged_ollama = dict(existing_ollama)

            provided_endpoint = str((request.config or {}).get("endpoint", "")).strip()
            if provided_endpoint:
                merged_ollama["endpoint"] = provided_endpoint

            provided_model = str((request.config or {}).get("model", "")).strip()
            if provided_model:
                merged_ollama["model"] = provided_model

            if merged_ollama:
                config_storage.set_ollama_config(merged_ollama)
        
        # Save provider preference
        config_storage.set_provider_preference(request.provider)

        # Build effective config after persistence changes.
        config = LLMConfig.from_env(dict(os.environ))
        config.provider = provider_type
        config.fallback_provider = (
            LLMProviderType.OPENAI if provider_type == LLMProviderType.OLLAMA else LLMProviderType.OLLAMA
        )

        if provider_type == LLMProviderType.OPENAI and not config.openai_api_key:
            return {
                "success": False,
                "current_provider": "none",
                "message": "OpenAI API key is missing. Set OPENAI_API_KEY in environment variables."
            }

        if provider_type == LLMProviderType.OLLAMA and not config.ollama_endpoint:
            return {
                "success": False,
                "current_provider": "none",
                "message": "Ollama endpoint is required to use Ollama provider"
            }

        llm_service = await get_llm_service()
        config_updated = await llm_service.update_config(config)

        if not config_updated:
            return {
                "success": False,
                "current_provider": "none",
                "message": f"Failed to initialize {request.provider} provider with current configuration"
            }

        switched = await llm_service.switch_provider(
            provider_type,
            # OpenAI health check defaults to config-only, so skip explicit probe to avoid false negatives.
            skip_health_check=(provider_type == LLMProviderType.OPENAI)
        )

        if not switched:
            current_provider = llm_service.get_current_provider()
            current_provider_name = (
                str(current_provider).split('.')[-1].lower() if current_provider else "none"
            )
            return {
                "success": False,
                "current_provider": current_provider_name,
                "message": f"Provider switch to {request.provider} failed health checks"
            }

        current_provider = llm_service.get_current_provider()
        current_provider_name = str(current_provider).split('.')[-1].lower() if current_provider else request.provider
        
        return {
            "success": True,
            "current_provider": current_provider_name,
            "message": f"Successfully switched to {request.provider}"
        }
            
    except Exception as e:
        print(f"Provider switch error: {e}")
        raise HTTPException(status_code=500, detail=f"Provider switch failed: {str(e)}") from e

class ConnectionTestRequest(BaseModel):
    provider: str
    config: dict = {}

@app.get("/api/llm/status")
async def get_llm_status():
    """Get current LLM provider status."""
    try:
        effective_config = LLMConfig.from_env(dict(os.environ))

        # Ensure we always inspect the latest service/config state.
        current_service = await get_llm_service()
        
        status = {
            "current_provider": None,
            "configured_provider": str(effective_config.provider),
            "configured_fallback_provider": str(effective_config.fallback_provider) if effective_config.fallback_provider else None,
            "providers": {
                "openai": {
                    "healthy": False,
                    "configured": bool(effective_config.openai_api_key),
                    "model": effective_config.openai_model,
                    "responseTime": None,
                    "error": None,
                },
                "ollama": {
                    "healthy": False,
                    "configured": bool(effective_config.ollama_endpoint),
                    "model": effective_config.ollama_model,
                    "endpoint": effective_config.ollama_endpoint,
                    "responseTime": None,
                    "error": None,
                }
            }
        }
        
        if current_service and current_service.is_initialized():
            current_provider_type = current_service.get_current_provider()
            if current_provider_type:
                status["current_provider"] = str(current_provider_type).split('.')[-1].lower()

            try:
                health_map = await current_service.health_check()
                for provider_type, health in health_map.items():
                    provider_name = str(provider_type).split('.')[-1].lower()
                    if provider_name not in status["providers"]:
                        continue

                    status["providers"][provider_name]["healthy"] = bool(health.is_healthy)
                    status["providers"][provider_name]["error"] = health.error
                    if health.model:
                        status["providers"][provider_name]["model"] = health.model
                    if health.response_time_ms is not None:
                        status["providers"][provider_name]["responseTime"] = int(health.response_time_ms)
            except Exception as e:
                logger.error(f"Error checking provider health: {e}")

        # Probe configured Ollama endpoint when health map is unavailable.
        if not status["providers"]["ollama"]["healthy"]:
            ollama_probe_url = f"{effective_config.ollama_endpoint.rstrip('/')}/api/tags"
            try:
                with urllib.request.urlopen(ollama_probe_url, timeout=2) as response:
                    if response.status == 200:
                        status["providers"]["ollama"]["healthy"] = True
                        response_payload = response.read().decode("utf-8")
                        tags_data = json.loads(response_payload)
                        if tags_data.get("models"):
                            status["providers"]["ollama"]["model"] = tags_data["models"][0]["name"]
            except Exception as probe_error:
                status["providers"]["ollama"]["error"] = str(probe_error)
            
        return status
        
    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        return {
            "current_provider": None,
            "configured_provider": None,
            "configured_fallback_provider": None,
            "providers": {
                "openai": {"healthy": False, "configured": False, "model": None, "responseTime": None, "error": str(e)},
                "ollama": {"healthy": False, "configured": False, "model": None, "endpoint": None, "responseTime": None, "error": str(e)}
            }
        }

@app.post("/api/llm/test-connection")
async def test_connection(request: ConnectionTestRequest):
    try:
        if request.provider not in ['openai', 'ollama']:
            raise HTTPException(status_code=400, detail="Invalid provider type")
        
        provider_type = LLMProviderType.OPENAI if request.provider == 'openai' else LLMProviderType.OLLAMA
        
        # Create a temporary provider with provided configuration for testing
        start_time = datetime.now()
        
        try:
            if provider_type == LLMProviderType.OPENAI:
                effective_config = LLMConfig.from_env(dict(os.environ))

                provided_api_key = str((request.config or {}).get('api_key', '')).strip()
                if provided_api_key and not any(ch.isalnum() for ch in provided_api_key):
                    provided_api_key = ''

                api_key = provided_api_key or (effective_config.openai_api_key or '').strip()
                if not api_key:
                    return {
                        "healthy": False,
                        "error": "OpenAI API key is required for testing"
                    }
                
                # Use provided model/base_url overrides, otherwise use persisted config.
                model = str(
                    (request.config or {}).get('model')
                    or effective_config.openai_model
                    or 'gpt-3.5-turbo'
                ).strip()
                base_url = str(
                    (request.config or {}).get('base_url')
                    or effective_config.openai_base_url
                    or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                ).strip()
                base_url = base_url.rstrip('/')
                models_url = base_url if base_url.endswith('/models') else f"{base_url}/models"

                openai_request = urllib.request.Request(
                    models_url,
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    },
                    method='GET'
                )

                with urllib.request.urlopen(openai_request, timeout=8) as response:
                    payload = json.loads(response.read().decode('utf-8') or '{}')
                    available_models = [item.get('id') for item in payload.get('data', []) if isinstance(item, dict)]
                    model_exists = model in available_models if available_models else None
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    "healthy": True,
                    "responseTime": int(response_time),
                    "model": model,
                    "modelAvailable": model_exists,
                    "error": None
                }
                
            elif provider_type == LLMProviderType.OLLAMA:
                # Use provided endpoint or default
                endpoint = request.config.get('endpoint', 'http://localhost:11434') if request.config else 'http://localhost:11434'

                test_provider = OllamaProvider(
                    endpoint=endpoint,
                    model="llama3.2:3b",  # Use default model for testing
                    max_tokens=10,  # Minimal tokens for test
                    temperature=0.7
                )
                
                # Initialize and test the provider
                await test_provider.initialize()
                provider_health = await test_provider.health_check()
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    "healthy": provider_health.is_healthy,
                    "responseTime": int(response_time),
                    "model": provider_health.model,
                    "error": provider_health.error if not provider_health.is_healthy else None
                }
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "healthy": False,
                "responseTime": int(response_time),
                "error": str(e)
            }
            
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }

# Configuration management endpoints
class ConfigUpdateRequest(BaseModel):
    openai: Optional[dict] = None
    ollama: Optional[dict] = None
    provider_preference: Optional[str] = None

@app.get("/api/config")
async def get_config():
    """Get current stored configuration."""
    try:
        config_storage = get_config_storage()
        effective_config = LLMConfig.from_env(dict(os.environ))

        openai_config = dict(config_storage.get_openai_config() or {})
        openai_config.pop("api_key", None)
        if effective_config.openai_api_key:
            openai_config["api_key"] = "***configured***"
        if not openai_config.get("model"):
            openai_config["model"] = effective_config.openai_model
        if not openai_config.get("embedding_model"):
            openai_config["embedding_model"] = effective_config.openai_embedding_model
        
        return {
            "openai": openai_config,
            "ollama": config_storage.get_ollama_config(),
            "provider_preference": config_storage.get_provider_preference()
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}") from e

@app.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    """Update stored configuration."""
    try:
        config_storage = get_config_storage()
        
        # Update configurations if provided
        if request.openai is not None:
            openai_payload = dict(request.openai)
            # Never persist API keys to user config files.
            openai_payload.pop("api_key", None)

            existing_openai = config_storage.get_openai_config() or {}
            merged_openai = dict(existing_openai)
            merged_openai.update(openai_payload)
            config_storage.set_openai_config(merged_openai)
        
        if request.ollama is not None:
            config_storage.set_ollama_config(request.ollama)
            
        if request.provider_preference is not None:
            config_storage.set_provider_preference(request.provider_preference)
        
        return {
            "success": True,
            "message": "Configuration updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}") from e

# Global workflow instance
_workflow = None
_graph = None

async def get_workflow():
    """Get or create the workflow instance."""
    global _workflow, _graph
    if _workflow is None:
        _workflow = AgentGraphWorkflow()
        _graph = _workflow.get_compiled_graph()
        logger.info("Successfully created LangGraph workflow")
    return _workflow

async def get_graph():
    """Get the compiled graph for langgraph dev."""
    global _graph
    if _graph is None:
        await get_workflow()
    return _graph

# Global cache for LangGraph dev
_dev_graph_cache = None

# Create LangGraph workflow instance for langgraph dev
def get_graph_for_dev():
    """
    Factory function to create the graph for LangGraph dev.
    This will be called by LangGraph dev when it needs the graph.
    Uses caching to avoid recreating the graph on every request.
    """
    global _dev_graph_cache
    
    # Return cached graph if available
    if _dev_graph_cache is not None:
        logger.debug("Returning cached graph for LangGraph dev")
        return _dev_graph_cache
    
    try:
        # Ensure agents are initialized when graph factory is called
        registry = get_agent_registry()
        agents = registry.get_all_agents()
        
        # If no agents, try to initialize them
        if not agents:
            logger.info("No agents found, initializing agent ecosystem for LangGraph dev")
            
            # Create a new event loop if none exists (for LangGraph dev context)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Initialize agents synchronously for LangGraph dev
            factory = AgentFactory()
            
            # Run initialization in the event loop
            if loop.is_running():
                # If loop is already running, we need to use a different approach
                logger.warning("Event loop already running, attempting direct agent creation")
                try:
                    # Try to create agents directly without async
                    from app.agents.enhanced_orchestrator import EnhancedOrchestratorAgent as OrchestratorAgent
                    from app.agents.specialized import HealthAgent, ProductivityAgent
                    
                    # Create and register agents directly
                    agents_to_create = [
                        (AgentType.ORCHESTRATOR, OrchestratorAgent),
                        (AgentType.PRODUCTIVITY, ProductivityAgent),
                        (AgentType.HEALTH, HealthAgent),
                        # Add other agent types as needed
                    ]
                    
                    for agent_type, agent_class in agents_to_create:
                        try:
                            agent = agent_class()
                            registry.register_agent(agent)
                            logger.info(f"Direct registered agent: {agent.agent_id}")
                        except Exception as e:
                            logger.warning(f"Failed to create {agent_type}: {e}")
                            
                except Exception as e:
                    logger.warning(f"Direct agent creation failed: {e}")
            else:
                # Run async initialization
                loop.run_until_complete(factory.initialize_agent_ecosystem())
            
            # Re-check for agents
            agents = registry.get_all_agents()
        
        if agents:
            logger.info(f"Creating graph with {len(agents)} agents for LangGraph dev")
            workflow = AgentGraphWorkflow()
            compiled_graph = workflow.get_compiled_graph()
            logger.info("Successfully created full agent ecosystem graph for LangGraph dev")
            
            # Cache the graph for future requests
            _dev_graph_cache = compiled_graph
            return compiled_graph
        else:
            logger.warning("Still no agents found after initialization attempt, creating placeholder graph")
            # Fallback: create a simple placeholder graph
            simple_graph = StateGraph(dict)
            simple_graph.add_node("placeholder", lambda x: {"response": "Agents not yet loaded"})
            simple_graph.add_edge(START, "placeholder")
            simple_graph.add_edge("placeholder", END)
            
            # Cache the fallback graph as well
            _dev_graph_cache = simple_graph.compile()
            return _dev_graph_cache
            
    except Exception as e:
        logger.warning(f"Error creating graph for dev: {e}")
        # Fallback: create a simple placeholder graph
        simple_graph = StateGraph(dict)
        simple_graph.add_node("error_placeholder", lambda x: {"response": f"Graph creation error: {str(e)}"})
        simple_graph.add_edge(START, "error_placeholder")
        simple_graph.add_edge("error_placeholder", END)
        
        # Cache the error fallback graph
        _dev_graph_cache = simple_graph.compile()
        return _dev_graph_cache

# Export the graph factory function for langgraph dev
graph = get_graph_for_dev

if __name__ == "__main__":
    import uvicorn
    logger.info("[main.py] FastAPI startup: initializing agent ecosystem.")
    uvicorn.run(app, host="0.0.0.0", port=8000)