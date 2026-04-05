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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.langgraph.workflow import AgentGraphWorkflow
from app.agents.registry import get_agent_registry
from app.api.knowledge import router as knowledge_router
from app.api.approval import router as approval_router
from app.agents.factory import initialize_agents, AgentFactory
from app.agents.base import AgentType
from app.llm import service as llm_service_module
from app.llm.service import LLMService, get_llm_service
from app.llm.base import LLMProviderType
from app.llm.config import LLMConfig
from app.llm.openai_provider import OpenAIProvider
from app.llm.ollama_provider import OllamaProvider
from app.utils.logging import get_api_category_logger
from app.services.config_storage import get_config_storage
from app.services.interaction_recorder import get_interaction_recorder
from app.services.knowledge_base import get_knowledge_base_service
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env file
load_dotenv()

# Use enhanced logging
logger = get_api_category_logger("main")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Initialize agents (LLM service will be initialized on-demand)
    await initialize_agents()
    
    # Initialize the interaction recorder with required service
    
    try:
        knowledge_service = get_knowledge_base_service()
        llm_service = get_llm_service()
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
    
        
        # Create new service with updated provider preference
        config = LLMConfig.from_env(dict(os.environ))
        config.provider = provider_type  # Set the preferred provider
        
        # Import config storage to save settings
        config_storage = get_config_storage()
        
        # Override config with frontend-provided values and save them
        if request.config:
            if provider_type == LLMProviderType.OPENAI and 'api_key' in request.config:
                config.openai_api_key = request.config['api_key']
                # Handle model selection
                if 'model' in request.config:
                    config.openai_model = request.config['model']
                # Save OpenAI config for persistence
                openai_config = {'api_key': request.config['api_key']}
                if 'model' in request.config:
                    openai_config['model'] = request.config['model']
                config_storage.set_openai_config(openai_config)
            elif provider_type == LLMProviderType.OLLAMA and 'endpoint' in request.config:
                config.ollama_endpoint = request.config['endpoint']
                # Save Ollama config for persistence
                config_storage.set_ollama_config({'endpoint': request.config['endpoint']})
        
        # Save provider preference
        config_storage.set_provider_preference(request.provider)
        
        # Create new service instance
        new_service = LLMService(config)
        new_service._initialized = True  # Mark as initialized to bypass initialize()
        
        # Create and add the specific provider directly to avoid config conflicts
        try:
            if provider_type == LLMProviderType.OLLAMA:
                # Create Ollama provider directly
                provider = OllamaProvider(
                    endpoint=config.ollama_endpoint,
                    model=config.ollama_model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                # Initialize Ollama provider (should not hit OpenAI)
                await provider.initialize()
                new_service.factory._providers[provider_type] = provider
                new_service.factory._current_provider = provider
                
            elif provider_type == LLMProviderType.OPENAI:
                # Only create OpenAI provider if API key is provided
                if not config.openai_api_key:
                    return {
                        "success": False,
                        "current_provider": "none",
                        "message": "OpenAI API key is required to use OpenAI provider"
                    }

                openai_base_url = config.openai_base_url or "https://api.openai.com/v1"
                openai_base_url = openai_base_url.rstrip('/')
                models_url = openai_base_url if openai_base_url.endswith('/models') else f"{openai_base_url}/models"

                connectivity_request = urllib.request.Request(
                    models_url,
                    headers={
                        'Authorization': f'Bearer {config.openai_api_key}',
                        'Content-Type': 'application/json'
                    },
                    method='GET'
                )

                try:
                    with urllib.request.urlopen(connectivity_request, timeout=8) as response:
                        if response.status >= 400:
                            return {
                                "success": False,
                                "current_provider": "none",
                                "message": f"OpenAI connectivity check failed with status {response.status}"
                            }
                except Exception as connectivity_error:
                    return {
                        "success": False,
                        "current_provider": "none",
                        "message": f"OpenAI connectivity check failed: {str(connectivity_error)}"
                    }
                
                # Create OpenAI provider directly
                provider = OpenAIProvider(
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    base_url=config.openai_base_url
                )
                # Initialize the provider to test the connection
                await provider.initialize()
                new_service.factory._providers[provider_type] = provider
                new_service.factory._current_provider = provider
            
            # Update the global service reference
            if llm_service_module._llm_service is not None:
                await llm_service_module._llm_service.shutdown()
            llm_service_module._llm_service = new_service
            
            return {
                "success": True,
                "current_provider": str(provider_type),
                "message": f"Successfully switched to {request.provider}"
            }
            
        except Exception as provider_error:
            return {
                "success": False,
                "current_provider": "none",
                "message": f"Failed to create {request.provider} provider: {str(provider_error)}"
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
        # Get current service if available
        current_service = llm_service_module._llm_service
        
        status = {
            "current_provider": None,
            "providers": {
                "openai": {"healthy": False, "model": None, "responseTime": None},
                "ollama": {"healthy": False, "model": None, "responseTime": None}
            }
        }
        
        if current_service and current_service._initialized:
            current_provider_type = current_service.get_current_provider()
            if current_provider_type:
                status["current_provider"] = str(current_provider_type).split('.')[-1].lower()
                
                # Check provider health
                try:
                    provider = current_service.factory._current_provider
                    if provider:
                        # Mark current provider as healthy if it exists
                        provider_name = str(current_provider_type).split('.')[-1].lower()
                        status["providers"][provider_name]["healthy"] = True
                        
                        # Get model info
                        if hasattr(provider, 'model'):
                            status["providers"][provider_name]["model"] = provider.model
                            
                except Exception as e:
                    logger.error(f"Error checking provider health: {e}")
        
        # Always check Ollama availability
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                if response.status != 200:
                    return status

                status["providers"]["ollama"]["healthy"] = True
                response_payload = response.read().decode("utf-8")
                tags_data = json.loads(response_payload)
                if tags_data.get("models"):
                    status["providers"]["ollama"]["model"] = tags_data["models"][0]["name"]
        except Exception:
            pass  # Ollama not available
            
        return status
        
    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        return {
            "current_provider": None,
            "providers": {
                "openai": {"healthy": False, "model": None, "responseTime": None},
                "ollama": {"healthy": False, "model": None, "responseTime": None}
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
                # Require API key for OpenAI testing
                api_key = request.config.get('api_key') if request.config else None
                if not api_key:
                    return {
                        "healthy": False,
                        "error": "OpenAI API key is required for testing"
                    }
                
                # Use provided model or default
                model = request.config.get('model', 'gpt-3.5-turbo') if request.config else 'gpt-3.5-turbo'
                base_url = request.config.get('base_url', os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')) if request.config else os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
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
        
        return {
            "openai": config_storage.get_openai_config(),
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
            config_storage.set_openai_config(request.openai)
        
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