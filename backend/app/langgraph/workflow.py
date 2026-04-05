"""
LangGraph workflow for multi-agent orchestration and structured logging.
"""
import logging
import json
from typing import Any
from langgraph.graph import START, StateGraph, END
from datetime import datetime

# If you get import errors for app.*, run from your project root or set PYTHONPATH to the parent of backend.
from app.agents.registry import get_agent_registry
from app.llm.service import get_llm_service
from app.llm.base import CompletionRequest, ChatMessage
from app.agents.prompts import PromptLibrary

logger = logging.getLogger("langgraph")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class AgentGraphWorkflow:
    def __init__(self):
        self.registry = get_agent_registry()
        self.graph = StateGraph(dict)
        
        # Add agent nodes
        agents = self.registry.get_all_agents()
        for agent in agents:
            # prefer .agent_id, then .id, then .name, then str(agent)
            node_name = None
            if hasattr(agent, "agent_id"):
                node_name = getattr(agent, "agent_id")
            elif hasattr(agent, "id"):
                node_name = getattr(agent, "id")
            elif hasattr(agent, "name"):
                node_name = getattr(agent, "name")
            else:
                node_name = str(agent)

            # Use the execute attribute if present, else the agent itself if callable
            node_callable = getattr(agent, "execute", None)
            if node_callable is None or not callable(node_callable):
                node_callable = agent if callable(agent) else None

            if node_callable is None:
                logger.warning(f"Agent {node_name} has no callable execute() - skipping")
                continue

            # Wrap the node callable to ensure proper state handling for LangGraph
            def create_node_wrapper(agent_execute_method):
                async def node_wrapper(state):
                    logger.debug(f"Node wrapper calling {agent_execute_method} with state: {state}")
                    result = await agent_execute_method(state)
                    logger.debug(f"Node wrapper received result: {result}")
                    return result
                return node_wrapper
            
            wrapped_callable = create_node_wrapper(node_callable)
            self.graph.add_node(node_name, wrapped_callable)

        # Add the response formatting node
        self.graph.add_node("format_response_final_step", self._format_response_final_step)

        # Route only through orchestrator - it will handle delegation internally
        agent_names = list(self.graph.nodes.keys())
        orchestrator_name = None
        
        logger.info(f"Searching for orchestrator in agent names: {agent_names}")
        
        # Find the orchestrator among registered agents
        for name in agent_names:
            logger.debug(f"Checking agent name: {name}, contains orchestrator: {'orchestrator' in name.lower()}")
            if "orchestrator" in name.lower():
                orchestrator_name = name
                logger.info(f"Found orchestrator: {orchestrator_name}")
                break
        
        if orchestrator_name:
            # Route through orchestrator then to response formatter
            self.graph.add_edge(START, orchestrator_name)
            self.graph.add_edge(orchestrator_name, "format_response_final_step")
            self.graph.add_edge("format_response_final_step", END)
        else:
            logger.warning("No orchestrator found - falling back to sequential execution")
            # Fallback to original sequential behavior if no orchestrator
            if agent_names:
                # Remove format node from agent_names for fallback
                fallback_agents = [name for name in agent_names if name != "format_response_final_step"]
                if fallback_agents:
                    self.graph.add_edge(START, fallback_agents[0])
                    for i in range(len(fallback_agents) - 1):
                        self.graph.add_edge(fallback_agents[i], fallback_agents[i + 1])
                    self.graph.add_edge(fallback_agents[-1], "format_response_final_step")
                    self.graph.add_edge("format_response_final_step", END)
        
        # Compile the graph for LangGraph dev
        self.compiled_graph = self.graph.compile()
    
    def get_compiled_graph(self):
        """Return the compiled LangGraph for langgraph dev server."""
        return self.compiled_graph

    async def _format_response_final_step(self, state):
        """Final step to format and enhance the response with context and personalization."""
        try:
            logger.info("Starting response formatting step")
            
            # Extract information from state
            user_input = state.get("user_input", "")
            raw_response = state.get("response", "")
            raw_response_text = self._normalize_response_text(raw_response)
            reasoning = state.get("reasoning", {})
            context = state.get("context", {})
            conversation_id = state.get("conversation_id", "")

            # If orchestrator returned an error object, keep it readable and skip formatter calls.
            if isinstance(raw_response, dict):
                response_status = str(raw_response.get("status", "")).strip().lower()
                if response_status == "error":
                    state["response"] = self._normalize_response_text(raw_response.get("response", raw_response))
                    state["formatting_applied"] = False
                    return state
            
            # Don't format if response is already well-formatted or if it's an error
            if not raw_response_text or "i apologize" in raw_response_text.lower():
                return state
            
            # Get the final agent that provided the response
            final_agent = reasoning.get("finalAgent", "orchestrator") if isinstance(reasoning, dict) else "orchestrator"
            
            # Create formatting prompt based on the agent type and context
            formatting_prompt = self._build_formatting_prompt(
                user_input=user_input,
                raw_response=raw_response_text,
                final_agent=final_agent,
                reasoning=reasoning,
                context=context
            )
            
            # Use LLM to format the response
            try:
                llm_service = await get_llm_service()
                
                request = CompletionRequest(
                    messages=[
                        ChatMessage(role="system", content=formatting_prompt),
                        ChatMessage(role="user", content=f"Please enhance and format this response:\n\n{raw_response_text}")
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                formatted_response = await llm_service.chat_completion(request)
                
                content_payload = getattr(formatted_response, "content", formatted_response)
                enhanced_response = self._normalize_response_text(content_payload)

                if not enhanced_response:
                    enhanced_response = self._apply_basic_formatting(raw_response_text, final_agent, user_input)
                
                # Update the state with the enhanced response
                state["response"] = enhanced_response
                state["formatting_applied"] = True
                
                logger.info("Response formatting completed successfully")
                
            except Exception as e:
                logger.warning(f"LLM formatting failed: {e}, using fallback formatting")
                # Fallback: basic formatting
                enhanced_response = self._apply_basic_formatting(raw_response_text, final_agent, user_input)
                state["response"] = enhanced_response
                state["formatting_applied"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error in format_response_final_step: {e}")
            # Return original state if formatting fails
            return state

    def _build_formatting_prompt(self, user_input: str, raw_response: str, final_agent: str, reasoning: dict, context: dict) -> str:
        """Build a comprehensive formatting prompt based on context."""
        
        base_prompt = f"""You are a response formatter for an AI agent ecosystem. Your job is to take a raw response and make it more helpful, engaging, and personalized.

**Context:**
- User asked: "{user_input}"
- Responding agent: {final_agent}
- Conversation context: {context.get('conversation_id', 'New conversation')}

**Your tasks:**
1. **Enhance Clarity**: Make the response clear and easy to understand
2. **Add Personality**: Match the tone to the {final_agent} agent's personality
3. **Structure Information**: Use proper formatting, bullets, or sections if helpful
4. **Add Actionability**: Include next steps or follow-up suggestions when appropriate
5. **Maintain Accuracy**: Don't change the core information, only improve presentation

**Agent Personalities:**
- orchestrator: Professional coordinator, helpful guide
- productivity: Energetic motivator, results-focused
- health: Caring wellness coach, supportive
- finance: Practical advisor, detail-oriented
- scheduling: Organized planner, time-conscious
- journal: Reflective companion, emotionally aware
- general: Knowledgeable assistant, adaptable

**Response Guidelines:**
- Keep the core information intact
- Use markdown formatting for structure
- Add relevant emojis sparingly
- Include actionable next steps
- Make it conversational but professional
- If it's a simple answer, don't over-complicate it

**Example transformations:**
- "Task added" → "✅ **Task Added Successfully!** I've added that to your task list. Would you like me to help you prioritize it or set a reminder?"
- "Budget is good" → "💰 **Great news!** Your budget is looking healthy this month. You're staying within your limits across all categories."

Now enhance the following response:"""

        return base_prompt

    def _normalize_response_text(self, payload: Any) -> str:
        """Normalize arbitrary workflow/LLM payloads to display-safe text."""
        if payload is None:
            return ""

        if isinstance(payload, str):
            return payload.strip()

        if isinstance(payload, dict):
            for key in ("content", "response", "message", "text", "output"):
                if key in payload:
                    candidate = self._normalize_response_text(payload.get(key))
                    if candidate:
                        return candidate
            try:
                return json.dumps(payload)
            except Exception:
                return str(payload).strip()

        if isinstance(payload, list):
            normalized_parts = [self._normalize_response_text(item) for item in payload]
            normalized_parts = [part for part in normalized_parts if part]
            return "\n".join(normalized_parts)

        return str(payload).strip()

    def _apply_basic_formatting(self, raw_response: str, final_agent: str, user_input: str) -> str:
        """Apply basic formatting when LLM formatting fails."""
        
        # Agent-specific emojis and greeting styles
        agent_styles = {
            "orchestrator": {"emoji": "🧠", "style": "Professional and coordinated"},
            "productivity": {"emoji": "⚡", "style": "Energetic and action-oriented"},
            "health": {"emoji": "🌿", "style": "Caring and supportive"},
            "finance": {"emoji": "💰", "style": "Practical and detailed"},
            "scheduling": {"emoji": "📅", "style": "Organized and time-focused"},
            "journal": {"emoji": "📝", "style": "Reflective and thoughtful"},
            "general": {"emoji": "🤖", "style": "Helpful and adaptable"}
        }
        
        raw_text = self._normalize_response_text(raw_response)
        if not raw_text:
            raw_text = "I could not generate a response right now."

        style = agent_styles.get(final_agent, agent_styles["general"])
        
        # Basic enhancement
        if len(raw_text) < 50:
            # Short response - add context
            enhanced = f"{style['emoji']} {raw_text}\n\nIs there anything else I can help you with?"
        else:
            # Longer response - just add emoji and ensure it ends well
            enhanced = f"{style['emoji']} {raw_text}"
            if not raw_text.endswith(("?", "!", ".")):
                enhanced += "."
            enhanced += "\n\nLet me know if you need any clarification or have other questions!"
        
        return enhanced

    async def run(self, state):
        log_steps = []
        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "step": "workflow_start",
            "input_state": state,
            "input_state_type": str(type(state))
        }, indent=2))
        workflow = self.graph.compile()
        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "step": "workflow_compiled",
            "nodes": list(workflow.nodes.keys())
        }, indent=2))
        result = await workflow.ainvoke(state)
        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "step": "workflow_complete",
            "result": result,
            "result_type": str(type(result))
        }, indent=2))
        # Guarantee result is a dict
        if isinstance(result, tuple) and len(result) == 2:
            result = {"response": result[0], "reasoning": result[1]}
        elif not isinstance(result, dict):
            result = {"response": str(result), "reasoning": None}

        # Normalize common alternate keys to a single response/reasoning
        response = result.get("response") or result.get("final_response") or result.get("final") or result.get("next_agent")
        reasoning = result.get("reasoning") or result.get("reason") or result.get("orchestrator_output")

        if isinstance(response, dict):
            nested_response = (
                response.get("response")
                or response.get("content")
                or response.get("message")
                or response.get("text")
            )

            if not reasoning and isinstance(response.get("reasoning"), dict):
                reasoning = response.get("reasoning")

            if nested_response is not None:
                response = nested_response
            else:
                try:
                    response = json.dumps(response)
                except Exception:
                    response = str(response)

        if isinstance(response, (list, tuple)):
            response = "\n".join(str(item) for item in response)

        if response is not None and not isinstance(response, str):
            response = str(response)

        # If it looks like the original state (no response produced), warn and return None
        if not response and isinstance(result, dict) and {"user_input", "conversation_id"}.issubset(set(result.keys())):
            logger.warning("Workflow returned original input-state (agents likely returned None or skipped processing).")
            logger.info(json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "step": "workflow_return",
                "response": None,
                "reasoning": reasoning
            }, indent=2))
            return None, reasoning

        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "step": "workflow_return",
            "response": response,
            "reasoning": reasoning
        }, indent=2))
        return response, reasoning
