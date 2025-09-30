"""
File Tools for Deep Agents
===========================

Tools for file-based context storage and retrieval, enabling agents to offload
detailed information while maintaining minimal working context.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState, DeepAgentStateManager


@tool(parse_docstring=True)
def write_file(
    filename: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    append: bool = False,
    add_metadata: bool = True
) -> Command:
    """Write content to a file in the agent's context storage.
    
    Use this tool to store detailed information, tool results, or context
    that needs to persist across agent interactions. This helps maintain
    token efficiency by offloading detailed content.
    
    Args:
        filename: Name of the file to write (will be auto-prefixed with timestamp if needed)
        content: Content to write to the file
        append: Whether to append to existing file (default: False)
        add_metadata: Whether to add metadata header (default: True)
    
    Returns:
        Command that updates the files in state and returns confirmation message
    """
    # Auto-generate filename if not provided or enhance with timestamp
    if not filename or filename == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_output_{timestamp}.md"
    elif not any(filename.endswith(ext) for ext in ['.md', '.txt', '.json', '.py']):
        # Add .md extension if no extension provided
        filename = f"{filename}.md"
    
    # Add metadata header if requested
    if add_metadata:
        metadata_header = f"""---
Generated: {datetime.now().isoformat()}
Agent: {state.get('current_agent', 'unknown')}
Session: {state.get('session_id', 'unknown')}
---

"""
        content = metadata_header + content
    
    # Get current files
    files = state.get("files", {})
    
    # Handle append mode
    if append and filename in files:
        files[filename] = files[filename] + "\n\n" + content
    else:
        files[filename] = content
    
    # Update agent context
    current_agent = state.get("current_agent")
    if current_agent:
        agent_contexts = state.get("agent_contexts", {})
        if current_agent in agent_contexts:
            agent_contexts[current_agent]["files_created"].append(filename)
            agent_contexts[current_agent]["last_activity"] = datetime.now().isoformat()
    
    return Command(
        update={
            "files": files,
            "agent_contexts": agent_contexts if current_agent else state.get("agent_contexts", {}),
            "messages": [
                ToolMessage(
                    content=f"✅ File '{filename}' written successfully. Size: {len(content)} characters.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True) 
def read_file(
    filename: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_chars: int = 2000
) -> Command:
    """Read content from a file in the agent's context storage.
    
    Use this tool to retrieve previously stored information, context,
    or detailed tool results when needed for decision making.
    
    Args:
        filename: Name of the file to read
        max_chars: Maximum characters to return (default: 2000)
    
    Returns:
        Command that returns file content or error message
    """
    files = state.get("files", {})
    
    if filename not in files:
        available_files = list(files.keys())
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ File '{filename}' not found. Available files: {available_files}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    content = files[filename]
    
    # Truncate if too long
    if len(content) > max_chars:
        truncated_content = content[:max_chars] + f"\n\n... (truncated, {len(content) - max_chars} more characters)"
    else:
        truncated_content = content
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"📄 **{filename}**\n\n{truncated_content}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def list_files(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    pattern: Optional[str] = None
) -> Command:
    """List all files in the agent's context storage.
    
    Use this tool to see what information has been stored and is available
    for retrieval. Helps maintain awareness of available context.
    
    Args:
        pattern: Optional pattern to filter files (e.g., "*.md" or "agent_*")
    
    Returns:
        Command that returns list of available files with metadata
    """
    files = state.get("files", {})
    
    if not files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📁 No files stored yet.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Filter files by pattern if provided
    file_list = list(files.keys())
    if pattern:
        import fnmatch
        file_list = [f for f in file_list if fnmatch.fnmatch(f, pattern)]
    
    # Create file listing with metadata
    file_info = []
    for filename in sorted(file_list):
        content = files[filename]
        size = len(content)
        
        # Try to extract creation time from metadata
        created = "Unknown"
        if content.startswith("---"):
            try:
                lines = content.split("\n")
                for line in lines[1:6]:  # Check first few lines for Generated field
                    if line.startswith("Generated:"):
                        created = line.split("Generated: ")[1]
                        break
            except:
                pass
        
        file_info.append(f"📄 {filename} ({size} chars, {created})")
    
    file_listing = "\n".join(file_info)
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"📁 **Available Files ({len(file_list)} total):**\n\n{file_listing}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def delete_file(
    filename: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Delete a file from the agent's context storage.
    
    Use this tool to clean up outdated or unnecessary files to maintain
    a clean context storage environment.
    
    Args:
        filename: Name of the file to delete
    
    Returns:
        Command that removes file and returns confirmation
    """
    files = state.get("files", {})
    
    if filename not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ File '{filename}' not found and cannot be deleted.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Remove the file
    del files[filename]
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🗑️ File '{filename}' deleted successfully.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def search_files(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: int = 5
) -> Command:
    """Search for content across all stored files.
    
    Use this tool to find specific information across your stored context
    when you need to locate relevant details for decision making.
    
    Args:
        query: Search query to find in file contents
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Command that returns search results with file names and relevant excerpts
    """
    files = state.get("files", {})
    
    if not files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="🔍 No files to search.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Search for query in file contents
    results = []
    query_lower = query.lower()
    
    for filename, content in files.items():
        if query_lower in content.lower():
            # Find the context around the match
            content_lines = content.split('\n')
            matching_lines = []
            
            for i, line in enumerate(content_lines):
                if query_lower in line.lower():
                    # Get context around the match (2 lines before and after)
                    start = max(0, i - 2)
                    end = min(len(content_lines), i + 3)
                    context = '\n'.join(content_lines[start:end])
                    matching_lines.append(context)
            
            if matching_lines:
                results.append({
                    'filename': filename,
                    'matches': matching_lines[:2]  # Limit to 2 matches per file
                })
    
    if not results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"🔍 No results found for query: '{query}'",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Format results
    results = results[:max_results]
    formatted_results = []
    
    for result in results:
        filename = result['filename']
        matches_text = '\n\n---\n\n'.join(result['matches'])
        formatted_results.append(f"📄 **{filename}**\n{matches_text}")
    
    search_results = '\n\n' + '='*50 + '\n\n'.join(formatted_results)
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"🔍 **Search Results for '{query}' ({len(results)} files):**{search_results}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# Convenience functions for non-tool usage
def create_file_tools():
    """Create a list of file management tools for agents."""
    return [write_file, read_file, list_files, delete_file, search_files]


def get_file_usage_instructions() -> str:
    """Get instructions for using file tools effectively."""
    return """
# FILE MANAGEMENT INSTRUCTIONS

Use these tools to manage context storage efficiently:

## When to Use Files:
- Store detailed tool results that don't need to be in working memory
- Save research findings, analysis results, or long-form content
- Preserve context between agent handoffs
- Maintain conversation history and important decisions

## Best Practices:
1. **Use descriptive filenames**: `health_analysis_2024.md` vs `output.txt`
2. **Store first, summarize later**: Save full details to file, return summary
3. **Regular cleanup**: Delete outdated files to maintain organization
4. **Search when needed**: Use search_files to find relevant information
5. **Read selectively**: Use max_chars parameter to avoid token overflow

## File Naming Conventions:
- Analysis results: `analysis_[topic]_[date].md`
- Tool outputs: `[tool_name]_output_[timestamp].md`
- Research: `research_[topic]_[date].md`
- Planning: `plan_[project]_[date].md`

## Context Offloading Pattern:
1. Execute tool/analysis
2. Store full results in file
3. Return only summary/key points to maintain token efficiency
4. Reference file for detailed follow-up when needed
"""