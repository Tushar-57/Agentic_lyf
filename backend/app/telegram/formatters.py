"""Response formatters for Telegram bot."""

import re
from typing import Dict, Any, Optional


class ResponseFormatter:
    """Format responses for Telegram with proper Markdown."""
    
    @staticmethod
    def format_response(
        response: str,
        reasoning: Optional[Dict[str, Any]] = None,
        platform: str = "telegram"
    ) -> str:
        """
        Format agent response for Telegram.
        
        Args:
            response: Raw response from agent
            reasoning: Optional reasoning data from agent
            platform: Platform identifier (telegram, web, etc.)
        
        Returns:
            Formatted response string
        """
        if not response:
            return "ℹ️ No response generated."
        
        # Clean up response for Telegram Markdown
        formatted = ResponseFormatter._clean_markdown(response)
        
        # Add agent info if reasoning available
        if reasoning and reasoning.get("finalAgent"):
            agent_name = reasoning["finalAgent"].replace("_", " ").title()
            formatted = f"🤖 *{agent_name}*\n\n{formatted}"
        
        return formatted
    
    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Clean and fix Markdown for Telegram."""
        # Telegram uses simpler Markdown - fix common issues
        
        # Remove HTML-style bold/italic if present
        text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)
        text = re.sub(r'<i>(.*?)</i>', r'*\1*', text)
        
        # Ensure code blocks are properly formatted
        text = re.sub(r'```(\w+)?\n', r'```\n', text)
        
        # Fix escaped characters that Telegram doesn't need
        # Telegram Markdown v2 requires escaping: _*[]()~`>#+-=|{}.!
        # But for MarkdownV1 (default), we use simpler formatting
        
        return text
    
    @staticmethod
    def format_profile(user_prefs: Dict[str, Any]) -> str:
        """Format user profile for display."""
        general = user_prefs.get("general", {})
        productivity = user_prefs.get("productivity", {})
        finance = user_prefs.get("finance", {})
        health = user_prefs.get("health", {})
        
        # Build profile text
        lines = ["👤 **Your Profile**\n"]
        
        # General info
        if general:
            mentor = general.get("mentor", {})
            lines.append("**General Settings:**")
            if general.get("role"):
                lines.append(f"• Role: {general['role'].title()}")
            if general.get("work_hours"):
                lines.append(f"• Work Hours: {general['work_hours']}")
            if general.get("timezone"):
                lines.append(f"• Timezone: {general['timezone']}")
            if mentor.get("style"):
                lines.append(f"• Mentor Style: {mentor['style']}")
            if general.get("priorities"):
                priorities = ", ".join(general["priorities"])
                lines.append(f"• Priorities: {priorities}")
            lines.append("")
        
        # Productivity goals
        if productivity and productivity.get("goals"):
            lines.append("**Productivity Goals:**")
            for goal in productivity["goals"][:3]:  # Show first 3
                title = goal.get("title", "Untitled")
                priority = goal.get("priority", "Normal")
                lines.append(f"• {title} ({priority})")
            lines.append("")
        
        # Finance info
        if finance:
            lines.append("**Finance:**")
            if finance.get("monthly_income"):
                lines.append(f"• Monthly Income: ${finance['monthly_income']}")
            if finance.get("savings_goal"):
                lines.append(f"• Savings Goal: {finance['savings_goal']}%")
            lines.append("")
        
        # Health info
        if health:
            lines.append("**Health:**")
            if health.get("diet_preference"):
                lines.append(f"• Diet: {health['diet_preference']}")
            if health.get("fitness_level"):
                lines.append(f"• Fitness: {health['fitness_level']}")
            lines.append("")
        
        if len(lines) == 1:
            lines.append("_No profile data available._")
            lines.append("\nVisit the web app to set up your profile!")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_error(error_message: str) -> str:
        """Format error message."""
        return f"❌ **Error**\n\n{error_message}"
    
    @staticmethod
    def format_success(message: str) -> str:
        """Format success message."""
        return f"✅ {message}"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 4096) -> str:
        """Truncate text to Telegram's message limit."""
        if len(text) <= max_length:
            return text
        
        return text[:max_length - 50] + "\n\n...(message truncated)"
