"""
Specialized Productivity Tools for ReAct Productivity Agent
=========================================================

Domain-specific tools for task management, goal tracking, time management,
and productivity optimization with deep agent integration.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState


@tool(parse_docstring=True)
def create_goal(
    title: str,
    description: str,
    target_date: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    category: str = "general",
    measurable_criteria: Optional[str] = None,
    action_steps: Optional[List[str]] = None,
    priority: str = "medium"
) -> Command:
    """Create a SMART goal with tracking capabilities.
    
    Create structured goals with specific criteria, deadlines, and action steps
    to enable effective goal tracking and achievement.
    
    Args:
        title: Clear, concise goal title
        description: Detailed description of what you want to achieve
        target_date: Target completion date in YYYY-MM-DD format
        category: Goal category (e.g., "career", "health", "finance", "personal")
        measurable_criteria: How you'll measure success
        action_steps: List of specific action steps to achieve the goal
        priority: Goal priority - "low", "medium", "high", "critical"
    
    Returns:
        Command that creates goal and generates action plan
    """
    # Validate target date
    try:
        target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Target date must be in YYYY-MM-DD format.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if target_datetime <= datetime.now():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Target date must be in the future.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Get or create goals data
    files = state.get("files", {})
    goals_file = "productivity_goals.json"
    
    if goals_file in files:
        try:
            goals_data = json.loads(files[goals_file])
        except json.JSONDecodeError:
            goals_data = {"goals": []}
    else:
        goals_data = {"goals": []}
    
    # Calculate time to target
    days_to_target = (target_datetime - datetime.now()).days
    
    # Create goal entry
    goal = {
        "id": len(goals_data["goals"]) + 1,
        "title": title,
        "description": description,
        "category": category.lower(),
        "target_date": target_date,
        "days_to_target": days_to_target,
        "measurable_criteria": measurable_criteria,
        "action_steps": action_steps or [],
        "priority": priority.lower(),
        "status": "active",
        "progress_percentage": 0,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "created_timestamp": datetime.now().isoformat(),
        "progress_updates": [],
        "milestones": []
    }
    
    goals_data["goals"].append(goal)
    
    # Update file
    files[goals_file] = json.dumps(goals_data, indent=2)
    
    # Create detailed goal plan
    goal_plan = [
        f"# Goal: {title}",
        f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Target Date:** {target_date} ({days_to_target} days)",
        f"**Category:** {category.title()}",
        f"**Priority:** {priority.title()}",
        "",
        "## Description",
        description,
        ""
    ]
    
    if measurable_criteria:
        goal_plan.extend([
            "## Success Criteria",
            measurable_criteria,
            ""
        ])
    
    if action_steps:
        goal_plan.extend([
            "## Action Steps",
            ""
        ])
        for i, step in enumerate(action_steps, 1):
            goal_plan.append(f"{i}. {step}")
        goal_plan.append("")
    
    # Add SMART framework analysis
    goal_plan.extend([
        "## SMART Analysis",
        f"- **Specific:** {title}",
        f"- **Measurable:** {measurable_criteria or 'Define specific metrics for tracking'}",
        f"- **Achievable:** Review if goal is realistic given timeframe",
        f"- **Relevant:** Ensure goal aligns with your priorities",
        f"- **Time-bound:** {target_date} ({days_to_target} days)",
        "",
        "## Recommended Next Steps",
        "1. Break down action steps into weekly milestones",
        "2. Schedule regular progress reviews (weekly/bi-weekly)",
        "3. Identify potential obstacles and mitigation strategies",
        "4. Set up accountability measures or tracking systems",
        ""
    ])
    
    # Time-based recommendations
    if days_to_target <= 30:
        goal_plan.extend([
            "## Short-term Goal Strategy",
            "- Create daily action items and track progress",
            "- Schedule focused work blocks for goal activities",
            "- Review progress every 2-3 days for quick adjustments"
        ])
    elif days_to_target <= 90:
        goal_plan.extend([
            "## Medium-term Goal Strategy", 
            "- Break into weekly milestones with specific deliverables",
            "- Schedule weekly progress reviews and planning sessions",
            "- Build habits and routines that support goal achievement"
        ])
    else:
        goal_plan.extend([
            "## Long-term Goal Strategy",
            "- Create quarterly milestones with measurable outcomes",
            "- Establish monthly progress reviews and course corrections",
            "- Build systems and processes for sustainable progress"
        ])
    
    goal_plan_text = "\n".join(goal_plan)
    
    # Save goal plan
    plan_filename = f"goal_plan_{goal['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[plan_filename] = goal_plan_text
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🎯 **Goal Created Successfully!**\n\n"
                           f"**Goal ID:** {goal['id']}\n"
                           f"**Title:** {title}\n"
                           f"**Target Date:** {target_date} ({days_to_target} days)\n"
                           f"**Category:** {category.title()}\n"
                           f"**Priority:** {priority.title()}\n\n"
                           f"📄 Detailed goal plan saved to {plan_filename}\n"
                           f"📊 Goal data saved to {goals_file}\n\n"
                           f"💡 Use update_goal_progress to track your advancement!",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def update_goal_progress(
    goal_id: int,
    progress_percentage: int,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    update_notes: Optional[str] = None,
    completed_actions: Optional[List[str]] = None,
    obstacles_faced: Optional[str] = None
) -> Command:
    """Update progress on an existing goal.
    
    Track progress, document achievements, and identify obstacles
    to maintain momentum toward goal completion.
    
    Args:
        goal_id: ID of the goal to update
        progress_percentage: Current progress as percentage (0-100)
        update_notes: Notes about recent progress or changes
        completed_actions: List of recently completed action items
        obstacles_faced: Description of any obstacles or challenges
    
    Returns:
        Command that updates goal progress and provides insights
    """
    if progress_percentage < 0 or progress_percentage > 100:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Progress percentage must be between 0 and 100.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    files = state.get("files", {})
    goals_file = "productivity_goals.json"
    
    if goals_file not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ No goals found. Create a goal first using create_goal tool.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    try:
        goals_data = json.loads(files[goals_file])
    except json.JSONDecodeError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Error reading goals data. Please recreate your goals.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Find the goal
    goal = None
    for g in goals_data.get("goals", []):
        if g["id"] == goal_id:
            goal = g
            break
    
    if not goal:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ Goal with ID {goal_id} not found.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Calculate progress change
    previous_progress = goal.get("progress_percentage", 0)
    progress_change = progress_percentage - previous_progress
    
    # Create progress update entry
    progress_update = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "previous_progress": previous_progress,
        "new_progress": progress_percentage,
        "progress_change": progress_change,
        "update_notes": update_notes,
        "completed_actions": completed_actions or [],
        "obstacles_faced": obstacles_faced
    }
    
    # Update goal
    goal["progress_percentage"] = progress_percentage
    goal["last_updated"] = datetime.now().isoformat()
    
    if "progress_updates" not in goal:
        goal["progress_updates"] = []
    goal["progress_updates"].append(progress_update)
    
    # Check if goal is completed
    if progress_percentage >= 100 and goal["status"] != "completed":
        goal["status"] = "completed"
        goal["completion_date"] = datetime.now().strftime("%Y-%m-%d")
    elif progress_percentage < 100 and goal["status"] == "completed":
        goal["status"] = "active"
        if "completion_date" in goal:
            del goal["completion_date"]
    
    # Update file
    files[goals_file] = json.dumps(goals_data, indent=2)
    
    # Generate progress insights
    target_date = datetime.strptime(goal["target_date"], "%Y-%m-%d")
    days_remaining = (target_date - datetime.now()).days
    days_elapsed = (datetime.now() - datetime.strptime(goal["created_date"], "%Y-%m-%d")).days
    total_days = (target_date - datetime.strptime(goal["created_date"], "%Y-%m-%d")).days
    
    expected_progress = (days_elapsed / total_days * 100) if total_days > 0 else 0
    progress_status = "ahead" if progress_percentage > expected_progress else "behind" if progress_percentage < expected_progress else "on track"
    
    # Create response with insights
    response = [
        f"📈 **Goal Progress Updated!**",
        f"",
        f"**Goal:** {goal['title']}",
        f"**Progress:** {previous_progress}% → {progress_percentage}% ({progress_change:+d}%)",
        f"**Target Date:** {goal['target_date']} ({days_remaining} days remaining)",
        f"**Status:** {progress_status.title()} (expected: {expected_progress:.1f}%)",
        ""
    ]
    
    if completed_actions:
        response.extend([
            "**Recent Achievements:**",
            ""
        ])
        for action in completed_actions:
            response.append(f"✅ {action}")
        response.append("")
    
    if obstacles_faced:
        response.extend([
            "**Obstacles Identified:**",
            f"⚠️ {obstacles_faced}",
            ""
        ])
    
    if update_notes:
        response.extend([
            "**Notes:**",
            update_notes,
            ""
        ])
    
    # Provide recommendations based on progress
    response.append("**Recommendations:**")
    
    if progress_percentage >= 100:
        response.append("🎉 Goal completed! Consider setting a new related goal or celebrating this achievement.")
    elif progress_status == "behind":
        response.extend([
            f"⚠️ Progress is behind schedule by {expected_progress - progress_percentage:.1f}%",
            "- Review and adjust action steps or timeline",
            "- Identify and address blocking factors",
            "- Consider breaking remaining work into smaller tasks"
        ])
    elif progress_status == "ahead":
        response.extend([
            f"🚀 Excellent! You're {progress_percentage - expected_progress:.1f}% ahead of schedule",
            "- Maintain current momentum and strategies",
            "- Consider if you can achieve the goal earlier",
            "- Use extra time to exceed original expectations"
        ])
    else:
        response.extend([
            "✅ Progress is on track with expectations",
            "- Continue current approach and strategies",
            "- Regular progress checks to maintain momentum",
            "- Stay alert for potential obstacles ahead"
        ])
    
    response_text = "\n".join(response)
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=response_text,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def track_time_spent(
    activity: str,
    duration_minutes: int,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    category: str = "work",
    productivity_level: Optional[int] = None,
    notes: Optional[str] = None,
    date: Optional[str] = None
) -> Command:
    """Track time spent on activities for productivity analysis.
    
    Record time allocation across different activities to identify
    patterns, optimize schedules, and improve time management.
    
    Args:
        activity: Description of the activity
        duration_minutes: Time spent in minutes
        category: Activity category (e.g., "work", "learning", "personal", "break")
        productivity_level: How productive you felt (1-5, where 5 is very productive)
        notes: Optional notes about the activity or session
        date: Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Command that saves time tracking data and provides insights
    """
    if duration_minutes <= 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Duration must be a positive number of minutes.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if productivity_level is not None and (productivity_level < 1 or productivity_level > 5):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Productivity level must be between 1 and 5.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get or create time tracking data
    files = state.get("files", {})
    time_file = "productivity_time_tracking.json"
    
    if time_file in files:
        try:
            time_data = json.loads(files[time_file])
        except json.JSONDecodeError:
            time_data = {"sessions": []}
    else:
        time_data = {"sessions": []}
    
    # Create time entry
    time_entry = {
        "id": len(time_data["sessions"]) + 1,
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "activity": activity,
        "category": category.lower(),
        "duration_minutes": duration_minutes,
        "productivity_level": productivity_level,
        "notes": notes
    }
    
    time_data["sessions"].append(time_entry)
    
    # Update file
    files[time_file] = json.dumps(time_data, indent=2)
    
    # Calculate daily and weekly insights
    today_sessions = [s for s in time_data["sessions"] if s["date"] == date]
    today_total = sum(s["duration_minutes"] for s in today_sessions)
    
    # Weekly total (last 7 days)
    week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    week_sessions = [
        s for s in time_data["sessions"]
        if s["date"] >= week_start
    ]
    week_total = sum(s["duration_minutes"] for s in week_sessions)
    
    # Category breakdown for today
    today_by_category = {}
    for session in today_sessions:
        cat = session["category"]
        today_by_category[cat] = today_by_category.get(cat, 0) + session["duration_minutes"]
    
    # Format duration
    def format_duration(minutes):
        hours = minutes // 60
        mins = minutes % 60
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"
    
    # Generate insights
    insights = []
    
    if productivity_level:
        # Calculate average productivity for the week
        productive_sessions = [s for s in week_sessions if s.get("productivity_level")]
        if productive_sessions:
            avg_productivity = sum(s["productivity_level"] for s in productive_sessions) / len(productive_sessions)
            insights.append(f"Average productivity (7 days): {avg_productivity:.1f}/5")
    
    # Find most productive time patterns
    if len(time_data["sessions"]) >= 5:
        # Analyze productive hours (simplified)
        insights.append("💡 Track more sessions to unlock time pattern insights")
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"⏱️ **Time Tracked Successfully!**\n\n"
                           f"**Activity:** {activity}\n"
                           f"**Duration:** {format_duration(duration_minutes)}\n"
                           f"**Category:** {category.title()}\n" +
                           (f"**Productivity Level:** {productivity_level}/5\n" if productivity_level else "") +
                           (f"**Notes:** {notes}\n" if notes else "") +
                           f"\n**Daily Summary ({date}):**\n"
                           f"- Total time tracked: {format_duration(today_total)}\n" +
                           "\n".join([f"- {cat.title()}: {format_duration(mins)}" 
                                    for cat, mins in today_by_category.items()]) +
                           f"\n\n**Weekly Total:** {format_duration(week_total)}\n" +
                           ("\n**Insights:**\n" + "\n".join([f"- {insight}" for insight in insights]) if insights else "") +
                           f"\n\n📄 Data saved to {time_file}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def analyze_productivity(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    period: str = "week",
    focus_category: Optional[str] = None
) -> Command:
    """Analyze productivity patterns and time allocation.
    
    Generate insights about time usage, productivity levels, and patterns
    to help optimize workflows and identify improvement opportunities.
    
    Args:
        period: Analysis period - "week", "month", or "all"
        focus_category: Specific category to analyze (if None, analyzes all)
    
    Returns:
        Command that generates comprehensive productivity analysis
    """
    files = state.get("files", {})
    time_file = "productivity_time_tracking.json"
    
    if time_file not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📊 No time tracking data found. Start tracking time to see analysis.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    try:
        time_data = json.loads(files[time_file])
    except json.JSONDecodeError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Error reading time tracking data. Please check the data format.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    sessions = time_data.get("sessions", [])
    if not sessions:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📊 No time tracking sessions found. Start tracking time to see analysis.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Filter sessions by period
    end_date = datetime.now()
    if period == "week":
        start_date = end_date - timedelta(days=7)
        period_name = "Last 7 Days"
    elif period == "month":
        start_date = end_date - timedelta(days=30)
        period_name = "Last 30 Days"
    else:  # "all"
        start_date = datetime.strptime(min(s["date"] for s in sessions), "%Y-%m-%d")
        period_name = "All Time"
    
    # Filter sessions to period and category
    period_sessions = [
        s for s in sessions
        if start_date <= datetime.strptime(s["date"], "%Y-%m-%d") <= end_date
    ]
    
    if focus_category:
        period_sessions = [s for s in period_sessions if s["category"] == focus_category.lower()]
    
    if not period_sessions:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"📊 No sessions found for {period_name}" + 
                               (f" in category '{focus_category}'" if focus_category else "") + ".",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Generate analysis
    analysis = [
        f"# Productivity Analysis - {period_name}",
        f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    if focus_category:
        analysis.append(f"**Category Focus:** {focus_category.title()}")
        analysis.append("")
    
    # Time allocation analysis
    total_minutes = sum(s["duration_minutes"] for s in period_sessions)
    total_sessions = len(period_sessions)
    avg_session_length = total_minutes / total_sessions if total_sessions > 0 else 0
    days_tracked = len(set(s["date"] for s in period_sessions))
    avg_daily_time = total_minutes / days_tracked if days_tracked > 0 else 0
    
    def format_duration(minutes):
        hours = minutes // 60
        mins = minutes % 60
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"
    
    analysis.extend([
        "## Summary",
        f"**Total Time Tracked:** {format_duration(total_minutes)}",
        f"**Total Sessions:** {total_sessions}",
        f"**Days with Activity:** {days_tracked}",
        f"**Average Session Length:** {format_duration(avg_session_length)}",
        f"**Average Daily Time:** {format_duration(avg_daily_time)}",
        ""
    ])
    
    # Category breakdown (if not filtering by category)
    if not focus_category:
        category_totals = {}
        for session in period_sessions:
            cat = session["category"]
            category_totals[cat] = category_totals.get(cat, 0) + session["duration_minutes"]
        
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        
        analysis.append("## Time by Category")
        for cat, minutes in sorted_categories:
            percentage = (minutes / total_minutes) * 100 if total_minutes > 0 else 0
            analysis.append(f"- **{cat.title()}:** {format_duration(minutes)} ({percentage:.1f}%)")
        analysis.append("")
    
    # Productivity level analysis
    productive_sessions = [s for s in period_sessions if s.get("productivity_level")]
    if productive_sessions:
        avg_productivity = sum(s["productivity_level"] for s in productive_sessions) / len(productive_sessions)
        
        # Productivity by category
        productivity_by_cat = {}
        for session in productive_sessions:
            cat = session["category"]
            if cat not in productivity_by_cat:
                productivity_by_cat[cat] = []
            productivity_by_cat[cat].append(session["productivity_level"])
        
        analysis.append("## Productivity Analysis")
        analysis.append(f"**Overall Average Productivity:** {avg_productivity:.1f}/5")
        analysis.append("")
        
        if not focus_category and len(productivity_by_cat) > 1:
            analysis.append("### Productivity by Category")
            for cat, levels in productivity_by_cat.items():
                cat_avg = sum(levels) / len(levels)
                analysis.append(f"- **{cat.title()}:** {cat_avg:.1f}/5 ({len(levels)} sessions)")
            analysis.append("")
        
        # Find most and least productive sessions
        most_productive = max(productive_sessions, key=lambda x: x["productivity_level"])
        least_productive = min(productive_sessions, key=lambda x: x["productivity_level"])
        
        analysis.append("### Productivity Insights")
        analysis.append(f"- **Most productive session:** {most_productive['activity']} ({most_productive['productivity_level']}/5)")
        analysis.append(f"- **Least productive session:** {least_productive['activity']} ({least_productive['productivity_level']}/5)")
    
    # Daily patterns
    daily_totals = {}
    for session in period_sessions:
        date = session["date"]
        daily_totals[date] = daily_totals.get(date, 0) + session["duration_minutes"]
    
    if daily_totals:
        analysis.append("\n## Daily Patterns")
        max_day = max(daily_totals.items(), key=lambda x: x[1])
        min_day = min(daily_totals.items(), key=lambda x: x[1])
        
        analysis.append(f"- **Most active day:** {max_day[0]} ({format_duration(max_day[1])})")
        analysis.append(f"- **Least active day:** {min_day[0]} ({format_duration(min_day[1])})")
        
        # Consistency analysis
        amounts = list(daily_totals.values())
        if len(amounts) > 1:
            avg_daily = sum(amounts) / len(amounts)
            variance = sum((x - avg_daily) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            consistency_score = max(0, 100 - (std_dev / avg_daily * 100)) if avg_daily > 0 else 0
            analysis.append(f"- **Time consistency:** {consistency_score:.1f}% (higher = more consistent)")
    
    # Recommendations
    analysis.append("\n## Recommendations")
    
    if total_minutes < 600:  # Less than 10 hours per week
        analysis.append("- 📈 **Increase tracking**: Track more activities to get better insights")
    
    if not focus_category and len(sorted_categories) > 0:
        top_category = sorted_categories[0]
        if (top_category[1] / total_minutes) > 0.6:
            analysis.append(f"- ⚖️ **Balance activities**: {top_category[0]} takes up {(top_category[1]/total_minutes)*100:.1f}% of tracked time")
    
    if productive_sessions:
        high_productivity_sessions = [s for s in productive_sessions if s["productivity_level"] >= 4]
        if high_productivity_sessions:
            # Find common patterns in high productivity sessions
            high_prod_categories = [s["category"] for s in high_productivity_sessions]
            most_common_productive_cat = max(set(high_prod_categories), key=high_prod_categories.count)
            analysis.append(f"- 🎯 **Optimize for productivity**: Your most productive category is {most_common_productive_cat}")
        
        if avg_productivity < 3:
            analysis.append("- 🔍 **Investigate low productivity**: Consider what factors contribute to lower productivity scores")
    
    if avg_session_length < 30:
        analysis.append("- 🕐 **Longer focus blocks**: Consider combining short sessions into longer focused blocks")
    elif avg_session_length > 180:
        analysis.append("- ⏸️ **Add breaks**: Very long sessions might benefit from scheduled breaks")
    
    analysis_text = "\n".join(analysis)
    
    # Save analysis
    analysis_filename = f"productivity_analysis_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[analysis_filename] = analysis_text
    
    # Create summary
    summary = f"📊 **Productivity Analysis Complete - {period_name}**\n\n"
    summary += f"**Total Time:** {format_duration(total_minutes)}\n"
    summary += f"**Sessions:** {total_sessions}\n"
    summary += f"**Average Daily:** {format_duration(avg_daily_time)}\n"
    if productive_sessions:
        summary += f"**Average Productivity:** {avg_productivity:.1f}/5\n"
    summary += f"\n📄 Detailed analysis saved to {analysis_filename}"
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=summary,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_productivity_tools():
    """Create a list of specialized productivity tools."""
    return [create_goal, update_goal_progress, track_time_spent, analyze_productivity]