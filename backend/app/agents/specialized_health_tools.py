"""
Specialized Health Tools for ReAct Health Agent
===============================================

Domain-specific tools for health, wellness, nutrition, and habit management.
These tools integrate with the deep agent system for context storage and planning.
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
def track_habit(
    habit_name: str,
    status: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    date: Optional[str] = None,
    notes: Optional[str] = None,
    value: Optional[float] = None,
    unit: Optional[str] = None
) -> Command:
    """Track a health habit for a specific date.
    
    Use this tool to record habit completion, progress, or measurements.
    Supports both boolean habits (done/not done) and quantitative habits.
    
    Args:
        habit_name: Name of the habit to track
        status: "completed", "partial", "missed", or "skipped"
        date: Date in YYYY-MM-DD format (defaults to today)
        notes: Optional notes about the habit
        value: Optional numeric value for quantitative habits
        unit: Unit for the value (e.g., "minutes", "glasses", "steps")
    
    Returns:
        Command that saves habit data and returns confirmation
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get or create habit tracking data
    files = state.get("files", {})
    habit_file = "health_habits_log.json"
    
    if habit_file in files:
        try:
            habit_data = json.loads(files[habit_file])
        except json.JSONDecodeError:
            habit_data = {"habits": {}}
    else:
        habit_data = {"habits": {}}
    
    # Initialize habit if not exists
    if habit_name not in habit_data["habits"]:
        habit_data["habits"][habit_name] = {"entries": []}
    
    # Create habit entry
    entry = {
        "date": date,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "notes": notes,
        "value": value,
        "unit": unit
    }
    
    # Remove existing entry for the same date if it exists
    habit_data["habits"][habit_name]["entries"] = [
        e for e in habit_data["habits"][habit_name]["entries"] if e["date"] != date
    ]
    
    # Add new entry
    habit_data["habits"][habit_name]["entries"].append(entry)
    
    # Sort entries by date
    habit_data["habits"][habit_name]["entries"].sort(key=lambda x: x["date"])
    
    # Update file
    files[habit_file] = json.dumps(habit_data, indent=2)
    
    # Calculate streak
    entries = habit_data["habits"][habit_name]["entries"]
    current_streak = 0
    for entry in reversed(entries):
        if entry["status"] == "completed":
            current_streak += 1
        else:
            break
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"✅ **Habit Tracked: {habit_name}**\n\n"
                           f"Date: {date}\n"
                           f"Status: {status}\n" +
                           (f"Value: {value} {unit}\n" if value else "") +
                           (f"Notes: {notes}\n" if notes else "") +
                           f"Current Streak: {current_streak} days\n\n"
                           f"💡 Habit data saved to {habit_file}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def analyze_habits(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    habit_name: Optional[str] = None,
    days: int = 30
) -> Command:
    """Analyze habit tracking data and generate insights.
    
    Provides analysis of habit completion rates, streaks, patterns,
    and recommendations for improvement.
    
    Args:
        habit_name: Specific habit to analyze (if None, analyzes all)
        days: Number of days to analyze (default: 30)
    
    Returns:
        Command that generates habit analysis and saves detailed report
    """
    files = state.get("files", {})
    habit_file = "health_habits_log.json"
    
    if habit_file not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📊 No habit data found. Start tracking habits to see analysis.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    try:
        habit_data = json.loads(files[habit_file])
    except json.JSONDecodeError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Error reading habit data. Please check the data format.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Filter habits
    habits_to_analyze = {}
    if habit_name:
        if habit_name in habit_data["habits"]:
            habits_to_analyze[habit_name] = habit_data["habits"][habit_name]
        else:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"❌ Habit '{habit_name}' not found in tracking data.",
                            tool_call_id=tool_call_id
                        )
                    ]
                }
            )
    else:
        habits_to_analyze = habit_data["habits"]
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    analysis = [
        f"# Habit Analysis Report",
        f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary"
    ]
    
    total_habits = len(habits_to_analyze)
    analysis.append(f"**Habits Tracked:** {total_habits}")
    
    # Analyze each habit
    habit_stats = []
    
    for habit, data in habits_to_analyze.items():
        entries = data["entries"]
        
        # Filter entries to date range
        relevant_entries = [
            entry for entry in entries 
            if start_date <= datetime.fromisoformat(entry["date"]) <= end_date
        ]
        
        if not relevant_entries:
            continue
        
        # Calculate statistics
        completed_days = len([e for e in relevant_entries if e["status"] == "completed"])
        total_tracked_days = len(relevant_entries)
        completion_rate = (completed_days / days) * 100 if days > 0 else 0
        
        # Calculate current streak
        current_streak = 0
        for entry in reversed(entries):
            if entry["status"] == "completed":
                current_streak += 1
            else:
                break
        
        # Calculate longest streak in period
        longest_streak = 0
        temp_streak = 0
        for entry in relevant_entries:
            if entry["status"] == "completed":
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 0
        
        habit_stats.append({
            "name": habit,
            "completion_rate": completion_rate,
            "completed_days": completed_days,
            "tracked_days": total_tracked_days,
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "recent_entries": relevant_entries[-7:]  # Last 7 entries
        })
    
    # Sort by completion rate
    habit_stats.sort(key=lambda x: x["completion_rate"], reverse=True)
    
    # Add overall statistics
    if habit_stats:
        avg_completion = sum(h["completion_rate"] for h in habit_stats) / len(habit_stats)
        analysis.append(f"**Average Completion Rate:** {avg_completion:.1f}%")
        
        best_habit = habit_stats[0]
        analysis.append(f"**Best Performing Habit:** {best_habit['name']} ({best_habit['completion_rate']:.1f}%)")
        
        if len(habit_stats) > 1:
            worst_habit = habit_stats[-1]
            analysis.append(f"**Needs Attention:** {worst_habit['name']} ({worst_habit['completion_rate']:.1f}%)")
    
    analysis.append("\n## Detailed Analysis")
    
    for stat in habit_stats:
        analysis.append(f"\n### {stat['name']}")
        analysis.append(f"- **Completion Rate:** {stat['completion_rate']:.1f}% ({stat['completed_days']}/{days} days)")
        analysis.append(f"- **Current Streak:** {stat['current_streak']} days")
        analysis.append(f"- **Longest Streak (period):** {stat['longest_streak']} days")
        analysis.append(f"- **Days Tracked:** {stat['tracked_days']}")
        
        # Pattern analysis
        recent_statuses = [e["status"] for e in stat["recent_entries"]]
        if recent_statuses:
            recent_completion = len([s for s in recent_statuses if s == "completed"])
            recent_rate = (recent_completion / len(recent_statuses)) * 100
            analysis.append(f"- **Recent Trend (7 days):** {recent_rate:.1f}% completion")
    
    # Recommendations
    analysis.append("\n## Recommendations")
    
    if habit_stats:
        # Find habits that need improvement
        low_performers = [h for h in habit_stats if h["completion_rate"] < 70]
        if low_performers:
            analysis.append("\n### Habits to Focus On:")
            for habit in low_performers[:3]:  # Top 3 that need attention
                analysis.append(f"- **{habit['name']}**: Consider reviewing your approach or reducing the target")
        
        # Find successful patterns
        high_performers = [h for h in habit_stats if h["completion_rate"] >= 80]
        if high_performers:
            analysis.append("\n### Success Patterns:")
            analysis.append("Apply strategies from your successful habits:")
            for habit in high_performers[:3]:
                analysis.append(f"- **{habit['name']}**: Excellent consistency - maintain this approach")
        
        # General recommendations
        analysis.append("\n### General Tips:")
        analysis.append("- Focus on 2-3 habits at a time for better success rates")
        analysis.append("- Review and adjust habits weekly based on your lifestyle")
        analysis.append("- Celebrate streaks and learn from missed days")
        analysis.append("- Consider habit stacking - linking new habits to existing ones")
    
    analysis_text = "\n".join(analysis)
    
    # Save detailed analysis
    analysis_filename = f"habit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[analysis_filename] = analysis_text
    
    # Create summary for response
    summary = f"📊 **Habit Analysis Complete**\n\n"
    if habit_stats:
        summary += f"**Period:** {days} days\n"
        summary += f"**Habits Analyzed:** {len(habit_stats)}\n"
        summary += f"**Average Completion:** {sum(h['completion_rate'] for h in habit_stats) / len(habit_stats):.1f}%\n"
        summary += f"**Best Habit:** {habit_stats[0]['name']} ({habit_stats[0]['completion_rate']:.1f}%)\n\n"
        summary += f"📄 Detailed analysis saved to {analysis_filename}"
    else:
        summary += "No habit data found for the specified period."
    
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


@tool(parse_docstring=True)
def create_meal_plan(
    duration_days: int,
    dietary_preferences: List[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    health_goals: Optional[List[str]] = None,
    restrictions: Optional[List[str]] = None,
    budget_level: str = "medium"
) -> Command:
    """Create a personalized meal plan based on preferences and goals.
    
    Generates a structured meal plan with recipes, shopping lists,
    and nutritional guidance tailored to user preferences.
    
    Args:
        duration_days: Number of days for the meal plan (1-14)
        dietary_preferences: List of dietary preferences (e.g., "vegetarian", "mediterranean", "low-carb")
        health_goals: Optional health goals (e.g., "weight_loss", "muscle_gain", "energy_boost")
        restrictions: Optional dietary restrictions (e.g., "gluten_free", "dairy_free", "nut_free")  
        budget_level: Budget level - "low", "medium", "high"
    
    Returns:
        Command that generates comprehensive meal plan and saves to file
    """
    if duration_days < 1 or duration_days > 14:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Duration must be between 1 and 14 days.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Generate meal plan structure
    meal_plan = [
        f"# {duration_days}-Day Meal Plan",
        f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dietary Preferences:** {', '.join(dietary_preferences)}",
        f"**Budget Level:** {budget_level}",
        ""
    ]
    
    if health_goals:
        meal_plan.append(f"**Health Goals:** {', '.join(health_goals)}")
    
    if restrictions:
        meal_plan.append(f"**Dietary Restrictions:** {', '.join(restrictions)}")
    
    meal_plan.append("")
    
    # Meal categories based on preferences
    meal_categories = {
        "breakfast": ["oatmeal", "smoothie bowl", "avocado toast", "yogurt parfait", "eggs"],
        "lunch": ["salad", "grain bowl", "soup", "wrap", "stir-fry"],
        "dinner": ["grilled protein", "pasta", "curry", "roasted vegetables", "casserole"],
        "snacks": ["nuts", "fruit", "vegetables with hummus", "yogurt", "trail mix"]
    }
    
    # Adjust based on dietary preferences
    if "vegetarian" in dietary_preferences:
        meal_categories["lunch"].extend(["lentil salad", "veggie burger", "quinoa bowl"])
        meal_categories["dinner"].extend(["bean curry", "vegetable pasta", "stuffed peppers"])
    
    if "mediterranean" in dietary_preferences:
        meal_categories["lunch"].extend(["greek salad", "mediterranean wrap"])
        meal_categories["dinner"].extend(["fish with herbs", "mediterranean bowl"])
        meal_categories["snacks"].extend(["olives", "mediterranean nuts"])
    
    if "low-carb" in dietary_preferences:
        meal_categories["breakfast"] = ["eggs", "greek yogurt", "avocado", "protein smoothie"]
        meal_categories["lunch"] = ["salad", "lettuce wraps", "zucchini noodles"]
        meal_categories["dinner"] = ["grilled protein", "cauliflower rice", "roasted vegetables"]
    
    # Generate daily meal plans
    meal_plan.append("## Daily Meal Plans")
    
    shopping_list = {"proteins": set(), "vegetables": set(), "grains": set(), "other": set()}
    
    for day in range(1, duration_days + 1):
        date = (datetime.now() + timedelta(days=day-1)).strftime("%A, %B %d")
        meal_plan.append(f"\n### Day {day} - {date}")
        
        # Generate meals for the day
        for meal_type in ["breakfast", "lunch", "dinner"]:
            import random
            meal_options = meal_categories[meal_type]
            selected_meal = random.choice(meal_options)
            
            meal_plan.append(f"\n**{meal_type.title()}:** {selected_meal.title()}")
            
            # Add basic ingredients to shopping list
            if "salad" in selected_meal:
                shopping_list["vegetables"].update(["lettuce", "tomatoes", "cucumbers"])
            elif "eggs" in selected_meal:
                shopping_list["proteins"].add("eggs")
            elif "yogurt" in selected_meal:
                shopping_list["other"].add("yogurt")
            # Add more ingredient mapping as needed
        
        # Add snacks
        snack_options = meal_categories["snacks"]
        import random
        selected_snacks = random.sample(snack_options, min(2, len(snack_options)))
        meal_plan.append(f"\n**Snacks:** {', '.join(selected_snacks)}")
    
    # Add shopping list
    meal_plan.append("\n## Shopping List")
    meal_plan.append("\n### Proteins")
    for item in sorted(shopping_list["proteins"]):
        meal_plan.append(f"- {item}")
    
    meal_plan.append("\n### Vegetables & Fruits")
    for item in sorted(shopping_list["vegetables"]):
        meal_plan.append(f"- {item}")
    
    meal_plan.append("\n### Grains & Carbs")
    for item in sorted(shopping_list["grains"]):
        meal_plan.append(f"- {item}")
    
    meal_plan.append("\n### Other Items")
    for item in sorted(shopping_list["other"]):
        meal_plan.append(f"- {item}")
    
    # Add nutritional guidance
    meal_plan.append("\n## Nutritional Guidelines")
    
    if health_goals:
        if "weight_loss" in health_goals:
            meal_plan.append("- Focus on portion control and high-fiber foods")
            meal_plan.append("- Include lean proteins with each meal")
            meal_plan.append("- Stay hydrated with 8+ glasses of water daily")
        
        if "muscle_gain" in health_goals:
            meal_plan.append("- Ensure adequate protein intake (0.8-1g per lb body weight)")
            meal_plan.append("- Include post-workout protein within 30 minutes")
            meal_plan.append("- Don't skip meals - consistent nutrition is key")
        
        if "energy_boost" in health_goals:
            meal_plan.append("- Balance complex carbs with protein")
            meal_plan.append("- Avoid sugar spikes with whole foods")
            meal_plan.append("- Consider meal timing around energy needs")
    
    # Add meal prep tips
    meal_plan.append("\n## Meal Prep Tips")
    meal_plan.append("- Prep vegetables and proteins in batches on weekends")
    meal_plan.append("- Cook grains in bulk and store in refrigerator")
    meal_plan.append("- Prepare grab-and-go snacks in portions")
    meal_plan.append("- Use glass containers for better food storage")
    
    meal_plan_text = "\n".join(meal_plan)
    
    # Save meal plan
    files = state.get("files", {})
    filename = f"meal_plan_{duration_days}day_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[filename] = meal_plan_text
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🍽️ **Meal Plan Created Successfully!**\n\n"
                           f"**Duration:** {duration_days} days\n"
                           f"**Dietary Style:** {', '.join(dietary_preferences)}\n"
                           f"**Budget Level:** {budget_level}\n\n"
                           f"📄 Complete meal plan saved to {filename}\n\n"
                           f"💡 Your plan includes daily meals, shopping list, and nutritional guidance. "
                           f"Use read_file to view the full plan when ready to start!",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def wellness_check_in(
    mood: str,
    energy_level: int,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    sleep_hours: Optional[float] = None,
    stress_level: Optional[int] = None,
    notes: Optional[str] = None,
    date: Optional[str] = None
) -> Command:
    """Record daily wellness check-in data.
    
    Track mood, energy, sleep, stress, and other wellness indicators
    to identify patterns and support health goals.
    
    Args:
        mood: Current mood - "excellent", "good", "neutral", "low", "poor"
        energy_level: Energy level from 1-10 (1=exhausted, 10=very energetic)
        sleep_hours: Hours of sleep last night
        stress_level: Stress level from 1-10 (1=no stress, 10=very stressed)
        notes: Optional notes about how you're feeling
        date: Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Command that saves wellness data and provides insights
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    if energy_level < 1 or energy_level > 10:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Energy level must be between 1 and 10.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if stress_level is not None and (stress_level < 1 or stress_level > 10):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Stress level must be between 1 and 10.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Get or create wellness data
    files = state.get("files", {})
    wellness_file = "wellness_checkins.json"
    
    if wellness_file in files:
        try:
            wellness_data = json.loads(files[wellness_file])
        except json.JSONDecodeError:
            wellness_data = {"checkins": []}
    else:
        wellness_data = {"checkins": []}
    
    # Remove existing entry for the same date
    wellness_data["checkins"] = [
        entry for entry in wellness_data["checkins"] if entry["date"] != date
    ]
    
    # Create new entry
    entry = {
        "date": date,
        "mood": mood,
        "energy_level": energy_level,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    
    wellness_data["checkins"].append(entry)
    
    # Sort by date
    wellness_data["checkins"].sort(key=lambda x: x["date"])
    
    # Update file
    files[wellness_file] = json.dumps(wellness_data, indent=2)
    
    # Generate insights from recent data
    recent_entries = wellness_data["checkins"][-7:]  # Last 7 days
    insights = []
    
    if len(recent_entries) >= 3:
        avg_energy = sum(e["energy_level"] for e in recent_entries) / len(recent_entries)
        insights.append(f"Average energy (7 days): {avg_energy:.1f}/10")
        
        if stress_level is not None:
            stress_entries = [e for e in recent_entries if e["stress_level"] is not None]
            if stress_entries:
                avg_stress = sum(e["stress_level"] for e in stress_entries) / len(stress_entries)
                insights.append(f"Average stress (7 days): {avg_stress:.1f}/10")
        
        if sleep_hours is not None:
            sleep_entries = [e for e in recent_entries if e["sleep_hours"] is not None]
            if sleep_entries:
                avg_sleep = sum(e["sleep_hours"] for e in sleep_entries) / len(sleep_entries)
                insights.append(f"Average sleep (7 days): {avg_sleep:.1f} hours")
                
                if avg_sleep < 7:
                    insights.append("💡 Consider prioritizing more sleep for better energy")
                elif avg_sleep > 9:
                    insights.append("💡 Monitor if extra sleep affects your daily energy")
    
    insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "Not enough data for insights yet."
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"✅ **Wellness Check-in Recorded**\n\n"
                           f"**Date:** {date}\n"
                           f"**Mood:** {mood}\n"
                           f"**Energy:** {energy_level}/10\n" +
                           (f"**Sleep:** {sleep_hours} hours\n" if sleep_hours else "") +
                           (f"**Stress:** {stress_level}/10\n" if stress_level else "") +
                           (f"**Notes:** {notes}\n" if notes else "") +
                           f"\n**Recent Insights:**\n{insights_text}\n\n"
                           f"📄 Data saved to {wellness_file}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_health_tools():
    """Create a list of specialized health tools."""
    return [track_habit, analyze_habits, create_meal_plan, wellness_check_in]