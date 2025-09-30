"""
Specialized Finance Tools for ReAct Finance Agent
================================================

Domain-specific tools for personal finance management, budgeting,
expense tracking, and financial planning with deep agent integration.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_HALF_UP
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState


@tool(parse_docstring=True)
def track_expense(
    amount: float,
    category: str,
    description: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    date: Optional[str] = None,
    payment_method: Optional[str] = None,
    tags: Optional[List[str]] = None,
    is_recurring: bool = False
) -> Command:
    """Track an expense with categorization and details.
    
    Record expenses with detailed categorization for budget tracking
    and financial analysis. Supports one-time and recurring expenses.
    
    Args:
        amount: Expense amount (positive number)
        category: Expense category (e.g., "food", "transportation", "housing", "entertainment")
        description: Description of the expense
        date: Date in YYYY-MM-DD format (defaults to today)
        payment_method: How payment was made (e.g., "cash", "credit", "debit", "transfer")
        tags: Optional tags for additional categorization
        is_recurring: Whether this is a recurring expense
    
    Returns:
        Command that saves expense data and provides spending insights
    """
    if amount <= 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Amount must be a positive number.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Date must be in YYYY-MM-DD format.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Get or create expense data
    files = state.get("files", {})
    expense_file = "finance_expenses.json"
    
    if expense_file in files:
        try:
            expense_data = json.loads(files[expense_file])
        except json.JSONDecodeError:
            expense_data = {"expenses": [], "categories": set()}
    else:
        expense_data = {"expenses": [], "categories": set()}
    
    # Ensure categories is a set for JSON serialization
    if isinstance(expense_data.get("categories"), list):
        expense_data["categories"] = set(expense_data["categories"])
    elif "categories" not in expense_data:
        expense_data["categories"] = set()
    
    # Create expense entry
    expense_entry = {
        "id": len(expense_data["expenses"]) + 1,
        "date": date,
        "amount": round(amount, 2),
        "category": category.lower(),
        "description": description,
        "payment_method": payment_method,
        "tags": tags or [],
        "is_recurring": is_recurring,
        "timestamp": datetime.now().isoformat()
    }
    
    expense_data["expenses"].append(expense_entry)
    expense_data["categories"].add(category.lower())
    
    # Convert set to list for JSON serialization
    expense_data["categories"] = list(expense_data["categories"])
    
    # Update file
    files[expense_file] = json.dumps(expense_data, indent=2)
    
    # Calculate monthly spending so far
    current_month = datetime.now().strftime("%Y-%m")
    monthly_expenses = [
        e for e in expense_data["expenses"] 
        if e["date"].startswith(current_month)
    ]
    monthly_total = sum(e["amount"] for e in monthly_expenses)
    
    # Calculate category spending this month
    category_total = sum(
        e["amount"] for e in monthly_expenses 
        if e["category"] == category.lower()
    )
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"💰 **Expense Tracked Successfully**\n\n"
                           f"**Amount:** ${amount:.2f}\n"
                           f"**Category:** {category}\n"
                           f"**Description:** {description}\n"
                           f"**Date:** {date}\n" +
                           (f"**Payment Method:** {payment_method}\n" if payment_method else "") +
                           (f"**Tags:** {', '.join(tags)}\n" if tags else "") +
                           f"\n**Monthly Summary:**\n"
                           f"- Total spent this month: ${monthly_total:.2f}\n"
                           f"- {category} category this month: ${category_total:.2f}\n\n"
                           f"📄 Data saved to {expense_file}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def analyze_spending(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    period: str = "month",
    category: Optional[str] = None
) -> Command:
    """Analyze spending patterns and generate financial insights.
    
    Provides detailed analysis of spending by category, trends over time,
    and recommendations for budget optimization.
    
    Args:
        period: Analysis period - "week", "month", "quarter", "year"
        category: Specific category to analyze (if None, analyzes all)
    
    Returns:
        Command that generates spending analysis and saves detailed report
    """
    files = state.get("files", {})
    expense_file = "finance_expenses.json"
    
    if expense_file not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📊 No expense data found. Start tracking expenses to see analysis.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    try:
        expense_data = json.loads(files[expense_file])
    except json.JSONDecodeError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Error reading expense data. Please check the data format.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    expenses = expense_data.get("expenses", [])
    if not expenses:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📊 No expenses recorded yet. Add some expenses to see analysis.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Calculate date range based on period
    end_date = datetime.now()
    if period == "week":
        start_date = end_date - timedelta(days=7)
        period_name = "Last 7 Days"
    elif period == "month":
        start_date = end_date.replace(day=1)
        period_name = "This Month"
    elif period == "quarter":
        start_date = end_date - timedelta(days=90)
        period_name = "Last 90 Days"
    elif period == "year":
        start_date = end_date.replace(month=1, day=1)
        period_name = "This Year"
    else:
        start_date = end_date - timedelta(days=30)
        period_name = "Last 30 Days"
    
    # Filter expenses to period
    period_expenses = [
        e for e in expenses
        if start_date <= datetime.strptime(e["date"], "%Y-%m-%d") <= end_date
    ]
    
    if category:
        period_expenses = [e for e in period_expenses if e["category"] == category.lower()]
    
    if not period_expenses:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"📊 No expenses found for {period_name}" + 
                               (f" in category '{category}'" if category else "") + ".",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Generate analysis
    analysis = [
        f"# Spending Analysis Report",
        f"**Period:** {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    if category:
        analysis.append(f"**Category Focus:** {category}")
        analysis.append("")
    
    # Overall statistics
    total_amount = sum(e["amount"] for e in period_expenses)
    avg_per_day = total_amount / ((end_date - start_date).days + 1)
    transaction_count = len(period_expenses)
    avg_per_transaction = total_amount / transaction_count if transaction_count > 0 else 0
    
    analysis.extend([
        "## Summary",
        f"**Total Spent:** ${total_amount:.2f}",
        f"**Number of Transactions:** {transaction_count}",
        f"**Average per Day:** ${avg_per_day:.2f}",
        f"**Average per Transaction:** ${avg_per_transaction:.2f}",
        ""
    ])
    
    # Category breakdown (if not filtering by category)
    if not category:
        category_totals = {}
        for expense in period_expenses:
            cat = expense["category"]
            category_totals[cat] = category_totals.get(cat, 0) + expense["amount"]
        
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        
        analysis.append("## Spending by Category")
        for cat, amount in sorted_categories:
            percentage = (amount / total_amount) * 100 if total_amount > 0 else 0
            analysis.append(f"- **{cat.title()}:** ${amount:.2f} ({percentage:.1f}%)")
        analysis.append("")
    
    # Daily spending trend
    daily_totals = {}
    for expense in period_expenses:
        date = expense["date"]
        daily_totals[date] = daily_totals.get(date, 0) + expense["amount"]
    
    analysis.append("## Daily Spending Trend")
    
    # Find highest and lowest spending days
    if daily_totals:
        max_day = max(daily_totals.items(), key=lambda x: x[1])
        min_day = min(daily_totals.items(), key=lambda x: x[1])
        
        analysis.append(f"- **Highest spending day:** {max_day[0]} (${max_day[1]:.2f})")
        analysis.append(f"- **Lowest spending day:** {min_day[0]} (${min_day[1]:.2f})")
        
        # Calculate spending consistency
        amounts = list(daily_totals.values())
        if len(amounts) > 1:
            avg_daily = sum(amounts) / len(amounts)
            variance = sum((x - avg_daily) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            consistency_score = max(0, 100 - (std_dev / avg_daily * 100)) if avg_daily > 0 else 0
            analysis.append(f"- **Spending consistency:** {consistency_score:.1f}% (100% = very consistent)")
    
    analysis.append("")
    
    # Payment method analysis
    payment_methods = {}
    for expense in period_expenses:
        method = expense.get("payment_method", "unknown")
        payment_methods[method] = payment_methods.get(method, 0) + expense["amount"]
    
    if any(method != "unknown" for method in payment_methods.keys()):
        analysis.append("## Payment Methods")
        for method, amount in sorted(payment_methods.items(), key=lambda x: x[1], reverse=True):
            if method != "unknown":
                percentage = (amount / total_amount) * 100 if total_amount > 0 else 0
                analysis.append(f"- **{method.title()}:** ${amount:.2f} ({percentage:.1f}%)")
        analysis.append("")
    
    # Top expenses
    top_expenses = sorted(period_expenses, key=lambda x: x["amount"], reverse=True)[:5]
    analysis.append("## Top 5 Expenses")
    for i, expense in enumerate(top_expenses, 1):
        analysis.append(f"{i}. **${expense['amount']:.2f}** - {expense['description']} ({expense['category']}) - {expense['date']}")
    analysis.append("")
    
    # Recommendations
    analysis.append("## Recommendations")
    
    if not category and len(sorted_categories) > 0:
        top_category = sorted_categories[0]
        top_category_pct = (top_category[1] / total_amount) * 100
        
        if top_category_pct > 40:
            analysis.append(f"- **High concentration in {top_category[0]}**: {top_category_pct:.1f}% of spending. Consider if this aligns with your priorities.")
        
        if len(sorted_categories) >= 3:
            bottom_categories = sorted_categories[-3:]
            small_categories = [cat for cat, amount in bottom_categories if (amount/total_amount)*100 < 5]
            if len(small_categories) >= 2:
                analysis.append(f"- **Small categories**: Consider consolidating tracking for categories under 5% of spending.")
    
    # Spending rate analysis
    if period == "month":
        days_in_month = (datetime.now().replace(month=datetime.now().month+1, day=1) - timedelta(days=1)).day
        days_passed = datetime.now().day
        projected_monthly = (total_amount / days_passed) * days_in_month
        
        analysis.append(f"- **Projected monthly total**: ${projected_monthly:.2f} based on current rate")
    
    # Budget suggestions
    if total_amount > 0:
        analysis.append("- **Budget allocation suggestion**: 50% needs, 30% wants, 20% savings")
        analysis.append("- **Track regularly**: Weekly reviews help maintain spending awareness")
        analysis.append("- **Set category limits**: Consider setting monthly limits for top spending categories")
    
    analysis_text = "\n".join(analysis)
    
    # Save detailed analysis
    analysis_filename = f"spending_analysis_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[analysis_filename] = analysis_text
    
    # Create summary for response
    summary = f"📊 **Spending Analysis Complete**\n\n"
    summary += f"**Period:** {period_name}\n"
    summary += f"**Total Spent:** ${total_amount:.2f}\n"
    summary += f"**Transactions:** {transaction_count}\n"
    summary += f"**Daily Average:** ${avg_per_day:.2f}\n\n"
    if not category and sorted_categories:
        summary += f"**Top Category:** {sorted_categories[0][0].title()} (${sorted_categories[0][1]:.2f})\n\n"
    summary += f"📄 Detailed analysis saved to {analysis_filename}"
    
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
def create_budget(
    monthly_income: float,
    budget_categories: Dict[str, float],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    budget_name: str = "Monthly Budget",
    savings_goal_percentage: float = 20.0
) -> Command:
    """Create a monthly budget with category allocations.
    
    Set up a comprehensive budget with income, expense categories,
    and savings goals for effective financial planning.
    
    Args:
        monthly_income: Total monthly income after taxes
        budget_categories: Dictionary of category names and their budget amounts
        budget_name: Name for this budget (default: "Monthly Budget")
        savings_goal_percentage: Percentage of income to save (default: 20%)
    
    Returns:
        Command that creates budget plan and saves to file
    """
    if monthly_income <= 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Monthly income must be a positive number.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if savings_goal_percentage < 0 or savings_goal_percentage > 100:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Savings goal percentage must be between 0 and 100.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Calculate totals and percentages
    total_budget = sum(budget_categories.values())
    savings_amount = monthly_income * (savings_goal_percentage / 100)
    available_for_expenses = monthly_income - savings_amount
    
    # Generate budget plan
    budget = [
        f"# {budget_name}",
        f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Monthly Income:** ${monthly_income:.2f}",
        f"**Savings Goal:** {savings_goal_percentage}% (${savings_amount:.2f})",
        f"**Available for Expenses:** ${available_for_expenses:.2f}",
        "",
        "## Budget Allocation"
    ]
    
    # Sort categories by amount (highest first)
    sorted_categories = sorted(budget_categories.items(), key=lambda x: x[1], reverse=True)
    
    for category, amount in sorted_categories:
        percentage = (amount / monthly_income) * 100 if monthly_income > 0 else 0
        budget.append(f"- **{category.title()}:** ${amount:.2f} ({percentage:.1f}% of income)")
    
    budget.append(f"\n**Total Budgeted:** ${total_budget:.2f}")
    budget.append(f"**Plus Savings:** ${savings_amount:.2f}")
    budget.append(f"**Total Allocated:** ${total_budget + savings_amount:.2f}")
    
    # Budget analysis
    remaining = monthly_income - total_budget - savings_amount
    budget.append(f"**Remaining/Deficit:** ${remaining:.2f}")
    
    budget.append("\n## Budget Analysis")
    
    if remaining > 0:
        budget.append(f"✅ **Budget Balanced** with ${remaining:.2f} buffer")
        budget.append("- Consider allocating extra funds to emergency savings or debt payoff")
    elif remaining < 0:
        budget.append(f"⚠️ **Budget Deficit** of ${abs(remaining):.2f}")
        budget.append("- Review and reduce category allocations or increase income")
        budget.append("- Consider reducing savings goal temporarily if needed")
    else:
        budget.append("✅ **Budget Perfectly Balanced**")
    
    # Budget recommendations
    budget.append("\n## Recommendations")
    
    # Check for 50/30/20 rule compliance
    needs_percentage = 0
    wants_percentage = 0
    
    # Common needs categories
    needs_categories = ["housing", "utilities", "food", "transportation", "insurance", "minimum_debt_payments"]
    wants_categories = ["entertainment", "dining_out", "shopping", "subscriptions", "hobbies"]
    
    for category, amount in budget_categories.items():
        cat_lower = category.lower()
        percentage = (amount / monthly_income) * 100
        
        if any(need in cat_lower for need in needs_categories):
            needs_percentage += percentage
        elif any(want in cat_lower for want in wants_categories):
            wants_percentage += percentage
    
    budget.append(f"- **Needs allocation:** {needs_percentage:.1f}% (recommended: ~50%)")
    budget.append(f"- **Wants allocation:** {wants_percentage:.1f}% (recommended: ~30%)")
    budget.append(f"- **Savings allocation:** {savings_goal_percentage:.1f}% (recommended: ~20%)")
    
    if needs_percentage > 60:
        budget.append("- ⚠️ High needs spending - look for cost reduction opportunities")
    if wants_percentage > 40:
        budget.append("- ⚠️ High wants spending - consider reducing discretionary expenses")
    if savings_goal_percentage < 10:
        budget.append("- 💡 Try to increase savings rate for better financial security")
    
    # Monthly tracking template
    budget.append("\n## Monthly Tracking Template")
    budget.append("\n| Category | Budgeted | Actual | Difference | % Used |")
    budget.append("|----------|----------|--------|------------|--------|")
    
    for category, amount in sorted_categories:
        budget.append(f"| {category.title()} | ${amount:.2f} | $0.00 | $0.00 | 0% |")
    
    budget.append(f"| **Savings** | ${savings_amount:.2f} | $0.00 | $0.00 | 0% |")
    budget.append(f"| **TOTAL** | ${total_budget + savings_amount:.2f} | $0.00 | $0.00 | 0% |")
    
    # Tips for success
    budget.append("\n## Budget Success Tips")
    budget.append("1. **Track weekly**: Regular check-ins prevent overspending")
    budget.append("2. **Use envelope method**: Allocate cash for discretionary categories")
    budget.append("3. **Automate savings**: Set up automatic transfers to savings accounts")
    budget.append("4. **Review monthly**: Adjust budget based on actual spending patterns")
    budget.append("5. **Plan for irregulars**: Include annual expenses divided by 12")
    budget.append("6. **Emergency fund**: Aim for 3-6 months of expenses in emergency savings")
    
    budget_text = "\n".join(budget)
    
    # Save budget
    files = state.get("files", {})
    budget_filename = f"budget_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[budget_filename] = budget_text
    
    # Also save as structured data for tracking
    budget_data = {
        "name": budget_name,
        "created_date": datetime.now().isoformat(),
        "monthly_income": monthly_income,
        "savings_goal_percentage": savings_goal_percentage,
        "savings_amount": savings_amount,
        "categories": budget_categories,
        "total_budget": total_budget,
        "remaining": remaining
    }
    
    budget_data_file = "current_budget.json"
    files[budget_data_file] = json.dumps(budget_data, indent=2)
    
    # Calculate if budget is realistic
    status_emoji = "✅" if remaining >= 0 else "⚠️"
    status_text = "Balanced" if remaining >= 0 else f"Needs ${abs(remaining):.2f} adjustment"
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"💰 **Budget Created Successfully!**\n\n"
                           f"**Budget Name:** {budget_name}\n"
                           f"**Monthly Income:** ${monthly_income:.2f}\n"
                           f"**Total Allocated:** ${total_budget + savings_amount:.2f}\n"
                           f"**Status:** {status_emoji} {status_text}\n"
                           f"**Savings Goal:** ${savings_amount:.2f} ({savings_goal_percentage}%)\n\n"
                           f"📄 Complete budget plan saved to {budget_filename}\n"
                           f"📊 Budget data saved to {budget_data_file}\n\n"
                           f"💡 Use track_expense to monitor spending against your budget!",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def budget_progress_check(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    month: Optional[str] = None
) -> Command:
    """Check progress against current budget for the specified month.
    
    Compare actual spending against budget allocations and provide
    insights on budget performance and recommendations.
    
    Args:
        month: Month to check in YYYY-MM format (defaults to current month)
    
    Returns:
        Command that generates budget progress report
    """
    if not month:
        month = datetime.now().strftime("%Y-%m")
    
    # Validate month format
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Month must be in YYYY-MM format (e.g., 2024-03).",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    files = state.get("files", {})
    
    # Load budget
    budget_file = "current_budget.json"
    if budget_file not in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ No budget found. Create a budget first using create_budget tool.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    try:
        budget_data = json.loads(files[budget_file])
    except json.JSONDecodeError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Error reading budget data. Please recreate your budget.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Load expenses
    expense_file = "finance_expenses.json"
    if expense_file not in files:
        expense_data = {"expenses": []}
    else:
        try:
            expense_data = json.loads(files[expense_file])
        except json.JSONDecodeError:
            expense_data = {"expenses": []}
    
    # Filter expenses to the specified month
    monthly_expenses = [
        e for e in expense_data.get("expenses", [])
        if e["date"].startswith(month)
    ]
    
    # Calculate actual spending by category
    actual_spending = {}
    for expense in monthly_expenses:
        category = expense["category"]
        actual_spending[category] = actual_spending.get(category, 0) + expense["amount"]
    
    # Generate progress report
    month_name = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
    
    report = [
        f"# Budget Progress Report - {month_name}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Budget:** {budget_data.get('name', 'Monthly Budget')}",
        "",
        "## Category Performance"
    ]
    
    budget_categories = budget_data.get("categories", {})
    total_budgeted = sum(budget_categories.values())
    total_spent = sum(actual_spending.values())
    
    # Calculate performance for each category
    category_performance = []
    
    for category, budgeted in budget_categories.items():
        spent = actual_spending.get(category, 0)
        remaining = budgeted - spent
        percentage_used = (spent / budgeted * 100) if budgeted > 0 else 0
        
        # Status indicators
        if percentage_used <= 75:
            status = "✅ On Track"
        elif percentage_used <= 95:
            status = "⚠️ Close to Limit"  
        elif percentage_used <= 100:
            status = "🔶 At Limit"
        else:
            status = "🔴 Over Budget"
        
        category_performance.append({
            "category": category,
            "budgeted": budgeted,
            "spent": spent,
            "remaining": remaining,
            "percentage_used": percentage_used,
            "status": status
        })
    
    # Sort by percentage used (highest first)
    category_performance.sort(key=lambda x: x["percentage_used"], reverse=True)
    
    for perf in category_performance:
        report.append(f"\n### {perf['category'].title()}")
        report.append(f"- **Budgeted:** ${perf['budgeted']:.2f}")
        report.append(f"- **Spent:** ${perf['spent']:.2f}")
        report.append(f"- **Remaining:** ${perf['remaining']:.2f}")
        report.append(f"- **Used:** {perf['percentage_used']:.1f}%")
        report.append(f"- **Status:** {perf['status']}")
    
    # Check for spending in non-budgeted categories
    non_budgeted_spending = {}
    for category, amount in actual_spending.items():
        if category not in budget_categories:
            non_budgeted_spending[category] = amount
    
    if non_budgeted_spending:
        report.append("\n## Non-Budgeted Spending")
        for category, amount in sorted(non_budgeted_spending.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{category.title()}:** ${amount:.2f}")
    
    # Overall summary
    overall_percentage = (total_spent / total_budgeted * 100) if total_budgeted > 0 else 0
    savings_target = budget_data.get("savings_amount", 0)
    
    report.append("\n## Overall Summary")
    report.append(f"- **Total Budgeted:** ${total_budgeted:.2f}")
    report.append(f"- **Total Spent:** ${total_spent:.2f}")
    report.append(f"- **Budget Used:** {overall_percentage:.1f}%")
    report.append(f"- **Remaining Budget:** ${total_budgeted - total_spent:.2f}")
    report.append(f"- **Savings Target:** ${savings_target:.2f}")
    
    # Performance insights
    report.append("\n## Insights & Recommendations")
    
    # Categories that need attention
    over_budget = [p for p in category_performance if p["percentage_used"] > 100]
    close_to_limit = [p for p in category_performance if 90 <= p["percentage_used"] <= 100]
    
    if over_budget:
        report.append("\n### ⚠️ Over Budget Categories:")
        for cat in over_budget:
            overage = cat["spent"] - cat["budgeted"]
            report.append(f"- **{cat['category'].title()}:** Over by ${overage:.2f}")
    
    if close_to_limit:
        report.append("\n### 🔶 Categories Close to Limit:")
        for cat in close_to_limit:
            remaining_days = (datetime.now().replace(month=datetime.now().month+1, day=1) - datetime.now()).days
            daily_budget = cat["remaining"] / max(remaining_days, 1)
            report.append(f"- **{cat['category'].title()}:** ${cat['remaining']:.2f} remaining (${daily_budget:.2f}/day)")
    
    # Positive performance
    well_managed = [p for p in category_performance if p["percentage_used"] <= 75]
    if well_managed:
        report.append("\n### ✅ Well-Managed Categories:")
        for cat in well_managed[:3]:  # Top 3
            report.append(f"- **{cat['category'].title()}:** {cat['percentage_used']:.1f}% used")
    
    # Monthly trend
    if overall_percentage > 100:
        report.append(f"\n### 🔴 Budget Overspending")
        report.append(f"Total overspend: ${total_spent - total_budgeted:.2f}")
        report.append("Consider reducing spending in over-budget categories.")
    elif overall_percentage > 90:
        report.append(f"\n### ⚠️ Approaching Budget Limit")
        remaining_days = (datetime.now().replace(month=datetime.now().month+1, day=1) - datetime.now()).days
        daily_remaining = (total_budgeted - total_spent) / max(remaining_days, 1)
        report.append(f"Daily budget remaining: ${daily_remaining:.2f}")
    else:
        report.append(f"\n### ✅ Budget on Track")
        report.append("Good spending discipline! Consider allocating unused budget to savings.")
    
    report_text = "\n".join(report)
    
    # Save progress report
    progress_filename = f"budget_progress_{month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    files[progress_filename] = report_text
    
    # Create summary
    summary_status = "🔴 Over Budget" if overall_percentage > 100 else "⚠️ Close to Limit" if overall_percentage > 90 else "✅ On Track"
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"📊 **Budget Progress Check - {month_name}**\n\n"
                           f"**Overall Status:** {summary_status}\n"
                           f"**Budget Used:** {overall_percentage:.1f}%\n"
                           f"**Total Spent:** ${total_spent:.2f} / ${total_budgeted:.2f}\n"
                           f"**Categories Over Budget:** {len(over_budget)}\n"
                           f"**Categories At Risk:** {len(close_to_limit)}\n\n"
                           f"📄 Detailed report saved to {progress_filename}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_finance_tools():
    """Create a list of specialized finance tools."""
    return [track_expense, analyze_spending, create_budget, budget_progress_check]