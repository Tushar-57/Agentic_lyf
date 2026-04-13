"""
Smart Time Context Analyzer - Analyzes time entries to provide rich context categorization.

Transforms raw time tracking data into actionable insights by categorizing work types,
identifying patterns, and detecting optimization opportunities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re

from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.SERVICE)


class WorkType(str, Enum):
    """Classification of work types for time entries."""
    DEEP_WORK = "deep_work"  # Focused, cognitively demanding work
    SHALLOW_WORK = "shallow_work"  # Admin, emails, quick tasks
    MEETINGS = "meetings"  # Synchronous communication
    LEARNING = "learning"  # Skill development, courses, LeetCode
    PLANNING = "planning"  # Strategy, goal setting, review
    CONTEXT_SWITCHING = "context_switching"  # Short entries suggesting fragmentation
    IDLE = "idle"  # Untracked or gap time


class EnergyPattern(str, Enum):
    """Energy pattern classification based on focus and duration."""
    HIGH_FOCUS = "high_focus"  # Sustained attention, high energy
    MODERATE_FOCUS = "moderate_focus"  # Steady work
    LOW_FOCUS = "low_focus"  # Struggling or distracted
    RECOVERY = "recovery"  # Break time or low cognitive load


@dataclass
class TimeEntryClassification:
    """Classification result for a single time entry."""
    entry_id: str
    work_type: WorkType
    energy_pattern: EnergyPattern
    focus_quality: float  # 0-10 score
    productivity_score: float  # 0-10 score
    goal_alignment: List[str]  # Which goals this serves
    recommended_optimization: Optional[str] = None


@dataclass
class TimeWindowAnalysis:
    """Comprehensive analysis of a time window (day, week, etc.)."""
    window_label: str
    total_minutes: float
    categorized_breakdown: Dict[WorkType, float]
    energy_distribution: Dict[EnergyPattern, float]
    focus_score_avg: float
    productivity_score_avg: float
    goal_coverage: Dict[str, float]  # Goal -> minutes spent
    gaps_detected: List[Dict[str, Any]]  # Idle periods
    optimization_opportunities: List[Dict[str, Any]]
    pattern_insights: List[str]


@dataclass
class WorkPatternProfile:
    """Long-term work pattern analysis."""
    deep_work_capacity: float  # Average sustained deep work minutes
    peak_performance_windows: List[Tuple[int, int]]  # Hours of day (start, end)
    context_switch_frequency: float  # Switches per hour
    focus_consistency_score: float  # 0-100
    learning_investment_ratio: float  # Learning / Total time
    shallow_work_ratio: float  # Admin / Total time
    recommended_adjustments: List[Dict[str, Any]]


class SmartTimeContextAnalyzer:
    """
    Analyzes time tracking data to provide rich, actionable context.
    
    Goes beyond "6000 minutes of work" to categorize:
    - Deep work vs meetings vs admin
    - Goal alignment patterns
    - Energy and focus patterns
    - Optimization opportunities
    """
    
    # Keywords for work type classification
    WORK_TYPE_PATTERNS = {
        WorkType.DEEP_WORK: [
            r'\b(coding|programming|development|debugging|architecture|design|implement|algorithm|leetcode|problem.solving)\b',
            r'\b(writing|drafting|creating|building|deep|focus|concentration)\b',
            r'\b(analysis|analytics|research|investigation|modeling)\b',
            r'\b(learning|studying|course|tutorial|practice)\b',
        ],
        WorkType.MEETINGS: [
            r'\b(meeting|call|standup|sync|discussion|review|interview|stand-up)\b',
            r'\b(presentation|demo|showcase|walkthrough)\b',
            r'\b(1:1|one.on.one|team|collaboration)\b',
        ],
        WorkType.SHALLOW_WORK: [
            r'\b(email|slack|message|communication|update|status)\b',
            r'\b(admin|documentation|organize|cleanup|maintenance)\b',
            r'\b(ticket|bug.fix|quick|urgent|firefight)\b',
        ],
        WorkType.PLANNING: [
            r'\b(plan|strategy|roadmap|prioritization|goal|objective)\b',
            r'\b(review|retrospective|assessment|evaluation|metrics)\b',
            r'\b(estimate|forecast|projection|budget)\b',
        ],
        WorkType.LEARNING: [
            r'\b(leetcode|coding.challenge|algorithm|data.structure)\b',
            r'\b(course|tutorial|lecture|read|book|article|learn)\b',
            r'\b(certification|exam|study|skill.development)\b',
        ],
    }
    
    # Goal alignment patterns
    GOAL_ALIGNMENT_PATTERNS = {
        "skill_development": [
            r'\b(leetcode|algorithm|data.structure|coding.challenge|skill|learn|practice)\b',
            r'\b(course|tutorial|certification|upskill)\b',
        ],
        "career_advancement": [
            r'\b(project|deliverable|milestone|achievement|impact|promotion)\b',
        ],
        "health_wellness": [
            r'\b(exercise|workout|gym|meditation|health|wellness|sleep)\b',
        ],
        "financial_goals": [
            r'\b(billable|client|revenue|income|freelance|contract)\b',
        ],
    }
    
    def __init__(self):
        self.classification_cache: Dict[str, TimeEntryClassification] = {}
    
    def classify_entry(
        self,
        entry: Dict[str, Any],
        user_priorities: Optional[List[str]] = None
    ) -> TimeEntryClassification:
        """
        Classify a single time entry into work type, energy pattern, and goal alignment.
        
        Args:
            entry: Time entry with project_name, description, duration_minutes, etc.
            user_priorities: User's stated priorities for goal alignment matching
            
        Returns:
            Classification with work type, energy pattern, and recommendations
        """
        entry_id = entry.get("entry_id", "unknown")
        description = str(entry.get("description", "")).lower()
        project = str(entry.get("project_name", "")).lower()
        duration = entry.get("duration_minutes", 0) or 0
        focus_score = entry.get("focus_score", 5) or 5
        energy_score = entry.get("energy_score", 5) or 5
        
        # Combine project and description for classification
        text_to_analyze = f"{project} {description}"
        
        # Determine work type
        work_type = self._classify_work_type(text_to_analyze, duration)
        
        # Determine energy pattern
        energy_pattern = self._classify_energy_pattern(
            duration, focus_score, energy_score, work_type
        )
        
        # Calculate quality scores
        focus_quality = self._calculate_focus_quality(
            duration, focus_score, work_type, energy_pattern
        )
        productivity_score = self._calculate_productivity_score(
            duration, work_type, energy_pattern, focus_quality
        )
        
        # Determine goal alignment
        goal_alignment = self._determine_goal_alignment(
            text_to_analyze, user_priorities or []
        )
        
        # Generate optimization recommendation if needed
        optimization = self._generate_optimization_recommendation(
            work_type, energy_pattern, duration, goal_alignment, user_priorities
        )
        
        classification = TimeEntryClassification(
            entry_id=entry_id,
            work_type=work_type,
            energy_pattern=energy_pattern,
            focus_quality=round(focus_quality, 1),
            productivity_score=round(productivity_score, 1),
            goal_alignment=goal_alignment,
            recommended_optimization=optimization
        )
        
        self.classification_cache[entry_id] = classification
        return classification
    
    def _classify_work_type(self, text: str, duration: float) -> WorkType:
        """Classify work type based on text patterns and duration."""
        scores = {wt: 0 for wt in WorkType}
        
        # Score based on keyword patterns
        for work_type, patterns in self.WORK_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[work_type] += 1
        
        # Duration heuristics
        if duration < 15:
            scores[WorkType.CONTEXT_SWITCHING] += 2
        elif duration >= 60 and scores[WorkType.DEEP_WORK] > 0:
            scores[WorkType.DEEP_WORK] += 1  # Boost deep work for long sessions
        
        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        
        # Default to shallow work if no strong signals
        if scores[best_type] == 0:
            if duration < 20:
                best_type = WorkType.CONTEXT_SWITCHING
            else:
                best_type = WorkType.SHALLOW_WORK
        
        return best_type
    
    def _classify_energy_pattern(
        self,
        duration: float,
        focus_score: float,
        energy_score: float,
        work_type: WorkType
    ) -> EnergyPattern:
        """Classify energy pattern based on metrics."""
        # High focus indicators
        if focus_score >= 7 and duration >= 45 and work_type == WorkType.DEEP_WORK:
            return EnergyPattern.HIGH_FOCUS
        
        # Low focus indicators
        if focus_score <= 4 or duration < 15:
            return EnergyPattern.LOW_FOCUS
        
        # Recovery patterns
        if work_type in [WorkType.SHALLOW_WORK] and duration < 30:
            return EnergyPattern.RECOVERY
        
        return EnergyPattern.MODERATE_FOCUS
    
    def _calculate_focus_quality(
        self,
        duration: float,
        focus_score: float,
        work_type: WorkType,
        energy_pattern: EnergyPattern
    ) -> float:
        """Calculate overall focus quality score (0-10)."""
        base_score = focus_score
        
        # Adjust for work type
        if work_type == WorkType.DEEP_WORK and duration >= 45:
            base_score += 1.5
        elif work_type == WorkType.MEETINGS:
            base_score -= 0.5
        elif work_type == WorkType.CONTEXT_SWITCHING:
            base_score -= 2
        
        # Adjust for energy pattern
        if energy_pattern == EnergyPattern.HIGH_FOCUS:
            base_score += 0.5
        elif energy_pattern == EnergyPattern.LOW_FOCUS:
            base_score -= 1
        
        return min(10, max(1, base_score))
    
    def _calculate_productivity_score(
        self,
        duration: float,
        work_type: WorkType,
        energy_pattern: EnergyPattern,
        focus_quality: float
    ) -> float:
        """Calculate productivity score (0-10)."""
        base_score = 5.0
        
        # Work type contribution
        work_type_weights = {
            WorkType.DEEP_WORK: 2.0,
            WorkType.LEARNING: 1.5,
            WorkType.PLANNING: 1.0,
            WorkType.SHALLOW_WORK: 0.0,
            WorkType.MEETINGS: -0.5,
            WorkType.CONTEXT_SWITCHING: -1.5,
        }
        
        base_score += work_type_weights.get(work_type, 0)
        base_score += (focus_quality - 5) * 0.3  # Focus quality contribution
        
        # Duration factor (sweet spot: 45-90 min)
        if 45 <= duration <= 90:
            base_score += 0.5
        elif duration < 15:
            base_score -= 1
        
        return min(10, max(1, round(base_score, 1)))
    
    def _determine_goal_alignment(
        self,
        text: str,
        user_priorities: List[str]
    ) -> List[str]:
        """Determine which goals this entry aligns with."""
        aligned_goals = []
        text_lower = text.lower()
        
        # Check against predefined patterns
        for goal, patterns in self.GOAL_ALIGNMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    aligned_goals.append(goal)
                    break
        
        # Check against user priorities (fuzzy match)
        for priority in user_priorities[:5]:  # Top 5 priorities
            priority_lower = priority.lower()
            priority_keywords = set(priority_lower.split())
            text_keywords = set(text_lower.split())
            
            # Simple overlap check
            if len(priority_keywords & text_keywords) > 0:
                aligned_goals.append(f"priority:{priority}")
        
        return list(set(aligned_goals))  # Deduplicate
    
    def _generate_optimization_recommendation(
        self,
        work_type: WorkType,
        energy_pattern: EnergyPattern,
        duration: float,
        goal_alignment: List[str],
        user_priorities: Optional[List[str]]
    ) -> Optional[str]:
        """Generate optimization recommendation if entry could be improved."""
        recommendations = []
        
        # Context switching detection
        if work_type == WorkType.CONTEXT_SWITCHING:
            recommendations.append("Consider batching similar short tasks")
        
        # Short deep work sessions
        if work_type == WorkType.DEEP_WORK and duration < 30:
            recommendations.append("Extend deep work blocks to 45+ min for flow state")
        
        # Low energy on important work
        if energy_pattern == EnergyPattern.LOW_FOCUS and work_type == WorkType.DEEP_WORK:
            recommendations.append("Schedule deep work during your peak energy hours")
        
        # Missing goal alignment
        if not goal_alignment and user_priorities:
            recommendations.append("Consider connecting this to your stated priorities")
        
        # Long meetings without focus
        if work_type == WorkType.MEETINGS and duration > 60:
            recommendations.append("Consider agenda timeboxing for long meetings")
        
        return "; ".join(recommendations) if recommendations else None
    
    def analyze_time_window(
        self,
        entries: List[Dict[str, Any]],
        window_label: str,
        user_priorities: Optional[List[str]] = None
    ) -> TimeWindowAnalysis:
        """
        Analyze a time window (day, week, etc.) comprehensively.
        
        Args:
            entries: List of time entries in the window
            window_label: Label for the window (e.g., "Today", "This Week")
            user_priorities: User's priorities for goal alignment analysis
            
        Returns:
            Comprehensive window analysis
        """
        if not entries:
            return TimeWindowAnalysis(
                window_label=window_label,
                total_minutes=0,
                categorized_breakdown={wt: 0 for wt in WorkType},
                energy_distribution={ep: 0 for ep in EnergyPattern},
                focus_score_avg=0,
                productivity_score_avg=0,
                goal_coverage={},
                gaps_detected=[],
                optimization_opportunities=[],
                pattern_insights=["No tracked time in this window"]
            )
        
        # Classify all entries
        classifications = [
            self.classify_entry(entry, user_priorities) for entry in entries
        ]
        
        # Calculate totals
        total_minutes = sum(
            e.get("duration_minutes", 0) or 0 for e in entries
        )
        
        # Categorized breakdown
        categorized_breakdown = {wt: 0.0 for wt in WorkType}
        for entry, classification in zip(entries, classifications):
            duration = entry.get("duration_minutes", 0) or 0
            categorized_breakdown[classification.work_type] += duration
        
        # Energy distribution
        energy_distribution = {ep: 0.0 for ep in EnergyPattern}
        for entry, classification in zip(entries, classifications):
            duration = entry.get("duration_minutes", 0) or 0
            energy_distribution[classification.energy_pattern] += duration
        
        # Average scores
        focus_score_avg = sum(c.focus_quality for c in classifications) / len(classifications)
        productivity_score_avg = sum(c.productivity_score for c in classifications) / len(classifications)
        
        # Goal coverage
        goal_coverage: Dict[str, float] = {}
        for entry, classification in zip(entries, classifications):
            duration = entry.get("duration_minutes", 0) or 0
            for goal in classification.goal_alignment:
                goal_coverage[goal] = goal_coverage.get(goal, 0) + duration
        
        # Detect gaps
        gaps_detected = self._detect_time_gaps(entries)
        
        # Find optimization opportunities
        optimization_opportunities = self._find_optimization_opportunities(
            entries, classifications, user_priorities or []
        )
        
        # Generate pattern insights
        pattern_insights = self._generate_pattern_insights(
            categorized_breakdown, energy_distribution, goal_coverage, total_minutes
        )
        
        return TimeWindowAnalysis(
            window_label=window_label,
            total_minutes=total_minutes,
            categorized_breakdown=categorized_breakdown,
            energy_distribution=energy_distribution,
            focus_score_avg=round(focus_score_avg, 1),
            productivity_score_avg=round(productivity_score_avg, 1),
            goal_coverage=goal_coverage,
            gaps_detected=gaps_detected,
            optimization_opportunities=optimization_opportunities,
            pattern_insights=pattern_insights
        )
    
    def _detect_time_gaps(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect idle gaps between time entries."""
        gaps = []
        
        # Sort entries by start time
        sorted_entries = sorted(
            entries,
            key=lambda e: e.get("start_time") or e.get("created_at", "")
        )
        
        for i in range(len(sorted_entries) - 1):
            current_end = self._parse_timestamp(sorted_entries[i].get("end_time"))
            next_start = self._parse_timestamp(sorted_entries[i + 1].get("start_time"))
            
            if current_end and next_start:
                gap_minutes = (next_start - current_end).total_seconds() / 60
                if gap_minutes > 30:  # Significant gap
                    gaps.append({
                        "start": current_end.isoformat(),
                        "end": next_start.isoformat(),
                        "duration_minutes": round(gap_minutes, 0),
                        "type": "untracked_idle" if gap_minutes > 60 else "short_break"
                    })
        
        return gaps
    
    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if not timestamp:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return None
    
    def _find_optimization_opportunities(
        self,
        entries: List[Dict[str, Any]],
        classifications: List[TimeEntryClassification],
        user_priorities: List[str]
    ) -> List[Dict[str, Any]]:
        """Find specific optimization opportunities."""
        opportunities = []
        
        # Calculate work type distribution
        total_minutes = sum(e.get("duration_minutes", 0) or 0 for e in entries)
        if total_minutes == 0:
            return opportunities
        
        deep_work_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type == WorkType.DEEP_WORK
        )
        
        learning_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type == WorkType.LEARNING
        )
        
        shallow_work_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type == WorkType.SHALLOW_WORK
        )
        
        # Deep work ratio analysis
        deep_work_ratio = deep_work_minutes / total_minutes
        if deep_work_ratio < 0.2:
            opportunities.append({
                "type": "deep_work_deficit",
                "severity": "high",
                "current_ratio": round(deep_work_ratio * 100, 1),
                "target_ratio": 30,
                "suggestion": "Your deep work is below 20%. Aim for 30% for high-impact output.",
                "action": "Block 90-min deep work sessions in your calendar"
            })
        
        # Learning investment analysis
        learning_ratio = learning_minutes / total_minutes
        if learning_ratio < 0.05 and any("skill" in p.lower() for p in user_priorities):
            opportunities.append({
                "type": "learning_deficit",
                "severity": "medium",
                "current_ratio": round(learning_ratio * 100, 1),
                "target_ratio": 10,
                "suggestion": "Skill development is a priority but under 5% of time invested.",
                "action": "Schedule 30-min daily learning blocks during identified idle gaps"
            })
        
        # Context switching analysis
        short_entries = [e for e in entries if (e.get("duration_minutes", 0) or 0) < 15]
        if len(short_entries) > len(entries) * 0.3:
            opportunities.append({
                "type": "fragmentation",
                "severity": "medium",
                "fragmentation_ratio": round(len(short_entries) / len(entries) * 100, 1),
                "suggestion": f"{len(short_entries)} short entries ({len(short_entries)/len(entries)*100:.0f}%) indicate context switching.",
                "action": "Batch similar tasks; aim for 45+ min blocks"
            })
        
        # Goal alignment gaps
        aligned_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.goal_alignment
        )
        
        if aligned_minutes / total_minutes < 0.4:
            opportunities.append({
                "type": "goal_misalignment",
                "severity": "high",
                "aligned_ratio": round(aligned_minutes / total_minutes * 100, 1),
                "suggestion": "Less than 40% of time aligns with your stated goals/priorities.",
                "action": "Review and tag entries with goal connections"
            })
        
        return opportunities
    
    def _generate_pattern_insights(
        self,
        categorized_breakdown: Dict[WorkType, float],
        energy_distribution: Dict[EnergyPattern, float],
        goal_coverage: Dict[str, float],
        total_minutes: float
    ) -> List[str]:
        """Generate human-readable pattern insights."""
        insights = []
        
        if total_minutes == 0:
            return insights
        
        # Work type insights
        deep_ratio = categorized_breakdown.get(WorkType.DEEP_WORK, 0) / total_minutes
        meeting_ratio = categorized_breakdown.get(WorkType.MEETINGS, 0) / total_minutes
        
        if deep_ratio > 0.3:
            insights.append(f"Strong deep work focus ({deep_ratio*100:.0f}% of time)")
        elif deep_ratio < 0.15:
            insights.append(f"Low deep work coverage ({deep_ratio*100:.0f}%) - consider protecting focus time")
        
        if meeting_ratio > 0.25:
            insights.append(f"High meeting load ({meeting_ratio*100:.0f}%) - may impact deep work capacity")
        
        # Energy insights
        high_focus_minutes = energy_distribution.get(EnergyPattern.HIGH_FOCUS, 0)
        low_focus_minutes = energy_distribution.get(EnergyPattern.LOW_FOCUS, 0)
        
        if high_focus_minutes > low_focus_minutes * 2:
            insights.append("Strong focus energy pattern detected")
        elif low_focus_minutes > total_minutes * 0.3:
            insights.append("Notable low-focus periods - consider energy management")
        
        # Goal coverage insights
        if goal_coverage:
            top_goal = max(goal_coverage.items(), key=lambda x: x[1])
            insights.append(f"Primary goal focus: {top_goal[0]} ({top_goal[1]/total_minutes*100:.0f}% of time)")
        
        return insights
    
    def generate_productivity_profile(
        self,
        entries: List[Dict[str, Any]],
        window_days: int = 14
    ) -> WorkPatternProfile:
        """
        Generate a long-term productivity profile.
        
        Args:
            entries: Time entries over the analysis period
            window_days: Number of days in the analysis window
            
        Returns:
            WorkPatternProfile with capacity and recommendations
        """
        if not entries:
            return WorkPatternProfile(
                deep_work_capacity=0,
                peak_performance_windows=[],
                context_switch_frequency=0,
                focus_consistency_score=0,
                learning_investment_ratio=0,
                shallow_work_ratio=0,
                recommended_adjustments=[]
            )
        
        # Classify all entries
        classifications = [self.classify_entry(entry) for entry in entries]
        
        # Calculate deep work capacity (longest sustained deep work sessions)
        deep_work_sessions = [
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type == WorkType.DEEP_WORK
        ]
        deep_work_capacity = max(deep_work_sessions) if deep_work_sessions else 0
        
        # Estimate peak performance windows (simplified)
        peak_windows = self._identify_peak_windows(entries, classifications)
        
        # Context switch frequency
        short_entries = [e for e in entries if (e.get("duration_minutes", 0) or 0) < 20]
        total_hours = sum(e.get("duration_minutes", 0) or 0 for e in entries) / 60
        switch_frequency = len(short_entries) / max(total_hours, 1)
        
        # Focus consistency
        focus_scores = [c.focus_quality for c in classifications]
        consistency_score = 100 - (max(focus_scores) - min(focus_scores)) * 10 if len(focus_scores) > 1 else 50
        
        # Learning investment
        total_minutes = sum(e.get("duration_minutes", 0) or 0 for e in entries)
        learning_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type == WorkType.LEARNING
        )
        learning_ratio = learning_minutes / max(total_minutes, 1)
        
        # Shallow work ratio
        shallow_minutes = sum(
            e.get("duration_minutes", 0) or 0
            for e, c in zip(entries, classifications)
            if c.work_type in [WorkType.SHALLOW_WORK, WorkType.CONTEXT_SWITCHING]
        )
        shallow_ratio = shallow_minutes / max(total_minutes, 1)
        
        # Generate recommendations
        adjustments = self._generate_profile_recommendations(
            deep_work_capacity, switch_frequency, learning_ratio, shallow_ratio, consistency_score
        )
        
        return WorkPatternProfile(
            deep_work_capacity=round(deep_work_capacity, 0),
            peak_performance_windows=peak_windows,
            context_switch_frequency=round(switch_frequency, 1),
            focus_consistency_score=round(consistency_score, 0),
            learning_investment_ratio=round(learning_ratio * 100, 1),
            shallow_work_ratio=round(shallow_ratio * 100, 1),
            recommended_adjustments=adjustments
        )
    
    def _identify_peak_windows(
        self,
        entries: List[Dict[str, Any]],
        classifications: List[TimeEntryClassification]
    ) -> List[Tuple[int, int]]:
        """Identify peak performance time windows (simplified)."""
        # Group entries by hour and score them
        hourly_scores: Dict[int, List[float]] = {}
        
        for entry, classification in zip(entries, classifications):
            start = self._parse_timestamp(entry.get("start_time"))
            if start:
                hour = start.hour
                if hour not in hourly_scores:
                    hourly_scores[hour] = []
                hourly_scores[hour].append(classification.focus_quality)
        
        # Find consecutive high-performing hours
        peak_windows = []
        current_window = []
        
        for hour in sorted(hourly_scores.keys()):
            avg_score = sum(hourly_scores[hour]) / len(hourly_scores[hour])
            if avg_score >= 7:  # High focus threshold
                current_window.append(hour)
            else:
                if len(current_window) >= 2:
                    peak_windows.append((current_window[0], current_window[-1] + 1))
                current_window = []
        
        if len(current_window) >= 2:
            peak_windows.append((current_window[0], current_window[-1] + 1))
        
        return peak_windows if peak_windows else [(9, 12)]  # Default morning peak
    
    def _generate_profile_recommendations(
        self,
        deep_work_capacity: float,
        switch_frequency: float,
        learning_ratio: float,
        shallow_ratio: float,
        consistency_score: float
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on profile metrics."""
        adjustments = []
        
        if deep_work_capacity < 45:
            adjustments.append({
                "area": "deep_work_capacity",
                "issue": f"Current max deep work session: {deep_work_capacity:.0f}min",
                "recommendation": "Gradually extend focus sessions by 10min/week to reach 90min",
                "priority": "high"
            })
        
        if switch_frequency > 4:
            adjustments.append({
                "area": "context_switching",
                "issue": f"{switch_frequency:.1f} context switches per hour",
                "recommendation": "Implement time-blocking; batch similar tasks",
                "priority": "high"
            })
        
        if learning_ratio < 0.05:
            adjustments.append({
                "area": "skill_development",
                "issue": f"Only {learning_ratio*100:.1f}% of time in learning",
                "recommendation": "Allocate 10% of work time to deliberate practice",
                "priority": "medium"
            })
        
        if shallow_ratio > 0.35:
            adjustments.append({
                "area": "shallow_work",
                "issue": f"{shallow_ratio*100:.1f}% of time in admin/shallow work",
                "recommendation": "Delegate or batch admin tasks; protect maker time",
                "priority": "medium"
            })
        
        return adjustments


# Singleton instance
_time_analyzer: Optional[SmartTimeContextAnalyzer] = None


def get_time_analyzer() -> SmartTimeContextAnalyzer:
    """Get or create the time analyzer singleton."""
    global _time_analyzer
    if _time_analyzer is None:
        _time_analyzer = SmartTimeContextAnalyzer()
    return _time_analyzer
