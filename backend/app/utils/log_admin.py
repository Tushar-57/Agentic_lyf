"""
Admin Dashboard Log Viewer and Analysis Utilities

Provides utilities for:
- Filtering logs by component, user, request, time range
- Aggregating metrics from logs
- Real-time log streaming for admin dashboard
- Log analysis and anomaly detection
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Callable
from collections import defaultdict
import asyncio


class LogFilter:
    """Filter criteria for log queries."""

    def __init__(
        self,
        components: Optional[List[str]] = None,
        levels: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        message_contains: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
    ):
        self.components = components
        self.levels = levels
        self.user_id = user_id
        self.request_id = request_id
        self.operation = operation
        self.start_time = start_time
        self.end_time = end_time
        self.message_contains = message_contains
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms

    def matches(self, log_entry: Dict[str, Any]) -> bool:
        """Check if a log entry matches the filter criteria."""
        # Component filter
        if self.components:
            if log_entry.get("component") not in self.components:
                return False

        # Level filter
        if self.levels:
            if log_entry.get("level") not in self.levels:
                return False

        # User filter
        if self.user_id:
            if log_entry.get("user_id") != self.user_id:
                return False

        # Request filter
        if self.request_id:
            if log_entry.get("request_id") != self.request_id:
                return False

        # Operation filter
        if self.operation:
            if log_entry.get("operation") != self.operation:
                return False

        # Time range filters
        if self.start_time or self.end_time:
            timestamp_str = log_entry.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if self.start_time and timestamp < self.start_time:
                    return False
                if self.end_time and timestamp > self.end_time:
                    return False
            except ValueError:
                return False

        # Message content filter
        if self.message_contains:
            message = log_entry.get("message", "")
            if self.message_contains.lower() not in message.lower():
                return False

        # Duration filters
        if self.min_duration_ms is not None or self.max_duration_ms is not None:
            duration = log_entry.get("duration_ms")
            if duration is None:
                return False
            if self.min_duration_ms is not None and duration < self.min_duration_ms:
                return False
            if self.max_duration_ms is not None and duration > self.max_duration_ms:
                return False

        return True


@dataclass
class LogMetrics:
    """Aggregated metrics from log analysis."""
    total_entries: int
    entries_by_level: Dict[str, int]
    entries_by_component: Dict[str, int]
    error_count: int
    warning_count: int
    avg_duration_ms: Optional[float]
    p95_duration_ms: Optional[float]
    p99_duration_ms: Optional[float]
    top_operations: List[tuple]
    slowest_operations: List[tuple]
    time_range_start: Optional[datetime]
    time_range_end: Optional[datetime]


class LogAnalyzer:
    """Analyze log entries and extract metrics."""

    def __init__(self, log_entries: List[Dict[str, Any]]):
        self.entries = log_entries

    def compute_metrics(self) -> LogMetrics:
        """Compute aggregated metrics from logs."""
        if not self.entries:
            return LogMetrics(
                total_entries=0,
                entries_by_level={},
                entries_by_component={},
                error_count=0,
                warning_count=0,
                avg_duration_ms=None,
                p95_duration_ms=None,
                p99_duration_ms=None,
                top_operations=[],
                slowest_operations=[],
                time_range_start=None,
                time_range_end=None,
            )

        # Count by level and component
        entries_by_level = defaultdict(int)
        entries_by_component = defaultdict(int)
        operation_counts = defaultdict(int)
        operation_durations = defaultdict(list)

        durations = []
        timestamps = []
        error_count = 0
        warning_count = 0

        for entry in self.entries:
            level = entry.get("level", "UNKNOWN")
            component = entry.get("component", "unknown")
            operation = entry.get("operation", "unknown")

            entries_by_level[level] += 1
            entries_by_component[component] += 1
            operation_counts[operation] += 1

            if level == "ERROR":
                error_count += 1
            elif level == "WARNING":
                warning_count += 1

            # Collect durations
            duration = entry.get("duration_ms")
            if duration is not None:
                durations.append(duration)
                operation_durations[operation].append(duration)

            # Collect timestamps
            timestamp_str = entry.get("timestamp")
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except ValueError:
                    pass

        # Calculate duration statistics
        if durations:
            durations_sorted = sorted(durations)
            avg_duration = sum(durations) / len(durations)
            p95_idx = int(len(durations_sorted) * 0.95)
            p99_idx = int(len(durations_sorted) * 0.99)
            p95_duration = durations_sorted[min(p95_idx, len(durations_sorted) - 1)]
            p99_duration = durations_sorted[min(p99_idx, len(durations_sorted) - 1)]
        else:
            avg_duration = None
            p95_duration = None
            p99_duration = None

        # Top operations by count
        top_operations = sorted(
            operation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Slowest operations by average duration
        operation_avg_durations = {
            op: sum(durs) / len(durs)
            for op, durs in operation_durations.items()
            if durs
        }
        slowest_operations = sorted(
            operation_avg_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Time range
        time_range_start = min(timestamps) if timestamps else None
        time_range_end = max(timestamps) if timestamps else None

        return LogMetrics(
            total_entries=len(self.entries),
            entries_by_level=dict(entries_by_level),
            entries_by_component=dict(entries_by_component),
            error_count=error_count,
            warning_count=warning_count,
            avg_duration_ms=avg_duration,
            p95_duration_ms=p95_duration,
            p99_duration_ms=p99_duration,
            top_operations=top_operations,
            slowest_operations=slowest_operations,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
        )

    def find_errors(self) -> List[Dict[str, Any]]:
        """Extract all error entries with context."""
        errors = []
        for entry in self.entries:
            if entry.get("level") == "ERROR":
                errors.append(entry)
        return errors

    def find_slow_operations(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Find operations exceeding duration threshold."""
        slow = []
        for entry in self.entries:
            duration = entry.get("duration_ms")
            if duration and duration >= threshold_ms:
                slow.append(entry)
        return sorted(slow, key=lambda x: x.get("duration_ms", 0), reverse=True)

    def get_user_activity(self, user_id: str) -> Dict[str, Any]:
        """Get activity summary for a specific user."""
        user_entries = [e for e in self.entries if e.get("user_id") == user_id]

        if not user_entries:
            return {"user_id": user_id, "activity_count": 0}

        operations = defaultdict(int)
        components = defaultdict(int)
        errors = 0

        for entry in user_entries:
            operations[entry.get("operation", "unknown")] += 1
            components[entry.get("component", "unknown")] += 1
            if entry.get("level") == "ERROR":
                errors += 1

        return {
            "user_id": user_id,
            "activity_count": len(user_entries),
            "operations": dict(operations),
            "components": dict(components),
            "error_count": errors,
        }


class LogStreamer:
    """Stream logs in real-time for admin dashboard."""

    def __init__(self, log_source: Callable[[], Iterator[Dict[str, Any]]]):
        self.log_source = log_source
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._running = False

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to log stream."""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from log stream."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def start(self):
        """Start streaming logs."""
        self._running = True
        while self._running:
            try:
                for entry in self.log_source():
                    if not self._running:
                        break
                    for subscriber in self.subscribers:
                        try:
                            subscriber(entry)
                        except Exception:
                            pass
                await asyncio.sleep(0.1)  # Brief pause between checks
            except Exception:
                await asyncio.sleep(1)  # Longer pause on error

    def stop(self):
        """Stop streaming logs."""
        self._running = False


class LogQueryEngine:
    """Query engine for searching and filtering logs."""

    def __init__(self, log_entries: List[Dict[str, Any]]):
        self.entries = log_entries

    def query(self, filter_criteria: LogFilter) -> List[Dict[str, Any]]:
        """Query logs with filter criteria."""
        results = []
        for entry in self.entries:
            if filter_criteria.matches(entry):
                results.append(entry)
        return results

    def query_recent(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Query recent logs within time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        filter_criteria = LogFilter(start_time=cutoff)
        return self.query(filter_criteria)

    def query_by_request(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific request (distributed trace)."""
        filter_criteria = LogFilter(request_id=request_id)
        results = self.query(filter_criteria)
        # Sort by timestamp
        return sorted(
            results,
            key=lambda x: x.get("timestamp", "")
        )

    def query_by_user(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs for a specific user."""
        filter_criteria = LogFilter(user_id=user_id)
        results = self.query(filter_criteria)
        # Sort by timestamp desc and limit
        sorted_results = sorted(
            results,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_results[:limit]

    def query_errors(self, since_minutes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query error logs."""
        start_time = None
        if since_minutes:
            start_time = datetime.utcnow() - timedelta(minutes=since_minutes)

        filter_criteria = LogFilter(
            levels=["ERROR", "CRITICAL"],
            start_time=start_time,
        )
        return self.query(filter_criteria)


class AdminDashboardSummary:
    """Generate summary for admin dashboard display."""

    @staticmethod
    def generate(log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dashboard summary from logs."""
        analyzer = LogAnalyzer(log_entries)
        metrics = analyzer.compute_metrics()

        # Recent errors (last 5 minutes)
        recent_errors = analyzer.find_errors()
        recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
        recent_errors = [
            e for e in recent_errors
            if AdminDashboardSummary._parse_timestamp(e.get("timestamp", "")) > recent_cutoff
        ]

        # Slow operations (>1s)
        slow_ops = analyzer.find_slow_operations(1000)

        # Health status
        health_status = "healthy"
        if metrics.error_count > 10:
            health_status = "critical"
        elif metrics.error_count > 0:
            health_status = "degraded"

        return {
            "health_status": health_status,
            "metrics": {
                "total_logs": metrics.total_entries,
                "error_count": metrics.error_count,
                "warning_count": metrics.warning_count,
                "avg_duration_ms": metrics.avg_duration_ms,
                "p95_duration_ms": metrics.p95_duration_ms,
            },
            "breakdown": {
                "by_level": metrics.entries_by_level,
                "by_component": metrics.entries_by_component,
            },
            "top_operations": metrics.top_operations,
            "slowest_operations": metrics.slowest_operations,
            "recent_errors": recent_errors[:5],  # Top 5 recent errors
            "slow_operations": slow_ops[:5],   # Top 5 slow operations
            "generated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse timestamp string."""
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min


# Convenience functions for admin dashboard API
def filter_logs(
    logs: List[Dict[str, Any]],
    components: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter logs with simple API."""
    filter_criteria = LogFilter(
        components=components,
        levels=levels,
        user_id=user_id,
        request_id=request_id,
    )
    engine = LogQueryEngine(logs)
    return engine.query(filter_criteria)


def get_dashboard_summary(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get dashboard summary."""
    return AdminDashboardSummary.generate(logs)


def analyze_performance(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance metrics."""
    analyzer = LogAnalyzer(logs)
    metrics = analyzer.compute_metrics()

    return {
        "avg_response_time": metrics.avg_duration_ms,
        "p95_response_time": metrics.p95_duration_ms,
        "p99_response_time": metrics.p99_duration_ms,
        "slowest_operations": metrics.slowest_operations,
        "by_component": metrics.entries_by_component,
    }


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON log line."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def load_logs_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load structured logs from a JSON Lines file."""
    logs = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                log_entry = parse_log_line(line.strip())
                if log_entry:
                    logs.append(log_entry)
    except FileNotFoundError:
        pass
    return logs
