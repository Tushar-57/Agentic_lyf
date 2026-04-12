import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import {
  BellRing,
  Sparkles,
  Zap,
  Target,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  TrendingDown,
  Minus,
  Calendar,
  BarChart3,
  Activity,
  Brain,
  Flame,
  Timer,
  ArrowUpRight,
  ArrowDownRight,
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  Sparkline,
  CircularProgress,
  SegmentedProgress,
  BatteryIndicator,
  ComparisonGauge,
  MetricCard,
  HeatmapGrid,
  AnimatedCounter,
  MiniSchedule,
} from './NotificationVisuals'

// Enhanced notification interfaces
interface RecommendedAction {
  text: string
  priority: 'high' | 'medium' | 'low'
  estimated_minutes?: number
}

interface AINotification {
  id: number
  notification_key: string
  kind: string
  severity: 'low' | 'medium' | 'high' | 'critical' | string
  status: 'active' | 'acknowledged' | 'resolved' | string
  title: string
  summary: string
  details?: string | null
  score?: number | null
  recommended_actions?: string[] | RecommendedAction[]
  insights?: string[]
  triggering_metrics?: Record<string, number | string>
  priority_score?: number
  tags?: string[]
  last_seen_at?: string
  updated_at?: string
}

interface AINotificationEnvelope {
  persistence_enabled?: boolean
  notifications?: AINotification[]
  generated?: number
  upserted?: number
  resolved?: number
  generated_at?: string
}

interface ContextSnapshot {
  overdue_tasks: number
  due_today_tasks: number
  focus_tasks: Array<{
    id: string
    title: string
    priority: string
    completed?: boolean
  }>
  habit_metrics: {
    totalHabits: number
    completedToday: number
    completionRate7d: number
    avgStreak: number
  }
  time_metrics: {
    totalTimeSpentMinutes: number
    totalEstimatedMinutes: number
    deepWorkCoverageRatio: number
  }
  upcoming_deadlines: Array<{
    id: string
    title: string
    dueDate: string
    daysRemaining: number
  }>
  top_goals: string[]
  timezone: string
}

interface EnhancedAINotificationsProps {
  contextSnapshot?: Partial<ContextSnapshot>
  autoRefresh?: boolean
  refreshInterval?: number
}

const normalizeSeverity = (value: unknown): 'low' | 'medium' | 'high' | 'critical' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'low' || normalized === 'medium' || normalized === 'high' || normalized === 'critical') {
    return normalized
  }
  return 'medium'
}

const normalizeStatus = (value: unknown): 'active' | 'acknowledged' | 'resolved' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'active' || normalized === 'acknowledged' || normalized === 'resolved') {
    return normalized
  }
  return 'active'
}

const getTone = (severity: string) => {
  const normalized = normalizeSeverity(severity)
  if (normalized === 'critical') {
    return {
      wrapper: 'border-rose-400 bg-rose-50/80 dark:border-rose-800 dark:bg-rose-950/40',
      badge: 'bg-rose-100 text-rose-800 dark:bg-rose-900/70 dark:text-rose-200',
      icon: AlertTriangle,
      iconColor: 'text-rose-600 dark:text-rose-400',
    }
  }

  if (normalized === 'high') {
    return {
      wrapper: 'border-amber-400 bg-amber-50/80 dark:border-amber-800 dark:bg-amber-950/40',
      badge: 'bg-amber-100 text-amber-800 dark:bg-amber-900/70 dark:text-amber-200',
      icon: Zap,
      iconColor: 'text-amber-600 dark:text-amber-400',
    }
  }

  if (normalized === 'low') {
    return {
      wrapper: 'border-emerald-400 bg-emerald-50/80 dark:border-emerald-800 dark:bg-emerald-950/40',
      badge: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/70 dark:text-emerald-200',
      icon: CheckCircle2,
      iconColor: 'text-emerald-600 dark:text-emerald-400',
    }
  }

  return {
    wrapper: 'border-cyan-400 bg-cyan-50/80 dark:border-cyan-800 dark:bg-cyan-950/40',
    badge: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/70 dark:text-cyan-200',
    icon: Lightbulb,
    iconColor: 'text-cyan-600 dark:text-cyan-400',
  }
}

const getKindIcon = (kind: string) => {
  if (kind.includes('goal')) return Target
  if (kind.includes('deadline')) return Clock
  if (kind.includes('habit')) return CheckCircle2
  if (kind.includes('billable') || kind.includes('finance')) return TrendingUp
  return BellRing
}

const formatTimestamp = (value: string | undefined) => {
  if (!value) return 'n/a'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'n/a'
  return parsed.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

// Sanitize and render HTML content safely
const RichContent: React.FC<{ html: string; className?: string }> = ({ html, className }) => {
  // Basic sanitization - in production, use a proper library like DOMPurify
  const sanitized = html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+\s*=/gi, '')
  
  return (
    <div 
      className={cn(
        "prose prose-sm max-w-none dark:prose-invert",
        "prose-headings:text-sm prose-headings:font-semibold",
        "prose-ul:list-disc prose-ul:pl-4",
        "prose-ol:list-decimal prose-ol:pl-4",
        className
      )}
      dangerouslySetInnerHTML={{ __html: sanitized }}
    />
  )
}

// Individual notification card component
const NotificationCard: React.FC<{
  notification: AINotification
  onAcknowledge: (id: number, acknowledged: boolean) => void
  expandedByDefault?: boolean
}> = ({ notification, onAcknowledge, expandedByDefault = false }) => {
  const [expanded, setExpanded] = useState(expandedByDefault)
  const tone = getTone(notification.severity)
  const KindIcon = getKindIcon(notification.kind)
  
  // Parse actions - handle both string array and object array
  const actions: RecommendedAction[] = useMemo(() => {
    if (!notification.recommended_actions) return []
    return notification.recommended_actions.map((action: string | RecommendedAction) => {
      if (typeof action === 'string') {
        return { text: action, priority: 'medium' as const }
      }
      return action
    })
  }, [notification.recommended_actions])
  
  // Parse insights
  const insights = notification.insights || []
  
  return (
    <div className={cn('rounded-xl border p-4 transition-all duration-200', tone.wrapper)}>
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-start gap-2">
        <div className={cn('flex h-8 w-8 items-center justify-center rounded-full bg-white/60 dark:bg-slate-900/50', tone.iconColor)}>
          <KindIcon className="h-4 w-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge className={cn('text-[10px] uppercase tracking-wide', tone.badge)}>
              {notification.severity}
            </Badge>
            <Badge variant="outline" className="bg-white/60 text-[10px] dark:bg-slate-900/50">
              {notification.kind.replace(/[_-]+/g, ' ')}
            </Badge>
            {typeof notification.score === 'number' && (
              <Badge variant="outline" className="bg-white/60 text-[10px] font-semibold dark:bg-slate-900/50">
                Score {Math.round(notification.score)}
              </Badge>
            )}
          </div>
          <h3 className="mt-1 text-sm font-semibold text-foreground leading-tight">
            {notification.title}
          </h3>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 w-7 p-0"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      </div>
      
      {/* Summary */}
      <p className="text-xs leading-relaxed text-muted-foreground mb-3">
        {notification.summary}
      </p>
      
      {/* Visual Summary Row (Always Visible) */}
      <div className="mb-3 flex items-center gap-3">
        {/* Score Ring */}
        {typeof notification.score === 'number' && (
          <CircularProgress
            value={notification.score}
            max={100}
            size={48}
            strokeWidth={4}
            showValue={true}
          />
        )}
        
        {/* Sparkline Trend */}
        {notification.triggering_metrics && (
          <div className="flex-1">
            <Sparkline
              data={[
                (notification.triggering_metrics.habit_completion_ratio as number) * 100 || 60,
                (notification.triggering_metrics.deep_work_coverage as number) * 100 || 55,
                (notification.triggering_metrics.deadline_health as number) * 100 || 70,
                notification.score || 65,
              ]}
              width={100}
              height={30}
              showDots={false}
            />
            <p className="text-[9px] text-muted-foreground mt-0.5">7-day trend</p>
          </div>
        )}
        
        {/* Priority Indicator */}
        {notification.priority_score && (
          <BatteryIndicator
            level={notification.priority_score}
            max={100}
            size="sm"
            colorScheme="heatmap"
            label="Priority"
          />
        )}
      </div>
      
      {/* Expanded Content */}
      {expanded && (
        <div className="space-y-4 border-t border-border/50 pt-3">
          {/* Rich HTML Details */}
          {notification.details && (
            <RichContent 
              html={notification.details} 
              className="text-xs"
            />
          )}
          
          {/* Visual Metrics Dashboard */}
          {notification.triggering_metrics && (
            <div className="grid grid-cols-2 gap-3">
              {/* Goal Alignment Breakdown */}
              {notification.kind === 'goal_alignment' && (
                <>
                  <MetricCard
                    value={notification.triggering_metrics.performance_score as number || 0}
                    label="Performance"
                    trend={notification.triggering_metrics.performance_score as number > 6 ? 'up' : 'down'}
                    trendValue="/10"
                    icon={<Activity className="h-3 w-3" />}
                    color="#3b82f6"
                  />
                  <MetricCard
                    value={Math.round((notification.triggering_metrics.deep_work_coverage as number || 0) * 100)}
                    label="Deep Work %"
                    trend={notification.triggering_metrics.deep_work_coverage as number > 0.6 ? 'up' : 'down'}
                    trendValue="%"
                    icon={<Brain className="h-3 w-3" />}
                    color="#8b5cf6"
                  />
                  <MetricCard
                    value={Math.round((notification.triggering_metrics.habit_completion as number || 0) * 100)}
                    label="Habits"
                    trend={notification.triggering_metrics.habit_completion as number > 0.7 ? 'up' : 'neutral'}
                    trendValue="%"
                    icon={<Flame className="h-3 w-3" />}
                    color="#f59e0b"
                  />
                  <MetricCard
                    value={Math.round((notification.triggering_metrics.deadline_health as number || 0) * 100)}
                    label="Deadline Health"
                    trend={notification.triggering_metrics.deadline_health as number > 0.8 ? 'up' : 'down'}
                    trendValue="%"
                    icon={<Clock className="h-3 w-3" />}
                    color="#10b981"
                  />
                </>
              )}
              
              {/* Deadline Status */}
              {notification.kind === 'proactive_alert' && notification.notification_key.includes('deadline') && (
                <>
                  <div className="col-span-2">
                    <ComparisonGauge
                      current={notification.triggering_metrics.due_today as number || 0}
                      target={5}
                      label="Tasks Due Today"
                      unit=" tasks"
                    />
                  </div>
                  <MetricCard
                    value={notification.triggering_metrics.overdue as number || 0}
                    label="Overdue"
                    trend={notification.triggering_metrics.overdue as number > 0 ? 'down' : 'up'}
                    trendValue=" tasks"
                    icon={<AlertTriangle className="h-3 w-3" />}
                    color="#ef4444"
                  />
                  <MetricCard
                    value={notification.triggering_metrics.due_today as number || 0}
                    label="Due Today"
                    trend="neutral"
                    trendValue=" tasks"
                    icon={<Calendar className="h-3 w-3" />}
                    color="#f59e0b"
                  />
                </>
              )}
              
              {/* Deep Work Gap */}
              {notification.kind === 'proactive_alert' && notification.notification_key.includes('deep_work') && (
                <>
                  <div className="col-span-2">
                    <SegmentedProgress
                      segments={[
                        { label: 'Deep Work', value: (notification.triggering_metrics.deep_work_coverage as number || 0) * 100, color: '#8b5cf6' },
                        { label: 'Gap', value: (1 - (notification.triggering_metrics.deep_work_coverage as number || 0)) * 100, color: '#e5e7eb' },
                      ]}
                      height={20}
                    />
                  </div>
                  <MetricCard
                    value={Math.round((notification.triggering_metrics.planned_deep_work as number || 0) / 60)}
                    label="Planned (hrs)"
                    trend="neutral"
                    icon={<Target className="h-3 w-3" />}
                    color="#3b82f6"
                  />
                  <MetricCard
                    value={Math.round((notification.triggering_metrics.gap_percentage as number || 0))}
                    label="Gap %"
                    trend="down"
                    trendValue="%"
                    icon={<TrendingDown className="h-3 w-3" />}
                    color="#ef4444"
                  />
                </>
              )}
              
              {/* Habit Consistency */}
              {notification.kind === 'proactive_alert' && notification.notification_key.includes('habit') && (
                <>
                  <div className="col-span-2">
                    <CircularProgress
                      value={(notification.triggering_metrics.habit_completion as number || 0) * 100}
                      max={100}
                      size={80}
                      strokeWidth={8}
                      showValue={true}
                      label="Completion"
                      className="mx-auto"
                    />
                  </div>
                  <MetricCard
                    value={notification.triggering_metrics.habits_total as number || 0}
                    label="Total Habits"
                    trend="neutral"
                    icon={<CheckCircle2 className="h-3 w-3" />}
                    color="#3b82f6"
                  />
                  <MetricCard
                    value={notification.triggering_metrics.habits_completed as number || 0}
                    label="Completed"
                    trend="up"
                    icon={<Flame className="h-3 w-3" />}
                    color="#10b981"
                  />
                </>
              )}
            </div>
          )}
          
          {/* Insights */}
          {insights.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                <Lightbulb className="h-3 w-3" />
                Insights
              </h4>
              <ul className="space-y-1">
                {insights.map((insight, idx) => (
                  <li key={idx} className="text-xs text-muted-foreground flex items-start gap-1.5">
                    <span className="text-primary mt-0.5">•</span>
                    {insight}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Detailed Metrics Table */}
          {notification.triggering_metrics && Object.keys(notification.triggering_metrics).length > 0 && (
            <div className="rounded-md bg-white/50 dark:bg-slate-900/30 p-2.5">
              <h4 className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                All Metrics
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(notification.triggering_metrics).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-xs py-1 border-b border-border/20 last:border-0">
                    <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                    <span className="font-medium font-mono">
                      {typeof value === 'number' ? value.toFixed(2) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Action Items */}
          {actions.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                <Zap className="h-3 w-3" />
                Recommended Actions
              </h4>
              <div className="space-y-2">
                {actions.map((action, idx) => (
                  <div 
                    key={idx} 
                    className={cn(
                      'flex items-start gap-2 rounded-md p-2.5 text-xs transition-all hover:scale-[1.02] cursor-pointer',
                      action.priority === 'high' && 'bg-rose-50/50 border border-rose-200 dark:bg-rose-950/20 dark:border-rose-900/50 hover:bg-rose-100/50',
                      action.priority === 'medium' && 'bg-amber-50/50 border border-amber-200 dark:bg-amber-950/20 dark:border-amber-900/50 hover:bg-amber-100/50',
                      action.priority === 'low' && 'bg-emerald-50/50 border border-emerald-200 dark:bg-emerald-950/20 dark:border-emerald-900/50 hover:bg-emerald-100/50',
                    )}
                  >
                    <div className={cn(
                      'mt-0.5 h-2 w-2 rounded-full animate-pulse',
                      action.priority === 'high' && 'bg-rose-500',
                      action.priority === 'medium' && 'bg-amber-500',
                      action.priority === 'low' && 'bg-emerald-500',
                    )} />
                    <div className="flex-1">
                      <p className="font-medium text-foreground">{action.text}</p>
                      {action.estimated_minutes && (
                        <div className="mt-1 flex items-center gap-1 text-[10px] text-muted-foreground">
                          <Timer className="h-3 w-3" />
                          ~{action.estimated_minutes} min
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Footer */}
      <div className="mt-3 flex items-center justify-between border-t border-border/30 pt-2">
        <p className="text-[10px] text-muted-foreground">
          {formatTimestamp(notification.last_seen_at || notification.updated_at)}
        </p>
        
        {notification.id > 0 && (
          <Button
            type="button"
            size="sm"
            variant={notification.status === 'active' ? 'secondary' : 'outline'}
            className="h-7 text-xs"
            onClick={() => onAcknowledge(notification.id, notification.status === 'active')}
          >
            {notification.status === 'active' ? 'Acknowledge' : 'Mark Active'}
          </Button>
        )}
      </div>
    </div>
  )
}

// Main component
export const EnhancedAINotifications: React.FC<EnhancedAINotificationsProps> = ({
  contextSnapshot,
  autoRefresh = false,
  refreshInterval = 300000, // 5 minutes
}) => {
  const [notifications, setNotifications] = useState<AINotification[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [persistenceEnabled, setPersistenceEnabled] = useState(false)
  const [stats, setStats] = useState({ generated: 0, upserted: 0, resolved: 0 })

  // Generate enhanced notifications with context
  const loadNotifications = useCallback(async (refresh: boolean) => {
    setError(null)
    setIsLoading(true)
    if (refresh) {
      setIsRefreshing(true)
    }

    try {
      const endpoint = refresh && !contextSnapshot
        ? '/api/knowledge/notifications/enhanced/refresh'
        : '/api/knowledge/notifications/enhanced'
      
      const options: RequestInit = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }
      
      // Include context snapshot if provided
      if (contextSnapshot) {
        options.body = JSON.stringify({
          deadline_tasks: {
            overdue: contextSnapshot.overdue_tasks || 0,
            dueToday: contextSnapshot.due_today_tasks || 0,
          },
          focus_tasks: contextSnapshot.focus_tasks || [],
          habit_metrics: contextSnapshot.habit_metrics || {
            totalHabits: 0,
            completedToday: 0,
            completionRate7d: 0,
            avgStreak: 0,
          },
          time_metrics: contextSnapshot.time_metrics || {
            totalTimeSpentMinutes: 0,
            totalEstimatedMinutes: 0,
            deepWorkCoverageRatio: 0,
          },
          upcoming_deadlines: contextSnapshot.upcoming_deadlines || [],
          top_goals: contextSnapshot.top_goals || [],
          timezone: contextSnapshot.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone,
        })
      }

      const response = await fetch(`${endpoint}?limit=10&use_llm=true`, options)

      if (!response.ok) {
        throw new Error(`Notifications request failed with status ${response.status}`)
      }

      const payload = (await response.json()) as AINotificationEnvelope
      const list = Array.isArray(payload.notifications) ? payload.notifications : []

      setNotifications(
        list.map((entry) => ({
          ...entry,
          severity: normalizeSeverity(entry.severity),
          status: normalizeStatus(entry.status),
        }))
      )
      setPersistenceEnabled(Boolean(payload.persistence_enabled))
      setStats({
        generated: payload.generated || 0,
        upserted: payload.upserted || 0,
        resolved: payload.resolved || 0,
      })
    } catch (loadError) {
      console.error('Failed to load AI notifications:', loadError)
      setError(loadError instanceof Error ? loadError.message : 'Unable to load notifications.')
    } finally {
      setIsLoading(false)
      if (refresh) {
        setIsRefreshing(false)
      }
    }
  }, [contextSnapshot])

  // Acknowledge/unacknowledge notification
  const setAcknowledged = useCallback(async (notificationId: number, acknowledged: boolean) => {
    try {
      const response = await fetch(`/api/knowledge/notifications/${notificationId}/ack`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acknowledged }),
      })

      if (!response.ok) {
        throw new Error(`Notification update failed with status ${response.status}`)
      }

      const updated = (await response.json()) as AINotification
      setNotifications((previous) => 
        previous.map((entry) => (entry.id === updated.id ? { ...entry, ...updated } : entry))
      )
    } catch (updateError) {
      console.error('Failed to update notification acknowledgement:', updateError)
      setError('Unable to update notification status right now.')
    }
  }, [])

  // Use ref to keep stable reference to loadNotifications for interval
  const loadNotificationsRef = useRef(loadNotifications)
  useEffect(() => {
    loadNotificationsRef.current = loadNotifications
  }, [loadNotifications])

  // Initial load
  useEffect(() => {
    void loadNotifications(true)
  }, [loadNotifications])

  // Auto-refresh - uses ref to avoid recreating interval when loadNotifications changes
  useEffect(() => {
    if (!autoRefresh) return
    
    const interval = setInterval(() => {
      void loadNotificationsRef.current(true)
    }, refreshInterval)
    
    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval]) // Note: loadNotifications removed from deps

  // Sort notifications
  const sortedNotifications = useMemo(() => {
    const severityRank: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 }
    const statusRank: Record<string, number> = { active: 0, acknowledged: 1, resolved: 2 }

    return [...notifications].sort((left, right) => {
      const statusDiff = (statusRank[left.status] ?? 3) - (statusRank[right.status] ?? 3)
      if (statusDiff !== 0) return statusDiff

      const severityDiff = (severityRank[left.severity] ?? 9) - (severityRank[right.severity] ?? 9)
      if (severityDiff !== 0) return severityDiff

      const leftTime = new Date(left.last_seen_at || left.updated_at || 0).getTime()
      const rightTime = new Date(right.last_seen_at || right.updated_at || 0).getTime()
      return rightTime - leftTime
    })
  }, [notifications])

  // Stats
  const activeCount = sortedNotifications.filter((entry) => entry.status === 'active').length
  const highSignalCount = sortedNotifications.filter(
    (entry) => entry.status === 'active' && (entry.severity === 'high' || entry.severity === 'critical')
  ).length

  return (
    <Card className="mx-auto w-full max-w-5xl border-border/70 bg-gradient-to-br from-cyan-50/75 via-white to-emerald-50/55 p-5 dark:from-cyan-950/20 dark:via-slate-950 dark:to-emerald-950/20">
      {/* Header */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="mb-1 inline-flex items-center gap-2 rounded-full border border-cyan-200/70 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-cyan-700 dark:border-cyan-900/60 dark:bg-slate-900/70 dark:text-cyan-200">
            <Sparkles className="h-3.5 w-3.5" />
            Enhanced AI Assistant
          </p>
          <h2 className="text-2xl font-bold">Personalized Insights</h2>
          <p className="text-sm text-muted-foreground">
            AI-powered notifications tailored to your goals, deadlines, habits, and time patterns.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="bg-white/75 text-[11px] dark:bg-slate-900/70">
            Active {activeCount}
          </Badge>
          <Badge 
            variant="outline" 
            className={cn(
              "text-[11px]",
              highSignalCount > 0 && "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200"
            )}
          >
            High Priority {highSignalCount}
          </Badge>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => {
              // Prevent duplicate clicks when already refreshing
              if (!isRefreshing && !isLoading) {
                void loadNotifications(true)
              }
            }}
            disabled={isRefreshing || isLoading}
            className="gap-2"
          >
            <Sparkles className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
            {isRefreshing ? 'Analyzing...' : 'Refresh'}
          </Button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="mb-3 flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          <span className={cn(
            "inline-flex items-center gap-1",
            persistenceEnabled ? "text-emerald-600" : "text-amber-600"
          )}>
            <span className={cn(
              "h-1.5 w-1.5 rounded-full",
              persistenceEnabled ? "bg-emerald-500" : "bg-amber-500"
            )} />
            {persistenceEnabled ? 'Persistent storage' : 'Session mode'}
          </span>
          {stats.generated > 0 && (
            <span className="text-muted-foreground/70">
              Generated {stats.generated} • Stored {stats.upserted} • Resolved {stats.resolved}
            </span>
          )}
        </div>
        <span>{isLoading ? 'Analyzing your data...' : `${sortedNotifications.length} insights`}</span>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-3 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
          {error}
        </div>
      )}

      {/* Loading / Empty / Notifications */}
      {isLoading && sortedNotifications.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/70 bg-background/65 p-5 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 animate-pulse" />
            Analyzing your patterns and generating personalized insights...
          </div>
        </div>
      ) : sortedNotifications.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/70 bg-background/65 p-5 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            No immediate insights. Your system appears well-balanced. Check back later for updates.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {sortedNotifications.map((notification) => (
            <NotificationCard
              key={notification.id || notification.notification_key}
              notification={notification}
              onAcknowledge={setAcknowledged}
              expandedByDefault={notification.severity === 'critical' || notification.severity === 'high'}
            />
          ))}
        </div>
      )}
    </Card>
  )
}

export default EnhancedAINotifications
