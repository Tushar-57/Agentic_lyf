import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'
import {
  TrendingUp,
  Activity,
  Brain,
  Target,
  Zap,
  Heart,
  DollarSign,
  BookOpen,
  BarChart3,
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
  Layers,
  Clock3,
  Sunrise,
  MoonStar,
  Compass,
  Flag,
  X,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  BellRing,
  Lock,
  CalendarClock,
  ClipboardCheck,
  ListTodo,
  AlertTriangle,
  ChevronDown,
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn, formatMinutesToHoursMinutes } from '@/lib/utils'
import './checkup.css'

// Wrapper class for the LLM-emitted morning/evening checkup HTML.
//
// The previous value (`prose prose-sm ...`) used Tailwind Typography, which
// resets default semantics and overrides any inner element styling — including
// the dc-* class set the backend prompt declares. Result: the rich semantic
// markup the model returned was being flattened into a tower of unstyled text.
//
// Now we just attach a stable wrapper class so checkup.css can scope styles
// safely and the .daily-checkup / .dc-* rules apply to the rendered output.
const PREMIUM_CHECKUP_HTML_CLASSNAMES = 'checkup-html-render text-sm leading-relaxed'

interface AnalyticsData {
  interactions: {
    daily: Array<{ date: string; count: number; agent: string }>
    weekly: Array<{ week: string; count: number }>
    by_agent: Array<{ agent: string; count: number; color: string }>
    by_category: Array<{ category: string; raw_category: string; count: number; color: string }>
  }
  patterns: {
    most_active_hours: Array<{ hour: number; interactions: number }>
    preference_changes: Array<{ date: string; category: string; changes: number }>
    knowledge_growth: Array<{ date: string; total_entries: number; new_entries: number }>
    category_focus: Array<{ date: string; category: string; count: number }>
  }
  insights: {
    total_interactions: number
    most_used_agent: string
    avg_daily_interactions: number
    knowledge_base_size: number
    preference_stability: number
    learning_velocity: number
    top_knowledge_category: string
    time_entry_records: number
    time_entry_billable_records: number
    avg_time_entry_minutes: number
    time_entry_total_minutes: number
    time_entry_count: number
    time_entry_daily?: Array<{ date: string; total_minutes: number; count: number }>
    habit_metrics?: {
      total_habits: number
      completed_today: number
      completion_rate_7d: number
      current_streak: number
      longest_streak: number
    }
  }
}

interface DailyCheckupResponse {
  date: string
  checkup_type: 'morning' | 'evening'
  coach_message: string
  coach_message_html?: string
  generated_with: 'llm' | 'fallback' | string
  focus_target?: string
  intent_note?: string | null
  reflection_note?: string | null
  wins?: string[]
  blockers?: string[]
  tomorrow_focus?: string[]
  stats?: Record<string, unknown>
}

interface DailyCheckupInsightEntry {
  content?: string
  created_at?: string
  updated_at?: string
  metadata?: Record<string, unknown>
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
  recommended_actions?: string[]
  payload?: Record<string, unknown>
  first_seen_at?: string
  last_seen_at?: string
  acknowledged_at?: string | null
  resolved_at?: string | null
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

interface AnalyticsDashboardProps {
  className?: string
}

interface OnboardingGoal {
  id?: string
  title?: string
  description?: string
  category?: string
  priority?: string
  smartCriteria?: Record<string, unknown>
  milestones?: Array<Record<string, unknown>>
}

interface OnboardingProfileData {
  goals?: OnboardingGoal[]
  schedule?: Record<string, unknown> | null
  planner?: Record<string, unknown> | null
  preferenceProfile?: Record<string, Record<string, unknown>> | null
}

interface CheckupFormState {
  topPriority: string
  focusTaskOne: string
  focusTaskTwo: string
  focusTaskThree: string
  blockers: string
  additionalNote: string
  plannedDeepWorkMinutes: number
  confidence: number
  selfRating: number
  completedTasksToday: number
  totalEstimatedMinutes: number
  totalTimeSpentMinutes: number
  habitsTotal: number
  habitsCompletedToday: number
  topPriorityCompleted: boolean
}

type CheckupSectionKey = 'context' | 'planning' | 'focus' | 'execution' | 'notes'

const DAILY_CHECKUP_MODAL_HINT_KEY = 'agentic-daily-checkup-modal-hint'

// Daily checkup card style variants using theme tokens
const dailyCheckupCardVariants = {
  card: 'rounded-2xl border border-border/70 bg-card/80 shadow-sm',
  header: 'rounded-t-2xl border border-border/80 bg-gradient-to-br from-muted/95 via-primary/5 to-background p-4',
  badgeRow: 'mb-2 flex items-center justify-between',
  kicker: 'inline-flex rounded-full bg-primary/20 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-primary',
  date: 'text-xs font-medium text-muted-foreground',
  focus: 'text-base font-semibold tracking-tight text-foreground',
  subtitle: 'mt-1 text-xs text-muted-foreground',
  metrics: 'grid grid-cols-1 gap-2 sm:grid-cols-3',
  metric: 'rounded-xl border border-border/80 bg-muted/80 px-3 py-2.5',
  metricLabel: 'text-[11px] font-medium uppercase tracking-wide text-muted-foreground',
  metricValue: 'mt-1 text-lg font-semibold text-foreground',
  panel: 'rounded-2xl border border-border/70 bg-card/80 p-4',
  panelTitle: 'text-xs font-semibold uppercase tracking-[0.12em] text-foreground',
  panelSubtitle: 'mt-1 text-xs text-muted-foreground',
  timeline: 'mt-3 space-y-2.5',
  block: 'list-none rounded-xl border p-3 shadow-sm',
  blockHigh: 'border-destructive/30 bg-destructive/10',
  blockMedium: 'border-amber-500/30 bg-amber-500/10',
  blockLow: 'border-emerald-500/30 bg-emerald-500/10',
  timeWrap: 'mb-1.5 flex items-center justify-between',
  time: 'text-[11px] font-semibold tracking-wide text-foreground',
  priority: 'text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground',
  blockTitle: 'text-sm font-semibold text-foreground',
  blockReason: 'mt-1 text-xs leading-relaxed text-muted-foreground',
  notes: 'mt-3 space-y-1.5 pl-4 [&>li]:list-disc [&>li]:text-xs [&>li]:leading-relaxed',
  journal: 'bg-gradient-to-br from-muted/90 to-primary/5',
  journalQ: 'mt-2 text-xs leading-relaxed text-foreground',
}

const createInitialCheckupForm = (): CheckupFormState => ({
  topPriority: '',
  focusTaskOne: '',
  focusTaskTwo: '',
  focusTaskThree: '',
  blockers: '',
  additionalNote: '',
  plannedDeepWorkMinutes: 120,
  confidence: 7,
  selfRating: 7,
  completedTasksToday: 0,
  totalEstimatedMinutes: 120,
  totalTimeSpentMinutes: 0,
  habitsTotal: 0,
  habitsCompletedToday: 0,
  topPriorityCompleted: false,
})

const toFiniteNumber = (value: unknown, fallback = 0): number => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const toDisplayLabel = (value: unknown, fallback = 'N/A'): string => {
  if (typeof value === 'string') {
    const normalized = value.trim()
    if (!normalized || normalized === '[object Object]') {
      return fallback
    }
    return normalized
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  if (Array.isArray(value)) {
    const joined = value
      .map((item) => toDisplayLabel(item, ''))
      .filter(Boolean)
      .join(', ')

    return joined || fallback
  }

  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const preferredKeys = ['title', 'name', 'label', 'category', 'agent', 'value']

    for (const key of preferredKeys) {
      if (key in record) {
        const candidate = toDisplayLabel(record[key], '')
        if (candidate) {
          return candidate
        }
      }
    }

    const scalarEntry = Object.values(record).find((entry) => {
      const entryType = typeof entry
      return entryType === 'string' || entryType === 'number' || entryType === 'boolean'
    })

    if (scalarEntry !== undefined) {
      return toDisplayLabel(scalarEntry, fallback)
    }
  }

  return fallback
}

const toColor = (value: unknown, fallback: string): string => {
  if (typeof value !== 'string') {
    return fallback
  }

  const normalized = value.trim()
  return normalized || fallback
}

const parseDateCandidate = (value: unknown): Date | null => {
  if (!value || typeof value !== 'string') {
    return null
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return null
  }

  return parsed
}

const resolveGoalDeadline = (goal: OnboardingGoal): Date | null => {
  const smartCriteria = goal.smartCriteria || {}
  const preferredKeys = [
    'dueDate',
    'due_date',
    'deadline',
    'targetDate',
    'target_date',
    'endDate',
    'end_date',
  ]

  for (const key of preferredKeys) {
    const candidate = parseDateCandidate(smartCriteria[key])
    if (candidate) {
      return candidate
    }
  }

  const milestones = Array.isArray(goal.milestones) ? goal.milestones : []
  for (const milestone of milestones) {
    if (!milestone || typeof milestone !== 'object') {
      continue
    }

    for (const key of preferredKeys) {
      const candidate = parseDateCandidate(milestone[key])
      if (candidate) {
        return candidate
      }
    }
  }

  return null
}

const formatCheckupTime = (checkupDate: string | undefined) => {
  if (!checkupDate) {
    return 'Not run yet'
  }

  const date = new Date(checkupDate)
  if (Number.isNaN(date.getTime())) {
    return 'Not run yet'
  }

  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const formatMinutesLabel = (value: unknown): string => {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return '0m'
  }
  return formatMinutesToHoursMinutes(numericValue)
}

const stripHtmlTags = (value?: string) => {
  const normalized = (value || '').trim()
  if (!normalized) {
    return ''
  }

  return normalized
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const sanitizeCheckupHtml = (value: string) => {
  if (typeof window === 'undefined') {
    return value
  }

  const parser = new window.DOMParser()
  const documentFragment = parser.parseFromString(value, 'text/html')

  documentFragment
    .querySelectorAll('script, style, iframe, object, embed, link, meta')
    .forEach((element) => element.remove())

  documentFragment.querySelectorAll('*').forEach((element) => {
    Array.from(element.attributes).forEach((attribute) => {
      const attributeName = attribute.name.toLowerCase()
      const attributeValue = attribute.value || ''

      if (attributeName.startsWith('on')) {
        element.removeAttribute(attribute.name)
        return
      }

      if (attributeName === 'style') {
        element.removeAttribute(attribute.name)
        return
      }

      if ((attributeName === 'href' || attributeName === 'src') && /^\s*javascript:/i.test(attributeValue)) {
        element.removeAttribute(attribute.name)
      }
    })
  })

  return documentFragment.body.innerHTML.trim()
}

const toRenderableCheckupHtml = (rawMessage?: string) => {
  const normalized = (rawMessage || '').trim()
  if (!normalized) {
    return ''
  }

  const looksLikeHtml = /<\/?[a-z][\s\S]*>/i.test(normalized)
  if (looksLikeHtml) {
    const sanitized = sanitizeCheckupHtml(normalized)
    if (sanitized) {
      return sanitized
    }

    const textFallback = stripHtmlTags(normalized)
    return textFallback ? `<p>${escapeHtml(textFallback)}</p>` : ''
  }

  return normalized
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join('')
}

const summarizeCheckupMessage = (message?: string) => {
  const normalized = stripHtmlTags(message).replace(/\s+/g, ' ')
  if (!normalized) {
    return 'No checkup result yet.'
  }

  return normalized.length > 140 ? `${normalized.slice(0, 137)}...` : normalized
}

const summarizeCheckupList = (items: string[] | undefined, label: string): string | null => {
  if (!Array.isArray(items) || items.length === 0) {
    return null
  }

  const normalizedItems = items
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 2)

  if (normalizedItems.length === 0) {
    return null
  }

  return `${label}: ${normalizedItems.join('; ')}`
}

const buildCheckupCardSummary = (
  checkup: DailyCheckupResponse | null,
  checkupType: 'morning' | 'evening',
): string => {
  if (!checkup) {
    return 'No checkup result yet.'
  }

  const fragments: string[] = []

  if (typeof checkup.focus_target === 'string' && checkup.focus_target.trim()) {
    fragments.push(`Focus: ${checkup.focus_target.trim()}`)
  }

  if (checkupType === 'morning' && typeof checkup.intent_note === 'string' && checkup.intent_note.trim()) {
    fragments.push(`Intent: ${checkup.intent_note.trim()}`)
  }

  if (checkupType === 'evening' && typeof checkup.reflection_note === 'string' && checkup.reflection_note.trim()) {
    fragments.push(`Reflection: ${checkup.reflection_note.trim()}`)
  }

  const winsSummary = summarizeCheckupList(checkup.wins, 'Wins')
  if (winsSummary) {
    fragments.push(winsSummary)
  }

  const blockersSummary = summarizeCheckupList(checkup.blockers, 'Blockers')
  if (blockersSummary) {
    fragments.push(blockersSummary)
  }

  const tomorrowSummary = summarizeCheckupList(checkup.tomorrow_focus, 'Tomorrow')
  if (tomorrowSummary) {
    fragments.push(tomorrowSummary)
  }

  const messageSource =
    (checkup.coach_message_html && checkup.coach_message_html.trim())
      ? checkup.coach_message_html
      : checkup.coach_message
  const messageSummary = summarizeCheckupMessage(messageSource)

  if (fragments.length < 3 && messageSummary !== 'No checkup result yet.') {
    fragments.push(messageSummary)
  }

  if (fragments.length === 0) {
    return messageSummary
  }

  const combined = fragments.join(' | ')
  return combined.length > 220 ? `${combined.slice(0, 217)}...` : combined
}

const normalizeNotificationSeverity = (value: unknown): 'low' | 'medium' | 'high' | 'critical' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'low' || normalized === 'medium' || normalized === 'high' || normalized === 'critical') {
    return normalized
  }
  return 'medium'
}

const normalizeNotificationStatus = (value: unknown): 'active' | 'acknowledged' | 'resolved' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'active' || normalized === 'acknowledged' || normalized === 'resolved') {
    return normalized
  }
  return 'active'
}

const getNotificationTone = (severity: string) => {
  const normalized = normalizeNotificationSeverity(severity)
  if (normalized === 'critical') {
    return {
      wrapper: 'border-rose-300/80 bg-rose-50/70 dark:border-rose-900/70 dark:bg-rose-950/30',
      badge: 'bg-rose-100 text-rose-700 dark:bg-rose-900/60 dark:text-rose-200',
    }
  }

  if (normalized === 'high') {
    return {
      wrapper: 'border-amber-300/80 bg-amber-50/70 dark:border-amber-900/70 dark:bg-amber-950/30',
      badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/60 dark:text-amber-200',
    }
  }

  if (normalized === 'low') {
    return {
      wrapper: 'border-emerald-300/80 bg-emerald-50/70 dark:border-emerald-900/70 dark:bg-emerald-950/30',
      badge: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-200',
    }
  }

  return {
    wrapper: 'border-cyan-300/80 bg-cyan-50/70 dark:border-cyan-900/70 dark:bg-cyan-950/30',
    badge: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/60 dark:text-cyan-200',
  }
}

const formatNotificationTimestamp = (value: string | undefined) => {
  if (!value) {
    return 'n/a'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return 'n/a'
  }

  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  className
}) => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTimeRange, setSelectedTimeRange] = useState<'7d' | '30d' | '90d'>('30d')
  const [checkupLoading, setCheckupLoading] = useState<'morning' | 'evening' | null>(null)
  const [checkupError, setCheckupError] = useState<string | null>(null)
  const [morningCheckup, setMorningCheckup] = useState<DailyCheckupResponse | null>(null)
  const [eveningCheckup, setEveningCheckup] = useState<DailyCheckupResponse | null>(null)
  const [isCheckupModalOpen, setIsCheckupModalOpen] = useState(false)
  const [activeCheckupFlow, setActiveCheckupFlow] = useState<'morning' | 'evening'>('morning')
  const [collapsedCheckupSections, setCollapsedCheckupSections] = useState<Record<CheckupSectionKey, boolean>>({
    context: false,
    planning: false,
    focus: false,
    execution: false,
    notes: false,
  })
  const [profileData, setProfileData] = useState<OnboardingProfileData | null>(null)
  const [morningForm, setMorningForm] = useState<CheckupFormState>(() => createInitialCheckupForm())
  const [eveningForm, setEveningForm] = useState<CheckupFormState>(() => createInitialCheckupForm())
  const [habitMetrics, setHabitMetrics] = useState<{
    totalHabits: number
    completedToday: number
    completionRate7d: number | null
  }>({ totalHabits: 0, completedToday: 0, completionRate7d: null })
  const [aiNotifications, setAiNotifications] = useState<AINotification[]>([])
  const [notificationsError, setNotificationsError] = useState<string | null>(null)
  const [notificationsPersistenceEnabled, setNotificationsPersistenceEnabled] = useState(false)
  const [isNotificationsLoading, setIsNotificationsLoading] = useState(false)
  const [isNotificationRefreshRunning, setIsNotificationRefreshRunning] = useState(false)

  // AbortControllers for canceling pending requests
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map())

  const cancelPendingRequests = useCallback(() => {
    abortControllersRef.current.forEach((controller) => controller.abort())
    abortControllersRef.current.clear()
  }, [])

  const fetchWithCancellation = useCallback(async (
    key: string,
    url: string,
    options?: RequestInit
  ): Promise<Response | null> => {
    // Cancel any existing request with same key
    const existing = abortControllersRef.current.get(key)
    if (existing) {
      existing.abort()
    }

    const controller = new AbortController()
    abortControllersRef.current.set(key, controller)

    try {
      const response = await fetch(url, { ...options, signal: controller.signal })
      abortControllersRef.current.delete(key)
      return response
    } catch (error) {
      abortControllersRef.current.delete(key)
      if (error instanceof Error && error.name === 'AbortError') {
        return null
      }
      throw error
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelPendingRequests()
    }
  }, [cancelPendingRequests])

  useEffect(() => {
    loadAnalyticsData()
    loadLatestCheckups()
    loadOnboardingProfile()
  }, [selectedTimeRange])

  const toStringArray = (value: unknown): string[] | undefined => {
    if (!Array.isArray(value)) {
      return undefined
    }

    const normalized = value
      .map((item) => (typeof item === 'string' ? item.trim() : String(item ?? '').trim()))
      .filter(Boolean)

    return normalized.length ? normalized : undefined
  }

  const parseStoredCheckup = (entry: DailyCheckupInsightEntry): DailyCheckupResponse | null => {
    const metadata = entry.metadata
    if (!metadata || typeof metadata !== 'object') {
      return null
    }

    const checkupType = metadata.checkup_type
    if (checkupType !== 'morning' && checkupType !== 'evening') {
      return null
    }

    const checkupDate =
      typeof metadata.checkup_date === 'string' && metadata.checkup_date.trim()
        ? metadata.checkup_date
        : (entry.updated_at || entry.created_at || new Date().toISOString()).slice(0, 10)

    const coachMessage =
      typeof metadata.coach_message === 'string' && metadata.coach_message.trim()
        ? metadata.coach_message
        : (entry.content || '').trim()

    const coachMessageHtml =
      typeof metadata.coach_message_html === 'string' && metadata.coach_message_html.trim()
        ? metadata.coach_message_html
        : undefined

    const normalizedCoachMessage = coachMessage || stripHtmlTags(coachMessageHtml)

    if (!normalizedCoachMessage && !coachMessageHtml) {
      return null
    }

    const generatedWith =
      typeof metadata.generated_with === 'string' && metadata.generated_with.trim()
        ? metadata.generated_with
        : 'stored'

    const focusTarget =
      typeof metadata.focus_target === 'string' && metadata.focus_target.trim()
        ? metadata.focus_target
        : undefined

    const intentNote =
      typeof metadata.intent_note === 'string' ? metadata.intent_note : null

    const reflectionNote =
      typeof metadata.reflection_note === 'string' ? metadata.reflection_note : null

    const stats =
      metadata.stats && typeof metadata.stats === 'object'
        ? (metadata.stats as Record<string, unknown>)
        : undefined

    return {
      date: checkupDate,
      checkup_type: checkupType,
      coach_message: normalizedCoachMessage || '',
      coach_message_html: coachMessageHtml,
      generated_with: generatedWith,
      focus_target: focusTarget,
      intent_note: intentNote,
      reflection_note: reflectionNote,
      wins: toStringArray(metadata.wins),
      blockers: toStringArray(metadata.blockers),
      tomorrow_focus: toStringArray(metadata.tomorrow_focus),
      stats,
    }
  }

  const normalizeNotification = (raw: unknown): AINotification | null => {
    if (!raw || typeof raw !== 'object') {
      return null
    }

    const source = raw as Record<string, unknown>
    const id = Number(source.id)
    const notificationKey = String(source.notification_key || '').trim()
    const title = String(source.title || '').trim()
    const summary = String(source.summary || '').trim()

    if (!Number.isFinite(id) || !notificationKey || !title || !summary) {
      return null
    }

    const recommendedActions = Array.isArray(source.recommended_actions)
      ? source.recommended_actions.map((item) => String(item || '').trim()).filter(Boolean)
      : []

    const payload = source.payload && typeof source.payload === 'object'
      ? (source.payload as Record<string, unknown>)
      : {}

    return {
      id,
      notification_key: notificationKey,
      kind: String(source.kind || 'signal').trim() || 'signal',
      severity: normalizeNotificationSeverity(source.severity),
      status: normalizeNotificationStatus(source.status),
      title,
      summary,
      details: source.details ? String(source.details).trim() : null,
      score: source.score === null || source.score === undefined ? null : toFiniteNumber(source.score, 0),
      recommended_actions: recommendedActions,
      payload,
      first_seen_at: source.first_seen_at ? String(source.first_seen_at) : undefined,
      last_seen_at: source.last_seen_at ? String(source.last_seen_at) : undefined,
      acknowledged_at: source.acknowledged_at ? String(source.acknowledged_at) : null,
      resolved_at: source.resolved_at ? String(source.resolved_at) : null,
      updated_at: source.updated_at ? String(source.updated_at) : undefined,
    }
  }

  const applyNotificationEnvelope = (payload: unknown) => {
    const envelope = payload && typeof payload === 'object'
      ? (payload as AINotificationEnvelope)
      : {}

    const notifications = Array.isArray(envelope.notifications)
      ? envelope.notifications.map((entry) => normalizeNotification(entry)).filter((entry): entry is AINotification => Boolean(entry))
      : []

    setAiNotifications(notifications)
    setNotificationsPersistenceEnabled(Boolean(envelope.persistence_enabled))
  }

  const loadAiNotifications = async (options?: { refresh?: boolean }) => {
    const shouldRefresh = Boolean(options?.refresh)

    setIsNotificationsLoading(true)
    setNotificationsError(null)
    if (shouldRefresh) {
      setIsNotificationRefreshRunning(true)
    }

    try {
      const refreshUrl = `/api/knowledge/notifications/refresh?limit=30&range=${selectedTimeRange}`
      const listUrl = '/api/knowledge/notifications?limit=30'
      const response = await fetch(shouldRefresh ? refreshUrl : listUrl, {
        method: shouldRefresh ? 'POST' : 'GET',
      })

      if (!response.ok) {
        throw new Error(`Notifications request failed with status ${response.status}`)
      }

      const payload = await response.json()
      applyNotificationEnvelope(payload)
    } catch (error) {
      console.error('Failed to load AI notifications:', error)
      const fallbackMessage = 'Unable to load AI notifications right now.'
      setNotificationsError(error instanceof Error && error.message ? error.message : fallbackMessage)
    } finally {
      setIsNotificationsLoading(false)
      if (shouldRefresh) {
        setIsNotificationRefreshRunning(false)
      }
    }
  }

  const setNotificationAcknowledgement = async (notificationId: number, acknowledged: boolean) => {
    try {
      const response = await fetch(`/api/knowledge/notifications/${notificationId}/ack`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ acknowledged }),
      })

      if (!response.ok) {
        throw new Error(`Notification update failed with status ${response.status}`)
      }

      const payload = await response.json()
      const normalized = normalizeNotification(payload)
      if (!normalized) {
        return
      }

      setAiNotifications((previous) =>
        previous.map((entry) => (entry.id === normalized.id ? normalized : entry)),
      )
    } catch (error) {
      console.error('Failed to update notification status:', error)
      setNotificationsError('Unable to update notification status right now.')
    }
  }

  const loadLatestCheckups = async (options?: { preserveCurrent?: boolean }) => {
    const applyResolvedCheckups = (latestMorning: DailyCheckupResponse | null, latestEvening: DailyCheckupResponse | null) => {
      if (options?.preserveCurrent) {
        setMorningCheckup((previous) => latestMorning || previous)
        setEveningCheckup((previous) => latestEvening || previous)
      } else {
        setMorningCheckup(latestMorning)
        setEveningCheckup(latestEvening)
      }
    }

    try {
      const latestResponse = await fetch('/api/knowledge/checkups/latest')
      if (latestResponse.ok) {
        const latestPayload = await latestResponse.json()
        const latestMorning = parseStoredCheckup({ metadata: latestPayload?.morning })
        const latestEvening = parseStoredCheckup({ metadata: latestPayload?.evening })

        // Null-check before applying to prevent downstream crashes
        if (latestMorning || latestEvening) {
          applyResolvedCheckups(latestMorning ?? null, latestEvening ?? null)
          return
        }
      }
    } catch (error) {
      console.error('Failed to load latest checkups from database endpoint:', error)
    }

    try {
      const response = await fetch('/api/knowledge/entries?entry_type=insight&category=daily_checkup')
      if (!response.ok) {
        throw new Error(`Checkup entries request failed with status ${response.status}`)
      }

      const payload = await response.json()
      const entries = Array.isArray(payload)
        ? (payload as DailyCheckupInsightEntry[])
        : Array.isArray(payload?.entries)
          ? (payload.entries as DailyCheckupInsightEntry[])
          : []

      const sortedEntries = [...entries].sort((a, b) => {
        const aTime = new Date(a.updated_at || a.created_at || 0).getTime()
        const bTime = new Date(b.updated_at || b.created_at || 0).getTime()
        return bTime - aTime
      })

      let latestMorning: DailyCheckupResponse | null = null
      let latestEvening: DailyCheckupResponse | null = null

      for (const entry of sortedEntries) {
        const parsed = parseStoredCheckup(entry)
        if (!parsed) {
          continue
        }

        if (parsed.checkup_type === 'morning' && !latestMorning) {
          latestMorning = parsed
        }

        if (parsed.checkup_type === 'evening' && !latestEvening) {
          latestEvening = parsed
        }

        if (latestMorning && latestEvening) {
          break
        }
      }

      applyResolvedCheckups(latestMorning, latestEvening)
    } catch (error) {
      console.error('Failed to load saved checkups:', error)
    }
  }

  const loadOnboardingProfile = async () => {
    try {
      const response = await fetch('/api/knowledge/onboarding/profile')
      if (!response.ok) {
        throw new Error(`Profile request failed with status ${response.status}`)
      }

      const payload = await response.json()
      const normalizedProfile: OnboardingProfileData = {
        goals: Array.isArray(payload?.goals) ? payload.goals : [],
        schedule: payload?.schedule && typeof payload.schedule === 'object' ? payload.schedule : null,
        planner: payload?.planner && typeof payload.planner === 'object' ? payload.planner : null,
        preferenceProfile:
          payload?.preferenceProfile && typeof payload.preferenceProfile === 'object'
            ? payload.preferenceProfile
            : null,
      }

      setProfileData(normalizedProfile)
    } catch (error) {
      console.error('Failed to load onboarding profile:', error)
      setProfileData(null)
    }
  }

  const loadHabitMetrics = async (): Promise<void> => {
    // Today's local date key (re-derived here so this helper has no React deps)
    const now = new Date()
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`

    const applyMetrics = (raw: Record<string, unknown> | null | undefined) => {
      if (!raw || typeof raw !== 'object') {
        return false
      }

      const summary = (raw.summary && typeof raw.summary === 'object'
        ? (raw.summary as Record<string, unknown>)
        : raw) as Record<string, unknown>

      const totalHabits = toFiniteNumber(summary.totalHabits ?? summary.total_habits ?? raw.totalHabits ?? raw.total_habits, NaN)

      const dailyCounts = (summary.dailyCompletionCounts
        || summary.daily_completion_counts
        || raw.dailyCompletionCounts
        || raw.daily_completion_counts) as Record<string, unknown> | undefined
      let completedToday = NaN
      if (dailyCounts && typeof dailyCounts === 'object') {
        completedToday = toFiniteNumber(dailyCounts[today], NaN)
      }
      if (!Number.isFinite(completedToday)) {
        completedToday = toFiniteNumber(
          summary.completedToday
          ?? summary.completed_today
          ?? raw.completedToday
          ?? raw.completed_today,
          NaN,
        )
      }

      const rate7d = toFiniteNumber(
        summary.completionRate7d
        ?? summary.completion_rate_7d
        ?? raw.completionRate7d
        ?? raw.completion_rate_7d,
        NaN,
      )

      if (!Number.isFinite(totalHabits) && !Number.isFinite(completedToday)) {
        return false
      }

      setHabitMetrics({
        totalHabits: Math.max(0, Number.isFinite(totalHabits) ? Math.round(totalHabits) : 0),
        completedToday: Math.max(0, Number.isFinite(completedToday) ? Math.round(completedToday) : 0),
        completionRate7d: Number.isFinite(rate7d) ? Math.max(0, Math.min(100, rate7d)) : null,
      })
      return true
    }

    // 1) Try analytics endpoint first — it might already expose habit_metrics.
    try {
      const analyticsResponse = await fetch(`/api/knowledge/analytics?range=30d`)
      if (analyticsResponse.ok) {
        const analyticsPayload = await analyticsResponse.json()
        const habitBlock =
          analyticsPayload?.habit_metrics
          ?? analyticsPayload?.insights?.habit_metrics
          ?? analyticsPayload?.habits
        if (applyMetrics(habitBlock as Record<string, unknown> | null | undefined)) {
          return
        }
      }
    } catch (error) {
      console.warn('Habit metrics: analytics endpoint failed, falling back to entries:', error)
    }

    // 2) Fall back to the latest habit_snapshot knowledge entry.
    try {
      const entriesResponse = await fetch('/api/knowledge/entries?category=habit_snapshot&limit=1')
      if (!entriesResponse.ok) {
        throw new Error(`habit_snapshot request failed with status ${entriesResponse.status}`)
      }
      const entriesPayload = await entriesResponse.json()
      const list: unknown[] = Array.isArray(entriesPayload)
        ? entriesPayload
        : Array.isArray(entriesPayload?.entries)
          ? entriesPayload.entries
          : Array.isArray(entriesPayload?.results)
            ? entriesPayload.results
            : []
      const latest = list[0]
      if (latest && typeof latest === 'object') {
        const record = latest as Record<string, unknown>
        const metadata = (record.metadata && typeof record.metadata === 'object'
          ? record.metadata
          : {}) as Record<string, unknown>
        const context = (metadata.context && typeof metadata.context === 'object'
          ? metadata.context
          : {}) as Record<string, unknown>

        // Try parsing the entry content as JSON if structured habit data was stored there.
        let parsedContent: Record<string, unknown> | null = null
        if (typeof record.content === 'string') {
          try {
            const candidate = JSON.parse(record.content)
            if (candidate && typeof candidate === 'object') {
              parsedContent = candidate as Record<string, unknown>
            }
          } catch {
            // content was not JSON — ignore.
          }
        }

        for (const candidate of [parsedContent, context, metadata, record]) {
          if (applyMetrics(candidate as Record<string, unknown> | null)) {
            return
          }
        }
      }
    } catch (error) {
      console.warn('Habit metrics: snapshot fallback failed, defaulting to 0/0:', error)
    }

    // 3) Default — keep prior state, but ensure we never block the UI.
    setHabitMetrics((previous) => previous)
  }

  const loadAnalyticsData = async (options?: { silent?: boolean }) => {
    const shouldToggleLoading = !options?.silent
    if (shouldToggleLoading) {
      setIsLoading(true)
    }

    try {
      const response = await fetch(`/api/knowledge/analytics?range=${selectedTimeRange}`)
      if (!response.ok) {
        throw new Error(`Analytics request failed with status ${response.status}`)
      }

      const data = await response.json()
      const normalizedData: AnalyticsData = {
        interactions: {
          daily: Array.isArray(data.interactions?.daily)
            ? data.interactions.daily.map((entry: Record<string, unknown>) => ({
                date: toDisplayLabel(entry?.date, new Date().toISOString()),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
                agent: toDisplayLabel(entry?.agent, 'Unknown'),
              }))
            : [],
          weekly: Array.isArray(data.interactions?.weekly)
            ? data.interactions.weekly.map((entry: Record<string, unknown>) => ({
                week: toDisplayLabel(entry?.week, 'Unknown'),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
              }))
            : [],
          by_agent: Array.isArray(data.interactions?.by_agent)
            ? data.interactions.by_agent.map((entry: Record<string, unknown>, index: number) => ({
                agent: toDisplayLabel(entry?.agent, `Agent ${index + 1}`),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
                color: toColor(entry?.color, '#3b82f6'),
              }))
            : [],
          by_category: Array.isArray(data.interactions?.by_category)
            ? data.interactions.by_category.map((entry: Record<string, unknown>, index: number) => ({
                category: toDisplayLabel(entry?.category, `Category ${index + 1}`),
                raw_category: toDisplayLabel(entry?.raw_category, `category_${index + 1}`),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
                color: toColor(entry?.color, '#06b6d4'),
              }))
            : [],
        },
        patterns: {
          most_active_hours: Array.isArray(data.patterns?.most_active_hours)
            ? data.patterns.most_active_hours.map((entry: Record<string, unknown>) => ({
                hour: Math.max(0, Math.min(23, Math.round(toFiniteNumber(entry?.hour, 0)))),
                interactions: Math.max(0, toFiniteNumber(entry?.interactions, 0)),
              }))
            : [],
          preference_changes: Array.isArray(data.patterns?.preference_changes)
            ? data.patterns.preference_changes.map((entry: Record<string, unknown>) => ({
                date: toDisplayLabel(entry?.date, new Date().toISOString()),
                category: toDisplayLabel(entry?.category, 'Unknown'),
                changes: Math.max(0, toFiniteNumber(entry?.changes, 0)),
              }))
            : [],
          knowledge_growth: Array.isArray(data.patterns?.knowledge_growth)
            ? data.patterns.knowledge_growth.map((entry: Record<string, unknown>) => ({
                date: toDisplayLabel(entry?.date, new Date().toISOString()),
                total_entries: Math.max(0, toFiniteNumber(entry?.total_entries, 0)),
                new_entries: Math.max(0, toFiniteNumber(entry?.new_entries, 0)),
              }))
            : [],
          category_focus: Array.isArray(data.patterns?.category_focus)
            ? data.patterns.category_focus.map((entry: Record<string, unknown>) => ({
                date: toDisplayLabel(entry?.date, new Date().toISOString()),
                category: toDisplayLabel(entry?.category, 'Unknown'),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
              }))
            : [],
        },
        insights: {
          total_interactions: Math.max(0, toFiniteNumber(data.insights?.total_interactions, 0)),
          most_used_agent: toDisplayLabel(data.insights?.most_used_agent, 'N/A'),
          avg_daily_interactions: Math.max(0, toFiniteNumber(data.insights?.avg_daily_interactions, 0)),
          knowledge_base_size: Math.max(0, toFiniteNumber(data.insights?.knowledge_base_size, 0)),
          preference_stability: Math.max(0, Math.min(1, toFiniteNumber(data.insights?.preference_stability, 0))),
          learning_velocity: Math.max(0, toFiniteNumber(data.insights?.learning_velocity, 0)),
          top_knowledge_category: toDisplayLabel(data.insights?.top_knowledge_category, 'N/A'),
          time_entry_records: Math.max(0, toFiniteNumber(data.insights?.time_entry_records, 0)),
          time_entry_billable_records: Math.max(0, toFiniteNumber(data.insights?.time_entry_billable_records, 0)),
          avg_time_entry_minutes: Math.max(0, toFiniteNumber(data.insights?.avg_time_entry_minutes, 0)),
          time_entry_total_minutes: Math.max(0, toFiniteNumber(data.insights?.time_entry_total_minutes, 0)),
          time_entry_count: Math.max(0, toFiniteNumber(data.insights?.time_entry_count, 0)),
          time_entry_daily: Array.isArray(data.insights?.time_entry_daily)
            ? data.insights.time_entry_daily.map((entry: Record<string, unknown>) => ({
                date: toDisplayLabel(entry?.date, ''),
                total_minutes: Math.max(0, toFiniteNumber(entry?.total_minutes, 0)),
                count: Math.max(0, toFiniteNumber(entry?.count, 0)),
              }))
            : undefined,
        },
      }

      setAnalyticsData(normalizedData)
      // Habit metrics live alongside analytics — fetch them in the background.
      void loadHabitMetrics()
    } catch (error) {
      console.error('Failed to load analytics data:', error)
      setAnalyticsData(null)
    } finally {
      if (shouldToggleLoading) {
        setIsLoading(false)
      }
    }
  }

  const runDailyCheckup = async (checkupType: 'morning' | 'evening') => {
    setCheckupError(null)
    setCheckupLoading(checkupType)

    try {
      const requestPayload = buildCheckupRequestPayload(checkupType)
      const response = await fetch(`/api/knowledge/checkups/${checkupType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      })

      if (!response.ok) {
        let errorDetail = `Checkup failed with status ${response.status}`
        try {
          const errorPayload = await response.json()
          const candidate =
            (typeof errorPayload?.detail === 'string' && errorPayload.detail.trim()) ||
            (typeof errorPayload?.message === 'string' && errorPayload.message.trim())

          if (candidate) {
            errorDetail = candidate
          }
        } catch {
          // Ignore JSON parse failures and keep status-based fallback detail.
        }

        throw new Error(errorDetail)
      }

      const payload = (await response.json()) as DailyCheckupResponse
      if (checkupType === 'morning') {
        setMorningCheckup(payload)
        setMorningForm((previous) => ({
          ...previous,
          blockers: '',
          additionalNote: '',
          confidence: Math.min(10, previous.confidence + 1),
        }))
      } else {
        setEveningCheckup(payload)
        setEveningForm((previous) => ({
          ...previous,
          blockers: '',
          additionalNote: '',
          selfRating: Math.min(10, previous.selfRating + 1),
        }))
      }

      // Keep the checkup UX stable: avoid full analytics refresh right after submit.
      // A background checkup-entry refresh is enough to confirm persistence.
      void loadLatestCheckups({ preserveCurrent: true })
    } catch (error) {
      console.error(`Failed to run ${checkupType} checkup:`, error)
      const fallbackMessage = 'Unable to run checkup right now. Please try again.'
      setCheckupError(error instanceof Error && error.message ? error.message : fallbackMessage)
    } finally {
      setCheckupLoading(null)
    }
  }

  const formatHour = (hour: number) => {
    return `${hour.toString().padStart(2, '0')}:00`
  }

  const getAgentIcon = (agent: string) => {
    switch (agent.toLowerCase()) {
      case 'orchestrator': return Brain
      case 'productivity': return Zap
      case 'health': return Heart
      case 'finance': return DollarSign
      case 'journal': return BookOpen
      default: return Activity
    }
  }

  const normalizeDateKey = (value?: string | null): string | null => {
    if (!value || typeof value !== 'string') {
      return null
    }

    const trimmed = value.trim()
    if (!trimmed) {
      return null
    }

    return trimmed.slice(0, 10)
  }

  const toLocalDateKey = (value: Date): string => {
    const year = value.getFullYear()
    const month = String(value.getMonth() + 1).padStart(2, '0')
    const day = String(value.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const todayDateKey = toLocalDateKey(new Date())
  const morningCompletedToday = normalizeDateKey(morningCheckup?.date) === todayDateKey
  const eveningCompletedToday = normalizeDateKey(eveningCheckup?.date) === todayDateKey
  const canRunEveningCheckup = morningCompletedToday || eveningCompletedToday

  const sortedNotifications = useMemo(() => {
    const severityRank: Record<string, number> = {
      critical: 0,
      high: 1,
      medium: 2,
      low: 3,
    }
    const statusRank: Record<string, number> = {
      active: 0,
      acknowledged: 1,
      resolved: 2,
    }

    return [...aiNotifications].sort((left, right) => {
      const statusDiff = (statusRank[left.status] ?? 3) - (statusRank[right.status] ?? 3)
      if (statusDiff !== 0) {
        return statusDiff
      }

      const severityDiff = (severityRank[left.severity] ?? 9) - (severityRank[right.severity] ?? 9)
      if (severityDiff !== 0) {
        return severityDiff
      }

      const leftTime = new Date(left.last_seen_at || left.updated_at || 0).getTime()
      const rightTime = new Date(right.last_seen_at || right.updated_at || 0).getTime()
      return rightTime - leftTime
    })
  }, [aiNotifications])

  const activeNotificationCount = sortedNotifications.filter((entry) => entry.status === 'active').length
  const highSeverityNotificationCount = sortedNotifications.filter(
    (entry) => entry.status === 'active' && (entry.severity === 'high' || entry.severity === 'critical'),
  ).length
  const acknowledgedNotificationCount = sortedNotifications.filter((entry) => entry.status === 'acknowledged').length

  const profileGoals = useMemo(() => {
    if (!Array.isArray(profileData?.goals)) {
      return [] as OnboardingGoal[]
    }

    return profileData.goals
  }, [profileData])

  const goalDeadlineSummary = useMemo(() => {
    const now = new Date()
    now.setHours(0, 0, 0, 0)

    const topGoals = profileGoals
      .map((goal) => String(goal.title || '').trim())
      .filter(Boolean)
      .slice(0, 3)

    const upcoming: Array<{ title: string; dueDate: string; daysLeft: number; priority: string }> = []
    let overdue = 0
    let dueToday = 0

    for (const goal of profileGoals) {
      const deadline = resolveGoalDeadline(goal)
      if (!deadline) {
        continue
      }

      const dueDate = new Date(deadline)
      dueDate.setHours(0, 0, 0, 0)
      const daysLeft = Math.round((dueDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24))

      if (daysLeft < 0) {
        overdue += 1
      }
      if (daysLeft === 0) {
        dueToday += 1
      }

      upcoming.push({
        title: String(goal.title || 'Untitled goal'),
        dueDate: dueDate.toISOString(),
        daysLeft,
        priority: String(goal.priority || 'Medium'),
      })
    }

    upcoming.sort((left, right) => left.daysLeft - right.daysLeft)

    return {
      topGoals,
      overdue,
      dueToday,
      upcoming: upcoming.slice(0, 6),
    }
  }, [profileGoals])

  const yesterdaySnapshot = useMemo(() => {
    if (!analyticsData) {
      return {
        interactions: 0,
        estimatedFocusedMinutes: 0,
        avgSessionMinutes: 0,
      }
    }

    const yesterdayKey = toLocalDateKey(new Date(Date.now() - 24 * 60 * 60 * 1000))
    const dailyEntries = [...analyticsData.interactions.daily].sort((a, b) => a.date.localeCompare(b.date))

    const exactYesterday = dailyEntries.find((entry) => entry.date.slice(0, 10) === yesterdayKey)
    const fallbackRecent = [...dailyEntries].reverse().find((entry) => entry.date.slice(0, 10) < todayDateKey)
    const baselineEntry = exactYesterday || fallbackRecent || null

    // interactions is the chat-interaction count for yesterday — kept for display only.
    // It MUST NOT be used as a proxy for focused work time.
    const interactions = baselineEntry ? toFiniteNumber(baselineEntry.count, 0) : 0

    // Prefer per-day breakdown if backend provides one (`insights.time_entry_daily`).
    // Otherwise fall back to the aggregate totals as a best-effort estimate.
    // TODO: When the backend exposes a daily time_entry breakdown, scope this strictly
    // to `yesterdayKey` only. Until then, the aggregate range total is the closest signal.
    const dailyTimeEntries = analyticsData.insights.time_entry_daily
    let totalTrackedMinutes = 0
    let sessionCount = 0
    if (Array.isArray(dailyTimeEntries) && dailyTimeEntries.length > 0) {
      const yesterdayBucket =
        dailyTimeEntries.find((entry) => entry.date.slice(0, 10) === yesterdayKey)
        || [...dailyTimeEntries].reverse().find((entry) => entry.date.slice(0, 10) < todayDateKey)
      if (yesterdayBucket) {
        totalTrackedMinutes = Math.max(0, toFiniteNumber(yesterdayBucket.total_minutes, 0))
        sessionCount = Math.max(0, toFiniteNumber(yesterdayBucket.count, 0))
      }
    } else {
      totalTrackedMinutes = Math.max(0, toFiniteNumber(analyticsData.insights.time_entry_total_minutes, 0))
      sessionCount = Math.max(0, toFiniteNumber(analyticsData.insights.time_entry_count, 0))
    }

    const avgSessionMinutes = sessionCount > 0 ? totalTrackedMinutes / sessionCount : 0
    // Cap at one day's worth of minutes to keep downstream math sane.
    const estimatedFocusedMinutes = Math.min(totalTrackedMinutes, 1440)

    return {
      interactions,
      estimatedFocusedMinutes,
      avgSessionMinutes,
    }
  }, [analyticsData, todayDateKey])

  const updateMorningForm = (updater: (previous: CheckupFormState) => CheckupFormState) => {
    setMorningForm((previous) => updater(previous))
  }

  const updateEveningForm = (updater: (previous: CheckupFormState) => CheckupFormState) => {
    setEveningForm((previous) => updater(previous))
  }

  // Use refs to track if forms have been initialized to prevent infinite loops
  const morningFormInitializedRef = useRef(false)
  const eveningFormInitializedRef = useRef(false)

  useEffect(() => {
    if (!analyticsData) {
      return
    }

    setMorningForm((previous) => {
      // Guard: only initialize once and only if fields are empty
      if (morningFormInitializedRef.current || previous.topPriority || previous.focusTaskOne || previous.additionalNote) {
        return previous
      }
      morningFormInitializedRef.current = true

      return {
        ...previous,
        topPriority: goalDeadlineSummary.topGoals[0] || '',
        plannedDeepWorkMinutes: Math.max(90, Math.min(480, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120))),
        totalEstimatedMinutes: Math.max(120, Math.min(1440, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120))),
        totalTimeSpentMinutes: Math.max(0, Math.min(1440, yesterdaySnapshot.estimatedFocusedMinutes)),
        habitsTotal: Math.max(previous.habitsTotal, habitMetrics.totalHabits),
        habitsCompletedToday: Math.max(previous.habitsCompletedToday, habitMetrics.completedToday),
      }
    })

    setEveningForm((previous) => {
      // Guard: only initialize once and only if fields are empty
      if (eveningFormInitializedRef.current || previous.focusTaskOne || previous.additionalNote) {
        return previous
      }
      eveningFormInitializedRef.current = true

      return {
        ...previous,
        focusTaskOne: goalDeadlineSummary.topGoals[0] || '',
        totalEstimatedMinutes: Math.max(120, Math.min(1440, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120))),
        habitsTotal: Math.max(previous.habitsTotal, habitMetrics.totalHabits),
        habitsCompletedToday: Math.max(previous.habitsCompletedToday, habitMetrics.completedToday),
      }
    })
  }, [
    analyticsData,
    goalDeadlineSummary.topGoals,
    yesterdaySnapshot.estimatedFocusedMinutes,
    habitMetrics.totalHabits,
    habitMetrics.completedToday,
  ])

  // Habit metrics may arrive after the initial form sync (separate fetch).
  // Backfill the form whenever the backend reports values and the user has
  // not yet entered their own.
  useEffect(() => {
    if (habitMetrics.totalHabits === 0 && habitMetrics.completedToday === 0) {
      return
    }
    setMorningForm((previous) => {
      if (previous.habitsTotal !== 0 || previous.habitsCompletedToday !== 0) {
        return previous
      }
      return {
        ...previous,
        habitsTotal: habitMetrics.totalHabits,
        habitsCompletedToday: habitMetrics.completedToday,
      }
    })
    setEveningForm((previous) => {
      if (previous.habitsTotal !== 0 || previous.habitsCompletedToday !== 0) {
        return previous
      }
      return {
        ...previous,
        habitsTotal: habitMetrics.totalHabits,
        habitsCompletedToday: habitMetrics.completedToday,
      }
    })
  }, [habitMetrics.totalHabits, habitMetrics.completedToday])

  const openCheckupFlow = (flow: 'morning' | 'evening') => {
    setActiveCheckupFlow(flow)
    setCheckupError(null)
    setIsCheckupModalOpen(true)
  }

  const activeCheckup = activeCheckupFlow === 'morning' ? morningCheckup : eveningCheckup
  const activeForm = activeCheckupFlow === 'morning' ? morningForm : eveningForm
  const activeCheckupRenderedHtml = useMemo(() => {
    const messageSource =
      (activeCheckup?.coach_message_html && activeCheckup.coach_message_html.trim())
        ? activeCheckup.coach_message_html
        : activeCheckup?.coach_message

    return toRenderableCheckupHtml(messageSource)
  }, [activeCheckup?.coach_message, activeCheckup?.coach_message_html])

  const updateActiveFormField = <K extends keyof CheckupFormState>(field: K, value: CheckupFormState[K]) => {
    if (activeCheckupFlow === 'morning') {
      updateMorningForm((previous) => ({ ...previous, [field]: value }))
      return
    }

    updateEveningForm((previous) => ({ ...previous, [field]: value }))
  }

  const activeFocusTasks = [
    activeForm.focusTaskOne,
    activeForm.focusTaskTwo,
    activeForm.focusTaskThree,
  ]

  const activeNote = activeForm.additionalNote

  const setActiveNote = (value: string) => {
    updateActiveFormField('additionalNote', value)
  }

  const toggleCheckupSection = (section: CheckupSectionKey) => {
    setCollapsedCheckupSections((previous) => ({
      ...previous,
      [section]: !previous[section],
    }))
  }

  const activeDeepWorkCoverage = activeForm.totalEstimatedMinutes > 0
    ? (activeForm.totalTimeSpentMinutes / activeForm.totalEstimatedMinutes) * 100
    : 0

  const activeHabitCompletionToday = activeForm.habitsTotal > 0
    ? (activeForm.habitsCompletedToday / activeForm.habitsTotal) * 100
    : 0

  const quickPromptsByFlow: Record<'morning' | 'evening', string[]> = {
    morning: [
      'My single highest-leverage outcome today is...',
      'I will protect deep-work time between...',
      'The risk that could derail today is...',
    ],
    evening: [
      'Today my strongest win was...',
      'The blocker I need to resolve tomorrow is...',
      'One habit I should tighten tomorrow is...',
    ],
  }

  const applyQuickPrompt = (prompt: string) => {
    const merged = activeNote.trim() ? `${activeNote.trim()}\n${prompt}` : prompt
    setActiveNote(merged)
  }

  const buildCheckupRequestPayload = (checkupType: 'morning' | 'evening') => {
    const form = checkupType === 'morning' ? morningForm : eveningForm
    const focusTasks = [form.focusTaskOne, form.focusTaskTwo, form.focusTaskThree]
      .map((task) => task.trim())
      .filter(Boolean)

    const blockers = form.blockers
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    const estimatedMinutes = Math.max(0, toFiniteNumber(form.totalEstimatedMinutes, 0))
    const spentMinutes = Math.max(0, toFiniteNumber(form.totalTimeSpentMinutes, 0))
    const deepWorkCoverageRatio = estimatedMinutes > 0
      ? Number((spentMinutes / estimatedMinutes).toFixed(2))
      : 0

    // Today's completion rate — derived strictly from today's totals.
    const habitCompletionRateToday = form.habitsTotal > 0
      ? Number(((form.habitsCompletedToday / form.habitsTotal) * 100).toFixed(1))
      : 0

    // 7-day rolling completion rate — sourced from the latest habit_snapshot
    // when available, falling back to today's rate if the backend hasn't
    // produced a 7d aggregate yet.
    const backendRate7d = habitMetrics.completionRate7d
    const habitCompletionRate7d = backendRate7d !== null && Number.isFinite(backendRate7d)
      ? Number(backendRate7d.toFixed(1))
      : habitCompletionRateToday

    const contextSnapshot = {
      priorityFocus: form.topPriority.trim(),
      topGoals: goalDeadlineSummary.topGoals,
      focusTasks: focusTasks.map((title, index) => ({
        id: `focus-task-${index + 1}`,
        title,
      })),
      deadlineTasks: {
        overdue: goalDeadlineSummary.overdue,
        dueToday: goalDeadlineSummary.dueToday,
      },
      upcomingDeadlines: goalDeadlineSummary.upcoming.map((deadline) => ({
        title: deadline.title,
        dueDate: deadline.dueDate,
        daysLeft: deadline.daysLeft,
        priority: deadline.priority,
      })),
      habitMetrics: {
        totalHabits: Math.max(0, toFiniteNumber(form.habitsTotal, 0)),
        completedToday: Math.max(0, toFiniteNumber(form.habitsCompletedToday, 0)),
        habitCompletionRateToday: Math.max(0, Math.min(100, habitCompletionRateToday)),
        habitCompletionRate7d: Math.max(0, Math.min(100, habitCompletionRate7d)),
        // Kept under the legacy key for backward compatibility with consumers
        // that still read `completionRate7d` from this payload.
        completionRate7d: Math.max(0, Math.min(100, habitCompletionRate7d)),
      },
      timeMetrics: {
        totalEstimatedMinutes: estimatedMinutes,
        totalTimeSpentMinutes: spentMinutes,
        deepWorkCoverageRatio,
      },
      completedTasksToday: Math.max(0, toFiniteNumber(form.completedTasksToday, 0)),
      plannedDeepWorkMinutes: Math.max(0, toFiniteNumber(form.plannedDeepWorkMinutes, 0)),
      blockers,
    }

    const perspective = checkupType === 'morning'
      ? {
          confidence: Math.max(0, Math.min(10, toFiniteNumber(form.confidence, 0))),
          plannedDeepWorkMinutes: Math.max(0, toFiniteNumber(form.plannedDeepWorkMinutes, 0)),
        }
      : {
          selfRating: Math.max(0, Math.min(10, toFiniteNumber(form.selfRating, 0))),
          topPriorityCompleted: Boolean(form.topPriorityCompleted),
          plannedDeepWorkMinutes: Math.max(0, toFiniteNumber(form.plannedDeepWorkMinutes, 0)),
        }

    const noteSections = [
      form.topPriority.trim() ? `Top priority: ${form.topPriority.trim()}` : '',
      focusTasks.length ? `Focus tasks: ${focusTasks.join(', ')}` : '',
      blockers.length ? `Likely blockers: ${blockers.join(', ')}` : '',
      form.additionalNote.trim(),
    ].filter(Boolean)

    const clientTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'

    return {
      date: todayDateKey,
      timezone: clientTimezone,
      note: noteSections.join('\n'),
      perspective,
      context_snapshot: contextSnapshot,
    }
  }

  const handleRunActiveCheckup = async () => {
    await runDailyCheckup(activeCheckupFlow)
  }

  const checkupHeaderLabel = activeCheckupFlow === 'morning'
    ? 'Morning Strategic Setup'
    : 'Evening Goal Progression Review'

  const checkupSubLabel = activeCheckupFlow === 'morning'
    ? 'Start with intent before your first serious work block. The output gives focus, constraints, and execution posture.'
    : 'Close the loop on goal progression and carry only the right priorities into tomorrow.'

  const isActiveCheckupRunning = checkupLoading === activeCheckupFlow
  const eveningRunLocked = activeCheckupFlow === 'evening' && !canRunEveningCheckup

  useEffect(() => {
    if (!isCheckupModalOpen) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsCheckupModalOpen(false)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isCheckupModalOpen])

  useEffect(() => {
    if (typeof window === 'undefined' || isLoading || morningCompletedToday) {
      return
    }

    const alreadyPromptedForToday = window.localStorage.getItem(DAILY_CHECKUP_MODAL_HINT_KEY) === todayDateKey
    if (alreadyPromptedForToday) {
      return
    }

    setActiveCheckupFlow('morning')
    setIsCheckupModalOpen(true)
    window.localStorage.setItem(DAILY_CHECKUP_MODAL_HINT_KEY, todayDateKey)
  }, [isLoading, morningCompletedToday, todayDateKey])

  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center h-64", className)}>
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    )
  }

  if (!analyticsData) {
    return (
      <div className={cn("text-center py-12", className)}>
        <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Analytics Data</h3>
        <p className="text-muted-foreground">
          Start interacting with agents to see your usage patterns and insights.
        </p>
      </div>
    )
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
          <p className="text-muted-foreground">
            Insights into your AI agent interactions and learning patterns
          </p>
        </div>
        
        <div className="flex flex-wrap gap-2">
          {(['7d', '30d', '90d'] as const).map((range) => (
            <button
              type="button"
              key={range}
              onClick={() => setSelectedTimeRange(range)}
              className={cn(
                "px-3 py-1 text-sm rounded-md transition-colors",
                selectedTimeRange === range
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
              )}
            >
              {range === '7d' ? 'Last 7 days' : range === '30d' ? 'Last 30 days' : 'Last 90 days'}
            </button>
          ))}
        </div>
      </div>

      <Card className="relative overflow-hidden border-border/70 bg-gradient-to-br from-amber-50/80 via-white to-orange-50/50 p-5 dark:from-amber-950/25 dark:via-slate-950 dark:to-orange-950/20">
        <div
          aria-hidden
          className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-amber-300/35 blur-3xl dark:bg-amber-700/20"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-20 left-1/4 h-56 w-56 rounded-full bg-indigo-200/30 blur-3xl dark:bg-indigo-700/15"
        />

        <div className="relative mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="mb-1 inline-flex items-center gap-2 rounded-full border border-amber-200/70 bg-white/75 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-amber-700 dark:border-amber-900/60 dark:bg-slate-900/65 dark:text-amber-300">
              <Sparkles className="h-3.5 w-3.5" />
              Daily Ritual Intelligence
            </p>
            <h3 className="text-lg font-semibold">Daily AI Checkup Studio</h3>
            <p className="text-sm text-muted-foreground">
              Morning creates execution strategy. Evening closes loop on progression and tomorrow focus.
            </p>
          </div>
          <Badge variant="outline" className="self-start bg-white/70 text-[11px] sm:self-auto dark:bg-slate-900/70">
            {morningCompletedToday && eveningCompletedToday
              ? 'Both Complete Today'
              : morningCompletedToday || eveningCompletedToday
                ? 'In Progress Today'
                : 'Not Started Today'}
          </Badge>
        </div>

        {/* At-a-glance stat strip — auto-populated from tracker data */}
        <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <div className="flex flex-col rounded-xl border border-border/60 bg-white/70 px-3 py-2 dark:bg-slate-900/60">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Yesterday Tracked</span>
            <span className="mt-0.5 text-base font-bold text-foreground">
              {yesterdaySnapshot.estimatedFocusedMinutes > 0
                ? formatMinutesLabel(yesterdaySnapshot.estimatedFocusedMinutes)
                : '—'}
            </span>
            <span className="text-[10px] text-muted-foreground">from time tracker</span>
          </div>
          <div className="flex flex-col rounded-xl border border-border/60 bg-white/70 px-3 py-2 dark:bg-slate-900/60">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Habits Today</span>
            <span className="mt-0.5 text-base font-bold text-foreground">
              {habitMetrics.totalHabits > 0
                ? `${habitMetrics.completedToday}/${habitMetrics.totalHabits}`
                : '—'}
            </span>
            <span className="text-[10px] text-muted-foreground">
              {habitMetrics.totalHabits > 0
                ? `${Math.round((habitMetrics.completedToday / habitMetrics.totalHabits) * 100)}% done`
                : 'sync needed'}
            </span>
          </div>
          <div className="flex flex-col rounded-xl border border-border/60 bg-white/70 px-3 py-2 dark:bg-slate-900/60">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Goals Active</span>
            <span className="mt-0.5 text-base font-bold text-foreground">
              {goalDeadlineSummary.topGoals.length > 0 ? goalDeadlineSummary.topGoals.length : '—'}
            </span>
            <span className="text-[10px] text-muted-foreground">
              {goalDeadlineSummary.overdue > 0
                ? `${goalDeadlineSummary.overdue} overdue`
                : goalDeadlineSummary.dueToday > 0
                  ? `${goalDeadlineSummary.dueToday} due today`
                  : goalDeadlineSummary.upcoming[0]
                    ? `next in ${goalDeadlineSummary.upcoming[0].daysLeft}d`
                    : 'no deadlines'}
            </span>
          </div>
          <div className="flex flex-col rounded-xl border border-border/60 bg-white/70 px-3 py-2 dark:bg-slate-900/60">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Avg Session</span>
            <span className="mt-0.5 text-base font-bold text-foreground">
              {yesterdaySnapshot.avgSessionMinutes > 0
                ? formatMinutesLabel(yesterdaySnapshot.avgSessionMinutes)
                : '—'}
            </span>
            <span className="text-[10px] text-muted-foreground">per work block</span>
          </div>
        </div>

        <div className="relative mb-4 rounded-2xl border border-border/70 bg-white/70 p-4 dark:bg-slate-900/65">
          <div className="grid grid-cols-[auto,1fr,auto,1fr,auto] items-center gap-3">
            <div className={cn(
              'flex h-9 w-9 items-center justify-center rounded-full border',
              morningCompletedToday
                ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/20 dark:text-emerald-300'
                : 'border-amber-500/50 bg-amber-500/10 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/20 dark:text-amber-300',
            )}>
              {morningCompletedToday ? <CheckCircle2 className="h-4 w-4" /> : <Sunrise className="h-4 w-4" />}
            </div>
            <div className="h-px bg-gradient-to-r from-amber-300/70 to-indigo-300/70 dark:from-amber-700/60 dark:to-indigo-700/60" />
            <div className={cn(
              'flex h-9 w-9 items-center justify-center rounded-full border',
              eveningCompletedToday
                ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/20 dark:text-emerald-300'
                : canRunEveningCheckup
                  ? 'border-indigo-500/50 bg-indigo-500/10 text-indigo-700 dark:border-indigo-500/30 dark:bg-indigo-500/20 dark:text-indigo-300'
                  : 'border-muted bg-muted text-muted-foreground',
            )}>
              {eveningCompletedToday ? <CheckCircle2 className="h-4 w-4" /> : canRunEveningCheckup ? <MoonStar className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
            </div>
            <div className="h-px bg-gradient-to-r from-indigo-300/70 to-emerald-300/70 dark:from-indigo-700/60 dark:to-emerald-700/60" />
            <div className="flex h-9 w-9 items-center justify-center rounded-full border border-emerald-500/50 bg-emerald-500/10 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/20 dark:text-emerald-300">
              <Flag className="h-4 w-4" />
            </div>
          </div>
          <div className="mt-2 grid grid-cols-3 text-[11px] text-muted-foreground">
            <p>Morning Strategy</p>
            <p className="text-center">Evening Reflection</p>
            <p className="text-right">Tomorrow Focus</p>
          </div>
        </div>

        <div className="relative grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div
            className={cn(
              'rounded-2xl border p-4 transition-shadow',
              morningCompletedToday
                ? 'border-emerald-500/30 bg-emerald-500/5'
                : 'border-amber-500/30 bg-amber-500/5 shadow-[0_0_0_1px_rgba(251,191,36,0.15)]',
            )}
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold">Morning Strategic Setup</p>
                <p className="text-xs text-muted-foreground">Run before your first high-focus block.</p>
              </div>
              <Sunrise className="h-5 w-5 text-amber-600 dark:text-amber-300" />
            </div>
            <p className="mb-2 text-xs text-muted-foreground">
              Last run: {formatCheckupTime(morningCheckup?.date)}
            </p>
            <p className="mb-3 rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
              {buildCheckupCardSummary(morningCheckup, 'morning')}
            </p>
            <Button onClick={() => openCheckupFlow('morning')} className="w-full gap-2">
              <Compass className="h-4 w-4" />
              {morningCompletedToday ? 'Review Morning Plan' : 'Start Morning Plan'}
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>

          <div className="rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-4">
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold">Evening Goal Progression</p>
                <p className="text-xs text-muted-foreground">Review execution and lock tomorrow’s narrow focus.</p>
              </div>
              <MoonStar className="h-5 w-5 text-indigo-600 dark:text-indigo-300" />
            </div>
            <p className="mb-2 text-xs text-muted-foreground">
              Last run: {formatCheckupTime(eveningCheckup?.date)}
            </p>
            <p className="mb-3 rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
              {canRunEveningCheckup
                ? buildCheckupCardSummary(eveningCheckup, 'evening')
                : 'Evening checkup unlocks after morning checkup to preserve strategic sequence.'}
            </p>
            <Button
              type="button"
              onClick={() => openCheckupFlow('evening')}
              variant="secondary"
              className="w-full gap-2"
            >
              {canRunEveningCheckup ? <Flag className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
              {eveningCompletedToday ? 'Review Evening Reflection' : 'Open Evening Checkup'}
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {checkupError && (
          <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {checkupError}
          </div>
        )}
      </Card>

      {isCheckupModalOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-background/80 p-2 pt-[calc(1rem+env(safe-area-inset-top))] backdrop-blur-sm sm:items-center sm:p-4">
          <div className="flex h-[100dvh] w-[min(1100px,96vw)] flex-col overflow-hidden rounded-xl border border-border/70 bg-background shadow-2xl sm:h-[88vh] sm:rounded-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-border/70 px-5 py-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Interactive Daily Checkup</p>
                <h4 className="text-xl font-semibold">{checkupHeaderLabel}</h4>
                <p className="mt-1 text-sm text-muted-foreground">{checkupSubLabel}</p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setIsCheckupModalOpen(false)}
                aria-label="Close checkup modal"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="border-b border-border/70 px-5 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setActiveCheckupFlow('morning')}
                  className={cn(
                    'rounded-full border px-3 py-1.5 text-xs font-semibold transition',
                    activeCheckupFlow === 'morning'
                      ? 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300'
                      : 'border-border/70 bg-background text-muted-foreground hover:bg-secondary/50',
                  )}
                >
                  Morning Strategy
                </button>
                <button
                  type="button"
                  onClick={() => setActiveCheckupFlow('evening')}
                  className={cn(
                    'rounded-full border px-3 py-1.5 text-xs font-semibold transition',
                    activeCheckupFlow === 'evening'
                      ? 'border-indigo-300 bg-indigo-50 text-indigo-700 dark:border-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-300'
                      : 'border-border/70 bg-background text-muted-foreground hover:bg-secondary/50',
                    !canRunEveningCheckup && !eveningCompletedToday && 'opacity-70',
                  )}
                >
                  Evening Reflection
                </button>
                {!canRunEveningCheckup && !eveningCompletedToday && (
                  <Badge variant="outline" className="text-[10px]">
                    Unlocks After Morning Checkup
                  </Badge>
                )}
              </div>
            </div>

            <div className="grid min-h-0 flex-1 overflow-y-auto lg:grid-cols-[1.15fr,1fr] lg:overflow-hidden">
              <div className="border-b border-border/60 p-5 lg:overflow-y-auto lg:border-b-0 lg:border-r">
                <div className="mb-4 rounded-xl border border-border/70 bg-card/60 p-4">
                  <p className="text-sm font-medium">
                    {activeCheckupFlow === 'morning'
                      ? 'What is your highest leverage outcome today?'
                      : 'How did your actions move your top goals forward today?'}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Add context to improve recommendation quality and tactical specificity.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="overflow-hidden rounded-xl border border-border/70 bg-background/70">
                    <button
                      type="button"
                      onClick={() => toggleCheckupSection('context')}
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Context Snapshot</p>
                      <ChevronDown className={cn('h-4 w-4 transition-transform', !collapsedCheckupSections.context && 'rotate-180')} />
                    </button>
                    {!collapsedCheckupSections.context && (
                      <div className="grid gap-3 border-t border-border/60 p-3 sm:grid-cols-2">
                        <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            <ClipboardCheck className="h-3.5 w-3.5" />
                            Yesterday · from tracker
                          </p>
                          <div className="flex items-baseline gap-1">
                            <p className="text-lg font-bold text-foreground">
                              {formatMinutesLabel(yesterdaySnapshot.estimatedFocusedMinutes)}
                            </p>
                            <span className="text-xs text-muted-foreground">tracked</span>
                          </div>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {yesterdaySnapshot.interactions > 0
                              ? `${yesterdaySnapshot.interactions} sessions · avg ${formatMinutesLabel(yesterdaySnapshot.avgSessionMinutes)} each`
                              : yesterdaySnapshot.avgSessionMinutes > 0
                                ? `Avg session: ${formatMinutesLabel(yesterdaySnapshot.avgSessionMinutes)}`
                                : 'No sessions logged yesterday'}
                          </p>
                        </div>

                        <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            <CalendarClock className="h-3.5 w-3.5" />
                            Goal & Deadline Pressure
                          </p>
                          <p className="truncate text-sm font-semibold text-foreground">
                            {goalDeadlineSummary.topGoals[0] || 'No top goal set yet'}
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {goalDeadlineSummary.overdue} overdue • {goalDeadlineSummary.dueToday} due today
                          </p>
                          {goalDeadlineSummary.upcoming[0] && (
                            <p className="text-xs text-muted-foreground">
                              Next: {goalDeadlineSummary.upcoming[0].title} ({goalDeadlineSummary.upcoming[0].daysLeft}d)
                            </p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="overflow-hidden rounded-xl border border-border/70 bg-background/70">
                    <button
                      type="button"
                      onClick={() => toggleCheckupSection('planning')}
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Planning Inputs</p>
                      <ChevronDown className={cn('h-4 w-4 transition-transform', !collapsedCheckupSections.planning && 'rotate-180')} />
                    </button>
                    {!collapsedCheckupSections.planning && (
                      <div className="grid gap-3 border-t border-border/60 p-3 sm:grid-cols-2">
                        <label className="space-y-1">
                          <span className="text-xs font-medium text-muted-foreground">Top Priority Outcome</span>
                          <input
                            type="text"
                            value={activeForm.topPriority}
                            onChange={(event) => updateActiveFormField('topPriority', event.target.value)}
                            placeholder={
                              activeCheckupFlow === 'morning'
                                ? 'Ship one meaningful result before noon'
                                : 'The most important result I aimed for today'
                            }
                            className="w-full rounded-lg border border-border/70 bg-background px-3 py-2 text-sm"
                          />
                        </label>

                        <div className="space-y-1">
                          <span className="text-xs font-medium text-muted-foreground">Planned Deep Work</span>
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <input
                                type="number"
                                min={0}
                                max={16}
                                step={1}
                                value={Math.floor(activeForm.plannedDeepWorkMinutes / 60)}
                                onChange={(event) => {
                                  const h = Math.max(0, Math.min(16, Number(event.target.value) || 0))
                                  const m = activeForm.plannedDeepWorkMinutes % 60
                                  updateActiveFormField('plannedDeepWorkMinutes', h * 60 + m)
                                }}
                                className="w-14 rounded-lg border border-border/70 bg-background px-2 py-2 text-center text-sm"
                              />
                              <span className="text-xs text-muted-foreground">h</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <input
                                type="number"
                                min={0}
                                max={59}
                                step={15}
                                value={activeForm.plannedDeepWorkMinutes % 60}
                                onChange={(event) => {
                                  const m = Math.max(0, Math.min(59, Number(event.target.value) || 0))
                                  const h = Math.floor(activeForm.plannedDeepWorkMinutes / 60)
                                  updateActiveFormField('plannedDeepWorkMinutes', h * 60 + m)
                                }}
                                className="w-14 rounded-lg border border-border/70 bg-background px-2 py-2 text-center text-sm"
                              />
                              <span className="text-xs text-muted-foreground">m</span>
                            </div>
                          </div>
                          <p className="text-[11px] text-muted-foreground">
                            = {formatMinutesToHoursMinutes(activeForm.plannedDeepWorkMinutes)} total focused work
                          </p>
                        </div>

                        <label className="space-y-1">
                          <span className="text-xs font-medium text-muted-foreground">
                            {activeCheckupFlow === 'morning' ? 'Confidence Score' : 'Self Rating'} (0-10)
                          </span>
                          <input
                            type="range"
                            min={0}
                            max={10}
                            step={1}
                            value={activeCheckupFlow === 'morning' ? activeForm.confidence : activeForm.selfRating}
                            onChange={(event) => {
                              const parsed = Number(event.target.value)
                              const normalized = Number.isFinite(parsed) ? Math.max(0, Math.min(10, parsed)) : 0
                              if (activeCheckupFlow === 'morning') {
                                updateActiveFormField('confidence', normalized)
                              } else {
                                updateActiveFormField('selfRating', normalized)
                              }
                            }}
                            className="w-full"
                          />
                          <p className="text-xs text-muted-foreground">
                            {activeCheckupFlow === 'morning' ? activeForm.confidence : activeForm.selfRating}/10
                          </p>
                        </label>

                        <label className="space-y-1">
                          <span className="text-xs font-medium text-muted-foreground">Completed Tasks Today</span>
                          <input
                            type="number"
                            min={0}
                            value={activeForm.completedTasksToday}
                            onChange={(event) => {
                              const parsed = Number(event.target.value)
                              updateActiveFormField(
                                'completedTasksToday',
                                Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0,
                              )
                            }}
                            className="w-full rounded-lg border border-border/70 bg-background px-3 py-2 text-sm"
                          />
                        </label>
                      </div>
                    )}
                  </div>

                  <div className="overflow-hidden rounded-xl border border-border/70 bg-background/70">
                    <button
                      type="button"
                      onClick={() => toggleCheckupSection('focus')}
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        <ListTodo className="h-3.5 w-3.5" />
                        Focus Tasks
                      </p>
                      <ChevronDown className={cn('h-4 w-4 transition-transform', !collapsedCheckupSections.focus && 'rotate-180')} />
                    </button>
                    {!collapsedCheckupSections.focus && (
                      <div className="grid gap-2 border-t border-border/60 p-3">
                        {activeFocusTasks.map((task, index) => {
                          const fieldKey = index === 0 ? 'focusTaskOne' : index === 1 ? 'focusTaskTwo' : 'focusTaskThree'
                          return (
                            <input
                              key={`focus-task-${index + 1}`}
                              type="text"
                              value={task}
                              onChange={(event) => updateActiveFormField(fieldKey, event.target.value)}
                              placeholder={`Focus task ${index + 1}`}
                              className="w-full rounded-lg border border-border/70 bg-background px-3 py-2 text-sm"
                            />
                          )
                        })}
                      </div>
                    )}
                  </div>

                  <div className="overflow-hidden rounded-xl border border-border/70 bg-background/70">
                    <button
                      type="button"
                      onClick={() => toggleCheckupSection('execution')}
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Execution Metrics</p>
                      <ChevronDown className={cn('h-4 w-4 transition-transform', !collapsedCheckupSections.execution && 'rotate-180')} />
                    </button>
                    {!collapsedCheckupSections.execution && (
                      <div className="grid gap-3 border-t border-border/60 p-3 sm:grid-cols-2">
                        <div className="rounded-xl border border-border/70 bg-background/70 p-3">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            <Clock3 className="h-3.5 w-3.5" />
                            Time Metrics
                          </p>
                          <p className="mb-2 text-[11px] leading-relaxed text-muted-foreground/90">
                            Estimated = planned focused minutes for today. Spent = actual tracked minutes completed so far.
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            <label className="space-y-1">
                              <span className="text-[11px] text-muted-foreground">Estimated</span>
                              <input
                                type="number"
                                min={0}
                                step={15}
                                value={activeForm.totalEstimatedMinutes}
                                onChange={(event) => {
                                  const parsed = Number(event.target.value)
                                  updateActiveFormField(
                                    'totalEstimatedMinutes',
                                    Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
                                  )
                                }}
                                className="w-full rounded-md border border-border/70 bg-background px-2 py-1.5 text-sm"
                              />
                              <p className="text-[11px] text-muted-foreground">
                                = {formatMinutesToHoursMinutes(activeForm.totalEstimatedMinutes)}
                              </p>
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px] text-muted-foreground">Spent</span>
                              <input
                                type="number"
                                min={0}
                                step={15}
                                value={activeForm.totalTimeSpentMinutes}
                                onChange={(event) => {
                                  const parsed = Number(event.target.value)
                                  updateActiveFormField(
                                    'totalTimeSpentMinutes',
                                    Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
                                  )
                                }}
                                className="w-full rounded-md border border-border/70 bg-background px-2 py-1.5 text-sm"
                              />
                              <p className="text-[11px] text-muted-foreground">
                                = {formatMinutesToHoursMinutes(activeForm.totalTimeSpentMinutes)}
                              </p>
                            </label>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            Deep-work coverage: {Math.round(activeDeepWorkCoverage)}% — spent {formatMinutesToHoursMinutes(activeForm.totalTimeSpentMinutes)} of {formatMinutesToHoursMinutes(activeForm.totalEstimatedMinutes)}
                          </p>
                        </div>

                        <div className="rounded-xl border border-border/70 bg-background/70 p-3">
                          <p className="mb-2 flex items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            <span className="flex items-center gap-2">
                              <Activity className="h-3.5 w-3.5" />
                              Habit Pulse · from tracker
                            </span>
                            {activeForm.habitsTotal === 0 && (
                              <span className="text-[10px] font-normal normal-case text-amber-600 dark:text-amber-400">
                                Sync habits from AlterEgo
                              </span>
                            )}
                          </p>
                          {activeForm.habitsTotal > 0 ? (
                            <>
                              <div className="flex items-end gap-2">
                                <span className="text-lg font-bold text-foreground">{activeForm.habitsCompletedToday}</span>
                                <span className="mb-0.5 text-sm text-muted-foreground">/ {activeForm.habitsTotal} done</span>
                              </div>
                              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                                <div
                                  className="h-full rounded-full bg-green-500 transition-all"
                                  style={{ width: `${Math.round(activeHabitCompletionToday)}%` }}
                                />
                              </div>
                              <p className="mt-1.5 text-xs text-muted-foreground">
                                {Math.round(activeHabitCompletionToday)}% today
                                {habitMetrics.completionRate7d !== null && (
                                  <span className="ml-2 text-muted-foreground/70">· {habitMetrics.completionRate7d.toFixed(0)}% this week</span>
                                )}
                              </p>
                            </>
                          ) : (
                            <p className="text-xs text-muted-foreground italic">No habit data synced yet. Open AlterEgo → Settings → Sync to Agentic.</p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="overflow-hidden rounded-xl border border-border/70 bg-background/70">
                    <button
                      type="button"
                      onClick={() => toggleCheckupSection('notes')}
                      className="flex w-full items-center justify-between px-3 py-2 text-left"
                    >
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Risks & Notes</p>
                      <ChevronDown className={cn('h-4 w-4 transition-transform', !collapsedCheckupSections.notes && 'rotate-180')} />
                    </button>
                    {!collapsedCheckupSections.notes && (
                      <div className="space-y-3 border-t border-border/60 p-3">
                        {activeCheckupFlow === 'evening' && (
                          <label className="flex items-center gap-2 rounded-xl border border-indigo-200/70 bg-indigo-50/60 px-3 py-2 text-sm dark:border-indigo-900/60 dark:bg-indigo-950/20">
                            <input
                              type="checkbox"
                              checked={activeForm.topPriorityCompleted}
                              onChange={(event) => updateActiveFormField('topPriorityCompleted', event.target.checked)}
                              className="h-4 w-4"
                            />
                            <span>I completed my top priority outcome today.</span>
                          </label>
                        )}

                        <div className="rounded-xl border border-rose-200/70 bg-rose-50/60 p-3 dark:border-rose-900/60 dark:bg-rose-950/20">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-rose-700 dark:text-rose-300">
                            <AlertTriangle className="h-3.5 w-3.5" />
                            Known Blockers
                          </p>
                          <input
                            type="text"
                            value={activeForm.blockers}
                            onChange={(event) => updateActiveFormField('blockers', event.target.value)}
                            placeholder="Comma-separated blockers (e.g., unclear requirements, dependency wait)"
                            className="w-full rounded-lg border border-rose-200/80 bg-background px-3 py-2 text-sm dark:border-rose-900/70"
                          />
                        </div>

                        <div>
                          <p className="mb-1 text-xs font-medium text-muted-foreground">Additional Note</p>
                          <textarea
                            value={activeNote}
                            onChange={(event) => setActiveNote(event.target.value)}
                            placeholder={
                              activeCheckupFlow === 'morning'
                                ? 'Example: I need two uninterrupted deep-work blocks for architecture decisions and one stakeholder sync.'
                                : 'Example: I shipped one key task, but reactive context switching slowed execution quality.'
                            }
                            className="min-h-[120px] w-full rounded-xl border border-border/70 bg-background px-4 py-3 text-sm"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-4 space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Quick Strategic Prompts</p>
                  <div className="flex flex-wrap gap-2">
                    {quickPromptsByFlow[activeCheckupFlow].map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => applyQuickPrompt(prompt)}
                        className="rounded-full border border-border/70 bg-secondary/40 px-3 py-1 text-xs transition hover:bg-secondary/70"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mt-5 flex flex-col gap-2 sm:flex-row">
                  <Button
                    type="button"
                    onClick={() => void handleRunActiveCheckup()}
                    disabled={isActiveCheckupRunning || eveningRunLocked}
                    className="gap-2"
                  >
                    {activeCheckupFlow === 'morning' ? <Sunrise className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
                    {isActiveCheckupRunning
                      ? `Running ${activeCheckupFlow === 'morning' ? 'Morning' : 'Evening'} Checkup...`
                      : `Run ${activeCheckupFlow === 'morning' ? 'Morning' : 'Evening'} Checkup`}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setActiveNote('')}
                    disabled={isActiveCheckupRunning}
                  >
                    Clear Note
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      if (activeCheckupFlow === 'morning') {
                        setMorningForm(createInitialCheckupForm())
                      } else {
                        setEveningForm(createInitialCheckupForm())
                      }
                    }}
                    disabled={isActiveCheckupRunning}
                  >
                    Reset Inputs
                  </Button>
                </div>
              </div>

              <div className="bg-secondary/20 p-5 lg:overflow-y-auto">
                <div className="mb-4 flex items-center justify-between">
                  <p className="text-sm font-semibold">Latest Output</p>
                  {activeCheckup && <Badge variant="secondary">{activeCheckup.generated_with}</Badge>}
                </div>

                {!activeCheckup ? (
                  <div className="rounded-xl border border-dashed border-border/70 bg-background/70 p-6 text-sm text-muted-foreground">
                    {activeCheckupFlow === 'morning'
                      ? 'Run your morning checkup to generate focus direction, constraints, and execution posture.'
                      : 'Run your evening checkup to reflect on progression, blockers, and tomorrow focus.'}
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="rounded-xl border border-border/70 bg-background/80 p-4">
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        {activeCheckupFlow === 'morning' ? 'Strategic Focus' : 'Reflection Summary'}
                      </p>
                      {activeCheckupRenderedHtml ? (
                        <div
                          className={PREMIUM_CHECKUP_HTML_CLASSNAMES}
                          dangerouslySetInnerHTML={{ __html: activeCheckupRenderedHtml }}
                        />
                      ) : (
                        <p className="text-sm leading-relaxed text-muted-foreground">No checkup output yet.</p>
                      )}
                    </div>

                    {activeCheckup.focus_target && (
                      <div className="rounded-xl border border-border/70 bg-background/80 p-4">
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Focus Target</p>
                        <p className="mt-1 text-sm font-medium text-foreground">{activeCheckup.focus_target}</p>
                      </div>
                    )}

                    {!!activeCheckup.wins?.length && (
                      <div className="rounded-xl border border-emerald-200/70 bg-emerald-50/50 p-4 dark:border-emerald-900/60 dark:bg-emerald-950/25">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">Wins</p>
                        <ul className="space-y-1 text-sm text-emerald-900 dark:text-emerald-100">
                          {activeCheckup.wins.map((item) => (
                            <li key={item}>• {item}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {!!activeCheckup.blockers?.length && (
                      <div className="rounded-xl border border-rose-200/70 bg-rose-50/50 p-4 dark:border-rose-900/60 dark:bg-rose-950/25">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-rose-700 dark:text-rose-300">Blockers</p>
                        <ul className="space-y-1 text-sm text-rose-900 dark:text-rose-100">
                          {activeCheckup.blockers.map((item) => (
                            <li key={item}>• {item}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {!!activeCheckup.tomorrow_focus?.length && (
                      <div className="rounded-xl border border-indigo-200/70 bg-indigo-50/50 p-4 dark:border-indigo-900/60 dark:bg-indigo-950/25">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-indigo-700 dark:text-indigo-300">Tomorrow Focus</p>
                        <ul className="space-y-1 text-sm text-indigo-900 dark:text-indigo-100">
                          {activeCheckup.tomorrow_focus.map((item) => (
                            <li key={item}>• {item}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col-reverse gap-3 border-t border-border/70 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs text-muted-foreground">
                {eveningRunLocked
                  ? 'Complete your morning checkup first to unlock evening progression review.'
                  : 'Checkups are persisted as knowledge insights and reused in future coaching context.'}
              </p>
              <div className="flex gap-2 self-end sm:self-auto">
                <Button type="button" variant="outline" onClick={() => setIsCheckupModalOpen(false)}>
                  Close
                </Button>
                {activeCheckupFlow === 'morning' && (
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => setActiveCheckupFlow('evening')}
                    disabled={!canRunEveningCheckup && !eveningCompletedToday}
                  >
                    Go To Evening Flow
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{analyticsData.insights.total_interactions}</p>
              <p className="text-sm text-muted-foreground">Total Interactions</p>
            </div>
          </div>
        </Card>
        
        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{analyticsData.insights.avg_daily_interactions.toFixed(1)}</p>
              <p className="text-sm text-muted-foreground">Daily Average</p>
            </div>
          </div>
        </Card>
        
        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{analyticsData.insights.knowledge_base_size}</p>
              <p className="text-sm text-muted-foreground">Knowledge Entries</p>
            </div>
          </div>
        </Card>
        
        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{(analyticsData.insights.preference_stability * 100).toFixed(0)}%</p>
              <p className="text-sm text-muted-foreground">Preference Stability</p>
            </div>
          </div>
        </Card>

        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-xl font-bold">{analyticsData.insights.top_knowledge_category}</p>
              <p className="text-sm text-muted-foreground">Top Category</p>
            </div>
          </div>
        </Card>

        <Card className="min-h-[96px] p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 flex items-center justify-center">
              <Clock3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{analyticsData.insights.time_entry_records}</p>
              <p className="text-sm text-muted-foreground">Time Entry Records</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Charts */}
      <Tabs defaultValue="interactions" className="space-y-6">
        <TabsList className="flex w-full overflow-x-auto">
          <TabsTrigger value="interactions" className="gap-2 shrink-0">
            <BarChart3 className="w-4 h-4" />
            Interactions
          </TabsTrigger>
          <TabsTrigger value="agents" className="gap-2 shrink-0">
            <PieChartIcon className="w-4 h-4" />
            Agent Usage
          </TabsTrigger>
          <TabsTrigger value="patterns" className="gap-2 shrink-0">
            <LineChartIcon className="w-4 h-4" />
            Patterns
          </TabsTrigger>
          <TabsTrigger value="categories" className="gap-2 shrink-0">
            <Layers className="w-4 h-4" />
            Categories
          </TabsTrigger>
          <TabsTrigger value="growth" className="gap-2 shrink-0">
            <TrendingUp className="w-4 h-4" />
            Growth
          </TabsTrigger>
        </TabsList>

        {/* Interactions Tab */}
        <TabsContent value="interactions" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Daily Interactions</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.interactions.daily}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Activity by Hour</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analyticsData.patterns.most_active_hours}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="hour" 
                    tickFormatter={formatHour}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => `${formatHour(value as number)}`}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="interactions" 
                    stroke="#10b981" 
                    fill="#10b981" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </TabsContent>

        {/* Agent Usage Tab */}
        <TabsContent value="agents" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Agent Usage Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={analyticsData.interactions.by_agent}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {analyticsData.interactions.by_agent.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Agent Rankings</h3>
              <div className="space-y-3">
                {analyticsData.interactions.by_agent
                  .sort((a, b) => b.count - a.count)
                  .map((agent, index) => {
                    const Icon = getAgentIcon(agent.agent)
                    return (
                      <div key={agent.agent} className="flex items-center gap-3">
                        <div className="flex items-center gap-2 flex-1">
                          <div 
                            className="w-8 h-8 rounded-lg flex items-center justify-center"
                            style={{ backgroundColor: agent.color }}
                          >
                            <Icon className="w-4 h-4 text-white" />
                          </div>
                          <span className="font-medium">{agent.agent}</span>
                        </div>
                        <Badge variant="secondary">{agent.count} interactions</Badge>
                        <div className="text-sm text-muted-foreground">#{index + 1}</div>
                      </div>
                    )
                  })}
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Patterns Tab */}
        <TabsContent value="patterns" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Preference Changes Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analyticsData.patterns.preference_changes}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="changes" 
                    stroke="#8b5cf6" 
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Dominant Category By Day</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.patterns.category_focus}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis />
                  <Tooltip
                    labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <Bar dataKey="count" fill="#06b6d4" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="categories" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Knowledge Categories</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.interactions.by_category}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {analyticsData.interactions.by_category.map((entry) => (
                      <Cell key={entry.raw_category} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-6 space-y-4">
              <h3 className="text-lg font-semibold">Time Entry Intelligence</h3>
              <div className="rounded-xl border bg-secondary/40 p-4">
                <p className="text-sm text-muted-foreground">Billable Entries</p>
                <p className="text-2xl font-bold">{analyticsData.insights.time_entry_billable_records}</p>
              </div>
              <div className="rounded-xl border bg-secondary/40 p-4">
                <p className="text-sm text-muted-foreground">Average Logged Time</p>
                <p className="text-2xl font-bold">{formatMinutesLabel(analyticsData.insights.avg_time_entry_minutes)}</p>
              </div>
              <div className="rounded-xl border bg-secondary/40 p-4">
                <p className="text-sm text-muted-foreground">Most Used Agent</p>
                <p className="truncate text-2xl font-bold">{analyticsData.insights.most_used_agent}</p>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Growth Tab */}
        <TabsContent value="growth" className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Knowledge Base Growth</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={analyticsData.patterns.knowledge_growth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Area 
                  type="monotone" 
                  dataKey="total_entries" 
                  stackId="1"
                  stroke="#3b82f6" 
                  fill="#3b82f6" 
                  fillOpacity={0.6}
                />
                <Area 
                  type="monotone" 
                  dataKey="new_entries" 
                  stackId="2"
                  stroke="#10b981" 
                  fill="#10b981" 
                  fillOpacity={0.8}
                />
                <Legend />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}