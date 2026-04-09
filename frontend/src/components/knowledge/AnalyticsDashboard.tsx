import React, { useState, useEffect, useMemo } from 'react'
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
import { cn } from '@/lib/utils'

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

const PREMIUM_CHECKUP_HTML_CLASSNAMES = [
  'text-sm leading-relaxed text-foreground',
  '[&_.daily-checkup]:space-y-4',
  '[&_.dc-header]:rounded-2xl',
  '[&_.dc-header]:border',
  '[&_.dc-header]:border-slate-200/80',
  '[&_.dc-header]:bg-gradient-to-br',
  '[&_.dc-header]:from-slate-50/95',
  '[&_.dc-header]:via-cyan-50/70',
  '[&_.dc-header]:to-white',
  '[&_.dc-header]:p-4',
  'dark:[&_.dc-header]:border-slate-700/70',
  'dark:[&_.dc-header]:from-slate-900/80',
  'dark:[&_.dc-header]:via-cyan-950/40',
  'dark:[&_.dc-header]:to-slate-900/90',
  '[&_.dc-badge-row]:mb-2',
  '[&_.dc-badge-row]:flex',
  '[&_.dc-badge-row]:items-center',
  '[&_.dc-badge-row]:justify-between',
  '[&_.dc-kicker]:inline-flex',
  '[&_.dc-kicker]:rounded-full',
  '[&_.dc-kicker]:bg-cyan-100',
  '[&_.dc-kicker]:px-2.5',
  '[&_.dc-kicker]:py-1',
  '[&_.dc-kicker]:text-[10px]',
  '[&_.dc-kicker]:font-semibold',
  '[&_.dc-kicker]:uppercase',
  '[&_.dc-kicker]:tracking-[0.12em]',
  '[&_.dc-kicker]:text-cyan-700',
  'dark:[&_.dc-kicker]:bg-cyan-900/50',
  'dark:[&_.dc-kicker]:text-cyan-100',
  '[&_.dc-date]:text-xs',
  '[&_.dc-date]:font-medium',
  '[&_.dc-date]:text-slate-500',
  'dark:[&_.dc-date]:text-slate-300',
  '[&_.dc-focus]:text-base',
  '[&_.dc-focus]:font-semibold',
  '[&_.dc-focus]:tracking-tight',
  '[&_.dc-subtitle]:mt-1',
  '[&_.dc-subtitle]:text-xs',
  '[&_.dc-subtitle]:text-slate-600',
  'dark:[&_.dc-subtitle]:text-slate-300',
  '[&_.dc-metrics]:grid',
  '[&_.dc-metrics]:grid-cols-1',
  '[&_.dc-metrics]:gap-2',
  'sm:[&_.dc-metrics]:grid-cols-3',
  '[&_.dc-metric]:rounded-xl',
  '[&_.dc-metric]:border',
  '[&_.dc-metric]:border-slate-200/80',
  '[&_.dc-metric]:bg-slate-50/80',
  '[&_.dc-metric]:px-3',
  '[&_.dc-metric]:py-2.5',
  'dark:[&_.dc-metric]:border-slate-700/70',
  'dark:[&_.dc-metric]:bg-slate-800/60',
  '[&_.dc-metric-label]:text-[11px]',
  '[&_.dc-metric-label]:font-medium',
  '[&_.dc-metric-label]:uppercase',
  '[&_.dc-metric-label]:tracking-wide',
  '[&_.dc-metric-label]:text-slate-500',
  'dark:[&_.dc-metric-label]:text-slate-300',
  '[&_.dc-metric-value]:mt-1',
  '[&_.dc-metric-value]:text-lg',
  '[&_.dc-metric-value]:font-semibold',
  '[&_.dc-metric-value]:text-slate-900',
  'dark:[&_.dc-metric-value]:text-slate-100',
  '[&_.dc-panel]:rounded-2xl',
  '[&_.dc-panel]:border',
  '[&_.dc-panel]:border-slate-200/70',
  '[&_.dc-panel]:bg-white/80',
  '[&_.dc-panel]:p-4',
  'dark:[&_.dc-panel]:border-slate-700/70',
  'dark:[&_.dc-panel]:bg-slate-900/60',
  '[&_.dc-panel-title]:text-xs',
  '[&_.dc-panel-title]:font-semibold',
  '[&_.dc-panel-title]:uppercase',
  '[&_.dc-panel-title]:tracking-[0.12em]',
  '[&_.dc-panel-title]:text-slate-600',
  'dark:[&_.dc-panel-title]:text-slate-200',
  '[&_.dc-panel-subtitle]:mt-1',
  '[&_.dc-panel-subtitle]:text-xs',
  '[&_.dc-panel-subtitle]:text-slate-500',
  'dark:[&_.dc-panel-subtitle]:text-slate-300',
  '[&_.dc-timeline]:mt-3',
  '[&_.dc-timeline]:space-y-2.5',
  '[&_.dc-block]:list-none',
  '[&_.dc-block]:rounded-xl',
  '[&_.dc-block]:border',
  '[&_.dc-block]:p-3',
  '[&_.dc-block]:shadow-sm',
  '[&_.dc-block--high]:border-rose-200',
  '[&_.dc-block--high]:bg-rose-50/80',
  '[&_.dc-block--medium]:border-amber-200',
  '[&_.dc-block--medium]:bg-amber-50/80',
  '[&_.dc-block--low]:border-emerald-200',
  '[&_.dc-block--low]:bg-emerald-50/80',
  'dark:[&_.dc-block--high]:border-rose-800/70',
  'dark:[&_.dc-block--high]:bg-rose-950/30',
  'dark:[&_.dc-block--medium]:border-amber-800/70',
  'dark:[&_.dc-block--medium]:bg-amber-950/30',
  'dark:[&_.dc-block--low]:border-emerald-800/70',
  'dark:[&_.dc-block--low]:bg-emerald-950/30',
  '[&_.dc-time-wrap]:mb-1.5',
  '[&_.dc-time-wrap]:flex',
  '[&_.dc-time-wrap]:items-center',
  '[&_.dc-time-wrap]:justify-between',
  '[&_.dc-time]:text-[11px]',
  '[&_.dc-time]:font-semibold',
  '[&_.dc-time]:tracking-wide',
  '[&_.dc-time]:text-slate-700',
  'dark:[&_.dc-time]:text-slate-100',
  '[&_.dc-priority]:text-[10px]',
  '[&_.dc-priority]:font-semibold',
  '[&_.dc-priority]:uppercase',
  '[&_.dc-priority]:tracking-[0.12em]',
  '[&_.dc-priority]:text-slate-500',
  'dark:[&_.dc-priority]:text-slate-300',
  '[&_.dc-block-title]:text-sm',
  '[&_.dc-block-title]:font-semibold',
  '[&_.dc-block-title]:text-slate-900',
  'dark:[&_.dc-block-title]:text-slate-100',
  '[&_.dc-block-reason]:mt-1',
  '[&_.dc-block-reason]:text-xs',
  '[&_.dc-block-reason]:leading-relaxed',
  '[&_.dc-block-reason]:text-slate-600',
  'dark:[&_.dc-block-reason]:text-slate-300',
  '[&_.dc-notes]:mt-3',
  '[&_.dc-notes]:space-y-1.5',
  '[&_.dc-notes]:pl-4',
  '[&_.dc-notes>li]:list-disc',
  '[&_.dc-notes>li]:text-xs',
  '[&_.dc-notes>li]:leading-relaxed',
  '[&_.dc-journal]:bg-gradient-to-br',
  '[&_.dc-journal]:from-slate-50/90',
  '[&_.dc-journal]:to-cyan-50/70',
  'dark:[&_.dc-journal]:from-slate-900/80',
  'dark:[&_.dc-journal]:to-cyan-950/40',
  '[&_.dc-journal-q]:mt-2',
  '[&_.dc-journal-q]:text-xs',
  '[&_.dc-journal-q]:leading-relaxed',
  '[&_.dc-journal-q]:text-slate-700',
  'dark:[&_.dc-journal-q]:text-slate-200',
].join(' ')

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

        if (latestMorning || latestEvening) {
          applyResolvedCheckups(latestMorning, latestEvening)
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
      const entries = Array.isArray(payload?.entries)
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
        },
      }

      setAnalyticsData(normalizedData)
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

    const interactions = baselineEntry ? toFiniteNumber(baselineEntry.count, 0) : 0
    const avgSessionMinutes = Math.max(0, toFiniteNumber(analyticsData.insights.avg_time_entry_minutes, 0))
    const estimatedFocusedMinutes = Math.round(interactions * avgSessionMinutes)

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

  useEffect(() => {
    if (!analyticsData) {
      return
    }

    setMorningForm((previous) => {
      if (previous.topPriority || previous.focusTaskOne || previous.additionalNote) {
        return previous
      }

      return {
        ...previous,
        topPriority: goalDeadlineSummary.topGoals[0] || '',
        plannedDeepWorkMinutes: Math.max(90, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120)),
        totalEstimatedMinutes: Math.max(120, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120)),
        totalTimeSpentMinutes: Math.max(0, yesterdaySnapshot.estimatedFocusedMinutes),
      }
    })

    setEveningForm((previous) => {
      if (previous.focusTaskOne || previous.additionalNote) {
        return previous
      }

      return {
        ...previous,
        focusTaskOne: goalDeadlineSummary.topGoals[0] || '',
        totalEstimatedMinutes: Math.max(120, Math.round(yesterdaySnapshot.estimatedFocusedMinutes || 120)),
      }
    })
  }, [analyticsData, goalDeadlineSummary.topGoals, yesterdaySnapshot.estimatedFocusedMinutes])

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

    const habitCompletionRate7d = form.habitsTotal > 0
      ? Number(((form.habitsCompletedToday / form.habitsTotal) * 100).toFixed(1))
      : 0

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

    return {
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

        <div className="relative mb-4 rounded-2xl border border-border/70 bg-white/70 p-4 dark:bg-slate-900/65">
          <div className="grid grid-cols-[auto,1fr,auto,1fr,auto] items-center gap-3">
            <div className={cn(
              'flex h-9 w-9 items-center justify-center rounded-full border',
              morningCompletedToday
                ? 'border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                : 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300',
            )}>
              {morningCompletedToday ? <CheckCircle2 className="h-4 w-4" /> : <Sunrise className="h-4 w-4" />}
            </div>
            <div className="h-px bg-gradient-to-r from-amber-300/70 to-indigo-300/70 dark:from-amber-700/60 dark:to-indigo-700/60" />
            <div className={cn(
              'flex h-9 w-9 items-center justify-center rounded-full border',
              eveningCompletedToday
                ? 'border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                : canRunEveningCheckup
                  ? 'border-indigo-300 bg-indigo-50 text-indigo-700 dark:border-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-300'
                  : 'border-slate-300 bg-slate-100 text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400',
            )}>
              {eveningCompletedToday ? <CheckCircle2 className="h-4 w-4" /> : canRunEveningCheckup ? <MoonStar className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
            </div>
            <div className="h-px bg-gradient-to-r from-indigo-300/70 to-emerald-300/70 dark:from-indigo-700/60 dark:to-emerald-700/60" />
            <div className="flex h-9 w-9 items-center justify-center rounded-full border border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300">
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
                ? 'border-emerald-200/70 bg-white/80 dark:border-emerald-900/50 dark:bg-slate-900/70'
                : 'border-amber-200/80 bg-amber-50/45 shadow-[0_0_0_1px_rgba(251,191,36,0.2)] dark:border-amber-900/60 dark:bg-amber-950/20',
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
              {summarizeCheckupMessage(morningCheckup?.coach_message)}
            </p>
            <Button onClick={() => openCheckupFlow('morning')} className="w-full gap-2">
              <Compass className="h-4 w-4" />
              {morningCompletedToday ? 'Review Morning Plan' : 'Start Morning Plan'}
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>

          <div className="rounded-2xl border border-indigo-200/75 bg-white/80 p-4 dark:border-indigo-900/60 dark:bg-slate-900/70">
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
                ? summarizeCheckupMessage(eveningCheckup?.coach_message)
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
          <div className="mt-4 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
            {checkupError}
          </div>
        )}
      </Card>

      {isCheckupModalOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-slate-950/60 p-2 pt-[calc(1rem+env(safe-area-inset-top))] backdrop-blur-sm sm:items-center sm:p-4">
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
                            Yesterday Snapshot
                          </p>
                          <p className="text-sm font-semibold text-foreground">
                            {yesterdaySnapshot.interactions} interactions logged
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            Estimated focused time: {Math.round(yesterdaySnapshot.estimatedFocusedMinutes)} min
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Avg session: {Math.round(yesterdaySnapshot.avgSessionMinutes)} min
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

                        <label className="space-y-1">
                          <span className="text-xs font-medium text-muted-foreground">Planned Deep Work (minutes)</span>
                          <input
                            type="number"
                            min={0}
                            step={15}
                            value={activeForm.plannedDeepWorkMinutes}
                            onChange={(event) => {
                              const parsed = Number(event.target.value)
                              updateActiveFormField(
                                'plannedDeepWorkMinutes',
                                Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
                              )
                            }}
                            className="w-full rounded-lg border border-border/70 bg-background px-3 py-2 text-sm"
                          />
                        </label>

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
                            </label>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            Deep-work coverage: {Math.round(activeDeepWorkCoverage)}%
                          </p>
                        </div>

                        <div className="rounded-xl border border-border/70 bg-background/70 p-3">
                          <p className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            <Activity className="h-3.5 w-3.5" />
                            Habit Pulse
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            <label className="space-y-1">
                              <span className="text-[11px] text-muted-foreground">Total</span>
                              <input
                                type="number"
                                min={0}
                                value={activeForm.habitsTotal}
                                onChange={(event) => {
                                  const parsed = Number(event.target.value)
                                  updateActiveFormField('habitsTotal', Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0)
                                }}
                                className="w-full rounded-md border border-border/70 bg-background px-2 py-1.5 text-sm"
                              />
                            </label>
                            <label className="space-y-1">
                              <span className="text-[11px] text-muted-foreground">Completed</span>
                              <input
                                type="number"
                                min={0}
                                value={activeForm.habitsCompletedToday}
                                onChange={(event) => {
                                  const parsed = Number(event.target.value)
                                  updateActiveFormField(
                                    'habitsCompletedToday',
                                    Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0,
                                  )
                                }}
                                className="w-full rounded-md border border-border/70 bg-background px-2 py-1.5 text-sm"
                              />
                            </label>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            Habit completion today: {Math.round(activeHabitCompletionToday)}%
                          </p>
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
                <p className="text-sm text-muted-foreground">Average Logged Minutes</p>
                <p className="text-2xl font-bold">{analyticsData.insights.avg_time_entry_minutes}</p>
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