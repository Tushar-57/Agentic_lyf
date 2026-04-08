import React, { useState, useEffect } from 'react'
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

const DAILY_CHECKUP_MODAL_HINT_KEY = 'agentic-daily-checkup-modal-hint'

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

const summarizeCheckupMessage = (message?: string) => {
  const normalized = (message || '').trim().replace(/\s+/g, ' ')
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
  const [morningNote, setMorningNote] = useState('')
  const [eveningNote, setEveningNote] = useState('')
  const [checkupLoading, setCheckupLoading] = useState<'morning' | 'evening' | null>(null)
  const [checkupError, setCheckupError] = useState<string | null>(null)
  const [morningCheckup, setMorningCheckup] = useState<DailyCheckupResponse | null>(null)
  const [eveningCheckup, setEveningCheckup] = useState<DailyCheckupResponse | null>(null)
  const [isCheckupModalOpen, setIsCheckupModalOpen] = useState(false)
  const [activeCheckupFlow, setActiveCheckupFlow] = useState<'morning' | 'evening'>('morning')

  useEffect(() => {
    loadAnalyticsData()
    loadLatestCheckups()
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

    if (!coachMessage) {
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
      coach_message: coachMessage,
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

  const loadLatestCheckups = async () => {
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

      setMorningCheckup(latestMorning)
      setEveningCheckup(latestEvening)
    } catch (error) {
      console.error('Failed to load saved checkups:', error)
    }
  }

  const loadAnalyticsData = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`/api/knowledge/analytics?range=${selectedTimeRange}`)
      if (!response.ok) {
        throw new Error(`Analytics request failed with status ${response.status}`)
      }

      const data = await response.json()
      const normalizedData: AnalyticsData = {
        interactions: {
          daily: data.interactions?.daily ?? [],
          weekly: data.interactions?.weekly ?? [],
          by_agent: data.interactions?.by_agent ?? [],
          by_category: data.interactions?.by_category ?? [],
        },
        patterns: {
          most_active_hours: data.patterns?.most_active_hours ?? [],
          preference_changes: data.patterns?.preference_changes ?? [],
          knowledge_growth: data.patterns?.knowledge_growth ?? [],
          category_focus: data.patterns?.category_focus ?? [],
        },
        insights: {
          total_interactions: data.insights?.total_interactions ?? 0,
          most_used_agent: data.insights?.most_used_agent ?? 'N/A',
          avg_daily_interactions: data.insights?.avg_daily_interactions ?? 0,
          knowledge_base_size: data.insights?.knowledge_base_size ?? 0,
          preference_stability: data.insights?.preference_stability ?? 0,
          learning_velocity: data.insights?.learning_velocity ?? 0,
          top_knowledge_category: data.insights?.top_knowledge_category ?? 'N/A',
          time_entry_records: data.insights?.time_entry_records ?? 0,
          time_entry_billable_records: data.insights?.time_entry_billable_records ?? 0,
          avg_time_entry_minutes: data.insights?.avg_time_entry_minutes ?? 0,
        },
      }

      setAnalyticsData(normalizedData)
    } catch (error) {
      console.error('Failed to load analytics data:', error)
      setAnalyticsData(null)
    } finally {
      setIsLoading(false)
    }
  }

  const runDailyCheckup = async (checkupType: 'morning' | 'evening') => {
    setCheckupError(null)
    setCheckupLoading(checkupType)

    try {
      const note = checkupType === 'morning' ? morningNote.trim() : eveningNote.trim()
      const response = await fetch(`/api/knowledge/checkups/${checkupType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ note: note || undefined }),
      })

      if (!response.ok) {
        throw new Error(`Checkup failed with status ${response.status}`)
      }

      const payload = (await response.json()) as DailyCheckupResponse
      if (checkupType === 'morning') {
        setMorningCheckup(payload)
        setMorningNote('')
      } else {
        setEveningCheckup(payload)
        setEveningNote('')
      }

      await Promise.all([loadAnalyticsData(), loadLatestCheckups()])
    } catch (error) {
      console.error(`Failed to run ${checkupType} checkup:`, error)
      setCheckupError('Unable to run checkup right now. Please try again.')
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

  const todayDateKey = new Date().toISOString().slice(0, 10)
  const morningCompletedToday = normalizeDateKey(morningCheckup?.date) === todayDateKey
  const eveningCompletedToday = normalizeDateKey(eveningCheckup?.date) === todayDateKey
  const canRunEveningCheckup = morningCompletedToday || eveningCompletedToday

  const openCheckupFlow = (flow: 'morning' | 'evening') => {
    setActiveCheckupFlow(flow)
    setCheckupError(null)
    setIsCheckupModalOpen(true)
  }

  const activeCheckup = activeCheckupFlow === 'morning' ? morningCheckup : eveningCheckup
  const activeNote = activeCheckupFlow === 'morning' ? morningNote : eveningNote

  const setActiveNote = (value: string) => {
    if (activeCheckupFlow === 'morning') {
      setMorningNote(value)
      return
    }

    setEveningNote(value)
  }

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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
          <div className="flex h-[80vh] w-[min(1100px,94vw)] flex-col overflow-hidden rounded-2xl border border-border/70 bg-background shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-border/70 px-5 py-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Interactive Daily Checkup</p>
                <h4 className="text-xl font-semibold">{checkupHeaderLabel}</h4>
                <p className="mt-1 text-sm text-muted-foreground">{checkupSubLabel}</p>
              </div>
              <Button
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

            <div className="grid flex-1 overflow-hidden lg:grid-cols-[1.15fr,1fr]">
              <div className="overflow-y-auto border-b border-border/60 p-5 lg:border-b-0 lg:border-r">
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

                <textarea
                  value={activeNote}
                  onChange={(event) => setActiveNote(event.target.value)}
                  placeholder={
                    activeCheckupFlow === 'morning'
                      ? 'Example: I need two uninterrupted deep-work blocks for architecture decisions and one stakeholder sync.'
                      : 'Example: I shipped one key task, but reactive context switching slowed execution quality.'
                  }
                  className="min-h-[210px] w-full rounded-xl border border-border/70 bg-background px-4 py-3 text-sm"
                />

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
                    variant="ghost"
                    onClick={() => setActiveNote('')}
                    disabled={isActiveCheckupRunning}
                  >
                    Clear Note
                  </Button>
                </div>
              </div>

              <div className="overflow-y-auto bg-secondary/20 p-5">
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
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                        {activeCheckup.coach_message}
                      </p>
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
                <Button variant="outline" onClick={() => setIsCheckupModalOpen(false)}>
                  Close
                </Button>
                {activeCheckupFlow === 'morning' && (
                  <Button
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
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        <Card className="p-4">
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
        
        <Card className="p-4">
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
        
        <Card className="p-4">
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
        
        <Card className="p-4">
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

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold">{analyticsData.insights.top_knowledge_category}</p>
              <p className="text-sm text-muted-foreground">Top Category</p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
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
                <p className="text-2xl font-bold">{analyticsData.insights.most_used_agent}</p>
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