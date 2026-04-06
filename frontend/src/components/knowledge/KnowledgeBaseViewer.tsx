import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Database, 
  Search, 
  Filter, 
  Edit3, 
  Trash2, 
  Plus, 
  RefreshCcw,
  Tag,
  Calendar,
  Clock,
  User,
  Brain,
  TrendingUp,
  BarChart3,
  Target
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'

interface KnowledgeEntry {
  entry_id: string
  user_id: string
  entry_type: 'preference' | 'user_preference' | 'interaction' | 'pattern' | 'insight' | 'memory' | string
  entry_sub_type: string
  category: string
  title: string
  content: string
  metadata: Record<string, any>
  tags: string[]
  created_at: string
  updated_at: string
}

interface UserPreferences {
  user_id: string
  productivity: Record<string, any>
  health: Record<string, any>
  finance: Record<string, any>
  journal: Record<string, any>
  llm_provider: Record<string, any>
  general: Record<string, any>
}

interface KnowledgeStats {
  total_entries: number
  entries_by_type: Record<string, number>
  entries_by_category: Record<string, number>
  last_updated: string
  embedding_model: string
}

interface OnboardingProfileSnapshot {
  role?: string | null
  mentor?: {
    name?: string
    archetype?: string
    style?: string
  }
  goals?: Array<{ id?: string; title?: string }>
  preferredTone?: string | null
  onboardingCompleted?: boolean
}

interface KnowledgeBaseViewerProps {
  className?: string
  onEditPreferences?: () => void
  onAddPreference?: () => void
  refreshKey?: number
}

interface DisplayKnowledgeEntry extends KnowledgeEntry {
  displayTitle: string
  displayType: string
  displayTypeLabel: string
  displayCategory: string
  displayCategoryLabel: string
}

const toDisplayLabel = (rawValue: string) => {
  return rawValue
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

const resolveEntryCategory = (entry: KnowledgeEntry): string => {
  const metadata = entry.metadata || {}
  const context = (metadata.context || {}) as Record<string, any>

  const source = String(context.source || '').toLowerCase()
  const sourceAction = String(context.source_action || '').toLowerCase()
  const category = String(entry.category || '').toLowerCase()

  if (
    category === 'time_entry' ||
    source === 'alterego_timetracker' ||
    sourceAction.includes('time_entry') ||
    context.time_entry_id
  ) {
    return 'time_entry'
  }

  return category || 'uncategorized'
}

const resolveEntryType = (entry: KnowledgeEntry, resolvedCategory: string): string => {
  const subType = String(entry.entry_sub_type || '').toLowerCase()

  if (resolvedCategory === 'time_entry') {
    return 'time_entry'
  }

  if (subType.includes('goal')) {
    return 'goal'
  }

  if (subType.includes('schedule')) {
    return 'schedule'
  }

  if (subType.includes('profile')) {
    return 'profile'
  }

  return String(entry.entry_type || 'memory').toLowerCase()
}

const shouldUseDerivedTimeEntryTitle = (title: string): boolean => {
  const normalized = String(title || '').trim().toLowerCase()
  if (!normalized) {
    return true
  }

  return normalized.startsWith('interaction with time entry')
}

const resolveDisplayTitle = (entry: KnowledgeEntry, displayType: string): string => {
  if (displayType !== 'time_entry') {
    return entry.title
  }

  if (!shouldUseDerivedTimeEntryTitle(entry.title)) {
    return entry.title
  }

  const metadata = entry.metadata || {}
  const context = (metadata.context || {}) as Record<string, any>

  const project = String(context.project_name || '').trim()
  const activity = String(context.description || context.task_name || '').trim()
  const rawDuration = Number(context.duration_minutes)
  const durationSuffix = Number.isFinite(rawDuration) && rawDuration > 0
    ? ` (${Math.round(rawDuration)}m)`
    : ''

  let base = ''
  if (project && activity && project.toLowerCase() !== activity.toLowerCase()) {
    base = `${project}: ${activity}`
  } else {
    base = activity || project
  }

  if (!base) {
    return `Time Entry${durationSuffix}`
  }

  const compact = base.length > 90 ? `${base.slice(0, 87)}...` : base
  return `Time Entry - ${compact}${durationSuffix}`
}

type HighlightItem = {
  label: string
  value: string
  icon?: React.ComponentType<{ className?: string }>
}

type EntryPresentation = {
  summary: string
  highlights: HighlightItem[]
  contentRows: Array<{ label: string; value: string }>
  metadataRows: Array<{ label: string; value: string }>
}

const isRecord = (value: unknown): value is Record<string, any> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const tryParseJson = (value: string): unknown => {
  const trimmed = value.trim()
  if (!trimmed || !/^[\[{]/.test(trimmed)) {
    return null
  }

  try {
    return JSON.parse(trimmed)
  } catch {
    return null
  }
}

const formatDurationMinutes = (value: unknown): string | null => {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return null
  }

  const hours = Math.floor(numericValue / 60)
  const minutes = Math.round(numericValue % 60)

  if (hours > 0 && minutes > 0) {
    return `${hours}h ${minutes}m`
  }

  if (hours > 0) {
    return `${hours}h`
  }

  return `${minutes}m`
}

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined || value === '') {
    return 'Not provided'
  }

  if (typeof value === 'string') {
    return value
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  if (Array.isArray(value)) {
    const compactArray = value.slice(0, 5).map((item) => formatValue(item))
    const suffix = value.length > 5 ? ` +${value.length - 5} more` : ''
    return `${compactArray.join(', ')}${suffix}`
  }

  if (isRecord(value)) {
    try {
      return JSON.stringify(value)
    } catch {
      return '[Object]'
    }
  }

  return String(value)
}

const toKeyValueRows = (
  record: Record<string, any>,
  preferredKeys: string[],
  limit: number,
  options?: {
    allowFallback?: boolean
    excludeKeys?: string[]
  },
): Array<{ label: string; value: string }> => {
  if (!isRecord(record)) {
    return []
  }

  const allowFallback = options?.allowFallback ?? true
  const excludedKeys = new Set((options?.excludeKeys || []).map((key) => key.toLowerCase()))

  const seenKeys = new Set<string>()
  const rows: Array<{ label: string; value: string }> = []

  const addRow = (key: string, value: unknown) => {
    if (excludedKeys.has(key.toLowerCase())) {
      return
    }

    if (rows.length >= limit || seenKeys.has(key)) {
      return
    }

    const formattedValue = formatValue(value)
    if (!formattedValue || formattedValue === 'Not provided') {
      return
    }

    rows.push({
      label: toDisplayLabel(key),
      value: formattedValue.length > 140 ? `${formattedValue.slice(0, 137)}...` : formattedValue,
    })
    seenKeys.add(key)
  }

  preferredKeys.forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(record, key)) {
      addRow(key, record[key])
    }
  })

  if (!allowFallback) {
    return rows
  }

  Object.entries(record).forEach(([key, value]) => {
    if (rows.length >= limit || seenKeys.has(key)) {
      return
    }

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const nested = Object.keys(value as Record<string, unknown>)
      if (nested.length === 0) {
        return
      }
    }

    addRow(key, value)
  })

  return rows
}

const pickFirstValue = (
  records: Array<Record<string, any>>,
  keys: string[],
): unknown => {
  for (const key of keys) {
    for (const record of records) {
      if (!isRecord(record)) {
        continue
      }

      const value = record[key]
      if (value !== undefined && value !== null && value !== '') {
        return value
      }
    }
  }

  return null
}

const deriveSummary = (
  entry: DisplayKnowledgeEntry,
  contentRecord: Record<string, any>,
): string => {
  const summaryKeys = [
    'summary',
    'description',
    'insight',
    'notes',
    'note',
    'details',
    'activity',
    'task',
    'message',
    'observation',
  ]

  const summaryValue = pickFirstValue([contentRecord, entry.metadata || {}], summaryKeys)
  if (typeof summaryValue === 'string' && summaryValue.trim().length > 0) {
    return summaryValue.trim().replace(/\s+/g, ' ')
  }

  const contentAsText = typeof entry.content === 'string' ? entry.content.trim() : ''
  if (contentAsText) {
    const parsed = tryParseJson(contentAsText)
    if (!parsed) {
      return contentAsText.replace(/\s+/g, ' ')
    }
  }

  if (entry.displayType === 'time_entry') {
    return 'Tracked work activity captured from AlterEgo Time Tracker and indexed for coaching context.'
  }

  return `Knowledge entry captured for ${entry.displayCategoryLabel.toLowerCase()} context.`
}

const buildPresentation = (entry: DisplayKnowledgeEntry): EntryPresentation => {
  const parsedContent =
    typeof entry.content === 'string'
      ? tryParseJson(entry.content)
      : entry.content

  const contentRecord = isRecord(parsedContent)
    ? parsedContent
    : isRecord(entry.content)
      ? entry.content
      : {}

  const metadataRecord = isRecord(entry.metadata) ? entry.metadata : {}
  const contextRecord = isRecord(metadataRecord.context) ? metadataRecord.context : {}

  const summary = deriveSummary(entry, contentRecord)

  const recordSources = [contentRecord, contextRecord, metadataRecord]
  const durationRaw = pickFirstValue(recordSources, ['duration_minutes', 'duration', 'minutes_spent', 'minutes'])
  const duration = formatDurationMinutes(durationRaw) || null
  const project = pickFirstValue(recordSources, ['project_name', 'project', 'workspace', 'client'])
  const task = pickFirstValue(recordSources, ['task_name', 'description', 'task', 'activity', 'title'])
  const source = pickFirstValue(recordSources, ['source', 'origin'])
  const sourceAction = pickFirstValue(recordSources, ['source_action', 'action'])
  const confidence = pickFirstValue(recordSources, ['confidence', 'similarity'])
  const startTime = pickFirstValue(recordSources, ['start_time'])
  const endTime = pickFirstValue(recordSources, ['end_time'])
  const billable = pickFirstValue(recordSources, ['billable'])
  const linkedGoal = pickFirstValue(recordSources, ['linked_goal'])
  const focusScore = pickFirstValue(recordSources, ['focus_score'])
  const energyScore = pickFirstValue(recordSources, ['energy_score'])

  const highlights: HighlightItem[] = []

  if (entry.displayType === 'time_entry') {
    if (task) {
      highlights.push({ label: 'What You Did', value: formatValue(task), icon: Brain })
    }
    if (project) {
      highlights.push({ label: 'Project', value: formatValue(project), icon: Tag })
    }
    if (duration) {
      highlights.push({ label: 'Duration', value: duration, icon: Clock })
    }
    if (entry.tags.length > 0) {
      highlights.push({ label: 'Tags', value: entry.tags.slice(0, 3).join(', '), icon: Tag })
    }
  } else {
    if (task) {
      highlights.push({ label: 'Focus', value: formatValue(task), icon: Brain })
    }

    if (source) {
      highlights.push({ label: 'Source', value: formatValue(source), icon: Database })
    }

    if (sourceAction) {
      highlights.push({ label: 'Action', value: formatValue(sourceAction), icon: TrendingUp })
    }

    if (confidence !== null && confidence !== undefined && confidence !== '') {
      const numericConfidence = Number(confidence)
      const confidenceValue = Number.isFinite(numericConfidence)
        ? `${Math.round(numericConfidence * (numericConfidence <= 1 ? 100 : 1))}%`
        : formatValue(confidence)
      highlights.push({ label: 'Confidence', value: confidenceValue, icon: BarChart3 })
    }
  }

  const contentRows = entry.displayType === 'time_entry'
    ? toKeyValueRows(
        {
          what_user_did: task,
          project_name: project,
          duration_minutes: durationRaw,
          start_time: startTime,
          end_time: endTime,
        },
        ['what_user_did', 'project_name', 'duration_minutes', 'start_time', 'end_time'],
        6,
        { allowFallback: false },
      )
    : toKeyValueRows(
        contentRecord,
        [
          'project_name',
          'task_name',
          'duration_minutes',
          'start_time',
          'end_time',
          'date',
          'summary',
          'notes',
          'status',
        ],
        8,
      )

  const metadataRows = entry.displayType === 'time_entry'
    ? toKeyValueRows(
        {
          billable,
          linked_goal: linkedGoal,
          focus_score: focusScore,
          energy_score: energyScore,
          tag_count: entry.tags.length,
        },
        ['billable', 'linked_goal', 'focus_score', 'energy_score', 'tag_count'],
        5,
        { allowFallback: false },
      )
    : toKeyValueRows(
        { ...metadataRecord, ...contextRecord },
        ['agent', 'agent_type', 'confidence', 'timestamp'],
        8,
        {
          excludeKeys: [
            'context',
            'source',
            'source_action',
            'sync_event_key',
            'time_entry_id',
            'user_input_length',
            'response_length',
          ],
        },
      )

  return {
    summary: summary.length > 220 ? `${summary.slice(0, 217)}...` : summary,
    highlights,
    contentRows,
    metadataRows,
  }
}

export const KnowledgeBaseViewer: React.FC<KnowledgeBaseViewerProps> = ({
  className,
  onEditPreferences,
  onAddPreference,
  refreshKey = 0,
}) => {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([])
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [profileSnapshot, setProfileSnapshot] = useState<OnboardingProfileSnapshot | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null)
  const [refreshScope, setRefreshScope] = useState<string | null>(null)

  // Load data on component mount
  useEffect(() => {
    void loadKnowledgeData()
  }, [refreshKey])

  const loadKnowledgeData = async (forceRefresh = false) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const cacheBust = Date.now().toString()

      const requestOptions: RequestInit = {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
          Pragma: 'no-cache',
        },
      }

      if (forceRefresh) {
        const refreshResponse = await fetch(`/api/knowledge/refresh?ts=${cacheBust}`, {
          method: 'POST',
          ...requestOptions,
        }).catch(() => null)

        if (refreshResponse && refreshResponse.ok) {
          const refreshData = await refreshResponse.json()
          const scope = refreshData?.user_scope?.storage_key
          setRefreshScope(typeof scope === 'string' && scope.trim() ? scope : null)
        }
      }

      // Load entries, preferences, and stats in parallel
      const [entriesRes, preferencesRes, statsRes, profileRes] = await Promise.all([
        fetch(`/api/knowledge/entries?ts=${cacheBust}`, requestOptions).catch(() => null),
        fetch(`/api/knowledge/preferences?ts=${cacheBust}`, requestOptions).catch(() => null),
        fetch(`/api/knowledge/stats?ts=${cacheBust}`, requestOptions).catch(() => null),
        fetch(`/api/knowledge/onboarding/profile?ts=${cacheBust}`, requestOptions).catch(() => null),
      ])

      if (entriesRes && entriesRes.ok) {
        const entriesData = await entriesRes.json()
        setEntries(Array.isArray(entriesData) ? entriesData : [])
      } else {
        setEntries([])
      }

      if (preferencesRes && preferencesRes.ok) {
        const preferencesData = await preferencesRes.json()
        setPreferences(preferencesData)
      } else {
        setPreferences(null)
      }

      if (statsRes && statsRes.ok) {
        const statsData = await statsRes.json()
        setStats(statsData)
      } else {
        setStats(null)
      }

      if (profileRes && profileRes.ok) {
        const profileData = await profileRes.json()
        setProfileSnapshot(profileData)
      } else {
        setProfileSnapshot(null)
      }

      setLastSyncedAt(new Date().toISOString())
    } catch (err) {
      console.error('Failed to load knowledge data:', err)
      setError('Failed to load knowledge base data')
      
      // Use empty arrays as fallback - the real data should come from the API
      console.warn('Using fallback demo data - API may not be available')
      setEntries([])
      setStats({
        total_entries: 0,
        entries_by_type: {},
        entries_by_category: {},
        last_updated: new Date().toISOString(),
        embedding_model: 'unknown'
      })
      
      // Set empty preferences as fallback
      setPreferences(null)
      setProfileSnapshot(null)
    } finally {
      setIsLoading(false)
    }
  }

  const displayEntries: DisplayKnowledgeEntry[] = entries.map((entry) => {
    const displayCategory = resolveEntryCategory(entry)
    const displayType = resolveEntryType(entry, displayCategory)
    const displayTitle = resolveDisplayTitle(entry, displayType)

    return {
      ...entry,
      displayTitle,
      displayType,
      displayTypeLabel: toDisplayLabel(displayType),
      displayCategory,
      displayCategoryLabel: toDisplayLabel(displayCategory),
    }
  })

  // Filter entries based on search and filters
  const filteredEntries = displayEntries.filter(entry => {
    const matchesSearch = searchQuery === '' || 
      entry.displayTitle.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategory === 'all' || entry.displayCategory === selectedCategory
    const matchesType = selectedType === 'all' || entry.displayType === selectedType
    
    return matchesSearch && matchesCategory && matchesType
  })

  const categories = ['all', ...new Set(displayEntries.map((entry) => entry.displayCategory))]
  const types = ['all', ...new Set(displayEntries.map((entry) => entry.displayType))]
  const totalEntriesCount = displayEntries.length

  const preferencesCount = displayEntries.filter((entry) =>
    entry.displayType === 'preference' || entry.displayType === 'user_preference'
  ).length
  const goalsCount = displayEntries.filter((entry) => entry.displayType === 'goal').length
  const timeEntriesCount = displayEntries.filter((entry) => entry.displayType === 'time_entry').length
  const patternsCount = displayEntries.filter((entry) => entry.displayType === 'pattern').length

  const getEntryIcon = (type: string) => {
    switch (type) {
      case 'preference': return User
      case 'user_preference': return User
      case 'goal': return Target
      case 'schedule': return Calendar
      case 'profile': return User
      case 'time_entry': return Calendar
      case 'interaction': return Brain
      case 'pattern': return TrendingUp
      case 'insight': return BarChart3
      case 'memory': return Database
      default: return Database
    }
  }

  const getEntryColor = (type: string) => {
    switch (type) {
      case 'preference': return 'from-teal-600 to-cyan-500'
      case 'user_preference': return 'from-teal-600 to-cyan-500'
      case 'goal': return 'from-fuchsia-600 to-violet-500'
      case 'schedule': return 'from-blue-600 to-cyan-500'
      case 'profile': return 'from-emerald-600 to-teal-500'
      case 'time_entry': return 'from-indigo-600 to-blue-500'
      case 'interaction': return 'from-emerald-500 to-green-500'
      case 'pattern': return 'from-sky-500 to-blue-500'
      case 'insight': return 'from-amber-500 to-orange-500'
      case 'memory': return 'from-slate-600 to-slate-500'
      default: return 'from-slate-500 to-slate-400'
    }
  }

  const getEntryTileStyle = (type: string) => {
    switch (type) {
      case 'preference':
      case 'user_preference':
        return 'border-teal-200/70 bg-gradient-to-br from-teal-50/60 via-white to-cyan-50/40 dark:border-teal-900/60 dark:from-teal-950/25 dark:to-slate-900/55'
      case 'goal':
        return 'border-violet-200/70 bg-gradient-to-br from-violet-50/65 via-white to-fuchsia-50/40 dark:border-violet-900/60 dark:from-violet-950/20 dark:to-slate-900/55'
      case 'time_entry':
        return 'border-indigo-200/70 bg-gradient-to-br from-indigo-50/65 via-white to-blue-50/45 dark:border-indigo-900/60 dark:from-indigo-950/20 dark:to-slate-900/55'
      case 'interaction':
        return 'border-emerald-200/70 bg-gradient-to-br from-emerald-50/65 via-white to-green-50/40 dark:border-emerald-900/60 dark:from-emerald-950/20 dark:to-slate-900/55'
      case 'pattern':
        return 'border-sky-200/70 bg-gradient-to-br from-sky-50/65 via-white to-cyan-50/45 dark:border-sky-900/60 dark:from-sky-950/20 dark:to-slate-900/55'
      case 'insight':
        return 'border-amber-200/70 bg-gradient-to-br from-amber-50/70 via-white to-orange-50/45 dark:border-amber-900/60 dark:from-amber-950/20 dark:to-slate-900/55'
      case 'memory':
        return 'border-slate-300/70 bg-gradient-to-br from-slate-50/75 via-white to-slate-100/45 dark:border-slate-800/80 dark:from-slate-900/65 dark:to-slate-950/70'
      default:
        return 'border-border/70 bg-white/75 dark:bg-slate-900/60'
    }
  }

  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center h-64", className)}>
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-muted-foreground">Loading knowledge base...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex flex-col justify-between gap-4 rounded-2xl border border-border/70 bg-card/70 p-5 sm:flex-row sm:items-center">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold">Knowledge Base</h2>
          <p className="text-muted-foreground">
            View and manage your AI agent's learned preferences and patterns
          </p>
          {lastSyncedAt && (
            <p className="mt-1 text-xs text-muted-foreground">
              Last synced: {new Date(lastSyncedAt).toLocaleTimeString()}
            </p>
          )}
          {refreshScope && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              Active storage scope: {refreshScope}
            </p>
          )}
        </div>
        <div className="flex gap-2 flex-shrink-0 flex-wrap">
          <Button 
            onClick={() => void loadKnowledgeData(true)} 
            variant="ghost" 
            size="icon"
            disabled={isLoading}
            className="gap-2"
            title="Force refresh data"
          >
            <RefreshCcw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
          <Button onClick={onAddPreference} variant="outline" className="gap-2">
            <Plus className="w-4 h-4" />
            <span className="hidden sm:inline">Add Preference</span>
            <span className="sm:hidden">Add</span>
          </Button>
          <Button onClick={onEditPreferences} className="gap-2">
            <Edit3 className="w-4 h-4" />
            <span className="hidden sm:inline">Edit Preferences</span>
            <span className="sm:hidden">Edit</span>
          </Button>
        </div>
      </div>

      {(preferences || profileSnapshot) && (
        <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold">Preference Snapshot</h3>
            <Badge variant="outline" className="text-xs">
              Live Preferences
            </Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 text-xs text-muted-foreground sm:grid-cols-2 xl:grid-cols-6">
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Primary Provider</p>
              <p>{String(preferences?.llm_provider?.provider || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Timezone</p>
              <p>{String(preferences?.general?.timezone || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Work Hours</p>
              <p>{String(preferences?.productivity?.work_hours || preferences?.general?.work_hours || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Check-In Time</p>
              <p>{String(preferences?.journal?.check_in_time || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Role</p>
              <p>{String(profileSnapshot?.role || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Mentor</p>
              <p>{String(profileSnapshot?.mentor?.name || 'not set')}</p>
            </div>
          </div>
        </Card>
      )}

      {error && (
        <Card className="border-red-300/60 bg-red-50/70 p-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/35 dark:text-red-200">
          {error}
        </Card>
      )}

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-600 to-cyan-500 flex items-center justify-center">
                <Database className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{totalEntriesCount}</p>
                <p className="text-sm text-muted-foreground">Total Entries</p>
              </div>
            </div>
          </Card>
          
          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-500 flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{preferencesCount}</p>
                <p className="text-sm text-muted-foreground">Preferences</p>
              </div>
            </div>
          </Card>

          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-fuchsia-600 to-violet-500 flex items-center justify-center">
                <Target className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{goalsCount}</p>
                <p className="text-sm text-muted-foreground">Goals</p>
              </div>
            </div>
          </Card>

          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-blue-500 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{timeEntriesCount}</p>
                <p className="text-sm text-muted-foreground">Time Entries</p>
              </div>
            </div>
          </Card>

          
          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-blue-500 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{patternsCount}</p>
                <p className="text-sm text-muted-foreground">Patterns</p>
              </div>
            </div>
          </Card>
          
          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {new Date(stats.last_updated).toLocaleDateString()}
                </p>
                <p className="text-sm text-muted-foreground">Last Updated</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Search and Filters */}
      <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search entries, tags, or content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          
          <div className="flex gap-2">
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="rounded-xl border border-border/70 bg-white/75 px-3 py-2 text-sm shadow-sm dark:bg-slate-900/60"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : toDisplayLabel(category)}
                </option>
              ))}
            </select>
            
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="rounded-xl border border-border/70 bg-white/75 px-3 py-2 text-sm shadow-sm dark:bg-slate-900/60"
            >
              {types.map(type => (
                <option key={type} value={type}>
                  {type === 'all' ? 'All Types' : toDisplayLabel(type)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* Entries List */}
      <div className="space-y-4">
        <AnimatePresence>
          {filteredEntries.map((entry, index) => {
            const Icon = getEntryIcon(entry.displayType)
            const colorClass = getEntryColor(entry.displayType)
            const tileStyle = getEntryTileStyle(entry.displayType)
            const presentation = buildPresentation(entry)
            
            return (
              <motion.div
                key={entry.entry_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className={cn(
                  'p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md',
                  tileStyle,
                )}>
                  <div className="flex items-start gap-4">
                    <div className={cn(
                      "w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br",
                      colorClass
                    )}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between mb-2 gap-4">
                        <div className="flex-1 min-w-0">
                          <h3 className="font-semibold text-lg break-words">{entry.displayTitle}</h3>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
                            <Badge variant="secondary" className="text-xs">
                              {entry.displayTypeLabel}
                            </Badge>
                            {entry.entry_sub_type && entry.displayType !== 'time_entry' && (
                              <Badge variant="outline" className="text-xs">
                                {toDisplayLabel(entry.entry_sub_type)}
                              </Badge>
                            )}
                            <span>•</span>
                            <span className="break-words">{entry.displayCategoryLabel}</span>
                            <span>•</span>
                            <span>{new Date(entry.created_at).toLocaleDateString()}</span>
                          </div>
                        </div>

                        {(entry.displayType === 'preference' || entry.displayType === 'user_preference') && onEditPreferences && (
                          <div className="flex gap-1 flex-shrink-0">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={onEditPreferences}
                            >
                              <Edit3 className="w-4 h-4" />
                            </Button>
                          </div>
                        )}
                      </div>
                      
                      <p className="text-muted-foreground mb-3 break-words whitespace-pre-wrap text-sm leading-relaxed">
                        {presentation.summary}
                      </p>

                      {presentation.highlights.length > 0 && (
                        <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                          {presentation.highlights.slice(0, 4).map((item, itemIndex) => {
                            const HighlightIcon = item.icon || Tag

                            return (
                              <div
                                key={`${entry.entry_id}-highlight-${itemIndex}`}
                                className="flex items-start gap-2 rounded-lg border border-border/60 bg-muted/30 px-2 py-1.5"
                              >
                                <HighlightIcon className="mt-0.5 h-3.5 w-3.5 text-muted-foreground" />
                                <div className="min-w-0">
                                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{item.label}</p>
                                  <p className="truncate text-xs font-medium text-foreground">{item.value}</p>
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                      
                      {entry.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {entry.tags.map((tag, tagIndex) => (
                            <Badge key={`${entry.entry_id}-tag-${tagIndex}`} variant="outline" className="text-xs">
                              <Tag className="w-3 h-3 mr-1" />
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              </motion.div>
            )
          })}
        </AnimatePresence>
        
        {filteredEntries.length === 0 && (
          <Card className="border-border/70 bg-white/75 p-8 text-center shadow-sm dark:bg-slate-900/60">
            <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No entries found</h3>
            <p className="text-muted-foreground">
              {searchQuery || selectedCategory !== 'all' || selectedType !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Your knowledge base is empty. Start interacting with agents to build your knowledge base.'}
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}