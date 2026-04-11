import React, { useState, useEffect, useCallback } from 'react'
import type { CSSProperties } from 'react'
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
  Target,
  ChevronDown,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { getAgenticBridgeUserKey } from '@/lib/agenticBridgeSession'

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

interface KnowledgeSnapshotCache {
  version: number
  fetchedAt: string
  entries: KnowledgeEntry[]
  preferences: UserPreferences | null
  stats: KnowledgeStats | null
  profileSnapshot: OnboardingProfileSnapshot | null
  refreshScope: string | null
  lastSyncedAt: string | null
}

const KNOWLEDGE_CACHE_KEY_PREFIX = 'agentic-knowledge-view-cache-v4'
const KNOWLEDGE_CACHE_VERSION = 4
const KNOWLEDGE_CACHE_TTL_MS = 90 * 1000

const resolveKnowledgeCacheKey = (): string => {
  if (typeof window === 'undefined') {
    return `${KNOWLEDGE_CACHE_KEY_PREFIX}.single_user`
  }

  const userKey = getAgenticBridgeUserKey()
  return `${KNOWLEDGE_CACHE_KEY_PREFIX}.${userKey || 'single_user'}`
}

const inMemoryKnowledgeSnapshots: Record<string, KnowledgeSnapshotCache> = {}

const readCachedSnapshot = (): KnowledgeSnapshotCache | null => {
  const cacheKey = resolveKnowledgeCacheKey()
  if (inMemoryKnowledgeSnapshots[cacheKey]) {
    return inMemoryKnowledgeSnapshots[cacheKey]
  }

  if (typeof window === 'undefined') {
    return null
  }

  try {
    const raw = window.localStorage.getItem(cacheKey)
    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as KnowledgeSnapshotCache
    if (!parsed || parsed.version !== KNOWLEDGE_CACHE_VERSION || !Array.isArray(parsed.entries)) {
      return null
    }

    inMemoryKnowledgeSnapshots[cacheKey] = parsed
    return parsed
  } catch {
    return null
  }
}

const writeCachedSnapshot = (snapshot: KnowledgeSnapshotCache) => {
  const cacheKey = resolveKnowledgeCacheKey()
  inMemoryKnowledgeSnapshots[cacheKey] = snapshot

  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(cacheKey, JSON.stringify(snapshot))
  } catch {
    // Cache write failures should never block UI rendering.
  }
}

const isSnapshotFresh = (snapshot: KnowledgeSnapshotCache): boolean => {
  const fetchedAtMs = new Date(snapshot.fetchedAt).getTime()
  if (!Number.isFinite(fetchedAtMs)) {
    return false
  }

  return Date.now() - fetchedAtMs <= KNOWLEDGE_CACHE_TTL_MS
}

const getEntryAccentStyle = (type: string): CSSProperties => {
  const paletteByType: Record<string, [string, string, string]> = {
    time_entry: ['rgba(56,189,248,0.55)', 'rgba(34,197,94,0.38)', 'rgba(59,130,246,0.34)'],
    goal: ['rgba(251,146,60,0.58)', 'rgba(244,63,94,0.4)', 'rgba(245,158,11,0.33)'],
    insight: ['rgba(248,113,113,0.58)', 'rgba(249,115,22,0.42)', 'rgba(59,130,246,0.28)'],
    pattern: ['rgba(45,212,191,0.56)', 'rgba(14,165,233,0.4)', 'rgba(56,189,248,0.3)'],
    preference: ['rgba(16,185,129,0.56)', 'rgba(20,184,166,0.4)', 'rgba(52,211,153,0.3)'],
    user_preference: ['rgba(16,185,129,0.56)', 'rgba(20,184,166,0.4)', 'rgba(52,211,153,0.3)'],
  }

  const [colorA, colorB, colorC] = paletteByType[type] || ['rgba(148,163,184,0.5)', 'rgba(100,116,139,0.34)', 'rgba(148,163,184,0.26)']

  return {
    backgroundImage: [
      `radial-gradient(135% 95% at 98% 8%, ${colorA} 0%, rgba(255,255,255,0) 58%)`,
      `radial-gradient(120% 135% at 85% 92%, ${colorB} 0%, rgba(255,255,255,0) 66%)`,
      `radial-gradient(90% 110% at 56% 48%, ${colorC} 0%, rgba(255,255,255,0) 72%)`,
      'linear-gradient(170deg, rgba(255,255,255,0.68) 0%, rgba(255,255,255,0) 50%, rgba(255,255,255,0.16) 100%)',
      'repeating-linear-gradient(126deg, rgba(255,255,255,0.2) 0px, rgba(255,255,255,0.2) 1px, rgba(255,255,255,0) 1px, rgba(255,255,255,0) 7px)',
    ].join(', '),
    mixBlendMode: 'normal',
  }
}

const toDisplayLabel = (rawValue: string) => {
  return rawValue
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

const formatDurationMinutes = (value: unknown): string | null => {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return null
  }

  const roundedMinutes = Math.round(numericValue)
  const hours = Math.floor(roundedMinutes / 60)
  const minutes = roundedMinutes % 60

  if (hours > 0 && minutes > 0) {
    return `${hours} hour${hours === 1 ? '' : 's'} ${minutes} minute${minutes === 1 ? '' : 's'}`
  }

  if (hours > 0) {
    return `${hours} hour${hours === 1 ? '' : 's'}`
  }

  return `${minutes} minute${minutes === 1 ? '' : 's'}`
}

const normalizeDurationTokensInText = (value: string): string => {
  return value.replace(/\b(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b/gi, (match, rawMinutes) => {
    const formatted = formatDurationMinutes(rawMinutes)
    return formatted || match
  })
}

const isDurationLikeKey = (key: string): boolean => {
  const normalizedKey = key.toLowerCase()
  return (
    normalizedKey === 'duration'
    || normalizedKey === 'minutes'
    || normalizedKey.includes('duration')
    || normalizedKey.endsWith('_minutes')
    || normalizedKey.includes('minutes_')
  )
}

const resolveEntryCategory = (entry: KnowledgeEntry): string => {
  const metadata = entry.metadata || {}
  const context = (metadata.context || {}) as Record<string, any>
  const approval = (metadata.approval && typeof metadata.approval === 'object'
    ? (metadata.approval as Record<string, any>)
    : {})

  const source = String(context.source || '').toLowerCase()
  const sourceAction = String(context.source_action || '').toLowerCase()
  const category = String(entry.category || '').toLowerCase()

  if (Boolean(context.approved_as_insight) || Boolean(approval.approved_as_insight)) {
    return 'insight'
  }

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

  if (resolvedCategory === 'habit_snapshot' || resolvedCategory === 'habit_progress') {
    return 'habit_snapshot'
  }

  if (resolvedCategory === 'insight' && String(entry.entry_type || '').toLowerCase() === 'interaction') {
    return 'insight'
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
    return normalizeDurationTokensInText(entry.title)
  }

  const metadata = entry.metadata || {}
  const context = (metadata.context || {}) as Record<string, any>

  const project = String(context.project_name || '').trim()
  const activity = String(context.description || context.task_name || '').trim()
  const durationLabel = formatDurationMinutes(context.duration_minutes)
  const durationSuffix = durationLabel
    ? ` (${durationLabel})`
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
  conversation?: {
    userInput?: string
    agentResponse?: string
  }
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

const flattenStructuredRecord = (record: Record<string, any>): string => {
  const preferredKeys = [
    'title',
    'name',
    'label',
    'summary',
    'message',
    'status',
    'checkup_type',
    'focus_target',
    'intent_prompt',
    'accountability_prompt',
  ]

  const highlightedParts: string[] = []

  preferredKeys.forEach((key) => {
    if (!Object.prototype.hasOwnProperty.call(record, key)) {
      return
    }

    const rawValue = record[key]
    if (rawValue === undefined || rawValue === null || rawValue === '') {
      return
    }

    if (typeof rawValue === 'object' && rawValue !== null) {
      return
    }

    const durationValue = isDurationLikeKey(key) ? formatDurationMinutes(rawValue) : null
    highlightedParts.push(`${toDisplayLabel(key)}: ${durationValue || formatValue(rawValue)}`)
  })

  if (highlightedParts.length > 0) {
    return highlightedParts.slice(0, 3).join(' | ')
  }

  const scalarPairs = Object.entries(record)
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .filter(([, value]) => typeof value !== 'object' || Array.isArray(value))
    .slice(0, 4)
    .map(([key, value]) => {
      const durationValue = isDurationLikeKey(key) ? formatDurationMinutes(value) : null
      return `${toDisplayLabel(key)}: ${durationValue || formatValue(value)}`
    })

  if (scalarPairs.length > 0) {
    return scalarPairs.join(' | ')
  }

  return 'Additional structured context is available.'
}

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined || value === '') {
    return 'Not provided'
  }

  if (typeof value === 'string') {
    const normalizedValue = value
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()

    if (!normalizedValue || normalizedValue === '[object Object]') {
      return 'Not provided'
    }

    return normalizeDurationTokensInText(normalizedValue)
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
    return flattenStructuredRecord(value)
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

    const formattedDuration = isDurationLikeKey(key) ? formatDurationMinutes(value) : null
    const formattedValue = formattedDuration || formatValue(value)
    if (!formattedValue || formattedValue === 'Not provided') {
      return
    }

    rows.push({
      label: toDisplayLabel(key),
      value: formattedValue,
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

const toShortTextList = (value: unknown, limit = 2): string[] => {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((item) => formatValue(item).replace(/\s+/g, ' ').trim())
    .filter((item) => item.length > 0 && item !== 'Not provided')
    .slice(0, limit)
}

const deriveDailyCheckupSummary = (
  contentRecord: Record<string, any>,
  metadataRecord: Record<string, any>,
): string | null => {
  const checkupType = String(
    pickFirstValue([contentRecord, metadataRecord], ['checkup_type']) || '',
  ).toLowerCase()

  const focusTarget = pickFirstValue([contentRecord, metadataRecord], ['focus_target'])
  const intentNote = pickFirstValue([contentRecord, metadataRecord], ['intent_note'])
  const reflectionNote = pickFirstValue([contentRecord, metadataRecord], ['reflection_note'])

  const journalingRoot = isRecord(metadataRecord.journaling)
    ? metadataRecord.journaling
    : isRecord(metadataRecord.reflection_journal)
      ? metadataRecord.reflection_journal
      : {}

  const focusCommitment = isRecord(journalingRoot.focus_commitment)
    ? journalingRoot.focus_commitment
    : isRecord(journalingRoot.evidence)
      ? journalingRoot.evidence
      : {}

  const decisionMetrics = isRecord(metadataRecord.decision_metrics)
    ? metadataRecord.decision_metrics
    : {}

  const deadlinePressure = isRecord(focusCommitment.deadline_pressure)
    ? focusCommitment.deadline_pressure
    : {}

  const overdue = Math.max(0, Math.round(Number(deadlinePressure.overdue ?? decisionMetrics.overdue_tasks ?? 0) || 0))
  const dueToday = Math.max(0, Math.round(Number(deadlinePressure.due_today ?? decisionMetrics.due_today_tasks ?? 0) || 0))
  const upcoming = Math.max(
    0,
    Math.round(Number(deadlinePressure.upcoming_7d ?? decisionMetrics.upcoming_deadlines_7d ?? 0) || 0),
  )

  const focusTasks = toShortTextList(
    pickFirstValue([focusCommitment, metadataRecord], ['focus_tasks']),
    2,
  )
  const wins = toShortTextList(
    pickFirstValue([metadataRecord, journalingRoot], ['wins']),
    2,
  )
  const blockers = toShortTextList(
    pickFirstValue([metadataRecord, journalingRoot], ['blockers', 'friction_points']),
    2,
  )
  const tomorrowFocus = toShortTextList(
    pickFirstValue([metadataRecord, journalingRoot], ['tomorrow_focus', 'tomorrow_commitments']),
    2,
  )

  const parts: string[] = []

  if (checkupType === 'morning') {
    parts.push('Morning strategy')
  } else if (checkupType === 'evening') {
    parts.push('Evening reflection')
  }

  if (focusTarget) {
    parts.push(`Focus: ${formatValue(focusTarget)}`)
  }

  if (checkupType === 'morning' && typeof intentNote === 'string' && intentNote.trim()) {
    parts.push(`Intent: ${formatValue(intentNote)}`)
  }

  if (checkupType === 'evening' && typeof reflectionNote === 'string' && reflectionNote.trim()) {
    parts.push(`Reflection: ${formatValue(reflectionNote)}`)
  }

  if (focusTasks.length > 0) {
    parts.push(`Focus tasks: ${focusTasks.join(', ')}`)
  }

  if (wins.length > 0) {
    parts.push(`Wins: ${wins.join('; ')}`)
  }

  if (blockers.length > 0) {
    parts.push(`Blockers: ${blockers.join('; ')}`)
  }

  if (tomorrowFocus.length > 0) {
    parts.push(`Tomorrow: ${tomorrowFocus.join('; ')}`)
  }

  if (overdue > 0 || dueToday > 0 || upcoming > 0) {
    const deadlineBits: string[] = []
    if (overdue > 0) {
      deadlineBits.push(`${overdue} overdue`)
    }
    if (dueToday > 0) {
      deadlineBits.push(`${dueToday} due today`)
    }
    if (upcoming > 0) {
      deadlineBits.push(`${upcoming} upcoming`)
    }

    parts.push(`Deadlines: ${deadlineBits.join(', ')}`)
  }

  if (parts.length === 0) {
    return null
  }

  const summary = parts.join(' | ')
  return summary.length > 340 ? `${summary.slice(0, 337)}...` : summary
}

const deriveSummary = (
  entry: DisplayKnowledgeEntry,
  contentRecord: Record<string, any>,
): string => {
  const metadataRecord = isRecord(entry.metadata) ? entry.metadata : {}
  const contextRecord = isRecord(metadataRecord.context) ? metadataRecord.context : {}
  const checkupSummary = entry.displayCategory === 'daily_checkup'
    ? deriveDailyCheckupSummary(contentRecord, metadataRecord)
    : null

  if (entry.displayCategory === 'habit_snapshot') {
    const totalHabits = Number(
      pickFirstValue([contentRecord, contextRecord, metadataRecord], ['total_habits']) || 0,
    )
    const totalEvents = Number(
      pickFirstValue([contentRecord, contextRecord, metadataRecord], ['total_completion_events']) || 0,
    )
    const activeDays = Number(
      pickFirstValue([contentRecord, contextRecord, metadataRecord], ['active_days']) || 0,
    )
    const longestRun = Number(
      pickFirstValue([contentRecord, contextRecord, metadataRecord], ['longest_run']) || 0,
    )

    const highlightsRaw = pickFirstValue([contentRecord, contextRecord, metadataRecord], ['habit_highlights'])
    const highlights = toShortTextList(highlightsRaw, 2)

    const parts: string[] = [
      'Habit progress snapshot',
      `${Math.max(0, Math.round(totalHabits))} habits`,
      `${Math.max(0, Math.round(totalEvents))} completion events`,
    ]

    if (activeDays > 0) {
      parts.push(`${Math.round(activeDays)} active days`)
    }

    if (longestRun > 0) {
      parts.push(`${Math.round(longestRun)}-day best run`)
    }

    if (highlights.length > 0) {
      parts.push(`Highlights: ${highlights.join('; ')}`)
    }

    return parts.join(' | ')
  }

  const summaryKeys = [
    'summary',
    'agent_response',
    'assistant_response',
    'ai_response',
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

  const summaryValue = pickFirstValue([contentRecord, metadataRecord], summaryKeys)
  if (typeof summaryValue === 'string' && summaryValue.trim().length > 0) {
    const normalized = normalizeDurationTokensInText(summaryValue.trim().replace(/\s+/g, ' '))
    if (checkupSummary) {
      const combinedSummary = `${checkupSummary} | ${normalized}`
      return combinedSummary.length > 360 ? `${combinedSummary.slice(0, 357)}...` : combinedSummary
    }

    return normalized
  }

  if (checkupSummary) {
    return checkupSummary
  }

  const contentAsText = typeof entry.content === 'string' ? entry.content.trim() : ''
  if (contentAsText) {
    const parsed = tryParseJson(contentAsText)
    if (!parsed) {
      return normalizeDurationTokensInText(contentAsText.replace(/\s+/g, ' '))
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
  const conversationUserInput = pickFirstValue(recordSources, ['user_input', 'user_query', 'prompt'])
  const conversationAgentResponse = pickFirstValue(recordSources, ['agent_response', 'assistant_response', 'ai_response', 'response'])
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
          start_time: startTime,
          end_time: endTime,
          date: pickFirstValue(recordSources, ['date']),
        },
        ['date', 'start_time', 'end_time'],
        3,
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

  let metadataRows: Array<{ label: string; value: string }> = []

  if (entry.displayType === 'time_entry') {
    metadataRows = toKeyValueRows(
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
  } else if (entry.displayCategory === 'daily_checkup') {
    const checkupTopRows = toKeyValueRows(
      {
        date: metadataRecord.date,
        checkup_type: metadataRecord.checkup_type,
        intent_note: metadataRecord.intent_note,
        reflection_note: metadataRecord.reflection_note,
        focus_target: metadataRecord.focus_target,
        generated_with: metadataRecord.generated_with,
        recommended_projects: metadataRecord.recommended_projects,
      },
      [
        'date',
        'checkup_type',
        'focus_target',
        'intent_note',
        'reflection_note',
        'generated_with',
        'recommended_projects',
      ],
      8,
      { allowFallback: false },
    )

    const statsRows = isRecord(metadataRecord.stats)
      ? toKeyValueRows(metadataRecord.stats, [], 8).map((row) => ({
          ...row,
          label: `Stats · ${row.label}`,
        }))
      : []

    const decisionRows = isRecord(metadataRecord.decision_metrics)
      ? toKeyValueRows(metadataRecord.decision_metrics, [], 12).map((row) => ({
          ...row,
          label: `Decision · ${row.label}`,
        }))
      : []

    const journalingRoot = isRecord(metadataRecord.journaling)
      ? metadataRecord.journaling
      : isRecord(metadataRecord.reflection_journal)
        ? metadataRecord.reflection_journal
        : {}

    const journalingRows = toKeyValueRows(
      journalingRoot,
      [
        'intent_prompt',
        'accountability_prompt',
        'recap_prompt',
        'tomorrow_prompt',
      ],
      6,
      {
        excludeKeys: ['focus_commitment', 'evidence', 'deadline_pressure', 'habit_anchor'],
      },
    ).map((row) => ({
      ...row,
      label: `Journaling · ${row.label}`,
    }))

    const focusCommitment = isRecord(journalingRoot.focus_commitment)
      ? journalingRoot.focus_commitment
      : isRecord(journalingRoot.evidence)
        ? journalingRoot.evidence
        : {}

    const commitmentRows = toKeyValueRows(
      focusCommitment,
      ['priority_focus', 'focus_tasks', 'goal_anchors', 'deep_work_coverage_ratio'],
      8,
      {
        excludeKeys: ['deadline_pressure', 'habit_anchor', 'habit_state'],
      },
    ).map((row) => ({
      ...row,
      label: `Focus Context · ${row.label}`,
    }))

    metadataRows = [
      ...checkupTopRows,
      ...statsRows,
      ...decisionRows,
      ...journalingRows,
      ...commitmentRows,
    ]
  } else if (entry.displayCategory === 'habit_snapshot') {
    metadataRows = toKeyValueRows(
      { ...contextRecord, ...metadataRecord },
      [
        'captured_at',
        'total_habits',
        'total_completion_events',
        'active_days',
        'current_run',
        'longest_run',
        'habit_highlights',
        'daily_completion_digest',
        'source_action',
      ],
      10,
      {
        excludeKeys: [
          'context',
          'summary',
          'habits',
          'daily_completion_counts',
          'user_input',
          'agent_response',
        ],
      },
    )
  } else {
    metadataRows = toKeyValueRows(
      { ...metadataRecord, ...contextRecord },
      ['agent', 'agent_type', 'confidence', 'timestamp'],
      12,
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
  }

  const conversation = {
    userInput: typeof conversationUserInput === 'string' && conversationUserInput.trim().length > 0
      ? conversationUserInput.trim()
      : undefined,
    agentResponse: typeof conversationAgentResponse === 'string' && conversationAgentResponse.trim().length > 0
      ? conversationAgentResponse.trim()
      : undefined,
  }

  return {
    summary: summary.length > 220 ? `${summary.slice(0, 217)}...` : summary,
    highlights,
    conversation: conversation.userInput || conversation.agentResponse ? conversation : undefined,
    contentRows,
    metadataRows,
  }
}

const formatCompactTime = (value: string): string => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'unknown time'
  }

  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const resolveCheckupStatus = (
  entries: KnowledgeEntry[],
  checkupType: 'morning' | 'evening',
): string => {
  const matchingEntries = entries
    .filter((entry) => {
      const metadata = isRecord(entry.metadata) ? entry.metadata : {}
      const category = String(entry.category || '').toLowerCase()
      const metadataCheckupType = String(metadata.checkup_type || '').toLowerCase()
      const subType = String(entry.entry_sub_type || '').toLowerCase()

      return category === 'daily_checkup' && (
        metadataCheckupType === checkupType || subType.includes(checkupType)
      )
    })
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())

  if (matchingEntries.length === 0) {
    return 'Not run yet'
  }

  const latest = matchingEntries[0]
  const metadata = isRecord(latest.metadata) ? latest.metadata : {}
  const checkupDateRaw = typeof metadata.checkup_date === 'string' && metadata.checkup_date.trim()
    ? metadata.checkup_date.trim()
    : latest.created_at

  const checkupDay = checkupDateRaw.slice(0, 10)
  const todayDay = new Date().toISOString().slice(0, 10)

  if (checkupDay === todayDay) {
    return `Completed today at ${formatCompactTime(latest.created_at)}`
  }

  const latestDate = new Date(latest.created_at)
  if (Number.isNaN(latestDate.getTime())) {
    return 'Completed previously'
  }

  return `Last run ${latestDate.toLocaleDateString()}`
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
  const [selectedCategories, setSelectedCategories] = useState<string[]>([])
  const [selectedTypes, setSelectedTypes] = useState<string[]>([])
  const [expandedEntries, setExpandedEntries] = useState<Record<string, boolean>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null)
  const [refreshScope, setRefreshScope] = useState<string | null>(null)
  const [isRefreshingInBackground, setIsRefreshingInBackground] = useState(false)

  const applySnapshot = useCallback((snapshot: KnowledgeSnapshotCache) => {
    setEntries(Array.isArray(snapshot.entries) ? snapshot.entries : [])
    setPreferences(snapshot.preferences || null)
    setStats(snapshot.stats || null)
    setProfileSnapshot(snapshot.profileSnapshot || null)
    setRefreshScope(snapshot.refreshScope || null)
    setLastSyncedAt(snapshot.lastSyncedAt || snapshot.fetchedAt)
  }, [])

  const revalidateFromApi = useCallback(
    async (forceRefresh: boolean, backgroundRefresh: boolean) => {
      try {
        const cacheBust = Date.now().toString()

        const requestOptions: RequestInit = {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        }

        let resolvedRefreshScope: string | null = refreshScope
        if (forceRefresh) {
          const refreshResponse = await fetch(`/api/knowledge/refresh?ts=${cacheBust}`, {
            method: 'POST',
            ...requestOptions,
          }).catch(() => null)

          if (refreshResponse && refreshResponse.ok) {
            const refreshData = await refreshResponse.json()
            const scope = refreshData?.user_scope?.storage_key
            resolvedRefreshScope = typeof scope === 'string' && scope.trim() ? scope : null
          }
        }

        const [entriesRes, preferencesRes, statsRes, profileRes] = await Promise.all([
          fetch(`/api/knowledge/entries?ts=${cacheBust}`, requestOptions).catch(() => null),
          fetch(`/api/knowledge/preferences?ts=${cacheBust}`, requestOptions).catch(() => null),
          fetch(`/api/knowledge/stats?ts=${cacheBust}`, requestOptions).catch(() => null),
          fetch(`/api/knowledge/onboarding/profile?ts=${cacheBust}`, requestOptions).catch(() => null),
        ])

        const entriesData = entriesRes && entriesRes.ok ? await entriesRes.json() : []
        const preferencesData = preferencesRes && preferencesRes.ok ? await preferencesRes.json() : null
        const statsData = statsRes && statsRes.ok ? await statsRes.json() : null
        const profileData = profileRes && profileRes.ok ? await profileRes.json() : null

        const syncedAt = new Date().toISOString()
        const snapshot: KnowledgeSnapshotCache = {
          version: KNOWLEDGE_CACHE_VERSION,
          fetchedAt: syncedAt,
          entries: Array.isArray(entriesData) ? entriesData : [],
          preferences: preferencesData,
          stats: statsData,
          profileSnapshot: profileData,
          refreshScope: resolvedRefreshScope,
          lastSyncedAt: syncedAt,
        }

        applySnapshot(snapshot)
        writeCachedSnapshot(snapshot)
      } catch (err) {
        console.error('Failed to load knowledge data:', err)
        if (!backgroundRefresh) {
          setError('Failed to load knowledge base data')

          setEntries([])
          setStats({
            total_entries: 0,
            entries_by_type: {},
            entries_by_category: {},
            last_updated: new Date().toISOString(),
            embedding_model: 'unknown',
          })
          setPreferences(null)
          setProfileSnapshot(null)
        }
      } finally {
        if (backgroundRefresh) {
          setIsRefreshingInBackground(false)
        } else {
          setIsLoading(false)
        }
      }
    },
    [applySnapshot, refreshScope],
  )

  const loadKnowledgeData = useCallback(
    async (forceRefresh = false) => {
      setError(null)

      if (!forceRefresh) {
        const cachedSnapshot = readCachedSnapshot()
        if (cachedSnapshot) {
          applySnapshot(cachedSnapshot)
          setIsLoading(false)

          if (isSnapshotFresh(cachedSnapshot)) {
            return
          }

          setIsRefreshingInBackground(true)
          void revalidateFromApi(false, true)
          return
        }
      }

      setIsLoading(true)
      await revalidateFromApi(forceRefresh, false)
    },
    [applySnapshot, revalidateFromApi],
  )

  // Load data on component mount
  useEffect(() => {
    void loadKnowledgeData()
  }, [loadKnowledgeData, refreshKey])

  const visibleEntries = entries.filter((entry) => {
    const entryType = String(entry.entry_type || '').toLowerCase()
    const category = String(entry.category || '').toLowerCase()
    const title = String(entry.title || '').trim().toLowerCase()

    return !(entryType === 'preference' && category === 'system' && title === 'user preferences')
  })

  const displayEntries: DisplayKnowledgeEntry[] = [...visibleEntries]
    .map((entry) => {
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
    .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())

  const categoryOptions = [...new Set(displayEntries.map((entry) => entry.displayCategory))].sort((left, right) =>
    left.localeCompare(right),
  )
  const typeOptions = [...new Set(displayEntries.map((entry) => entry.displayType))].sort((left, right) =>
    left.localeCompare(right),
  )

  const toggleCategorySelection = (category: string) => {
    setSelectedCategories((previous) => (
      previous.includes(category)
        ? previous.filter((item) => item !== category)
        : [...previous, category]
    ))
  }

  const toggleTypeSelection = (type: string) => {
    setSelectedTypes((previous) => (
      previous.includes(type)
        ? previous.filter((item) => item !== type)
        : [...previous, type]
    ))
  }

  const clearAllFilters = () => {
    setSelectedCategories([])
    setSelectedTypes([])
  }

  // Filter entries based on search and filters
  const filteredEntries = displayEntries.filter(entry => {
    const matchesSearch = searchQuery === '' || 
      entry.displayTitle.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategories.length === 0 || selectedCategories.includes(entry.displayCategory)
    const matchesType = selectedTypes.length === 0 || selectedTypes.includes(entry.displayType)
    
    return matchesSearch && matchesCategory && matchesType
  })
  const totalEntriesCount = displayEntries.length

  const preferencesCount = displayEntries.filter((entry) =>
    entry.displayType === 'preference' || entry.displayType === 'user_preference'
  ).length
  const goalsCount = displayEntries.filter((entry) => entry.displayType === 'goal').length
  const timeEntriesCount = displayEntries.filter((entry) => entry.displayType === 'time_entry').length
  const insightsCount = displayEntries.filter((entry) => entry.displayType === 'insight').length
  const prioritiesCount = displayEntries.filter((entry) => {
    const metadata = isRecord(entry.metadata) ? entry.metadata : {}
    const context = isRecord(metadata.context) ? metadata.context : {}
    const prioritySignal = pickFirstValue(
      [context, metadata],
      ['priority', 'priority_level', 'importance', 'urgency', 'is_priority'],
    )

    if (prioritySignal !== null && prioritySignal !== undefined && prioritySignal !== '') {
      return true
    }

    const searchableText = [
      entry.displayType,
      entry.displayCategory,
      entry.entry_sub_type,
      ...entry.tags,
    ]
      .join(' ')
      .toLowerCase()

    return searchableText.includes('priority') || searchableText.includes('urgent')
  }).length

  const morningCheckInStatus = resolveCheckupStatus(entries, 'morning')
  const eveningCheckInStatus = resolveCheckupStatus(entries, 'evening')
  const latestUpdateLabel = stats?.last_updated
    ? new Date(stats.last_updated).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : lastSyncedAt
      ? new Date(lastSyncedAt).toLocaleString([], {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        })
      : 'Not synced yet'

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
      case 'habit_snapshot': return TrendingUp
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
            Loaded instantly from your latest stored state, then refreshed quietly in the background.
          </p>
          {refreshScope && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              Active storage scope: {refreshScope}
            </p>
          )}
          {isRefreshingInBackground && (
            <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">
              Refreshing in background...
            </p>
          )}
        </div>
        <div className="flex gap-2 flex-shrink-0 flex-wrap items-center justify-end">
          <div className="rounded-xl border border-border/70 bg-white/75 px-3 py-2 text-right text-xs shadow-sm dark:bg-slate-900/60">
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Last Updated</p>
            <p className="font-semibold text-foreground">{latestUpdateLabel}</p>
          </div>
          <Button 
            onClick={() => void loadKnowledgeData(true)} 
            variant="ghost" 
            size="icon"
            disabled={isLoading || isRefreshingInBackground}
            className="gap-2"
            title="Force refresh data"
          >
            <RefreshCcw className={`w-4 h-4 ${(isLoading || isRefreshingInBackground) ? 'animate-spin' : ''}`} />
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
              <p className="font-medium text-slate-700 dark:text-slate-200">Work Hours</p>
              <p>{String(preferences?.productivity?.work_hours || preferences?.general?.work_hours || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Preferred Check-In Time</p>
              <p>{String(preferences?.journal?.check_in_time || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Morning Check-In</p>
              <p>{morningCheckInStatus}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Evening Check-In</p>
              <p>{eveningCheckInStatus}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Mentor</p>
              <p>{String(profileSnapshot?.mentor?.name || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Communication Tone</p>
              <p>{String(profileSnapshot?.preferredTone || 'not set')}</p>
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
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{insightsCount}</p>
                <p className="text-sm text-muted-foreground">Insights</p>
              </div>
            </div>
          </Card>

          <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-orange-500 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-2xl font-bold">{prioritiesCount}</p>
                <p className="text-sm text-muted-foreground">Priorities</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Search and Filters */}
      <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
        <div className="flex flex-col gap-4 sm:flex-row">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search entries, tags, or content..."
              value={searchQuery}
              onChange={(event: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(event.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex flex-wrap items-start gap-2">
            <details className="group relative">
              <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl border border-border/70 bg-white/75 px-3 py-2 text-sm shadow-sm dark:bg-slate-900/60">
                <Filter className="h-4 w-4 text-muted-foreground" />
                Categories
                <Badge variant="secondary" className="ml-1 text-[10px]">
                  {selectedCategories.length === 0 ? 'All' : selectedCategories.length}
                </Badge>
                <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
              </summary>
              <div className="absolute right-0 z-20 mt-2 w-64 rounded-xl border border-border/70 bg-background p-3 shadow-xl">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Categories</p>
                  <button
                    type="button"
                    onClick={() => setSelectedCategories([])}
                    className="text-[11px] font-medium text-primary hover:underline"
                  >
                    Clear
                  </button>
                </div>
                <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
                  {categoryOptions.map((category) => (
                    <label key={category} className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-muted/60">
                      <input
                        type="checkbox"
                        checked={selectedCategories.includes(category)}
                        onChange={() => toggleCategorySelection(category)}
                        className="h-3.5 w-3.5"
                      />
                      <span className="truncate">{toDisplayLabel(category)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </details>

            <details className="group relative">
              <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl border border-border/70 bg-white/75 px-3 py-2 text-sm shadow-sm dark:bg-slate-900/60">
                <Tag className="h-4 w-4 text-muted-foreground" />
                Types
                <Badge variant="secondary" className="ml-1 text-[10px]">
                  {selectedTypes.length === 0 ? 'All' : selectedTypes.length}
                </Badge>
                <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
              </summary>
              <div className="absolute right-0 z-20 mt-2 w-64 rounded-xl border border-border/70 bg-background p-3 shadow-xl">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Types</p>
                  <button
                    type="button"
                    onClick={() => setSelectedTypes([])}
                    className="text-[11px] font-medium text-primary hover:underline"
                  >
                    Clear
                  </button>
                </div>
                <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
                  {typeOptions.map((type) => (
                    <label key={type} className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-muted/60">
                      <input
                        type="checkbox"
                        checked={selectedTypes.includes(type)}
                        onChange={() => toggleTypeSelection(type)}
                        className="h-3.5 w-3.5"
                      />
                      <span className="truncate">{toDisplayLabel(type)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </details>

            {(selectedCategories.length > 0 || selectedTypes.length > 0) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAllFilters}
                className="h-9 rounded-xl px-3 text-xs"
              >
                Clear Filters
              </Button>
            )}
          </div>
        </div>

        {(selectedCategories.length > 0 || selectedTypes.length > 0) && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {selectedCategories.map((category) => (
              <button
                key={`selected-category-${category}`}
                type="button"
                onClick={() => toggleCategorySelection(category)}
                className="rounded-full border border-border/70 bg-muted/40 px-2 py-1 text-[11px]"
              >
                Category: {toDisplayLabel(category)}
              </button>
            ))}
            {selectedTypes.map((type) => (
              <button
                key={`selected-type-${type}`}
                type="button"
                onClick={() => toggleTypeSelection(type)}
                className="rounded-full border border-border/70 bg-muted/40 px-2 py-1 text-[11px]"
              >
                Type: {toDisplayLabel(type)}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Entries List */}
      <div className="space-y-4">
        <AnimatePresence>
          {filteredEntries.map((entry, index) => {
            const Icon = getEntryIcon(entry.displayType)
            const colorClass = getEntryColor(entry.displayType)
            const tileStyle = getEntryTileStyle(entry.displayType)
            const presentation = buildPresentation(entry)
            const isEntryExpanded = Boolean(expandedEntries[entry.entry_id])
            const summaryPreview = presentation.summary.length > 220
              ? `${presentation.summary.slice(0, 217)}...`
              : presentation.summary
            const previewHighlights = presentation.highlights.slice(0, 2)
            
            return (
              <motion.div
                key={entry.entry_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className={cn(
                  'group relative overflow-hidden p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md',
                  tileStyle,
                )}>
                  <div
                    aria-hidden
                    className="pointer-events-none absolute inset-y-0 right-0 hidden w-32 md:block"
                  >
                    <div
                      className="relative h-full w-full border-l border-white/45 transition-transform duration-300 group-hover:translate-x-0.5 dark:border-slate-800/70"
                    >
                      <div className="absolute inset-0" style={getEntryAccentStyle(entry.displayType)} />
                      <div className="absolute -right-8 top-4 h-24 w-24 rounded-full bg-white/45 blur-2xl dark:bg-white/10" />
                      <div className="absolute -left-10 bottom-8 h-20 w-20 rounded-full bg-white/35 blur-2xl dark:bg-slate-400/10" />
                      <div className="absolute inset-x-2 bottom-3 space-y-1 rounded-lg border border-white/35 bg-white/45 px-2 py-1.5 backdrop-blur-sm dark:border-slate-700/60 dark:bg-slate-900/45">
                        <p className="truncate text-[10px] font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-200">
                          {entry.displayTypeLabel}
                        </p>
                        <p className="truncate text-[10px] text-slate-600 dark:text-slate-300">
                          {new Date(entry.updated_at).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="relative flex items-start gap-4 md:pr-32">
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

                        <div className="flex flex-shrink-0 items-center gap-1">
                          {(entry.displayType === 'preference' || entry.displayType === 'user_preference') && onEditPreferences && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={onEditPreferences}
                            >
                              <Edit3 className="w-4 h-4" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 gap-1 px-2 text-[11px]"
                            onClick={() => {
                              setExpandedEntries((previous) => ({
                                ...previous,
                                [entry.entry_id]: !previous[entry.entry_id],
                              }))
                            }}
                          >
                            {isEntryExpanded ? 'Collapse' : 'Expand'}
                            <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', isEntryExpanded && 'rotate-180')} />
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-muted-foreground mb-3 break-words whitespace-pre-wrap text-sm leading-relaxed">
                        {isEntryExpanded ? presentation.summary : summaryPreview}
                      </p>

                      {!isEntryExpanded && previewHighlights.length > 0 && (
                        <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                          {previewHighlights.map((item, itemIndex) => (
                            <div
                              key={`${entry.entry_id}-preview-highlight-${itemIndex}`}
                              className="rounded-lg border border-border/60 bg-muted/20 px-2 py-1.5"
                            >
                              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{item.label}</p>
                              <p className="truncate text-xs font-medium text-foreground">{item.value}</p>
                            </div>
                          ))}
                        </div>
                      )}

                      {isEntryExpanded && (
                        <>
                          {presentation.conversation?.userInput && (
                            <div className="mb-3 rounded-lg border border-blue-200/60 bg-blue-50/50 p-3 dark:border-blue-900/60 dark:bg-blue-950/25">
                              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-300">
                                User Input
                              </p>
                              <p className="max-h-24 overflow-y-auto whitespace-pre-wrap break-words text-sm text-blue-900 dark:text-blue-100">
                                {presentation.conversation.userInput}
                              </p>
                            </div>
                          )}

                          {presentation.conversation?.agentResponse && (
                            <div className="mb-3 rounded-lg border border-emerald-200/60 bg-emerald-50/40 p-3 dark:border-emerald-900/60 dark:bg-emerald-950/20">
                              <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">
                                AI Response
                              </p>
                              <p className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words text-sm text-emerald-900 dark:text-emerald-100">
                                {presentation.conversation.agentResponse}
                              </p>
                            </div>
                          )}

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

                          {presentation.contentRows.length > 0 && (
                            <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                              {presentation.contentRows.map((row, rowIndex) => (
                                <div
                                  key={`${entry.entry_id}-content-row-${rowIndex}`}
                                  className="rounded-lg border border-border/60 bg-muted/20 px-2 py-1.5"
                                >
                                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{row.label}</p>
                                  <p className="break-words text-xs font-medium text-foreground">{row.value}</p>
                                </div>
                              ))}
                            </div>
                          )}

                          {presentation.metadataRows.length > 0 && (
                            <details className="mb-3 rounded-lg border border-border/60 bg-muted/15 px-3 py-2">
                              <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                AI Additional Context
                              </summary>
                              <div className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2">
                                {presentation.metadataRows.map((row, rowIndex) => (
                                  <div key={`${entry.entry_id}-metadata-row-${rowIndex}`} className="min-w-0">
                                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{row.label}</p>
                                    <p className="break-words whitespace-pre-wrap text-xs text-foreground">{row.value}</p>
                                  </div>
                                ))}
                              </div>
                            </details>
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
                        </>
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
              {searchQuery || selectedCategories.length > 0 || selectedTypes.length > 0
                ? 'Try adjusting your search or filters'
                : 'Your knowledge base is empty. Start interacting with agents to build your knowledge base.'}
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}