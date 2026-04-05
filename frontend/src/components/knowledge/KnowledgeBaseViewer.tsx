import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Database, 
  Search, 
  Filter, 
  Edit3, 
  Trash2, 
  Plus, 
  Eye, 
  Tag,
  Calendar,
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

interface KnowledgeBaseViewerProps {
  className?: string
  onEditPreferences?: () => void
  onAddPreference?: () => void
  refreshKey?: number
}

interface DisplayKnowledgeEntry extends KnowledgeEntry {
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

export const KnowledgeBaseViewer: React.FC<KnowledgeBaseViewerProps> = ({
  className,
  onEditPreferences,
  onAddPreference,
  refreshKey = 0,
}) => {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([])
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null)

  // Load data on component mount
  useEffect(() => {
    void loadKnowledgeData()
  }, [refreshKey])

  const loadKnowledgeData = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const requestOptions: RequestInit = {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
        },
      }

      // Load entries, preferences, and stats in parallel
      const [entriesRes, preferencesRes, statsRes] = await Promise.all([
        fetch('/api/knowledge/entries', requestOptions).catch(() => null),
        fetch('/api/knowledge/preferences', requestOptions).catch(() => null),
        fetch('/api/knowledge/stats', requestOptions).catch(() => null)
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
    } finally {
      setIsLoading(false)
    }
  }

  const displayEntries: DisplayKnowledgeEntry[] = entries.map((entry) => {
    const displayCategory = resolveEntryCategory(entry)
    const displayType = resolveEntryType(entry, displayCategory)

    return {
      ...entry,
      displayType,
      displayTypeLabel: toDisplayLabel(displayType),
      displayCategory,
      displayCategoryLabel: toDisplayLabel(displayCategory),
    }
  })

  // Filter entries based on search and filters
  const filteredEntries = displayEntries.filter(entry => {
    const matchesSearch = searchQuery === '' || 
      entry.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entry.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategory === 'all' || entry.displayCategory === selectedCategory
    const matchesType = selectedType === 'all' || entry.displayType === selectedType
    
    return matchesSearch && matchesCategory && matchesType
  })

  const categories = ['all', ...new Set(displayEntries.map((entry) => entry.displayCategory))]
  const types = ['all', ...new Set(displayEntries.map((entry) => entry.displayType))]

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
        </div>
        <div className="flex gap-2 flex-shrink-0 flex-wrap">
          <Button 
            onClick={loadKnowledgeData} 
            variant="ghost" 
            size="icon"
            disabled={isLoading}
            className="gap-2"
            title="Refresh data"
          >
            <Database className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
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

      {preferences && (
        <Card className="border-border/70 bg-white/75 p-4 shadow-sm dark:bg-slate-900/60">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-sm font-semibold">Preference Snapshot</h3>
            <Badge variant="outline" className="text-xs">
              Live Preferences
            </Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Primary Provider</p>
              <p>{String(preferences.llm_provider?.provider || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Timezone</p>
              <p>{String(preferences.general?.timezone || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Work Hours</p>
              <p>{String(preferences.productivity?.work_hours || preferences.general?.work_hours || 'not set')}</p>
            </div>
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Check-In Time</p>
              <p>{String(preferences.journal?.check_in_time || 'not set')}</p>
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
                <p className="text-2xl font-bold">{stats.total_entries}</p>
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
            
            return (
              <motion.div
                key={entry.entry_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="border-border/70 bg-white/75 p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md dark:bg-slate-900/60">
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
                          <h3 className="font-semibold text-lg break-words">{entry.title}</h3>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
                            <Badge variant="secondary" className="text-xs">
                              {entry.displayTypeLabel}
                            </Badge>
                            {entry.entry_sub_type && (
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
                        
                        <div className="flex gap-1 flex-shrink-0">
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-8 w-8"
                            onClick={() => {
                              // Show entry details in a modal or expand inline
                              console.log('View entry:', entry.entry_id)
                              alert(`Entry Details:\n\nTitle: ${entry.title}\nContent: ${entry.content}\nTags: ${entry.tags.join(', ')}`)
                            }}
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-8 w-8"
                            onClick={() => {
                              if ((entry.displayType === 'preference' || entry.displayType === 'user_preference') && onEditPreferences) {
                                onEditPreferences()
                              } else {
                                alert(`Editing for ${entry.displayTypeLabel} entries is not available yet.`)
                              }
                            }}
                          >
                            <Edit3 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-muted-foreground mb-3 break-words whitespace-pre-wrap">
                        {entry.content.length > 300 
                          ? `${entry.content.substring(0, 300)}...` 
                          : entry.content
                        }
                      </p>
                      
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