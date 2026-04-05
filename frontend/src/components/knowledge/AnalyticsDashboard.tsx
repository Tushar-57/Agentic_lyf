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

interface AnalyticsDashboardProps {
  className?: string
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

  useEffect(() => {
    loadAnalyticsData()
  }, [selectedTimeRange])

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
      } else {
        setEveningCheckup(payload)
      }

      loadAnalyticsData()
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

      <Card className="p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold">Daily AI Checkups</h3>
            <p className="text-sm text-muted-foreground">
              Run a morning intention plan and an evening reflection from your real time-entry context.
            </p>
          </div>
          <Badge variant="outline">Morning + Evening</Badge>
        </div>

        {checkupError && (
          <div className="mb-4 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {checkupError}
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="space-y-3 rounded-xl border p-4">
            <div>
              <h4 className="font-medium">Morning Intent</h4>
              <p className="text-xs text-muted-foreground">Tell AI what matters most today.</p>
            </div>
            <textarea
              value={morningNote}
              onChange={(event) => setMorningNote(event.target.value)}
              placeholder="Example: Deep focus on API refactor and 2 important bug fixes."
              className="min-h-[88px] w-full rounded-md border bg-background px-3 py-2 text-sm"
            />
            <Button
              onClick={() => runDailyCheckup('morning')}
              disabled={checkupLoading === 'morning'}
              className="w-full"
            >
              {checkupLoading === 'morning' ? 'Running Morning Checkup...' : 'Run Morning Checkup'}
            </Button>

            {morningCheckup && (
              <div className="space-y-2 rounded-md bg-secondary/40 p-3 text-sm">
                <div className="flex items-center justify-between">
                  <p className="font-medium">Focus: {morningCheckup.focus_target || 'N/A'}</p>
                  <Badge variant="secondary">{morningCheckup.generated_with}</Badge>
                </div>
                <p className="whitespace-pre-wrap text-muted-foreground">{morningCheckup.coach_message}</p>
              </div>
            )}
          </div>

          <div className="space-y-3 rounded-xl border p-4">
            <div>
              <h4 className="font-medium">Evening Reflection</h4>
              <p className="text-xs text-muted-foreground">Close the day with a grounded review.</p>
            </div>
            <textarea
              value={eveningNote}
              onChange={(event) => setEveningNote(event.target.value)}
              placeholder="Example: Felt productive but context switching hurt focus."
              className="min-h-[88px] w-full rounded-md border bg-background px-3 py-2 text-sm"
            />
            <Button
              onClick={() => runDailyCheckup('evening')}
              disabled={checkupLoading === 'evening'}
              className="w-full"
            >
              {checkupLoading === 'evening' ? 'Running Evening Checkup...' : 'Run Evening Checkup'}
            </Button>

            {eveningCheckup && (
              <div className="space-y-2 rounded-md bg-secondary/40 p-3 text-sm">
                <div className="flex items-center justify-between">
                  <p className="font-medium">{eveningCheckup.date}</p>
                  <Badge variant="secondary">{eveningCheckup.generated_with}</Badge>
                </div>
                <p className="whitespace-pre-wrap text-muted-foreground">{eveningCheckup.coach_message}</p>
                {!!eveningCheckup.tomorrow_focus?.length && (
                  <div>
                    <p className="font-medium">Tomorrow Focus</p>
                    <ul className="list-disc pl-5 text-muted-foreground">
                      {eveningCheckup.tomorrow_focus.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </Card>

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