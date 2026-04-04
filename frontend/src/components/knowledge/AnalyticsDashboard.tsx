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
  LineChart as LineChartIcon
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface AnalyticsData {
  interactions: {
    daily: Array<{ date: string; count: number; agent: string }>
    weekly: Array<{ week: string; count: number }>
    by_agent: Array<{ agent: string; count: number; color: string }>
  }
  patterns: {
    most_active_hours: Array<{ hour: number; interactions: number }>
    preference_changes: Array<{ date: string; category: string; changes: number }>
    knowledge_growth: Array<{ date: string; total_entries: number; new_entries: number }>
  }
  insights: {
    total_interactions: number
    most_used_agent: string
    avg_daily_interactions: number
    knowledge_base_size: number
    preference_stability: number
    learning_velocity: number
  }
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
      setAnalyticsData(data)
    } catch (error) {
      console.error('Failed to load analytics data:', error)
      setAnalyticsData(null)
    } finally {
      setIsLoading(false)
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

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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