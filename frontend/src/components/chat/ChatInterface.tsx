import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Bot, User, Sparkles, Zap, Brain, Settings, Database, ChevronDown, ChevronUp, X, Calendar, Clock, Target, Play, CheckCircle, AlertCircle, RefreshCw, ListTodo, Leaf, Flame } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface ChatContextSummary {
  tasks_count: number
  habits_pending: number
  habits_total: number
  goals_count: number
  overdue_tasks: number
  due_today_tasks: number
  active_timer: string | null
  last_updated: string
}

interface ActionableSuggestion {
  id: string
  title: string
  description: string
  actionType: 'schedule' | 'reminder' | 'workflow' | 'manual'
  parameters: Record<string, any>
  estimatedImpact: string
  timeRequired?: number
  workflowId?: string
}

interface Message {
  id: string
  content: string | any  // Allow any type but convert to string during rendering
  role: 'user' | 'assistant' | 'system'
  timestamp: Date
  agent?: string
  reasoning?: string | AgentThinking
  isStreaming?: boolean
  actionableData?: {
    summary?: string
    insights?: string[]
    suggestions?: ActionableSuggestion[]
    workflowTriggered?: boolean
    timeAnalysis?: {
      windowLabel: string
      totalMinutes: number
      breakdown: Record<string, number>
      percentages: Record<string, number>
      focusScoreAvg: number
      productivityScoreAvg: number
    }
  }
}

interface AgentThinking {
  agent_outputs?: Array<{
    agent: string
    response_preview: string
  }>
  steps?: Array<{
    agent: string
    action: string
    result?: string
    timestamp?: Date
  }>
  finalAgent?: string
  classification?: {
    agent_type: string
    confidence: number
    reason: string
  }
  handoff?: {
    from: string
    to: string
    reason: string
  }
  knowledge_sources?: Array<{
    type: string
    content: string
    similarity?: number
    created_at?: string
    category?: string
    metadata?: Record<string, any>
  }>
  intent?: {
    agent_type?: string
    confidence?: number
    reason?: string
  }
  execution_path?: string[]
  data_points_used?: {
    role?: string
    priorities?: string[]
    knowledge_context_summary?: string
  }
  error?: string
}

interface ChatInterfaceProps {
  className?: string
  onSendMessage?: (message: string) => void
  messages?: Message[]
  isLoading?: boolean
  currentAgent?: string
  currentProvider?: 'openai' | 'ollama'
}

const agentIcons = {
  orchestrator: Brain,
  productivity: Zap,
  health: Sparkles,
  finance: Settings,
  general: Bot,
}

const AgentThinkingDisplay = React.memo(({ thinking }: { thinking: AgentThinking }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const [isMobileLayout, setIsMobileLayout] = useState(
    () => typeof window !== 'undefined' && window.innerWidth < 768
  )

  useEffect(() => {
    const handleResize = () => {
      setIsMobileLayout(window.innerWidth < 768)
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    if (!(isMobileLayout && isExpanded)) {
      return
    }

    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = originalOverflow
    }
  }, [isExpanded, isMobileLayout])

  const toggleThinking = () => {
    if (isMobileLayout) {
      setIsExpanded(true)
      return
    }

    setIsExpanded((previous) => !previous)
  }

  const closeThinking = () => {
    setIsExpanded(false)
  }

  const quickAgent = thinking.classification?.agent_type || thinking.finalAgent
  const quickStepCount = thinking.steps?.length || 0
  const quickSourceCount = thinking.knowledge_sources?.length || 0
  const quickOutputCount = thinking.agent_outputs?.length || 0
  const derivedClassification = thinking.classification || (
    thinking.intent?.agent_type
      ? {
          agent_type: thinking.intent.agent_type,
          confidence: Number(thinking.intent.confidence || 0),
          reason: thinking.intent.reason || '',
        }
      : undefined
  )
  const executionPath = Array.isArray(thinking.execution_path) ? thinking.execution_path : []
  const prioritiesUsed = Array.isArray(thinking.data_points_used?.priorities)
    ? thinking.data_points_used?.priorities || []
    : []
  const hasRoutingContext = Boolean(
    executionPath.length > 0
      || thinking.data_points_used?.role
      || prioritiesUsed.length > 0
      || thinking.data_points_used?.knowledge_context_summary
      || thinking.intent?.reason
  )

  const hasAnySections = Boolean(
    derivedClassification
      || hasRoutingContext
      || (thinking.knowledge_sources && thinking.knowledge_sources.length > 0)
        || (thinking.agent_outputs && thinking.agent_outputs.length > 0)
      || thinking.handoff
      || (thinking.steps && thinking.steps.length > 0)
      || thinking.finalAgent
      || thinking.error
  )

  const thinkingSections = (
    <>
      {derivedClassification && (
        <div className="mb-3 rounded border-l-4 border-teal-500 bg-teal-50/80 p-2 dark:bg-teal-950/30">
          <div className="text-sm font-medium text-teal-900 dark:text-teal-200">Intent Classification</div>
          <div className="mt-1 text-xs text-teal-700 dark:text-teal-300">
            Routed to: <span className="font-medium capitalize">{derivedClassification.agent_type}</span>
            <span className="ml-2 opacity-75">({Math.round((derivedClassification.confidence || 0) * 100)}% confidence)</span>
          </div>
          {derivedClassification.reason && (
            <div className="mt-1 text-xs text-teal-600 dark:text-teal-400">{derivedClassification.reason}</div>
          )}
        </div>
      )}

      {hasRoutingContext && (
        <div className="mb-3 rounded border-l-4 border-indigo-500 bg-indigo-50/80 p-2 dark:bg-indigo-950/25">
          <div className="text-sm font-medium text-indigo-900 dark:text-indigo-200">Routing And Context Used</div>

          {thinking.data_points_used?.role && (
            <div className="mt-1 text-xs text-indigo-700 dark:text-indigo-300">
              Role: <span className="font-medium">{thinking.data_points_used.role}</span>
            </div>
          )}

          {prioritiesUsed.length > 0 && (
            <div className="mt-1 text-xs text-indigo-700 dark:text-indigo-300">
              Priorities: <span className="font-medium">{prioritiesUsed.join(', ')}</span>
            </div>
          )}

          {thinking.data_points_used?.knowledge_context_summary && (
            <div className="mt-1 text-xs text-indigo-600 dark:text-indigo-400">
              {thinking.data_points_used.knowledge_context_summary}
            </div>
          )}

          {executionPath.length > 0 && (
            <div className="mt-2 space-y-1 text-xs text-indigo-700 dark:text-indigo-300">
              {executionPath.map((pathStep, index) => (
                <div key={`${pathStep}-${index}`} className="rounded bg-white/80 px-2 py-1 dark:bg-slate-800/70">
                  {index + 1}. {pathStep}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {thinking.knowledge_sources && thinking.knowledge_sources.length > 0 && (
        <div className="mb-3 rounded border-l-4 border-amber-500 bg-amber-50/80 p-2 dark:bg-amber-950/25">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-amber-700 dark:text-amber-300" />
              <div className="font-medium text-sm text-amber-900 dark:text-amber-200">
                Knowledge Sources ({thinking.knowledge_sources.length})
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSources(!showSources)}
              className="h-6 w-6 p-0 text-amber-700 dark:text-amber-300"
            >
              {showSources ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </Button>
          </div>

          {showSources && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-2 max-h-40 overflow-y-auto"
            >
              {thinking.knowledge_sources.map((source, index) => (
                <div
                  key={index}
                  className="bg-white dark:bg-slate-800 p-2 rounded text-xs border"
                >
                  <div className="flex items-center justify-between mb-1">
                    <Badge variant="outline" className="text-xs">
                      {source.type}
                    </Badge>
                    {source.similarity && (
                      <span className="text-amber-700 dark:text-amber-300">
                        {Math.round(source.similarity * 100)}% match
                      </span>
                    )}
                  </div>
                  <p className="text-slate-700 dark:text-slate-300 break-words">
                    {source.content}
                  </p>
                  {source.created_at && (
                    <p className="text-slate-500 dark:text-slate-500 mt-1">
                      {new Date(source.created_at).toLocaleDateString()}
                    </p>
                  )}
                </div>
              ))}
            </motion.div>
          )}
        </div>
      )}

      {thinking.handoff && (
        <div className="mb-3 rounded border-l-4 border-cyan-500 bg-cyan-50/80 p-2 dark:bg-cyan-950/25">
          <div className="text-sm font-medium text-cyan-900 dark:text-cyan-200">Agent Handoff</div>
          <div className="mt-1 text-xs text-cyan-700 dark:text-cyan-300">
            {thinking.handoff.from} → {thinking.handoff.to}
          </div>
          <div className="mt-1 text-xs text-cyan-600 dark:text-cyan-400">{thinking.handoff.reason}</div>
        </div>
      )}

      {thinking.agent_outputs && thinking.agent_outputs.length > 0 && (
        <div className="mb-3 rounded border-l-4 border-violet-500 bg-violet-50/80 p-2 dark:bg-violet-950/25">
          <div className="text-sm font-medium text-violet-900 dark:text-violet-200">Agent Responses</div>
          <div className="mt-2 space-y-2">
            {thinking.agent_outputs.map((output, index) => (
              <div key={`${output.agent}-${index}`} className="rounded border bg-white/80 p-2 dark:bg-slate-800/70">
                <div className="text-xs font-medium capitalize text-violet-800 dark:text-violet-300">
                  {output.agent} Agent
                </div>
                <div className="mt-1 text-xs text-violet-700 dark:text-violet-200 break-words">
                  {output.response_preview}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {thinking.steps && thinking.steps.length > 0 && (
        <div className="space-y-2">
          <div className="font-medium text-sm text-slate-700 dark:text-slate-300">Processing Steps:</div>
          {thinking.steps.map((step, index) => (
            <div key={index} className="flex items-start gap-2 p-2 bg-white dark:bg-slate-800 rounded border">
              <div className="w-2 h-2 rounded-full bg-green-400 mt-1.5 flex-shrink-0"></div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-slate-700 dark:text-slate-300 capitalize">
                  {step.agent} Agent
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                  {step.action}
                </div>
                {step.result && (
                  <div className="text-xs text-slate-500 dark:text-slate-500 mt-1 whitespace-pre-wrap break-words">
                    {step.result}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {thinking.finalAgent && (
        <div className="mt-3 p-2 bg-green-50 dark:bg-green-900/20 rounded border-l-4 border-green-400">
          <div className="font-medium text-sm text-green-800 dark:text-green-200">Final Response</div>
          <div className="text-xs text-green-600 dark:text-green-300 mt-1">
            Generated by: <span className="font-medium capitalize">{thinking.finalAgent} Agent</span>
          </div>
        </div>
      )}

      {thinking.error && (
        <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded border-l-4 border-red-400">
          <div className="font-medium text-sm text-red-800 dark:text-red-200">Error</div>
          <div className="text-xs text-red-600 dark:text-red-300 mt-1">{thinking.error}</div>
        </div>
      )}

      {!hasAnySections && (
        <div className="rounded border border-dashed border-slate-300 bg-white/70 p-2 text-xs text-slate-600 dark:border-slate-700 dark:bg-slate-900/70 dark:text-slate-300">
          No detailed reasoning sections were provided for this response.
        </div>
      )}
    </>
  )

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      className="mt-2 overflow-hidden rounded-xl border border-border/70 bg-cyan-50/70 dark:bg-slate-900/70"
    >
      <button
        onClick={toggleThinking}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium text-slate-700 hover:bg-cyan-100/70 dark:text-slate-200 dark:hover:bg-slate-800"
      >
        <span className="flex flex-col">
          <span className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            Agent Thinking Process
          </span>
          {(quickAgent || quickStepCount > 0 || quickOutputCount > 0 || quickSourceCount > 0) && (
            <span className="mt-0.5 text-[11px] font-normal text-slate-500 dark:text-slate-400">
              {quickAgent ? `Agent: ${quickAgent}` : ''}
              {quickStepCount > 0 ? `${quickAgent ? ' | ' : ''}${quickStepCount} steps` : ''}
              {quickOutputCount > 0 ? `${quickAgent || quickStepCount > 0 ? ' | ' : ''}${quickOutputCount} outputs` : ''}
              {quickSourceCount > 0 ? `${quickAgent || quickStepCount > 0 || quickOutputCount > 0 ? ' | ' : ''}${quickSourceCount} sources` : ''}
            </span>
          )}
        </span>
        <motion.div
          animate={{ rotate: !isMobileLayout && isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </motion.div>
      </button>

      {!isMobileLayout && (
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="px-3 pb-3"
            >
              {thinkingSections}
            </motion.div>
          )}
        </AnimatePresence>
      )}

      {isMobileLayout && (
        <AnimatePresence>
          {isExpanded && (
            <>
              <motion.button
                type="button"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={closeThinking}
                className="fixed inset-0 z-40 bg-black/45"
                aria-label="Close thinking details"
              />

              <motion.div
                initial={{ y: 60, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 60, opacity: 0 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
                className="fixed inset-x-0 bottom-0 z-50 max-h-[78vh] overflow-y-auto rounded-t-2xl border border-border/70 bg-white p-3 shadow-2xl dark:bg-slate-900"
              >
                <div className="sticky top-0 z-10 mb-3 flex items-center justify-between border-b border-border/70 bg-white/95 pb-2 dark:bg-slate-900/95">
                  <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">Agent Thinking Process</div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={closeThinking}
                    className="h-8 w-8"
                    aria-label="Close thinking details"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                <div className="pb-2">
                  {thinkingSections}
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      )}
    </motion.div>
  )
})

AgentThinkingDisplay.displayName = 'AgentThinkingDisplay'

const ActionableSuggestions = React.memo(({ 
  suggestions, 
  onExecute 
}: { 
  suggestions: ActionableSuggestion[]
  onExecute?: (suggestion: ActionableSuggestion) => void 
}) => {
  const [executing, setExecuting] = useState<string | null>(null)
  const [executed, setExecuted] = useState<string[]>([])

  const handleExecute = async (suggestion: ActionableSuggestion) => {
    setExecuting(suggestion.id)
    
    try {
      // Call API to execute the action
      if (suggestion.actionType === 'schedule') {
        const response = await fetch('/api/productivity/quick-schedule', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(suggestion.parameters)
        })
        if (response.ok) {
          setExecuted(prev => [...prev, suggestion.id])
        }
      } else if (suggestion.actionType === 'workflow' && suggestion.workflowId) {
        const response = await fetch('/api/productivity/execute-workflow', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            workflow_id: suggestion.workflowId,
            context: suggestion.parameters
          })
        })
        if (response.ok) {
          const data = await response.json()
          setExecuted(prev => [...prev, suggestion.id])
          // Could show workflow execution status
        }
      }
      
      onExecute?.(suggestion)
    } catch (error) {
      console.error('Failed to execute suggestion:', error)
    } finally {
      setExecuting(null)
    }
  }

  const getIcon = (actionType: string) => {
    switch (actionType) {
      case 'schedule': return Calendar
      case 'reminder': return Clock
      case 'workflow': return Play
      default: return Target
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-3 space-y-2"
    >
      <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
        <Target className="w-4 h-4 text-teal-600 dark:text-teal-400" />
        <span>Actionable Suggestions</span>
      </div>
      
      {suggestions.map((suggestion, index) => {
        const Icon = getIcon(suggestion.actionType)
        const isExecuting = executing === suggestion.id
        const isExecuted = executed.includes(suggestion.id)
        
        return (
          <motion.div
            key={suggestion.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={cn(
              "rounded-lg border p-3 transition-all",
              isExecuted 
                ? "border-green-200 bg-green-50/80 dark:border-green-900 dark:bg-green-950/30" 
                : "border-amber-200 bg-amber-50/80 dark:border-amber-900 dark:bg-amber-950/30 hover:border-amber-300 dark:hover:border-amber-700"
            )}
          >
            <div className="flex items-start gap-3">
              <div className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
                isExecuted
                  ? "bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-300"
                  : "bg-amber-100 text-amber-600 dark:bg-amber-900 dark:text-amber-300"
              )}>
                {isExecuted ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {suggestion.title}
                  </h4>
                  {suggestion.timeRequired && (
                    <Badge variant="outline" className="text-xs">
                      {suggestion.timeRequired}min
                    </Badge>
                  )}
                </div>
                
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                  {suggestion.description}
                </p>
                
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
                      <span className="font-medium">📈 Impact:</span>
                      <span>{suggestion.estimatedImpact}</span>
                    </div>
                  </div>
                  
                  {!isExecuted && (
                    <Button
                      size="sm"
                      variant={suggestion.actionType === 'workflow' ? "default" : "outline"}
                      className={cn(
                        "h-7 text-xs",
                        suggestion.actionType === 'workflow' 
                          ? "bg-teal-600 hover:bg-teal-700" 
                          : "border-amber-300 hover:bg-amber-100 dark:border-amber-700 dark:hover:bg-amber-900"
                      )}
                      onClick={() => handleExecute(suggestion)}
                      disabled={isExecuting}
                    >
                      {isExecuting ? (
                        <>
                          <div className="mr-1 h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Executing...
                        </>
                      ) : suggestion.actionType === 'workflow' ? (
                        <>
                          <Play className="mr-1 h-3 w-3" />
                          Run Workflow
                        </>
                      ) : (
                        <>
                          <Calendar className="mr-1 h-3 w-3" />
                          Schedule
                        </>
                      )}
                    </Button>
                  )}
                  
                  {isExecuted && (
                    <span className="text-xs font-medium text-green-600 dark:text-green-400">
                      ✓ Done
                    </span>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )
      })}
    </motion.div>
  )
})

ActionableSuggestions.displayName = 'ActionableSuggestions'

const ContextPill = React.memo(({
  summary,
  isLoading,
  onRefresh,
}: {
  summary: ChatContextSummary | null
  isLoading: boolean
  onRefresh: () => void
}) => {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 border-b border-border/50 bg-gray-50/70 px-4 py-2 dark:bg-gray-800/40">
        <div className="h-3 w-3 animate-pulse rounded-full bg-gray-300 dark:bg-gray-600" />
        <div className="h-3 w-16 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-3 w-20 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-3 w-20 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-3 w-16 animate-pulse rounded bg-gray-200 dark:bg-gray-700" />
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="flex items-center gap-2 border-b border-border/50 bg-gray-50/70 px-4 py-2 dark:bg-gray-800/40">
        <Brain className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Context loading...</span>
      </div>
    )
  }

  const isEmpty =
    summary.tasks_count === 0 &&
    summary.habits_total === 0 &&
    summary.goals_count === 0 &&
    !summary.active_timer

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/50 bg-gray-50/70 px-4 py-1.5 dark:bg-gray-800/40">
      {/* Label */}
      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 dark:text-slate-300">
        <Brain className="h-3.5 w-3.5 text-teal-600 dark:text-teal-400" />
        <span>AI Context</span>
      </div>

      {isEmpty ? (
        <span className="text-xs text-muted-foreground italic">No data loaded yet</span>
      ) : (
        <>
          {summary.tasks_count > 0 && (
            <div
              className="group relative flex cursor-default items-center gap-1 rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium shadow-sm ring-1 ring-border/50 dark:bg-slate-800/70"
              title={`${summary.tasks_count} active task${summary.tasks_count !== 1 ? 's' : ''} loaded from task board`}
            >
              <ListTodo className="h-3 w-3 text-blue-500" />
              <span>{summary.tasks_count} task{summary.tasks_count !== 1 ? 's' : ''}</span>
            </div>
          )}

          {summary.habits_total > 0 && (
            <div
              className="group relative flex cursor-default items-center gap-1 rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium shadow-sm ring-1 ring-border/50 dark:bg-slate-800/70"
              title={`${summary.habits_pending} of ${summary.habits_total} habits still pending today`}
            >
              <Leaf className="h-3 w-3 text-green-500" />
              <span>{summary.habits_pending} habit{summary.habits_pending !== 1 ? 's' : ''} pending</span>
            </div>
          )}

          {summary.goals_count > 0 && (
            <div
              className="group relative flex cursor-default items-center gap-1 rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium shadow-sm ring-1 ring-border/50 dark:bg-slate-800/70"
              title={`${summary.goals_count} active goal${summary.goals_count !== 1 ? 's' : ''} tracked`}
            >
              <Target className="h-3 w-3 text-violet-500" />
              <span>Goals: {summary.goals_count}</span>
            </div>
          )}

          {summary.overdue_tasks > 0 && (
            <div
              className="flex cursor-default items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700 ring-1 ring-red-200 dark:bg-red-950/40 dark:text-red-400 dark:ring-red-900"
              title={`${summary.overdue_tasks} task${summary.overdue_tasks !== 1 ? 's' : ''} past their due date`}
            >
              <AlertCircle className="h-3 w-3" />
              <span>Overdue: {summary.overdue_tasks}</span>
            </div>
          )}

          {summary.active_timer && (
            <div
              className="flex cursor-default items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700 ring-1 ring-amber-200 dark:bg-amber-950/40 dark:text-amber-400 dark:ring-amber-900"
              title={`Timer active: ${summary.active_timer}`}
            >
              <Flame className="h-3 w-3" />
              <span className="max-w-[120px] truncate">Tracking: {summary.active_timer}</span>
            </div>
          )}
        </>
      )}

      {/* Refresh button */}
      <button
        onClick={onRefresh}
        className="ml-auto flex items-center rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground"
        title="Refresh context"
        aria-label="Refresh AI context"
      >
        <RefreshCw className="h-3 w-3" />
      </button>
    </div>
  )
})
ContextPill.displayName = 'ContextPill'

const TypingIndicator = () => (
  <div className="flex items-center space-x-1 p-4">
    <div className="flex space-x-1">
      <div className="typing-indicator"></div>
      <div className="typing-indicator"></div>
      <div className="typing-indicator"></div>
    </div>
    <span className="text-sm text-muted-foreground ml-2">AI is thinking...</span>
  </div>
)

const MessageBubble = React.forwardRef<HTMLDivElement, { message: Message; isLast: boolean; coachAvatar?: string }>(({ message, isLast: _, coachAvatar = '' }, ref) => {
  const isUser = message.role === 'user'
  const AgentIcon = message.agent ? agentIcons[message.agent as keyof typeof agentIcons] || Bot : Bot

  // Parse reasoning to structured format
  const parseReasoning = (reasoning: string | AgentThinking | undefined): AgentThinking | null => {
    if (!reasoning) return null

    const safeString = (value: any): string => {
      if (value === null || value === undefined) return ''
      if (typeof value === 'string') return value
      return String(value)
    }

    const normalizeAgentType = (value: any): string => {
      if (!value) return ''
      if (typeof value === 'string') return value.toLowerCase()
      if (typeof value === 'object') {
        if (typeof value.value === 'string') return value.value.toLowerCase()
        if (typeof value.name === 'string') return value.name.toLowerCase()
      }
      return String(value).toLowerCase()
    }

    const normalizeReasoningObject = (rawObj: Record<string, any>): AgentThinking | null => {
      const normalized: AgentThinking = {}

      const normalizeStepItems = (rawSteps: any): any[] => {
        if (typeof rawSteps === 'string') {
          const singleStep = safeString(rawSteps).trim()
          return singleStep ? [singleStep] : []
        }

        if (!Array.isArray(rawSteps)) {
          return []
        }

        const looksLikeCharacterArray =
          rawSteps.length > 12
          && rawSteps.every((item: any) => typeof item === 'string' && item.length <= 1)

        if (looksLikeCharacterArray) {
          const joined = rawSteps.join('').trim()
          return joined ? [joined] : []
        }

        return rawSteps
      }

      const normalizedRawSteps = normalizeStepItems(rawObj.steps)
      if (normalizedRawSteps.length > 0) {
        normalized.steps = normalizedRawSteps
          .map((step: any) => ({
            agent: normalizeAgentType(step?.agent || step?.from || 'orchestrator') || 'orchestrator',
            action: safeString(step?.action || step?.description || step),
            result: safeString(step?.result || step?.status || ''),
          }))
          .filter((step: { action: string }) => Boolean(step.action))
      }

      if (rawObj.classification && typeof rawObj.classification === 'object') {
        const classificationAgent = normalizeAgentType(rawObj.classification.agent_type || rawObj.classification.agent)
        normalized.classification = {
          agent_type: classificationAgent || 'general',
          confidence: Number(rawObj.classification.confidence || 0),
          reason: safeString(rawObj.classification.reason || ''),
        }
      }

      if (Array.isArray(rawObj.agent_outputs)) {
        const agentOutputs = rawObj.agent_outputs
          .map((output: any) => ({
            agent: normalizeAgentType(output?.agent || output?.agent_type || 'orchestrator') || 'orchestrator',
            response_preview: safeString(output?.response_preview || output?.response || ''),
          }))
          .filter((output: { response_preview: string }) => Boolean(output.response_preview))

        if (agentOutputs.length > 0) {
          normalized.agent_outputs = agentOutputs
        }
      }

      if (rawObj.intent && typeof rawObj.intent === 'object') {
        const intentAgent = normalizeAgentType(rawObj.intent.agent_type || rawObj.intent.agent)
        if (!normalized.classification && intentAgent) {
          normalized.classification = {
            agent_type: intentAgent,
            confidence: Number(rawObj.intent.confidence || 0),
            reason: safeString(rawObj.intent.reason || ''),
          }
        }
        normalized.intent = {
          agent_type: intentAgent || undefined,
          confidence: Number(rawObj.intent.confidence || 0),
          reason: safeString(rawObj.intent.reason || ''),
        }
      }

      const executionPath = Array.isArray(rawObj.execution_path)
        ? rawObj.execution_path
        : (typeof rawObj.execution_path === 'string' && rawObj.execution_path.trim()
          ? [rawObj.execution_path]
          : [])
      if (executionPath.length > 0) {
        const existingSteps = normalized.steps ? [...normalized.steps] : []
        executionPath.forEach((pathStep: any, index: number) => {
          const action = safeString(pathStep)
          if (action) {
            existingSteps.push({
              agent: 'orchestrator',
              action,
              result: index === executionPath.length - 1 ? 'Completed' : undefined,
            })
          }
        })
        normalized.execution_path = executionPath.map((item: any) => safeString(item)).filter(Boolean)
        if (existingSteps.length > 0) {
          normalized.steps = existingSteps
        }
      }

      if (rawObj.plan?.steps && Array.isArray(rawObj.plan.steps)) {
        const existingSteps = normalized.steps ? [...normalized.steps] : []
        rawObj.plan.steps.forEach((planStep: any) => {
          const action = safeString(planStep?.action || planStep?.description)
          if (action) {
            existingSteps.push({
              agent: normalizeAgentType(planStep?.agent || 'orchestrator') || 'orchestrator',
              action,
              result: safeString(planStep?.estimated_time ? `~${planStep.estimated_time} min` : ''),
            })
          }
        })
        if (existingSteps.length > 0) {
          normalized.steps = existingSteps
        }
      }

      const knowledgeSources: AgentThinking['knowledge_sources'] = []
      if (Array.isArray(rawObj.knowledge_sources)) {
        rawObj.knowledge_sources.forEach((source: any) => {
          if (!source) return
          knowledgeSources.push({
            type: safeString(source.type || 'knowledge'),
            content: safeString(source.content || source.summary || source),
            similarity: typeof source.similarity === 'number' ? source.similarity : undefined,
            created_at: source.created_at,
            category: source.category,
            metadata: source.metadata,
          })
        })
      }

      if (rawObj.data_points_used && typeof rawObj.data_points_used === 'object') {
        const dataPoints = rawObj.data_points_used
        normalized.data_points_used = {
          role: safeString(dataPoints.role || ''),
          priorities: Array.isArray(dataPoints.priorities)
            ? dataPoints.priorities.map((item: any) => safeString(item)).filter(Boolean)
            : [],
          knowledge_context_summary: safeString(dataPoints.knowledge_context_summary || ''),
        }
      }

      if (knowledgeSources.length > 0) {
        normalized.knowledge_sources = knowledgeSources
      }

      if (rawObj.handoff && typeof rawObj.handoff === 'object') {
        normalized.handoff = {
          from: safeString(rawObj.handoff.from || 'orchestrator'),
          to: safeString(rawObj.handoff.to || ''),
          reason: safeString(rawObj.handoff.reason || ''),
        }
      }

      const finalAgent = normalizeAgentType(rawObj.finalAgent || rawObj.final_agent || rawObj.agent)
      if (finalAgent) {
        normalized.finalAgent = finalAgent
      } else if (normalized.classification?.agent_type) {
        normalized.finalAgent = normalized.classification.agent_type
      }

      const error = safeString(rawObj.error || '')
      if (error) {
        normalized.error = error
      }

      return Object.keys(normalized).length > 0 ? normalized : null
    }
    
    if (typeof reasoning === 'object') {
      return normalizeReasoningObject(reasoning as Record<string, any>)
    }
    
    if (typeof reasoning === 'string') {
      try {
        // Try to parse as JSON first
        const parsed = JSON.parse(reasoning)
        if (parsed && typeof parsed === 'object') {
          return normalizeReasoningObject(parsed)
        }
      } catch {
        // If not JSON, check for specific patterns and create structured thinking
        const thinking: AgentThinking = {}
        
        // Look for agent mentions
        const agentMatch = reasoning.match(/(\w+)\s+agent/i)
        if (agentMatch) {
          thinking.finalAgent = agentMatch[1].toLowerCase()
        }
        
        // Look for classification patterns
        const classificationMatch = reasoning.match(/classified as (\w+)/i)
        if (classificationMatch) {
          thinking.classification = {
            agent_type: classificationMatch[1].toLowerCase(),
            confidence: 0.8,
            reason: reasoning
          }
        }
        
        // Look for handoff patterns
        const handoffMatch = reasoning.match(/handoff from (\w+) to (\w+)/i)
        if (handoffMatch) {
          thinking.handoff = {
            from: handoffMatch[1],
            to: handoffMatch[2],
            reason: reasoning
          }
        }
        
        // If it looks like an error
        if (reasoning.toLowerCase().includes('error') || reasoning.toLowerCase().includes('failed')) {
          thinking.error = reasoning
        }
        
        return Object.keys(thinking).length > 0 ? thinking : null
      }
    }
    
    return null
  }

  const structuredThinking = parseReasoning(message.reasoning)

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "group flex gap-3 px-3 py-4 sm:px-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center overflow-hidden",
        isUser 
          ? "bg-primary text-primary-foreground" 
          : coachAvatar 
            ? "bg-transparent" 
            : "bg-gradient-to-br from-teal-700 via-cyan-600 to-amber-500 text-white"
      )}>
        {isUser ? (
          <User className="w-4 h-4" />
        ) : coachAvatar ? (
          <img src={coachAvatar} alt="Coach" className="w-full h-full object-cover" />
        ) : (
          <AgentIcon className="w-4 h-4" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex min-w-0 flex-col max-w-[92%] sm:max-w-[80%]",
        isUser ? "items-end" : "items-start"
      )}>
        {/* Agent Name */}
        {!isUser && message.agent && (
          <div className="text-xs text-muted-foreground mb-1 capitalize flex items-center gap-2">
            <span>{message.agent} Agent</span>
            {(() => {
              let reasoningText = '';
              if (typeof message.reasoning === 'string') {
                reasoningText = message.reasoning;
              } else if (message.reasoning && typeof message.reasoning === 'object') {
                reasoningText = JSON.stringify(message.reasoning);
              }
              return reasoningText.includes('openai') ? (
                <span className="px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 rounded text-xs">
                  OpenAI
                </span>
              ) : null;
            })()}
            {(() => {
              let reasoningText = '';
              if (typeof message.reasoning === 'string') {
                reasoningText = message.reasoning;
              } else if (message.reasoning && typeof message.reasoning === 'object') {
                reasoningText = JSON.stringify(message.reasoning);
              }
              return reasoningText.includes('ollama') ? (
                <span className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded text-xs">
                  Ollama
                </span>
              ) : null;
            })()}
          </div>
        )}

        {/* Message Bubble */}
        <Card className={cn(
          "max-w-full overflow-hidden p-3 shadow-md transition-all duration-200",
          isUser
            ? "ml-auto border-primary/30 bg-primary text-primary-foreground"
            : "border-border/70 bg-card/85 hover:shadow-lg",
          message.isStreaming && "animate-pulse"
        )}>
          <div className="prose prose-sm max-w-none break-words overflow-x-hidden dark:prose-invert">
            {(() => {
              const content = message.content
              const contentStr = typeof content === 'string'
                ? content
                : content && typeof content === 'object'
                  ? JSON.stringify(content, null, 2)
                  : String(content || 'No content')

              // Check if content contains daily checkup HTML - render as raw HTML
              if (contentStr.includes('class="daily-checkup"') || contentStr.includes("class='daily-checkup'")) {
                return (
                  <div
                    className="daily-checkup-wrapper"
                    dangerouslySetInnerHTML={{ __html: contentStr }}
                  />
                )
              }

              // Regular markdown content
              return (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => <p className="mb-2 break-words last:mb-0">{children}</p>,
                    code: ({ children, className }) => (
                      <code className={cn(
                        "break-all px-1.5 py-0.5 rounded text-xs font-mono bg-muted/80",
                        className
                      )}>
                        {children}
                      </code>
                    ),
                    pre: ({ children }) => (
                      <pre className={cn(
                        "max-w-full whitespace-pre-wrap break-words p-3 rounded-lg overflow-x-auto text-xs bg-muted/80"
                      )}>
                        {children}
                      </pre>
                    ),
                  }}
                >
                  {contentStr}
                </ReactMarkdown>
              )
            })()}
          </div>
        </Card>

        {/* Actionable Suggestions */}
        {message.actionableData?.suggestions && message.actionableData.suggestions.length > 0 && (
          <ActionableSuggestions 
            suggestions={message.actionableData.suggestions}
          />
        )}

        {/* Reasoning (if available) */}
        {message.reasoning && (
          structuredThinking ? (
            <AgentThinkingDisplay thinking={structuredThinking} />
          ) : (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mt-2 p-2 bg-muted/50 rounded-lg text-xs text-muted-foreground max-w-full"
            >
              <div className="font-medium mb-1">💭 Agent Reasoning:</div>
              <div>
                {typeof message.reasoning === 'string'
                  ? message.reasoning
                  : <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{JSON.stringify(message.reasoning, null, 2)}</pre>
                }
              </div>
            </motion.div>
          )
        )}

        {/* Timestamp */}
        <div className="text-xs text-muted-foreground mt-1">
          {message.timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>
    </motion.div>
  )
})

const MemoizedMessageBubble = React.memo(MessageBubble)
MemoizedMessageBubble.displayName = "MessageBubble"

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  className,
  onSendMessage,
  messages = [],
  isLoading = false,
  currentAgent = 'orchestrator',
  currentProvider = 'openai'
}) => {
  const [inputValue, setInputValue] = useState('')
  const [isComposing, setIsComposing] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  
  // User profile state
  const [userProfile, setUserProfile] = useState<any>(null)
  const [userAvatar, setUserAvatar] = useState<string>('')
  const [communicationStyle, setCommunicationStyle] = useState<string>('Direct')

  // Chat context summary state
  const [chatContextSummary, setChatContextSummary] = useState<ChatContextSummary | null>(null)
  const [contextSummaryLoading, setContextSummaryLoading] = useState<boolean>(true)

  // Derive first name from profile role string or a name field
  const firstName = useMemo(() => {
    if (!userProfile) return ''
    const nameField =
      userProfile.name ||
      userProfile.firstName ||
      userProfile.first_name ||
      (typeof userProfile.role === 'string' ? userProfile.role.split(' ')[0] : '') ||
      ''
    return typeof nameField === 'string' ? nameField.trim() : ''
  }, [userProfile])

  // Time-based greeting
  const greeting = useMemo(() => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Good morning'
    if (hour < 17) return 'Good afternoon'
    return 'Good evening'
  }, [])

  // Smart jump-back-in actions derived from real context
  const jumpBackInActions = useMemo(() => {
    const actions: Array<{ label: string; message: string; isPrimary?: boolean }> = [
      { label: 'What should I focus on?', message: 'What should I focus on right now?', isPrimary: true },
    ]
    if (chatContextSummary) {
      if (chatContextSummary.habits_pending > 0) {
        actions.push({
          label: `Check in on habits (${chatContextSummary.habits_pending} remaining)`,
          message: `I have ${chatContextSummary.habits_pending} habit${chatContextSummary.habits_pending !== 1 ? 's' : ''} pending today. Help me check in on them.`,
        })
      }
      if (chatContextSummary.overdue_tasks > 0) {
        actions.push({
          label: `${chatContextSummary.overdue_tasks} overdue task${chatContextSummary.overdue_tasks !== 1 ? 's' : ''} — help me tackle them`,
          message: `I have ${chatContextSummary.overdue_tasks} overdue task${chatContextSummary.overdue_tasks !== 1 ? 's' : ''}. Help me prioritize and tackle them.`,
        })
      }
      if (chatContextSummary.active_timer) {
        actions.push({
          label: `Log time for ${chatContextSummary.active_timer}`,
          message: `I'm currently tracking time for "${chatContextSummary.active_timer}". Help me wrap up this session.`,
        })
      }
    }
    return actions
  }, [chatContextSummary])

  // Load user profile callback
  const loadProfile = useCallback(async () => {
    try {
      const response = await fetch('/api/knowledge/onboarding/profile', {
        headers: {
          'Accept': 'application/json'
        }
      });
      if (response.ok) {
        const data = await response.json() as Record<string, unknown> | null;
        // Null check to prevent crashes from malformed data
        if (!data || typeof data !== 'object') {
          console.log('Invalid profile data received');
          return;
        }
        setUserProfile(data);
        // Set avatar from mentor selection with safe navigation
        const coachAvatar = typeof data?.coachAvatar === 'string' ? data.coachAvatar : '';
        const mentorAvatar = data?.mentor && typeof data.mentor === 'object' 
          ? String((data.mentor as Record<string, unknown>)?.avatar || '') 
          : '';
        setUserAvatar(coachAvatar || mentorAvatar);
        // Set communication style with safe navigation
        const mentorStyle = data?.mentor && typeof data.mentor === 'object'
          ? (data.mentor as Record<string, unknown>)?.style
          : undefined;
        setCommunicationStyle(typeof mentorStyle === 'string' ? mentorStyle : 'Direct');
      }
    } catch (error) {
      console.log('No user profile found, using defaults');
    }
  }, []);

  // Load user profile on mount
  useEffect(() => {
    void loadProfile();
  }, [loadProfile]);

  // Load chat context summary
  const loadContextSummary = useCallback(async () => {
    setContextSummaryLoading(true)
    try {
      const res = await fetch('/api/knowledge/chat-context-summary', {
        headers: { 'Accept': 'application/json' },
      })
      if (res.ok) {
        const data = await res.json() as ChatContextSummary
        setChatContextSummary(data)
      }
    } catch {
      // silently fail — pill shows "Context loading..." state
    } finally {
      setContextSummaryLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadContextSummary()
    // Refresh every 5 minutes
    const interval = setInterval(() => { void loadContextSummary() }, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [loadContextSummary])

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      const container = messagesContainerRef.current
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
      return
    }

    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage?.(inputValue.trim())
      setInputValue('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={cn("flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background/40", className)}>
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/70 bg-white/60 p-3 backdrop-blur-xl dark:bg-slate-950/50 sm:items-center sm:p-4">
        <div className="flex items-center gap-3">
          <div className={cn(
            "h-10 w-10 overflow-hidden rounded-2xl border border-border/70 flex items-center justify-center",
            userAvatar ? "bg-transparent" : "bg-gradient-to-br from-teal-700 via-cyan-600 to-amber-500"
          )}>
            {userAvatar ? (
              <img src={userAvatar} alt="Coach" className="w-full h-full object-cover" />
            ) : (
              <Brain className="w-5 h-5 text-white" />
            )}
          </div>
          <div>
            <h2 className="text-sm font-semibold sm:text-base">Your Coaching Session</h2>
            <p className="text-xs capitalize text-muted-foreground sm:text-sm">
              {currentAgent === 'orchestrator' ? 'Coach' : currentAgent} is here to help
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Communication Style Indicator */}
          {communicationStyle && communicationStyle !== 'Direct' && (
            <div className="flex items-center gap-2 rounded-full bg-amber-100/80 px-2 py-1 dark:bg-amber-900/25">
              <span className="text-xs font-semibold text-amber-800 dark:text-amber-300">
                {communicationStyle} Style
              </span>
            </div>
          )}
          {/* Provider Indicator */}
          <div className="flex items-center gap-2 rounded-full border border-border/70 bg-white/70 px-2 py-1 dark:bg-slate-900/60">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs text-muted-foreground font-medium">
              Ready
            </span>
          </div>
          <div className="hidden sm:flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-xs text-muted-foreground">Online</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        data-app-scroll-region="true"
        className="custom-scrollbar min-h-0 flex-1 overflow-x-hidden overflow-y-auto"
      >
        <AnimatePresence mode="popLayout">
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex h-full flex-col items-start justify-start gap-4 overflow-y-auto p-6"
            >
              {/* Header row */}
              <div className="flex w-full max-w-2xl items-center gap-4">
                <div className={cn(
                  "flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl shadow-md",
                  userAvatar ? "overflow-hidden border border-border/50 bg-transparent" : "bg-gradient-to-br from-teal-700 via-cyan-600 to-amber-500"
                )}>
                  {userAvatar ? (
                    <img src={userAvatar} alt="Coach" className="h-full w-full object-cover" />
                  ) : (
                    <Sparkles className="h-7 w-7 text-white" />
                  )}
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                    {greeting}{firstName ? `, ${firstName}` : ''}!
                  </h2>
                  <p className="text-sm text-muted-foreground">Here's where things stand today</p>
                </div>
              </div>

              {/* Stats row */}
              {contextSummaryLoading ? (
                <div className="grid w-full max-w-2xl grid-cols-3 gap-3">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="h-20 animate-pulse rounded-2xl bg-gray-100 dark:bg-slate-800" />
                  ))}
                </div>
              ) : chatContextSummary && (chatContextSummary.tasks_count > 0 || chatContextSummary.habits_total > 0 || chatContextSummary.goals_count > 0) ? (
                <div className="grid w-full max-w-2xl grid-cols-3 gap-3">
                  {/* Tasks card */}
                  <div className="rounded-2xl border border-border/70 bg-white/75 p-3 shadow-sm backdrop-blur-lg dark:bg-slate-900/65">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <ListTodo className="h-3.5 w-3.5 text-blue-500" />
                      Tasks
                    </div>
                    <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-slate-100">{chatContextSummary.tasks_count}</p>
                    <p className="text-xs text-muted-foreground">active</p>
                    {chatContextSummary.overdue_tasks > 0 && (
                      <p className="mt-1 flex items-center gap-1 text-xs font-medium text-red-600 dark:text-red-400">
                        <AlertCircle className="h-3 w-3" />
                        {chatContextSummary.overdue_tasks} overdue
                      </p>
                    )}
                  </div>

                  {/* Habits card */}
                  <div className="rounded-2xl border border-border/70 bg-white/75 p-3 shadow-sm backdrop-blur-lg dark:bg-slate-900/65">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <Leaf className="h-3.5 w-3.5 text-green-500" />
                      Habits
                    </div>
                    <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-slate-100">
                      {chatContextSummary.habits_total - chatContextSummary.habits_pending}
                      <span className="text-base font-normal text-muted-foreground">/{chatContextSummary.habits_total}</span>
                    </p>
                    <p className="text-xs text-muted-foreground">done today</p>
                    {chatContextSummary.habits_pending > 0 && (
                      <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">{chatContextSummary.habits_pending} pending</p>
                    )}
                  </div>

                  {/* Goals card */}
                  <div className="rounded-2xl border border-border/70 bg-white/75 p-3 shadow-sm backdrop-blur-lg dark:bg-slate-900/65">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      <Target className="h-3.5 w-3.5 text-violet-500" />
                      Goals
                    </div>
                    <p className="mt-1 text-2xl font-bold text-slate-900 dark:text-slate-100">{chatContextSummary.goals_count}</p>
                    <p className="text-xs text-muted-foreground">active</p>
                    {chatContextSummary.active_timer && (
                      <p className="mt-1 flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400 truncate">
                        <Flame className="h-3 w-3 shrink-0" />
                        <span className="truncate">{chatContextSummary.active_timer}</span>
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                !contextSummaryLoading && (
                  <div className="w-full max-w-2xl rounded-2xl border border-dashed border-border/70 bg-white/50 p-4 text-center text-sm text-muted-foreground dark:bg-slate-900/40">
                    Start by telling your coach about your goals for today.
                  </div>
                )
              )}

              {/* Jump back in */}
              <div className="w-full max-w-2xl">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Jump back in
                </p>
                <div className="flex flex-col gap-2">
                  {contextSummaryLoading ? (
                    [0, 1, 2].map((i) => (
                      <div key={i} className="h-9 animate-pulse rounded-xl bg-gray-100 dark:bg-slate-800" />
                    ))
                  ) : (
                    jumpBackInActions.map((action) => (
                      <button
                        key={action.label}
                        type="button"
                        onClick={() => {
                          setInputValue(action.message)
                          setTimeout(() => {
                            if (!isLoading) {
                              onSendMessage?.(action.message)
                              setInputValue('')
                            }
                          }, 80)
                        }}
                        className={cn(
                          "w-full rounded-xl px-4 py-2.5 text-left text-sm font-medium transition-all",
                          action.isPrimary
                            ? "border border-teal-500/40 bg-teal-50 text-teal-800 hover:bg-teal-100 dark:border-teal-700/50 dark:bg-teal-950/40 dark:text-teal-200 dark:hover:bg-teal-950/70"
                            : "border border-border/70 bg-white/75 text-slate-700 hover:border-teal-400/50 hover:bg-teal-50/50 dark:bg-slate-900/65 dark:text-slate-300 dark:hover:bg-slate-800/80"
                        )}
                      >
                        {action.label}
                      </button>
                    ))
                  )}
                </div>
              </div>

              {/* Or start fresh */}
              <div className="w-full max-w-2xl">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Or start fresh
                </p>
                <div className="flex flex-wrap gap-2">
                  {['Plan my day', 'Health check-in', 'Quick win', 'Weekly review'].map((label) => (
                    <button
                      key={label}
                      type="button"
                      onClick={() => {
                        setInputValue(label)
                        setTimeout(() => {
                          if (!isLoading) {
                            onSendMessage?.(label)
                            setInputValue('')
                          }
                        }, 80)
                      }}
                      className="rounded-full border border-border/70 bg-white/70 px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:border-teal-400/50 hover:bg-teal-50/60 dark:bg-slate-900/60 dark:text-slate-400 dark:hover:bg-teal-950/30"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            messages.map((message, index) => (
              <MemoizedMessageBubble
                key={message.id}
                message={message}
                isLast={index === messages.length - 1}
                coachAvatar={userAvatar}
              />
            ))
          )}
        </AnimatePresence>

        {isLoading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border/70 bg-white/65 p-4 backdrop-blur-xl dark:bg-slate-950/60">
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <Input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              onCompositionStart={() => setIsComposing(true)}
              onCompositionEnd={() => setIsComposing(false)}
              placeholder="Ask me anything..."
              className="min-h-[46px] resize-none rounded-xl border-border/70 bg-white/80 pr-12 shadow-sm dark:bg-slate-900/70"
              disabled={isLoading}
            />
          </div>
          <Button
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
            size="icon"
            className="h-11 w-11 shrink-0 rounded-xl"
            variant="gradient"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        
        {/* Quick Actions */}
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          {(() => {
            const getQuickActions = (agent: string) => {
              switch (agent) {
                case 'productivity':
                  return [
                    "Add a new task",
                    "View my progress",
                    "Build a consistent morning routine",
                    "Plan weekly meals"
                  ]
                case 'health':
                  return [
                    "Health check-in",
                    "Log exercise",
                    "Track sleep quality",
                    "Track a daily health habit"
                  ]
                case 'finance':
                  return [
                    "Add an expense",
                    "Financial summary",
                    "Budget review",
                    "Savings tips"
                  ]
                case 'scheduling':
                  return [
                    "Check my calendar",
                    "Schedule a meeting",
                    "Reschedule conflicts",
                    "Time optimization"
                  ]
                case 'journal':
                  return [
                    "Daily reflection",
                    "Mood check-in",
                    "Celebrate achievement",
                    "Weekly review"
                  ]
                default:
                  return [
                    "What can you help me with?",
                    "Show my coaches",
                    "System status",
                    "Get suggestions"
                  ]
              }
            }
            
            return getQuickActions(currentAgent).map((suggestion, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                className="whitespace-nowrap rounded-full border-border/70 bg-white/75 text-xs font-medium shadow-sm transition-colors hover:border-teal-500/40 hover:bg-teal-50 dark:bg-slate-900/70 dark:hover:bg-teal-950/30"
                onClick={() => {
                  setInputValue(suggestion.replace(/^[^\s]+ /, '')) // Remove emoji prefix
                  // Auto-send the message
                  setTimeout(() => {
                    if (!isLoading) {
                      onSendMessage?.(suggestion.replace(/^[^\s]+ /, ''))
                      setInputValue('')
                    }
                  }, 100)
                }}
              >
                {suggestion}
              </Button>
            ))
          })()}
        </div>
      </div>
    </div>
  )
}