import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Bot, User, Sparkles, Zap, Brain, Settings, Database, ChevronDown, ChevronUp, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Message {
  id: string
  content: string | any  // Allow any type but convert to string during rendering
  role: 'user' | 'assistant' | 'system'
  timestamp: Date
  agent?: string
  reasoning?: string | AgentThinking
  isStreaming?: boolean
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

const AgentThinkingDisplay = ({ thinking }: { thinking: AgentThinking }) => {
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
                  <div className="text-xs text-slate-500 dark:text-slate-500 mt-1 truncate">
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
}

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

        if (normalized.data_points_used.role) {
          knowledgeSources.push({
            type: 'Profile',
            content: `Role: ${normalized.data_points_used.role}`,
          })
        }

        if (normalized.data_points_used.priorities && normalized.data_points_used.priorities.length > 0) {
          knowledgeSources.push({
            type: 'Priorities',
            content: normalized.data_points_used.priorities.join(', '),
          })
        }

        const contextSummary = normalized.data_points_used.knowledge_context_summary || ''
        if (contextSummary && !contextSummary.toLowerCase().startsWith('no previous context found')) {
          knowledgeSources.push({
            type: 'Knowledge Context',
            content: contextSummary,
          })
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
                <span className="px-1.5 py-0.5 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded text-xs">
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
                <span className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded text-xs">
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
            ? "ml-auto border-primary/30 bg-gradient-to-r from-teal-700 to-cyan-600 text-white" 
            : "border-border/70 bg-card/85 hover:shadow-lg",
          message.isStreaming && "animate-pulse"
        )}>
          <div className="prose prose-sm max-w-none break-words overflow-x-hidden dark:prose-invert">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-2 break-words last:mb-0">{children}</p>,
                code: ({ children, className }) => (
                  <code className={cn(
                    "break-all px-1.5 py-0.5 rounded text-xs font-mono",
                    isUser ? "bg-primary-foreground/20" : "bg-muted",
                    className
                  )}>
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre className={cn(
                    "max-w-full whitespace-pre-wrap break-words p-3 rounded-lg overflow-x-auto text-xs",
                    isUser ? "bg-primary-foreground/20" : "bg-muted"
                  )}>
                    {children}
                  </pre>
                ),
              }}
            >
              {(() => {
                const content = message.content
                if (typeof content === 'string') {
                  return content
                } else if (content && typeof content === 'object') {
                  console.warn('Message content is object, converting to string:', content)
                  return JSON.stringify(content, null, 2)
                } else {
                  console.warn('Message content is unexpected type:', typeof content, content)
                  return String(content || 'No content')
                }
              })()}
            </ReactMarkdown>
          </div>
        </Card>

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

MessageBubble.displayName = "MessageBubble"

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

  const starterCapabilities = [
    {
      title: 'Strategic Routing',
      description: 'Routes each request to the best specialist agent automatically.',
    },
    {
      title: 'Live Memory Context',
      description: 'Uses your onboarding and preference memory for grounded guidance.',
    },
    {
      title: 'Action-Ready Plans',
      description: 'Turns broad goals into next-step execution plans quickly.',
    },
  ]

  const starterTasks = [
    'Plan tomorrow\'s top 3 priorities',
    'Block focused work from 9:00 AM to 11:00 AM',
    'Prepare weekly meal plan and grocery list',
  ]

  const starterHabits = [
    'Wake up by 6:30 AM daily',
    'Log every meal and water intake',
    'Walk for 20 minutes after dinner',
  ]

  // Load user profile on mount
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const response = await fetch('/api/knowledge/onboarding/profile');
        if (response.ok) {
          const data = await response.json();
          setUserProfile(data);
          // Set avatar from mentor selection
          setUserAvatar(data.coachAvatar || data.mentor?.avatar || '');
          // Set communication style
          setCommunicationStyle(data.mentor?.style || 'Direct');
        }
      } catch (error) {
        console.log('No user profile found, using defaults');
      }
    };
    loadProfile();
  }, []);

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
            <h2 className="text-sm font-semibold sm:text-base">Agentic Conversation Desk</h2>
            <p className="text-xs capitalize text-muted-foreground sm:text-sm">
              {currentAgent} agent active
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
            <span className="text-xs text-muted-foreground font-medium capitalize">
              {currentProvider}
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
              className="flex h-full flex-col items-center justify-center p-8 text-center"
            >
              <div className="floating-animation mb-4 flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-teal-700 via-cyan-600 to-amber-500 shadow-lg shadow-cyan-500/30">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Start A Meaningful Session</h3>
              <p className="max-w-md text-muted-foreground">
                Ask anything, and Agentic will route context to the right specialist so responses stay actionable and personal.
              </p>
              <div className="mt-6 grid w-full max-w-3xl gap-3 sm:grid-cols-3">
                {starterCapabilities.map((item) => (
                  <div
                    key={item.title}
                    className="rounded-2xl border border-border/70 bg-white/70 p-3 text-left shadow-sm backdrop-blur-lg dark:bg-slate-900/65"
                  >
                    <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{item.title}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{item.description}</p>
                  </div>
                ))}
              </div>

              <div className="mt-6 grid w-full max-w-3xl gap-3 text-left sm:grid-cols-2">
                <div className="rounded-2xl border border-border/70 bg-white/75 p-4 shadow-sm backdrop-blur-lg dark:bg-slate-900/65">
                  <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">Sample Tasks</p>
                  <ul className="mt-2 space-y-2 text-xs text-muted-foreground">
                    {starterTasks.map((task) => (
                      <li key={task} className="flex items-start gap-2">
                        <span className="mt-[2px] h-1.5 w-1.5 rounded-full bg-teal-500" />
                        <button
                          type="button"
                          onClick={() => setInputValue(`Add task: ${task}`)}
                          className="text-left transition-colors hover:text-teal-700 dark:hover:text-teal-300"
                        >
                          {task}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="rounded-2xl border border-border/70 bg-white/75 p-4 shadow-sm backdrop-blur-lg dark:bg-slate-900/65">
                  <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">Sample Habits</p>
                  <ul className="mt-2 space-y-2 text-xs text-muted-foreground">
                    {starterHabits.map((habit) => (
                      <li key={habit} className="flex items-start gap-2">
                        <span className="mt-[2px] h-1.5 w-1.5 rounded-full bg-amber-500" />
                        <button
                          type="button"
                          onClick={() => setInputValue(`Track habit: ${habit}`)}
                          className="text-left transition-colors hover:text-amber-700 dark:hover:text-amber-300"
                        >
                          {habit}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </motion.div>
          ) : (
            messages.map((message, index) => (
              <MessageBubble
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
              placeholder="Type your message..."
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
                    "📋 Add a new task",
                    "📊 Show my productivity stats",
                    "🌅 Add task: Wake up by 6:30 AM",
                    "🍽️ Add task: Plan this week's meals"
                  ]
                case 'health':
                  return [
                    "❤️ Health check-in",
                    "💪 Log a workout",
                    "😴 Track sleep quality",
                    "🥗 Track habit: Log every meal"
                  ]
                case 'finance':
                  return [
                    "💰 Add an expense",
                    "📈 Financial summary",
                    "🎯 Budget review",
                    "💡 Savings tips"
                  ]
                case 'scheduling':
                  return [
                    "📅 Check my calendar",
                    "⏰ Schedule a meeting",
                    "🔄 Reschedule conflicts",
                    "⚡ Time optimization"
                  ]
                case 'journal':
                  return [
                    "📝 Daily reflection",
                    "😊 Mood check-in",
                    "🎉 Celebrate achievement",
                    "💭 Weekly review"
                  ]
                default:
                  return [
                    "❓ What can you help me with?",
                    "🤖 Show available agents",
                    "📊 System status",
                    "💡 Get suggestions"
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