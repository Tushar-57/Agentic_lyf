/**
 * Deep Agent Status Component
 * 
 * Displays the status of the Deep Agent system including:
 * - Current agent and sub-agent hierarchy
 * - Active planning and context offloading
 * - Task delegation and human-in-the-loop status
 * - File-based context and TODO management
 */

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  FileText, 
  CheckCircle, 
  Clock, 
  Users, 
  Eye,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  User,
  Bot,
  Bell
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { ApprovalInterface } from './ApprovalInterface'

interface DeepAgentState {
  current_agent: string
  agent_hierarchy: string[]
  active_planning: {
    id: string
    title: string
    status: 'planning' | 'executing' | 'completed'
    complexity: 'simple' | 'moderate' | 'complex' | 'expert'
    created_at: string
  }[]
  files: Record<string, string>
  todos: {
    id: number
    title: string
    description: string
    status: 'not-started' | 'in-progress' | 'completed'
  }[]
  approval_requests: {
    id: string
    agent_id: string
    action_type: string
    description: string
    priority: 'low' | 'medium' | 'high' | 'critical'
    status: 'pending' | 'approved' | 'denied' | 'timeout'
    created_at: string
  }[]
  guidance_requests: {
    id: string
    agent_id: string
    question: string
    urgency: 'low' | 'normal' | 'high'
    status: 'pending' | 'answered'
    created_at: string
  }[]
  escalations: {
    id: string
    agent_id: string
    issue: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    status: 'pending' | 'resolved'
    created_at: string
  }[]
}

interface DeepAgentStatusProps {
  agentState: DeepAgentState | null
  className?: string
  onApprovalResponse?: (requestId: string, decision: 'approve' | 'deny' | 'modify', feedback?: string) => void
  onGuidanceResponse?: (requestId: string, guidance: string) => void
  onEscalationResponse?: (escalationId: string, resolution: string) => void
}

const priorityVariants = {
  low: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300',
  normal: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300',
  medium: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300',
  high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300',
  critical: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300'
}

const statusVariants = {
  'not-started': 'bg-muted text-muted-foreground',
  'in-progress': 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300',
  'completed': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300',
  'planning': 'bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-300',
  'executing': 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300',
  'pending': 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300',
  'approved': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300',
  'denied': 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300',
  'timeout': 'bg-muted text-muted-foreground',
  'answered': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300',
  'resolved': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300'
}

const ApprovalRequestCard = ({ 
  request, 
  onResponse 
}: { 
  request: DeepAgentState['approval_requests'][0], 
  onResponse: (requestId: string, decision: 'approve' | 'deny' | 'modify', feedback?: string) => void 
}) => {
  const [feedback, setFeedback] = useState('')
  const [showFeedback, setShowFeedback] = useState(false)

  return (
    <Card className="p-4 border-l-4 border-orange-400">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="font-medium text-sm text-foreground">
            Approval Required
          </h4>
          <p className="text-xs text-muted-foreground mt-1">
            {request.agent_id} • {request.action_type}
          </p>
        </div>
        <Badge className={priorityVariants[request.priority]}>
          {request.priority}
        </Badge>
      </div>
      
      <p className="text-sm text-muted-foreground mb-4">
        {request.description}
      </p>
      
      <div className="space-y-2">
        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={() => onResponse(request.id, 'approve', feedback)}
            className="bg-emerald-600 hover:bg-emerald-700 text-white dark:bg-emerald-700 dark:hover:bg-emerald-600"
          >
            Approve
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => onResponse(request.id, 'deny', feedback)}
            className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950/30"
          >
            Deny
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setShowFeedback(!showFeedback)}
          >
            Add Feedback
          </Button>
        </div>
        
        {showFeedback && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Optional feedback or modifications..."
              className="w-full p-2 text-sm border rounded resize-none"
              rows={2}
            />
          </motion.div>
        )}
      </div>
    </Card>
  )
}

const TodoCard = ({ todo }: { todo: DeepAgentState['todos'][0] }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  
  return (
    <motion.div
      layout
      className="border rounded-lg p-3 bg-card"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-0 h-auto"
          >
            {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </Button>
          <span className="font-medium text-sm">{todo.title}</span>
        </div>
        <Badge className={statusVariants[todo.status]}>
          {todo.status.replace('-', ' ')}
        </Badge>
      </div>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 pt-2 border-t"
          >
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {todo.description}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export const DeepAgentStatus: React.FC<DeepAgentStatusProps> = ({
  agentState,
  className,
  onApprovalResponse,
  onGuidanceResponse,
  onEscalationResponse
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'planning' | 'files' | 'todos' | 'human' | 'approval'>('overview')
  const [pendingApprovalCount, setPendingApprovalCount] = useState(0)

  // Fetch approval stats from the new system
  const fetchApprovalStats = async () => {
    try {
      const response = await fetch('/api/approval/stats')
      if (response.ok) {
        const data = await response.json()
        setPendingApprovalCount(data.pending_interactions || 0)
      }
    } catch (error) {
      console.error('Failed to fetch approval stats:', error)
    }
  }

  useEffect(() => {
    fetchApprovalStats()
    // Poll every 5 seconds for updates
    const interval = setInterval(fetchApprovalStats, 5000)
    return () => clearInterval(interval)
  }, [])

  // Check if there's any activity from old system or new approval system
  const hasOldSystemActivity = agentState && (
    (agentState.approval_requests?.length > 0) ||
    (agentState.guidance_requests?.length > 0) ||
    (agentState.escalations?.length > 0) ||
    (agentState.todos?.length > 0)
  )
  
  const hasNewApprovalActivity = pendingApprovalCount > 0
  const hasAnyActivity = hasOldSystemActivity || hasNewApprovalActivity

  const pendingApprovals = agentState?.approval_requests?.filter(req => req.status === 'pending') || []
  const pendingGuidance = agentState?.guidance_requests?.filter(req => req.status === 'pending') || []
  const pendingEscalations = agentState?.escalations?.filter(esc => esc.status === 'pending') || []
  const activeTodos = agentState?.todos?.filter(todo => todo.status !== 'completed') || []
  const completedTodos = agentState?.todos?.filter(todo => todo.status === 'completed') || []

  const hasHumanInteraction = pendingApprovals.length > 0 || pendingGuidance.length > 0 || pendingEscalations.length > 0 || pendingApprovalCount > 0

  // Auto-switch to approval tab when there are pending approvals and no other activity
  useEffect(() => {
    if (pendingApprovalCount > 0 && !hasOldSystemActivity && activeTab === 'overview') {
      setActiveTab('approval')
    }
  }, [pendingApprovalCount, hasOldSystemActivity, activeTab])

  if (!hasAnyActivity) {
    return (
      <Card className={cn("p-4", className)}>
        <div className="flex items-center gap-2 text-gray-500">
          <Bot className="w-5 h-5" />
          <span>No deep agent activity</span>
        </div>
      </Card>
    )
  }

  return (
    <Card className={cn("p-4", className)}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-blue-600" />
            <span className="font-medium">Deep Agent System</span>
          </div>
          <Badge variant="outline" className="text-xs">
            Agent: {agentState?.current_agent || 'Deep Agent'}
          </Badge>
        </div>

        {/* Quick Status */}
        {/* Quick Status */}
        <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
          <div className="p-2 rounded bg-blue-50 dark:bg-blue-900/20">
            <div className="text-lg font-bold text-blue-600">{Object.keys(agentState?.files || {}).length}</div>
            <div className="text-xs text-blue-600">Files</div>
          </div>
          <div className="p-2 rounded bg-green-50 dark:bg-green-900/20">
            <div className="text-lg font-bold text-green-600">{completedTodos.length}</div>
            <div className="text-xs text-green-600">Done</div>
          </div>
          <div className="p-2 rounded bg-purple-50 dark:bg-purple-900/20">
            <div className="text-lg font-bold text-purple-600">{agentState?.active_planning?.length || 0}</div>
            <div className="text-xs text-purple-600">Planning</div>
          </div>
          <div className="p-2 rounded bg-orange-50 dark:bg-orange-900/20">
            <div className="text-lg font-bold text-orange-600">{pendingApprovals.length + pendingApprovalCount}</div>
            <div className="text-xs text-orange-600">Approvals</div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1 overflow-x-auto rounded-lg bg-gray-100 p-1 dark:bg-gray-800">
          {[
            { id: 'overview', label: 'Overview', icon: Eye },
            { id: 'planning', label: 'Planning', icon: Brain },
            { id: 'files', label: 'Files', icon: FileText },
            { id: 'todos', label: 'Tasks', icon: CheckCircle },
            { id: 'human', label: 'Human', icon: User, hasNotification: hasHumanInteraction },
            { id: 'approval', label: 'Approval', icon: Bell, hasNotification: pendingApprovalCount > 0 }
          ].map(tab => (
            <Button
              key={tab.id}
              variant={activeTab === tab.id ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                "relative min-w-[5.75rem] shrink-0 text-xs sm:min-w-0 sm:flex-1",
                activeTab === tab.id && "bg-white dark:bg-gray-700 shadow-sm"
              )}
            >
              <tab.icon className="w-3 h-3 mr-1" />
              {tab.label}
              {tab.hasNotification && (
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" />
              )}
            </Button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="space-y-3">
          {activeTab === 'overview' && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Bot className="w-4 h-4" />
                <span>Current Agent: </span>
                <Badge variant="outline">{agentState?.current_agent || 'Deep Agent'}</Badge>
              </div>
              
              {agentState?.agent_hierarchy && agentState.agent_hierarchy.length > 1 && (
                <div className="text-xs text-muted-foreground">
                  Agent Hierarchy:
                  <div className="flex items-center gap-1 mt-1">
                    {agentState.agent_hierarchy.map((agent, index) => (
                      <React.Fragment key={agent}>
                        <span className="text-xs bg-gray-100 dark:bg-gray-800 px-1 rounded">{agent}</span>
                        {index < agentState.agent_hierarchy.length - 1 && (
                          <ChevronRight className="h-3 w-3" />
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              )}

              {hasHumanInteraction && (
                <div className="p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
                  <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="font-medium text-sm">Human Input Required</span>
                  </div>
                  <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
                    {pendingApprovals.length} approvals, {pendingGuidance.length} guidance requests, {pendingEscalations.length} escalations
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'planning' && (
            <div className="space-y-2">
              {agentState?.active_planning && agentState.active_planning.length > 0 ? (
                agentState.active_planning.map(plan => (
                  <div key={plan.id} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm">{plan.title}</span>
                      <Badge className={statusVariants[plan.status]}>
                        {plan.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                      <Badge variant="outline" className="text-xs">
                        {plan.complexity}
                      </Badge>
                      <Clock className="w-3 h-3" />
                      <span>{new Date(plan.created_at).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 py-4">
                  <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No active planning</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'files' && (
            <div className="space-y-2">
              {Object.keys(agentState?.files || {}).length > 0 ? (
                Object.entries(agentState?.files || {}).map(([filename, content]) => (
                  <div key={filename} className="p-3 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-sm">{filename}</span>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {content.slice(0, 100)}...
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-center text-gray-500 py-4">
                  <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No files created</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'todos' && (
            <div className="space-y-3">
              {activeTodos.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2">Active Tasks</h4>
                  <div className="space-y-2">
                    {activeTodos.map(todo => (
                      <TodoCard key={todo.id} todo={todo} />
                    ))}
                  </div>
                </div>
              )}
              
              {completedTodos.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2">Completed</h4>
                  <div className="space-y-2">
                    {completedTodos.slice(0, 3).map(todo => (
                      <TodoCard key={todo.id} todo={todo} />
                    ))}
                    {completedTodos.length > 3 && (
                      <p className="text-xs text-gray-500 text-center">
                        +{completedTodos.length - 3} more completed
                      </p>
                    )}
                  </div>
                </div>
              )}

              {activeTodos.length === 0 && completedTodos.length === 0 && (
                <div className="text-center text-gray-500 py-4">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No tasks yet</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'human' && (
            <div className="space-y-4">
              {/* Approval Requests */}
              {pendingApprovals.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2 flex items-center gap-2">
                    <User className="w-4 h-4" />
                    Approval Requests
                  </h4>
                  <div className="space-y-2">
                    {pendingApprovals.map(request => (
                      <ApprovalRequestCard
                        key={request.id}
                        request={request}
                        onResponse={onApprovalResponse || (() => {})}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Guidance Requests */}
              {pendingGuidance.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2">Guidance Requests</h4>
                  <div className="space-y-2">
                    {pendingGuidance.map(request => (
                      <Card key={request.id} className="p-4 border-l-4 border-blue-400">
                        <div className="flex items-start justify-between mb-2">
                          <span className="font-medium text-sm">{request.agent_id}</span>
                          <Badge className={priorityVariants[request.urgency as keyof typeof priorityVariants] || priorityVariants.medium}>
                            {request.urgency}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                          {request.question}
                        </p>
                        <Button
                          size="sm"
                          onClick={() => onGuidanceResponse?.(request.id, '')}
                        >
                          Provide Guidance
                        </Button>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {/* Escalations */}
              {pendingEscalations.length > 0 && (
                <div>
                  <h4 className="font-medium text-sm mb-2">Escalations</h4>
                  <div className="space-y-2">
                    {pendingEscalations.map(escalation => (
                      <Card key={escalation.id} className="p-4 border-l-4 border-red-400">
                        <div className="flex items-start justify-between mb-2">
                          <span className="font-medium text-sm">{escalation.agent_id}</span>
                          <Badge className={priorityVariants[escalation.severity]}>
                            {escalation.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                          {escalation.issue}
                        </p>
                        <Button
                          size="sm"
                          onClick={() => onEscalationResponse?.(escalation.id, '')}
                        >
                          Resolve
                        </Button>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {!hasHumanInteraction && (
                <div className="text-center text-gray-500 py-4">
                  <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No human interaction required</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'approval' && (
            <div className="space-y-4">
              <ApprovalInterface 
                onApprovalChange={() => {
                  // Refresh approval stats when approval is processed
                  fetchApprovalStats()
                  console.log('Approval processed - refreshing state')
                }}
              />
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}