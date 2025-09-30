import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, Clock, User, Bot, Database, ChevronDown, ChevronUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'

interface KnowledgeSource {
  type: string
  content: string
  similarity?: number
  created_at?: string
  category?: string
  metadata?: Record<string, any>
}

interface PendingInteraction {
  id: string
  user_input: string
  agent_response: string
  agent_type: string
  timestamp: string
  status: string
  knowledge_sources?: KnowledgeSource[]
}

interface ApprovalInterfaceProps {
  onApprovalChange?: () => void
}

export const ApprovalInterface = ({ onApprovalChange }: ApprovalInterfaceProps) => {
  const [pendingInteractions, setPendingInteractions] = useState<PendingInteraction[]>([])
  const [loading, setLoading] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
    const [showSources, setShowSources] = useState<Record<string, boolean>>({});

  const fetchPendingInteractions = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/approval/pending')
      if (response.ok) {
        const data = await response.json()
        setPendingInteractions(data)
      }
    } catch (error) {
      console.error('Failed to fetch pending interactions:', error)
    }
  }

  useEffect(() => {
    fetchPendingInteractions()
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchPendingInteractions, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleApproval = async (interactionId: string, approved: boolean) => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/approval/approve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interaction_id: interactionId,
          approved: approved
        })
      })

      if (response.ok) {
        // Remove the approved/rejected interaction from the list
        setPendingInteractions(prev => prev.filter(item => item.id !== interactionId))
        
        toast.success(
          approved 
            ? '✅ Response approved and saved to knowledge base' 
            : '❌ Response rejected - not saved'
        )
        
        // Notify parent component
        onApprovalChange?.()
      } else {
        throw new Error('Failed to process approval')
      }
    } catch (error) {
      console.error('Approval error:', error)
      toast.error('Failed to process approval')
    } finally {
      setLoading(false)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getAgentColor = (agentType: string) => {
    switch (agentType) {
      case 'health': return 'bg-green-500'
      case 'finance': return 'bg-blue-500'
      case 'productivity': return 'bg-purple-500'
      case 'scheduling': return 'bg-orange-500'
      case 'journal': return 'bg-pink-500'
      default: return 'bg-gray-500'
    }
  }

  if (pendingInteractions.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            All Caught Up!
          </CardTitle>
          <CardDescription>
            No pending interactions require your approval.
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Clock className="h-5 w-5" />
          Pending Approvals ({pendingInteractions.length})
        </h3>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={fetchPendingInteractions}
          disabled={loading}
        >
          Refresh
        </Button>
      </div>

      <div className="h-[400px] overflow-y-auto">
        <div className="space-y-3">
          <AnimatePresence>
            {pendingInteractions.map((interaction) => (
              <motion.div
                key={interaction.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
              >
                <Card className="border-2 border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge 
                          className={`${getAgentColor(interaction.agent_type)} text-white`}
                        >
                          {interaction.agent_type}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {formatTimestamp(interaction.timestamp)}
                        </span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setExpandedId(
                          expandedId === interaction.id ? null : interaction.id
                        )}
                      >
                        {expandedId === interaction.id ? 'Collapse' : 'View Details'}
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {expandedId === interaction.id && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-4"
                      >
                        {/* User Input */}
                        <div className="flex gap-3">
                          <User className="h-4 w-4 mt-1 text-blue-500" />
                          <div className="flex-1">
                            <p className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-1">
                              Your Question:
                            </p>
                            <p className="text-sm bg-blue-50 dark:bg-blue-950 p-3 rounded">
                              {interaction.user_input}
                            </p>
                          </div>
                        </div>

                        {/* Agent Response */}
                        <div className="flex gap-3">
                          <Bot className="h-4 w-4 mt-1 text-green-500" />
                          <div className="flex-1">
                            <p className="text-sm font-medium text-green-700 dark:text-green-300 mb-1">
                              Agent Response:
                            </p>
                            <div className="text-sm bg-green-50 dark:bg-green-950 p-3 rounded max-h-32 overflow-y-auto">
                              {interaction.agent_response.split('\n').map((line, i) => (
                                <p key={i} className="mb-1">{line}</p>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Knowledge Sources Section */}
                        {interaction.knowledge_sources && interaction.knowledge_sources.length > 0 && (
                          <div className="border-t pt-3">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Database className="h-4 w-4 text-purple-500" />
                                <p className="text-sm font-medium text-purple-700 dark:text-purple-300">
                                  Knowledge Sources ({interaction.knowledge_sources.length})
                                </p>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowSources(prev => ({
                                  ...prev,
                                  [interaction.id]: !prev[interaction.id]
                                }))}
                                className="h-6 w-6 p-0"
                              >
                                {showSources[interaction.id] ? (
                                  <ChevronUp className="h-3 w-3" />
                                ) : (
                                  <ChevronDown className="h-3 w-3" />
                                )}
                              </Button>
                            </div>
                            
                            {showSources[interaction.id] && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="space-y-2 max-h-40 overflow-y-auto"
                              >
                                {interaction.knowledge_sources.map((source, index) => (
                                  <div 
                                    key={index}
                                    className="bg-purple-50 dark:bg-purple-950 p-2 rounded text-xs"
                                  >
                                    <div className="flex items-center justify-between mb-1">
                                      <Badge variant="outline" className="text-xs">
                                        {source.type}
                                      </Badge>
                                      {source.similarity && (
                                        <span className="text-purple-600 dark:text-purple-400">
                                          {Math.round(source.similarity * 100)}% match
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-purple-700 dark:text-purple-300 break-words">
                                      {source.content}
                                    </p>
                                    {source.created_at && (
                                      <p className="text-purple-500 dark:text-purple-500 mt-1">
                                        {new Date(source.created_at).toLocaleDateString()}
                                      </p>
                                    )}
                                  </div>
                                ))}
                              </motion.div>
                            )}
                          </div>
                        )}
                      </motion.div>
                    )}

                    {/* Approval Actions */}
                    <div className="flex gap-3 pt-2">
                      <Button
                        onClick={() => handleApproval(interaction.id, true)}
                        disabled={loading}
                        className="flex-1 bg-green-600 hover:bg-green-700 text-white"
                      >
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Approve & Save
                      </Button>
                      <Button
                        onClick={() => handleApproval(interaction.id, false)}
                        disabled={loading}
                        variant="destructive"
                        className="flex-1"
                      >
                        <XCircle className="h-4 w-4 mr-2" />
                        Reject
                      </Button>
                    </div>

                    <p className="text-xs text-center text-muted-foreground">
                      💡 Approving helps the agent learn your preferences for better future responses
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}