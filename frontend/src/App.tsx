import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Menu } from 'lucide-react'
import { Toaster } from 'sonner'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { Sidebar } from '@/components/layout/Sidebar'
import { SettingsPanel } from '@/components/settings/SettingsPanel'
import { KnowledgeManagement } from '@/components/knowledge/KnowledgeManagement'
import ChatOnboarding from '@/components/onboarding/ChatOnboarding'
import { DeepAgentStatus } from '@/components/chat/DeepAgentStatus'
import { FloatingApprovalNotification } from '@/components/chat/FloatingApprovalNotification'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import './globals.css'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant' | 'system'
  timestamp: Date
  agent?: string
  reasoning?: string
  isStreaming?: boolean
  deepAgentState?: DeepAgentState  // Add deep agent state
}

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

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [currentAgent, setCurrentAgent] = useState('orchestrator')
  const [currentView, setCurrentView] = useState('chat')
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') === 'dark' || 
             (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)
    }
    return false
  })
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [currentProvider, setCurrentProvider] = useState<'openai' | 'ollama'>('ollama')  // Default to Ollama
  const [providerStatus, setProviderStatus] = useState({
    openai: { healthy: false, model: 'gpt-4', responseTime: 0 },
    ollama: { healthy: false, model: 'llama3.2:3b', responseTime: 0 }
  })
  
  // Deep Agent System State
  const [deepAgentState, setDeepAgentState] = useState<DeepAgentState | null>(null)
  const [showDeepAgentPanel, setShowDeepAgentPanel] = useState(false)

  // Theme management
  useEffect(() => {
    const root = window.document.documentElement
    if (isDarkMode) {
      root.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [isDarkMode])

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024
      setIsMobile(mobile)
      if (!mobile) {
        setMobileSidebarOpen(false)
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    if (mobileSidebarOpen) {
      document.body.style.overflow = 'hidden'
      return () => {
        document.body.style.overflow = ''
      }
    }

    document.body.style.overflow = ''
  }, [mobileSidebarOpen])

  // Settings panel event listener
  useEffect(() => {
    const handleOpenSettings = () => setSettingsOpen(true)
    window.addEventListener('openSettings', handleOpenSettings)
    return () => window.removeEventListener('openSettings', handleOpenSettings)
  }, [])

  // Load provider status on mount with retry logic
  useEffect(() => {
    const loadProviderStatus = async (retryCount = 0) => {
      try {
        const response = await fetch('/api/llm/status', {  // Fixed endpoint
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        })
        
        if (response.ok) {
          const data = await response.json()
          setCurrentProvider(data.current_provider || 'ollama')  // Default to Ollama
          
          // Update provider status based on response
          setProviderStatus({
            openai: {
              healthy: data.providers?.openai?.healthy || false,
              model: data.providers?.openai?.model || 'gpt-3.5-turbo',
              responseTime: data.providers?.openai?.responseTime || 0
            },
            ollama: {
              healthy: data.providers?.ollama?.healthy || false,
              model: data.providers?.ollama?.model || 'llama3.2:3b',
              responseTime: data.providers?.ollama?.responseTime || 0
            }
          })
          console.log('✅ Provider status loaded successfully:', data)
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
      } catch (error) {
        console.error('Failed to load provider status:', error)
        
        // Retry up to 3 times with increasing delay
        if (retryCount < 3) {
          console.log(`🔄 Retrying provider status load (attempt ${retryCount + 1}/3)...`)
          setTimeout(() => loadProviderStatus(retryCount + 1), (retryCount + 1) * 1000)
          return
        }
        
        // Final fallback in deployed environments: do not probe loopback from browser.
        console.log('📱 Using fallback provider status detection')
        setProviderStatus({
          openai: { healthy: false, model: 'gpt-3.5-turbo', responseTime: 0 },
          ollama: { healthy: false, model: 'llama3.2:3b', responseTime: 0 }
        })
      }
    }
    
    // Add a small delay to ensure backend is ready
    setTimeout(() => loadProviderStatus(), 500)
  }, [])

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  const handleProviderChange = async (provider: 'openai' | 'ollama') => {
    setCurrentProvider(provider)
    
    // Add a system message about the provider switch
    const systemMessage: Message = {
      id: `system-${Date.now()}`,
      content: `🔄 Switched to **${provider.toUpperCase()}** provider. ${
        provider === 'openai' 
          ? 'Now using OpenAI\'s cloud-based models for enhanced capabilities.' 
          : 'Now using local Ollama models for privacy-focused AI interactions.'
      }`,
      role: 'system',
      timestamp: new Date(),
      agent: currentAgent
    }
    
    setMessages(prev => [...prev, systemMessage])
  }

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Call backend API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          agent: currentAgent,
          conversation_id: 'demo-conversation'
        })
      })

      if (response.ok) {
        const data = await response.json()
        
        // Ensure content is always a string
        let responseContent = data.response
        if (typeof responseContent !== 'string') {
          if (responseContent && typeof responseContent === 'object') {
            responseContent = JSON.stringify(responseContent, null, 2)
          } else {
            responseContent = String(responseContent || 'No response received')
          }
        }
        
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          content: responseContent,
          role: 'assistant',
          timestamp: new Date(),
          agent: data.agent || currentAgent,
          reasoning: data.reasoning,
          deepAgentState: data.deep_agent_state  // Include deep agent state from response
        }

        setMessages(prev => [...prev, assistantMessage])
        
        // Update deep agent state if provided
        if (data.deep_agent_state) {
          setDeepAgentState(data.deep_agent_state)
          
          // Show deep agent panel if there are pending human interactions
          const hasHumanInteraction = 
            (data.deep_agent_state.approval_requests?.some((req: any) => req.status === 'pending')) ||
            (data.deep_agent_state.guidance_requests?.some((req: any) => req.status === 'pending')) ||
            (data.deep_agent_state.escalations?.some((esc: any) => esc.status === 'pending'))
          
          if (hasHumanInteraction) {
            setShowDeepAgentPanel(true)
          }
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      
      // Demo response when backend is not available
      const demoResponse: Message = {
        id: `demo-${Date.now()}`,
        content: `Hello! I'm the **${currentAgent}** agent. I'm currently I had to switch to demo mode since the backend is chilling somewhere with connected yet. 
        Here's what I can help you with once fully connected:
        ${currentAgent === 'orchestrator' ? `
        - 🧠 **Coordinate** between different AI agents
        - 🎯 **Route** your requests to the right specialist
        - 📊 **Provide** unified insights across all domains
        - 🔄 **Manage** complex multi-step workflows
        ` : currentAgent === 'productivity' ? `
        - ⚡ **Track** your tasks and goals
        - 📈 **Analyze** your productivity patterns  
        - 🎯 **Suggest** optimizations for your workflow
        - 📋 **Integrate** with your task management tools
        ` : currentAgent === 'health' ? `
        - ❤️ **Monitor** your wellness metrics
        - 🏃 **Track** your fitness activities
        - 😴 **Analyze** your sleep patterns
        - 🥗 **Suggest** healthy lifestyle changes
        ` : currentAgent === 'finance' ? `
        - 💰 **Track** your expenses and income
        - 📊 **Analyze** your spending patterns
        - 🎯 **Help** with budgeting and financial goals
        - 📈 **Provide** investment insights
        ` : `
        - 🤖 **Assist** with general queries
        - 💡 **Provide** helpful suggestions
        - 🔍 **Search** for information
        - 📝 **Help** with various tasks
        `}

      Try switching between different agents using the sidebar to see how each one specializes in different areas, or just trust me, and let me do my Orchestration Job!`,
        role: 'assistant',
        timestamp: new Date(),
        agent: currentAgent,
        reasoning: `Selected ${currentAgent} agent based on current context. Providing demo capabilities overview.`
      }

      setMessages(prev => [...prev, demoResponse])
    } finally {
      setIsLoading(false)
    }
  }

  // Human-in-the-Loop Response Handlers
  const handleApprovalResponse = async (requestId: string, decision: 'approve' | 'deny' | 'modify', feedback?: string) => {
    try {
      const response = await fetch('/api/human-approval', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          request_id: requestId,
          decision,
          feedback: feedback || ''
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        
        // Update deep agent state with the response
        if (data.deep_agent_state) {
          setDeepAgentState(data.deep_agent_state)
        }
        
        // Add a system message about the approval decision
        const systemMessage: Message = {
          id: `system-approval-${Date.now()}`,
          content: `✅ Approval ${decision}${decision === 'approve' ? 'ed' : decision === 'deny' ? 'ied' : 'ied with modifications'} for request ${requestId}${feedback ? `\n\n**Feedback:** ${feedback}` : ''}`,
          role: 'system',
          timestamp: new Date()
        }
        
        setMessages(prev => [...prev, systemMessage])
      }
    } catch (error) {
      console.error('Error handling approval response:', error)
    }
  }

  const handleGuidanceResponse = async (requestId: string, guidance: string) => {
    try {
      const response = await fetch('/api/human-guidance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          request_id: requestId,
          guidance
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        
        // Update deep agent state
        if (data.deep_agent_state) {
          setDeepAgentState(data.deep_agent_state)
        }
        
        // Add system message
        const systemMessage: Message = {
          id: `system-guidance-${Date.now()}`,
          content: `💭 Guidance provided for request ${requestId}\n\n**Guidance:** ${guidance}`,
          role: 'system',
          timestamp: new Date()
        }
        
        setMessages(prev => [...prev, systemMessage])
      }
    } catch (error) {
      console.error('Error handling guidance response:', error)
    }
  }

  const handleEscalationResponse = async (escalationId: string, resolution: string) => {
    try {
      const response = await fetch('/api/human-escalation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          escalation_id: escalationId,
          resolution
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        
        // Update deep agent state
        if (data.deep_agent_state) {
          setDeepAgentState(data.deep_agent_state)
        }
        
        // Add system message
        const systemMessage: Message = {
          id: `system-escalation-${Date.now()}`,
          content: `🚨 Escalation ${escalationId} resolved\n\n**Resolution:** ${resolution}`,
          role: 'system',
          timestamp: new Date()
        }
        
        setMessages(prev => [...prev, systemMessage])
      }
    } catch (error) {
      console.error('Error handling escalation response:', error)
    }
  }

  const handleAgentChange = (agentId: string) => {
    setCurrentAgent(agentId)
    
    // Add system message about agent switch
    const systemMessage: Message = {
      id: `system-${Date.now()}`,
      content: `Switched to **${agentId}** agent. This agent specializes in ${
        agentId === 'orchestrator' ? 'coordinating between different specialized agents to give the best answer to your query.' :
        agentId === 'productivity' ? 'task management, goal tracking, and productivity optimization' :
        agentId === 'health' ? 'wellness tracking, fitness monitoring, and healthy lifestyle guidance' :
        agentId === 'finance' ? 'expense tracking, budgeting, and financial planning' :
        agentId === 'scheduling' ? 'calendar management and appointment scheduling' :
        agentId === 'journal' ? 'reflection facilitation and personal insights' :
        'general assistance and information'
      }.`,
      role: 'system',
      timestamp: new Date(),
      agent: agentId
    }
    
    setMessages(prev => [...prev, systemMessage])
  }

  return (
    <div className={cn(
      "flex min-h-screen bg-background text-foreground transition-colors duration-300",
      isDarkMode && "dark"
    )}>
      {isMobile && mobileSidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50"
          onClick={() => setMobileSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        currentAgent={currentAgent}
        onAgentChange={handleAgentChange}
        currentView={currentView}
        onViewChange={setCurrentView}
        isDarkMode={isDarkMode}
        onToggleTheme={toggleTheme}
        isMobile={isMobile}
        mobileOpen={mobileSidebarOpen}
        onMobileClose={() => setMobileSidebarOpen(false)}
      />

      {/* Main Content */}
      <motion.div
        initial={false}
        animate={{ 
          marginLeft: 0,
          width: '100%'
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="flex-1 flex flex-col min-w-0"
      >
        {isMobile && (
          <div className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-card/95 px-4 backdrop-blur-lg lg:hidden">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Open navigation menu"
            >
              <Menu className="h-5 w-5" />
            </Button>
            <div className="text-sm font-semibold">AI Ecosystem</div>
            <div className="w-9" aria-hidden="true" />
          </div>
        )}

        {/* Main Content */}
        {currentView === 'chat' && (
          <div className="flex-1 flex relative">
            {/* Chat Interface */}
            <div className={cn(
              "flex-1 transition-all duration-300",
              showDeepAgentPanel && !isMobile ? "mr-96" : ""
            )}>
              <ChatInterface
                messages={messages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                currentAgent={currentAgent}
                currentProvider={currentProvider}
                className="h-full"
              />
            </div>
            
            {/* Deep Agent Toggle Button */}
            {deepAgentState && (
              <Button
                onClick={() => setShowDeepAgentPanel(!showDeepAgentPanel)}
                className={cn(
                  "fixed bottom-4 right-4 h-11 w-11 p-0 rounded-full shadow-lg z-30 transition-all sm:bottom-20 sm:right-6 sm:h-12 sm:w-12",
                  "bg-blue-600 hover:bg-blue-700 text-white",
                  showDeepAgentPanel && !isMobile ? "right-[25rem]" : "right-4 sm:right-6"
                )}
                title="Toggle Deep Agent Panel"
              >
                <div className="relative">
                  <Brain className="w-6 h-6" />
                  {/* Notification indicator for pending human interactions */}
                  {((deepAgentState.approval_requests?.some(req => req.status === 'pending')) ||
                    (deepAgentState.guidance_requests?.some(req => req.status === 'pending')) ||
                    (deepAgentState.escalations?.some(esc => esc.status === 'pending'))) && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-white" />
                  )}
                </div>
              </Button>
            )}
            
            {/* Deep Agent Status Panel */}
            {showDeepAgentPanel && (
              <motion.div
                initial={isMobile ? { y: 80, opacity: 0 } : { x: 384, opacity: 0 }}
                animate={isMobile ? { y: 0, opacity: 1 } : { x: 0, opacity: 1 }}
                exit={isMobile ? { y: 80, opacity: 0 } : { x: 384, opacity: 0 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className={cn(
                  "bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
                  isMobile
                    ? "fixed inset-x-0 bottom-0 top-14 z-40 w-full border-t"
                    : "w-96 border-l"
                )}
              >
                <div className="p-4 border-b flex items-center justify-between">
                  <h3 className="font-semibold text-lg">Deep Agent System</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowDeepAgentPanel(false)}
                    className="h-8 w-8 p-0"
                  >
                    ✕
                  </Button>
                </div>
                <div className="h-[calc(100dvh-8rem)] overflow-y-auto md:h-[calc(100vh-8rem)]">
                  <DeepAgentStatus
                    agentState={deepAgentState}
                    onApprovalResponse={handleApprovalResponse}
                    onGuidanceResponse={handleGuidanceResponse}
                    onEscalationResponse={handleEscalationResponse}
                    className="border-0"
                  />
                </div>
              </motion.div>
            )}
          </div>
        )}
        
        {currentView === 'knowledge' && (
          <KnowledgeManagement className="flex-1" />
        )}
        
        {currentView === 'onboarding' && (
          <ChatOnboarding 
            onComplete={(data) => {
              console.log('Onboarding completed:', data);
              setCurrentView('chat');
            }}
          />
        )}
        
        {currentView === 'analytics' && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Analytics</h2>
              <p className="text-muted-foreground">Analytics dashboard is being prepared. Please check back shortly.</p>
            </div>
          </div>
        )}
        
        {currentView === 'activity' && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Activity</h2>
              <p className="text-muted-foreground">Your recent activity feed will appear here soon.</p>
            </div>
          </div>
        )}
        
        {currentView === 'profile' && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Profile</h2>
              <p className="text-muted-foreground">Profile management tools are on the way.</p>
            </div>
          </div>
        )}
      </motion.div>

      {/* Settings Panel */}
      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        currentProvider={currentProvider}
        onProviderChange={handleProviderChange}
        providerStatus={providerStatus}
      />

      {/* Toast Notifications */}
      <Toaster 
        position="top-right"
        theme={isDarkMode ? 'dark' : 'light'}
        richColors
        closeButton
      />

      {/* Floating Approval Notification */}
      <FloatingApprovalNotification 
        onOpenApprovalPanel={() => setShowDeepAgentPanel(true)}
      />
    </div>
  )
}

export default App