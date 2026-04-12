import { useState, useEffect, useMemo, useRef } from 'react'
import { useBodyScrollLock, useThrottledResizeHandler } from '@/hooks/useBodyScrollLock'
import { motion } from 'framer-motion'
import { ArrowLeft, BarChart3, Brain, Database, Menu, MessageSquare, Sparkles, User, X } from 'lucide-react'
import { Toaster } from 'sonner'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { Sidebar } from '@/components/layout/Sidebar'
import { SettingsPanel } from '@/components/settings/SettingsPanel'
import { KnowledgeManagement } from '@/components/knowledge/KnowledgeManagement'
import ChatOnboarding from '@/components/onboarding/ChatOnboarding'
import ProfileWorkspace from '@/components/profile/ProfileWorkspace'
import { DeepAgentStatus } from '@/components/chat/DeepAgentStatus'
import { FloatingApprovalNotification } from '@/components/chat/FloatingApprovalNotification'
import { OfflineIndicator, BackendStatusIndicator } from '@/components/ui/offline-indicator'
import { Button } from '@/components/ui/button'
import { prefetchEmbeddingsVisualization } from '@/lib/embeddingsCache'
import { hapticButtonPress } from '@/lib/hapticFeedback'
import { cn } from '@/lib/utils'
import { AnalyticsDashboard } from '@/components/knowledge/AnalyticsDashboard'
import { AINotificationsCenter } from '@/components/knowledge/AINotificationsCenter'
import { getOrCreateConversationId } from '@/lib/agenticBridgeSession'
import './globals.css'

const BUILTIN_ALTEREGO_RETURN_ORIGINS = [
  'https://alterego-conventional.vercel.app',
  'https://alterego.vercel.app',
  'http://localhost:5173',
  'http://localhost:3000',
]
const SUPPORTED_VIEWS = ['chat', 'onboarding', 'knowledge', 'analytics', 'notifications', 'activity', 'profile'] as const

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
  const appShellRef = useRef<HTMLDivElement | null>(null)
  const viewScrollPositionsRef = useRef<Record<string, number>>({})
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
  const [conversationId] = useState<string>(() => getOrCreateConversationId())
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
  const isEmbedMode = useMemo(() => {
    if (typeof window === 'undefined') {
      return false
    }
    return new URLSearchParams(window.location.search).get('embed') === '1'
  }, [])

  const integrationContext = useMemo(() => {
    if (typeof window === 'undefined') {
      return { isAlterEgo: false, returnUrl: null as string | null }
    }

    const params = new URLSearchParams(window.location.search)
    const source = params.get('from')?.toLowerCase()
    const rawReturnUrl = params.get('return_url')

    const referrerOrigin = (() => {
      try {
        return document.referrer ? new URL(document.referrer).origin : null
      } catch {
        return null
      }
    })()

    const sourceIsAlterEgo = source === 'alterego'

    const envOrigins = ((import.meta.env.VITE_ALLOWED_RETURN_ORIGINS as string | undefined) || '')
      .split(',')
      .map((origin) => origin.trim())
      .filter((origin) => origin.length > 0)

    const trackerAppOrigin = (import.meta.env.VITE_TIMETRACKER_APP_ORIGIN as string | undefined)?.trim()
    const trustedAlterEgoHost = (hostname: string) => {
      const normalizedHost = hostname.toLowerCase()
      return (
        normalizedHost.includes('alterego') ||
        normalizedHost.includes('localhost') ||
        normalizedHost.includes('127.0.0.1')
      )
    }

    const allowedOrigins = new Set<string>([
      window.location.origin,
      ...envOrigins,
      ...BUILTIN_ALTEREGO_RETURN_ORIGINS,
    ])

    if (trackerAppOrigin) {
      allowedOrigins.add(trackerAppOrigin)
    }

    if (referrerOrigin) {
      allowedOrigins.add(referrerOrigin)
    }

    let returnUrl: string | null = null
    if (rawReturnUrl) {
      try {
        const parsed = new URL(rawReturnUrl, window.location.origin)
        const allowByOrigin = allowedOrigins.has(parsed.origin)
        const allowByAlterEgoHost = sourceIsAlterEgo && trustedAlterEgoHost(parsed.hostname)

        if (allowByOrigin || allowByAlterEgoHost) {
          returnUrl = parsed.toString()
        }
      } catch {
        returnUrl = null
      }
    }

    const isAlterEgo = sourceIsAlterEgo || !!returnUrl || (referrerOrigin?.includes('alterego') ?? false)

    const fallbackReturnUrl = (() => {
      if (returnUrl) {
        return returnUrl
      }

      if (!sourceIsAlterEgo || !trackerAppOrigin) {
        return null
      }

      try {
        const parsedFallback = new URL('/coach', trackerAppOrigin)
        return parsedFallback.toString()
      } catch {
        return null
      }
    })()

    return {
      isAlterEgo,
      returnUrl: fallbackReturnUrl,
    }
  }, [])

  const mobileViews = useMemo(
    () => [
      { id: 'chat', label: 'Talk', icon: MessageSquare },
      { id: 'onboarding', label: 'Start', icon: Sparkles },
      { id: 'knowledge', label: 'Memory', icon: Database },
      { id: 'analytics', label: 'Insights', icon: BarChart3 },
      { id: 'profile', label: 'Me', icon: User },
    ],
    []
  )

  const activeMobileView = useMemo(
    () => mobileViews.find((view) => view.id === currentView),
    [mobileViews, currentView]
  )

  // Standardized mobile safe area classes
  const mobileSafeAreaClasses = {
    bottomNav: 'pb-[calc(5rem+env(safe-area-inset-bottom))]',
    header: 'pt-[env(safe-area-inset-top)]',
    content: 'pb-[env(safe-area-inset-bottom)]',
  }
  
  const mobileContentInsetClass = isMobile && !isEmbedMode
    ? mobileSafeAreaClasses.bottomNav
    : ''

  const handleViewChange = (nextView: string) => {
    const normalizedView = nextView === 'activity' ? 'notifications' : nextView

    if (normalizedView === currentView) {
      return
    }

    const activeRegion = document.querySelector<HTMLElement>('[data-app-scroll-region="true"]')
    if (activeRegion) {
      viewScrollPositionsRef.current[currentView] = activeRegion.scrollTop
    }

    setCurrentView(normalizedView)
  }

  useEffect(() => {
    const requestedView = new URLSearchParams(window.location.search).get('view')
    if (!requestedView) {
      return
    }

    if ((SUPPORTED_VIEWS as readonly string[]).includes(requestedView)) {
      setCurrentView(requestedView === 'activity' ? 'notifications' : requestedView)
    }
  }, [])

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

  // Use throttled resize handler for performance
  useThrottledResizeHandler((width: number) => {
    const mobile = width < 1024 // lg breakpoint
    setIsMobile(mobile)
    if (!mobile) {
      setMobileSidebarOpen(false)
    }
  }, 100)

  // Use safe scroll lock hook instead of direct DOM manipulation
  useBodyScrollLock(mobileSidebarOpen)

  useEffect(() => {
    const restoreScrollPosition = () => {
      const activeRegion = document.querySelector<HTMLElement>('[data-app-scroll-region="true"]')
      if (!activeRegion) {
        return
      }

      const savedScrollTop = viewScrollPositionsRef.current[currentView]
      activeRegion.scrollTop = Number.isFinite(savedScrollTop) ? savedScrollTop : 0
    }

    const raf = window.requestAnimationFrame(restoreScrollPosition)
    return () => window.cancelAnimationFrame(raf)
  }, [currentView])

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

  useEffect(() => {
    if (!integrationContext.isAlterEgo) {
      return
    }

    const prefetchCoachData = async () => {
      await Promise.allSettled([
        fetch('/api/knowledge/onboarding/profile').catch(() => null),
        fetch('/api/knowledge/stats').catch(() => null),
        prefetchEmbeddingsVisualization(),
      ])
    }

    void prefetchCoachData()
  }, [integrationContext.isAlterEgo])

  const handleReturnToAlterEgo = () => {
    if (integrationContext.returnUrl) {
      window.location.assign(integrationContext.returnUrl)
      return
    }

    const trackerAppOrigin = (import.meta.env.VITE_TIMETRACKER_APP_ORIGIN as string | undefined)?.trim()
    if (trackerAppOrigin) {
      try {
        const trackerCoachUrl = new URL('/coach', trackerAppOrigin)
        window.location.assign(trackerCoachUrl.toString())
        return
      } catch {
        // Fall through to referrer/history fallback.
      }
    }

    if (typeof document !== 'undefined' && document.referrer) {
      window.location.assign(document.referrer)
      return
    }

    window.history.back()
  }

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
  }

  const handleProviderChange = async (provider: 'openai' | 'ollama') => {
    setCurrentProvider(provider)
    
    // Add a system message about the provider switch
    const systemMessage: Message = {
      id: `system-${Date.now()}`,
      content: `🔄 Now using **${provider === 'openai' ? 'OpenAI' : 'Local AI'}**. ${
        provider === 'openai' 
          ? 'Connected to OpenAI for enhanced capabilities.' 
          : 'Running locally for privacy-focused conversations.'
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
          conversation_id: conversationId,
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
      const errorText = error instanceof Error ? error.message : 'Unknown error'
      const providerUnavailable = /llm_provider_unavailable|no healthy providers available|llm service not initialized/i.test(errorText)
      
      // Demo response when backend is not available
      const demoResponse: Message = {
        id: `demo-${Date.now()}`,
        content: `${providerUnavailable
          ? "I can't connect to the AI right now. Check your settings and try again."
          : "I'm having trouble connecting. Running in offline mode."
        }\n\nIn the meantime, here's what I can help you with:\n\n${currentAgent === 'orchestrator' ? `
        - 🧠 **Coordinate** across all your life domains
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
        `}\n\nTry switching between different agents using the sidebar to explore each specialization.`,
        role: 'assistant',
        timestamp: new Date(),
        agent: currentAgent,
        reasoning: `Fallback response after backend error: ${errorText}`
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
    <div
      ref={appShellRef}
      className={cn(
      "relative flex min-h-[100svh] w-full overflow-x-hidden bg-background text-foreground transition-colors duration-300",
      isDarkMode && "dark"
      )}
    >
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_12%,rgba(45,212,191,0.18),transparent_32%),radial-gradient(circle_at_82%_10%,rgba(245,158,11,0.18),transparent_34%)]" />
        <div className="absolute -top-20 left-[-88px] h-64 w-64 rounded-full bg-cyan-300/35 blur-3xl dark:bg-cyan-500/20" />
        <div className="absolute bottom-[-120px] right-[-64px] h-72 w-72 rounded-full bg-amber-300/30 blur-3xl dark:bg-amber-500/20" />
      </div>

      {isMobile && mobileSidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50"
          onClick={() => setMobileSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      {!isEmbedMode && (
        <div className={cn("relative z-20", !isMobile && "sticky top-0 h-[100dvh] self-start")}>
          <Sidebar
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
            currentAgent={currentAgent}
            onAgentChange={handleAgentChange}
            currentView={currentView}
            onViewChange={handleViewChange}
            isDarkMode={isDarkMode}
            onToggleTheme={toggleTheme}
            isMobile={isMobile}
            mobileOpen={mobileSidebarOpen}
            onMobileClose={() => setMobileSidebarOpen(false)}
          />
        </div>
      )}

      {/* Main Content */}
      <motion.div
        initial={false}
        animate={{ 
          marginLeft: 0
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className={cn(
          "relative z-10 flex min-h-0 min-w-0 max-w-full flex-1 flex-col overflow-x-hidden",
          isMobile && !isEmbedMode && mobileSafeAreaClasses.bottomNav
        )}
      >
        {isMobile && !isEmbedMode && (
          <div className={cn(
            "sticky top-0 z-30 flex items-center justify-between border-b border-border/70 bg-white/80 px-4 backdrop-blur-xl dark:bg-slate-950/75 lg:hidden",
            "h-[calc(3.5rem+env(safe-area-inset-top))]",
            mobileSafeAreaClasses.header
          )}>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileSidebarOpen((previous) => !previous)}
              aria-label="Open navigation menu"
              className="h-10 w-10 rounded-xl border border-border/70 bg-white/80 shadow-sm dark:bg-slate-900/70"
            >
              {mobileSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <div className="text-center">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">
                AlterEgo
              </div>
              <div className="text-sm font-semibold">
                Your AI Partner
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 capitalize">
                {activeMobileView?.label || currentView.replace('_', ' ')}
              </div>
            </div>
            <div className="w-9" aria-hidden="true" />
          </div>
        )}

        {integrationContext.isAlterEgo && (
          <div className="border-b border-border/70 bg-gradient-to-r from-teal-50/90 via-cyan-50/80 to-amber-50/85 px-3 py-2 dark:from-slate-900/90 dark:via-teal-950/40 dark:to-amber-950/35 sm:px-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-teal-700 dark:text-teal-200">
                  Connected
                </p>
                <p className="text-sm text-slate-800 dark:text-slate-100">AlterEgo Link Active</p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="gap-2 border-teal-300 bg-white/80 text-teal-800 hover:bg-teal-100 dark:border-teal-800 dark:bg-slate-900/60 dark:text-teal-200"
                onClick={handleReturnToAlterEgo}
              >
                <ArrowLeft className="h-4 w-4" />
                Back to AlterEgo
              </Button>
            </div>
          </div>
        )}

        {/* Main Content */}
        {currentView === 'chat' && (
          <div
            data-app-scroll-region="true"
            className={cn(
              "relative mx-auto flex min-h-0 w-full max-w-[1600px] flex-1 overflow-x-hidden px-2 pt-2 sm:px-4",
              isMobile && !isEmbedMode ? mobileContentInsetClass : "pb-2 sm:pb-4"
            )}
          >
            {/* Chat Interface */}
            <div className={cn(
              "min-h-0 min-w-0 flex-1 overflow-hidden transition-all duration-300"
            )}>
              <ChatInterface
                messages={messages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                currentAgent={currentAgent}
                currentProvider={currentProvider}
                className="h-full w-full rounded-[28px] border border-border/70 bg-card/80 shadow-[0_24px_70px_-50px_rgba(15,23,42,0.7)] backdrop-blur-xl"
              />
            </div>
            
            {/* Deep Agent Toggle Button */}
            {deepAgentState && (
              <Button
                onClick={() => setShowDeepAgentPanel(!showDeepAgentPanel)}
                className={cn(
                  "fixed right-4 h-11 w-11 p-0 rounded-full shadow-lg z-30 transition-all sm:right-6 sm:h-12 sm:w-12",
                  isMobile && !isEmbedMode
                    ? "bottom-[calc(4rem+env(safe-area-inset-bottom))]"
                    : "bottom-4 sm:bottom-20",
                  "bg-gradient-to-r from-teal-700 to-cyan-600 text-white hover:from-teal-800 hover:to-cyan-700",
                  showDeepAgentPanel && !isMobile ? "right-[25rem]" : "right-4 sm:right-6"
                )}
                title="Show/Hide Action Panel"
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
                  "flex h-full flex-col bg-background/95 backdrop-blur-xl supports-[backdrop-filter]:bg-background/70",
                  isMobile
                    ? isEmbedMode
                      ? "fixed inset-0 z-40 w-full border-t"
                      : "fixed inset-x-0 z-40 w-full border-t " + mobileSafeAreaClasses.bottomNav + " top-[calc(3.5rem+env(safe-area-inset-top))]"
                    : "w-96 rounded-l-3xl border-l border-border/70"
                )}
              >
                <div className="p-4 border-b flex items-center justify-between">
                  <h3 className="font-semibold text-lg">Advanced Actions</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowDeepAgentPanel(false)}
                    className="h-8 w-8 p-0"
                  >
                    ✕
                  </Button>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto">
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
          <div data-app-scroll-region="true" className={cn("flex min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-3 sm:px-4 sm:pb-5 sm:pt-5", mobileContentInsetClass)}>
            <KnowledgeManagement className="mx-auto h-full w-full max-w-[1600px]" />
          </div>
        )}
        
        {currentView === 'onboarding' && (
          <div data-app-scroll-region="true" className={cn("flex min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-3 sm:px-4 sm:pb-5 sm:pt-5", mobileContentInsetClass)}>
            <div className="panel-surface mx-auto h-full w-full max-w-[1600px] overflow-hidden">
              <ChatOnboarding 
                onComplete={() => {
                  console.log('Onboarding completed');
                  handleViewChange('chat');
                }}
              />
            </div>
          </div>
        )}
        
        {currentView === 'analytics' && (
          <div data-app-scroll-region="true" className={cn("flex min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-3 sm:px-4 sm:pb-5 sm:pt-5", mobileContentInsetClass)}>
            <div className="panel-surface mx-auto h-full w-full max-w-[1600px] overflow-y-auto p-4 sm:p-5">
              <AnalyticsDashboard />
            </div>
          </div>
        )}
        
        {(currentView === 'notifications' || currentView === 'activity') && (
          <div data-app-scroll-region="true" className={cn("flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-3 pb-4 pt-3 sm:px-4 sm:pb-5 sm:pt-5", mobileContentInsetClass)}>
            <div className="mx-auto w-full max-w-[1600px]">
              <AINotificationsCenter />
            </div>
          </div>
        )}
        
        {currentView === 'profile' && (
          <div data-app-scroll-region="true" className={cn("flex min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-3 sm:px-4 sm:pb-5 sm:pt-5", mobileContentInsetClass)}>
            <div className="panel-surface mx-auto h-full w-full max-w-[1600px] overflow-auto">
              <ProfileWorkspace
                onStartOnboarding={() => handleViewChange('onboarding')}
                onContinueToChat={() => handleViewChange('chat')}
              />
            </div>
          </div>
        )}
      </motion.div>

      {isMobile && !isEmbedMode && (
        <nav className="fixed bottom-0 left-0 right-0 z-30 grid grid-cols-5 border-t border-border/70 bg-white/90 px-1 pb-[env(safe-area-inset-bottom)] pt-2 backdrop-blur-xl dark:bg-slate-950/85 lg:hidden">
          {mobileViews.map((view) => {
            const isActive = currentView === view.id
            return (
              <button
                key={view.id}
                type="button"
                onClick={() => {
                  hapticButtonPress() // Haptic feedback on mobile
                  handleViewChange(view.id)
                }}
                className={cn(
                  "flex flex-col items-center justify-center rounded-xl min-h-[44px] text-[11px] font-semibold transition",
                  isActive
                    ? "bg-gradient-to-r from-teal-700 via-cyan-600 to-amber-500 text-white shadow-md"
                    : "text-muted-foreground hover:bg-accent"
                )}
                aria-label={view.label}
              >
                <view.icon className="mb-1 h-5 w-5" />
                {view.label}
              </button>
            )
          })}
        </nav>
      )}

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

      {/* Offline/Connectivity Indicators */}
      <OfflineIndicator />
      <BackendStatusIndicator />

      {/* Floating Approval Notification */}
      <FloatingApprovalNotification 
        onOpenApprovalPanel={() => setShowDeepAgentPanel(true)}
      />
    </div>
  )
}

export default App