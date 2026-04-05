import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  Heart, 
  DollarSign, 
  Calendar, 
  BookOpen, 
  Settings, 
  Moon, 
  Sun,
  ChevronLeft,
  ChevronRight,
  Activity,
  BarChart3,
  MessageSquare,
  User,
  Database,
  Sparkles,
  X
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface SidebarProps {
  className?: string
  collapsed?: boolean
  onToggleCollapse?: () => void
  currentAgent?: string
  onAgentChange?: (agent: string) => void
  isDarkMode?: boolean
  onToggleTheme?: () => void
  currentView?: string
  onViewChange?: (view: string) => void
  isMobile?: boolean
  mobileOpen?: boolean
  onMobileClose?: () => void
}

const agents = [
  {
    id: 'orchestrator',
    name: 'Orchestrator',
    icon: Brain,
    description: 'Main coordination agent',
    color: 'from-teal-600 to-cyan-500',
    status: 'active'
  },
  {
    id: 'productivity',
    name: 'Productivity',
    icon: Zap,
    description: 'Task and goal management',
    color: 'from-amber-500 to-orange-500',
    status: 'active'
  },
  {
    id: 'health',
    name: 'Health',
    icon: Heart,
    description: 'Wellness and habits',
    color: 'from-rose-500 to-red-500',
    status: 'active'
  },
  {
    id: 'finance',
    name: 'Finance',
    icon: DollarSign,
    description: 'Budget and expenses',
    color: 'from-emerald-500 to-green-500',
    status: 'active'
  },
  {
    id: 'scheduling',
    name: 'Scheduling',
    icon: Calendar,
    description: 'Calendar management',
    color: 'from-sky-500 to-blue-500',
    status: 'active'
  },
  {
    id: 'journal',
    name: 'Journal',
    icon: BookOpen,
    description: 'Reflection and insights',
    color: 'from-slate-600 to-slate-500',
    status: 'active'
  }
]

const navigationItems = [
  {
    id: 'chat',
    name: 'Chat',
    icon: MessageSquare,
    path: '/chat'
  },
  {
    id: 'onboarding',
    name: 'Onboarding',
    icon: Sparkles,
    path: '/onboarding'
  },
  {
    id: 'knowledge',
    name: 'Knowledge Base',
    icon: Database,
    path: '/knowledge'
  },
  {
    id: 'analytics',
    name: 'Analytics',
    icon: BarChart3,
    path: '/analytics'
  },
  {
    id: 'activity',
    name: 'Activity',
    icon: Activity,
    path: '/activity'
  },
  {
    id: 'profile',
    name: 'Profile',
    icon: User,
    path: '/profile'
  }
]

export const Sidebar: React.FC<SidebarProps> = ({
  className,
  collapsed = false,
  onToggleCollapse,
  currentAgent = 'orchestrator',
  onAgentChange,
  isDarkMode = false,
  onToggleTheme,
  currentView = 'chat',
  onViewChange,
  isMobile = false,
  mobileOpen = false,
  onMobileClose
}) => {
  const [hoveredAgent, setHoveredAgent] = useState<string | null>(null)
  const isCollapsed = isMobile ? false : collapsed
  const mobileDrawerWidth = 304

  return (
    <motion.div
      initial={false}
      animate={isMobile ? { x: mobileOpen ? 0 : -mobileDrawerWidth, width: mobileDrawerWidth } : { width: isCollapsed ? 80 : 280 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className={cn(
        "relative flex h-[100dvh] flex-col border-r border-border/70 bg-white/75 backdrop-blur-xl dark:bg-slate-950/70",
        isMobile && "pb-[env(safe-area-inset-bottom)] pt-[env(safe-area-inset-top)]",
        isMobile && "fixed inset-y-0 left-0 z-50 shadow-2xl",
        className
      )}
    >
      {/* Header */}
      <div className="border-b border-border/70 p-4">
        <div className="flex items-center justify-between">
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-3"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-700 via-cyan-600 to-amber-500 shadow-lg shadow-cyan-500/30">
                  <Brain className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h1 className="text-sm font-semibold">Agentic Workspace</h1>
                  <p className="text-[11px] text-muted-foreground">Mission Control</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={() => {
              if (isMobile) {
                onMobileClose?.()
                return
              }
              onToggleCollapse?.()
            }}
            className="h-9 w-9 rounded-xl border border-border/70 bg-white/80 shadow-sm dark:bg-slate-900/70"
          >
            {isMobile ? (
              <X className="w-4 h-4" />
            ) : isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <div className="border-b border-border/70 p-4">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.h2
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground"
            >
              Navigation
            </motion.h2>
          )}
        </AnimatePresence>
        
        <div className="space-y-1">
          {navigationItems.map((item) => (
            <Button
              key={item.id}
              variant="ghost"
              onClick={() => {
                onViewChange?.(item.id)
                if (isMobile) {
                  onMobileClose?.()
                }
              }}
              className={cn(
                "h-10 w-full justify-start rounded-xl text-sm",
                isCollapsed && "justify-center px-0",
                currentView === item.id
                  ? "bg-gradient-to-r from-teal-700 via-cyan-600 to-amber-500 text-white shadow-md"
                  : "hover:bg-accent/70"
              )}
            >
              <item.icon className="w-4 h-4" />
              <AnimatePresence mode="wait">
                {!isCollapsed && (
                  <motion.span
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ duration: 0.2 }}
                    className="ml-3 text-sm"
                  >
                    {item.name}
                  </motion.span>
                )}
              </AnimatePresence>
            </Button>
          ))}
        </div>
      </div>

      {/* Agents */}
      <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.h2
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground"
            >
              AI Agents
            </motion.h2>
          )}
        </AnimatePresence>

        <div className="space-y-2">
          {agents.map((agent) => {
            const Icon = agent.icon
            const isActive = currentAgent === agent.id
            const isHovered = hoveredAgent === agent.id

            return (
              <motion.div
                key={agent.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Card
                  className={cn(
                    "cursor-pointer border border-border/70 bg-white/70 p-3 shadow-sm transition-all duration-200 dark:bg-slate-900/60",
                    isActive && "border-teal-500/70 bg-teal-50/70 shadow-md dark:bg-teal-950/30",
                    !isActive && "hover:-translate-y-0.5 hover:bg-accent/40",
                    isCollapsed && "p-2"
                  )}
                  onClick={() => {
                    onAgentChange?.(agent.id)
                    if (isMobile) {
                      onMobileClose?.()
                    }
                  }}
                  onMouseEnter={() => setHoveredAgent(agent.id)}
                  onMouseLeave={() => setHoveredAgent(null)}
                >
                  <div className={cn(
                    "flex items-center gap-3",
                    isCollapsed && "justify-center"
                  )}>
                    <div className={cn(
                      "w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br",
                      agent.color,
                      isActive && "shadow-lg"
                    )}>
                      <Icon className="w-4 h-4 text-white" />
                    </div>
                    
                    <AnimatePresence mode="wait">
                      {!isCollapsed && (
                        <motion.div
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                          className="flex-1 min-w-0"
                        >
                          <div className="flex items-center justify-between">
                            <h3 className="font-medium text-sm truncate">
                              {agent.name}
                            </h3>
                            <div className={cn(
                              "w-2 h-2 rounded-full",
                              agent.status === 'active' ? "bg-emerald-500" : "bg-gray-400"
                            )} />
                          </div>
                          <p className="truncate text-xs text-muted-foreground">
                            {agent.description}
                          </p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </Card>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border/70 p-4">
        <div className={cn(
          "flex items-center gap-2",
          isCollapsed && "justify-center"
        )}>
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleTheme}
            className="h-9 w-9 rounded-xl border border-border/70 bg-white/80 dark:bg-slate-900/70"
          >
            {isDarkMode ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </Button>
          
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 rounded-xl border border-border/70 bg-white/80 dark:bg-slate-900/70"
                  onClick={() => {
                    // This will be handled by the parent component
                    const event = new CustomEvent('openSettings')
                    window.dispatchEvent(event)
                  }}
                >
                  <Settings className="w-4 h-4" />
                </Button>
                
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="flex-1"
                >
                  <p className="text-xs font-medium text-muted-foreground">
                    v1.0.0 • Live
                  </p>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  )
}