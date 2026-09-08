import React, { useState, useMemo } from 'react'
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
 BellRing,
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

// Static data defined outside component to prevent recreation
const AGENTS_DATA = [
 {
 id: 'orchestrator',
 name: 'Coach',
 icon: Brain,
 description: 'Routes requests to the right specialist for you',
 color: 'from-teal-600 to-cyan-500',
 status: 'active'
 },
 {
 id: 'productivity',
 name: 'Goals',
 icon: Zap,
 description: 'Tracks tasks and helps you hit your targets',
 color: 'from-amber-500 to-orange-500',
 status: 'active'
 },
 {
 id: 'health',
 name: 'Wellness',
 icon: Heart,
 description: 'Supports your physical and mental wellbeing',
 color: 'from-rose-500 to-red-500',
 status: 'active'
 },
 {
 id: 'finance',
 name: 'Money',
 icon: DollarSign,
 description: 'Manages budgets and spending insights',
 color: 'from-emerald-500 to-green-500',
 status: 'active'
 },
 {
 id: 'scheduling',
 name: 'Calendar',
 icon: Calendar,
 description: 'Handles your schedule and appointments',
 color: 'from-sky-500 to-blue-500',
 status: 'active'
 },
 {
 id: 'journal',
 name: 'Reflection',
 icon: BookOpen,
 description: 'Captures your thoughts and growth moments',
 color: 'from-slate-600 to-slate-500',
 status: 'active'
 }
] as const

const NAVIGATION_DATA = [
 {
 id: 'chat',
 name: 'Talk',
 icon: MessageSquare,
 path: '/chat'
 },
 {
 id: 'onboarding',
 name: 'Getting Started',
 icon: Sparkles,
 path: '/onboarding'
 },
 {
 id: 'knowledge',
 name: 'My Memory',
 icon: Database,
 path: '/knowledge'
 },
 {
 id: 'analytics',
 name: 'Insights',
 icon: BarChart3,
 path: '/analytics'
 },
 {
 id: 'notifications',
 name: 'Alerts',
 icon: BellRing,
 path: '/notifications'
 },
 {
 id: 'profile',
 name: 'About Me',
 icon: User,
 path: '/profile'
 }
] as const

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

 // Memoize static arrays to prevent unnecessary re-renders
 const agents = useMemo(() => AGENTS_DATA, [])
 const navigationItems = useMemo(() => NAVIGATION_DATA, [])

 return (
 <motion.div
 initial={false}
 animate={isMobile ? { x: mobileOpen ? 0 : '-100%' } : { width: isCollapsed ? 80 : 280 }}
 transition={{ duration: 0.3, ease:"easeInOut" }}
 className={cn("relative flex h-[100dvh] flex-col border-r border-border/70 bg-white/75 backdrop-blur-xl /70",
 isMobile &&"pb-[env(safe-area-inset-bottom)] pt-[env(safe-area-inset-top)]",
 isMobile &&"fixed inset-y-0 left-0 z-50 w-[min(88vw,20rem)] max-w-[20rem] min-w-[15.5rem] shadow-2xl",
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
 <h1 className="text-sm font-semibold">AlterEgo</h1>
 <p className="text-[11px] text-muted-foreground">Your AI Partner</p>
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
 className="h-9 w-9 rounded-xl border border-border/70 bg-white/80 shadow-sm /70"
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
 className={cn("w-full justify-start rounded-xl text-sm",
 // Mobile touch target: 44px minimum (h-11 = 44px)
 isMobile ?"h-11" :"h-10",
 isCollapsed &&"justify-center px-0",
 currentView === item.id
 ?"bg-gradient-to-r from-teal-700 via-cyan-600 to-amber-500 text-white shadow-md"
 :"hover:bg-accent/70"
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
 Your Coaches
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
 className={cn("cursor-pointer border border-border/70 bg-card/70 shadow-sm transition-all duration-200",
 // Mobile touch target: 44px minimum padding
 isMobile ?"p-3.5" :"p-3",
 isActive &&"border-primary/70 bg-primary/10 shadow-md",
 !isActive &&"hover:-translate-y-0.5 hover:bg-accent/40",
 isCollapsed && (isMobile ?"p-3" :"p-2")
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
 <div className={cn("flex items-center gap-3",
 isCollapsed &&"justify-center"
 )}>
 <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br",
 agent.color,
 isActive &&"shadow-lg"
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
 <div className={cn("w-2 h-2 rounded-full",
 agent.status === 'active' ?"bg-muted0 dark:bg-emerald-400" :"bg-muted-foreground/50"
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
 <div className={cn("flex items-center gap-2",
 isCollapsed &&"justify-center"
 )}>
 <Button
 variant="ghost"
 size="icon"
 onClick={onToggleTheme}
 className="h-9 w-9 rounded-xl border border-border/70 bg-card/80"
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
 className="h-9 w-9 rounded-xl border border-border/70 bg-card/80"
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