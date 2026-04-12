import { useState, useCallback, useRef, ReactNode } from 'react'
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from 'framer-motion'
import { RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { hapticSuccess } from '@/lib/hapticFeedback'

interface PullToRefreshProps {
  onRefresh: () => Promise<void>
  children: ReactNode
  className?: string
  pullDistance?: number
  disabled?: boolean
}

const PULL_THRESHOLD = 80
const MAX_PULL = 150

export function PullToRefresh({
  onRefresh,
  children,
  className,
  pullDistance = PULL_THRESHOLD,
  disabled = false,
}: PullToRefreshProps) {
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [pullProgress, setPullProgress] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const startYRef = useRef(0)
  const isPullingRef = useRef(false)

  const y = useMotionValue(0)
  const smoothY = useSpring(y, { damping: 20, stiffness: 300 })
  const rotate = useTransform(smoothY, [0, pullDistance], [0, 360])
  const opacity = useTransform(smoothY, [0, pullDistance * 0.5], [0, 1])

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (disabled || isRefreshing) return
    
    // Only start pull if at top of scroll
    const scrollTop = containerRef.current?.scrollTop ?? 0
    if (scrollTop > 0) return

    startYRef.current = e.touches[0].clientY
    isPullingRef.current = true
  }, [disabled, isRefreshing])

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isPullingRef.current || disabled || isRefreshing) return

    const currentY = e.touches[0].clientY
    const diff = currentY - startYRef.current

    if (diff > 0) {
      // Calculate resistance - harder to pull as you go further
      const resisted = Math.min(diff * 0.6, MAX_PULL)
      y.set(resisted)
      setPullProgress(Math.min(resisted / pullDistance, 1))
      
      // Haptic feedback at threshold
      if (resisted >= pullDistance && pullProgress < 1) {
        hapticSuccess()
      }
    }
  }, [disabled, isRefreshing, pullDistance, pullProgress, y])

  const handleTouchEnd = useCallback(async () => {
    if (!isPullingRef.current || disabled) return
    
    isPullingRef.current = false
    const currentPull = y.get()

    if (currentPull >= pullDistance && !isRefreshing) {
      // Trigger refresh
      setIsRefreshing(true)
      y.set(pullDistance * 0.5) // Stay at indicator position
      
      try {
        await onRefresh()
      } finally {
        setIsRefreshing(false)
        y.set(0)
        setPullProgress(0)
      }
    } else {
      // Snap back
      y.set(0)
      setPullProgress(0)
    }
  }, [disabled, isRefreshing, onRefresh, pullDistance, y])

  // Don't enable on desktop
  if (typeof window !== 'undefined' && !('ontouchstart' in window)) {
    return <div className={className}>{children}</div>
  }

  return (
    <div className={cn("relative overflow-hidden", className)}>
      {/* Pull indicator */}
      <motion.div
        style={{ opacity, rotate }}
        className="absolute top-0 left-0 right-0 z-10 flex items-center justify-center pointer-events-none"
      >
        <div className="flex flex-col items-center gap-2 py-4">
          <motion.div
            animate={isRefreshing ? { rotate: 360 } : {}}
            transition={isRefreshing ? { repeat: Infinity, duration: 1, ease: "linear" } : {}}
          >
            <RefreshCw className={cn(
              "h-6 w-6 transition-colors",
              pullProgress >= 1 ? "text-primary" : "text-muted-foreground"
            )} />
          </motion.div>
          <span className="text-xs text-muted-foreground">
            {isRefreshing ? 'Refreshing...' : pullProgress >= 1 ? 'Release to refresh' : 'Pull to refresh'}
          </span>
        </div>
      </motion.div>

      {/* Content container */}
      <motion.div
        ref={containerRef}
        style={{ y: smoothY }}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        className="touch-pan-y"
      >
        {children}
      </motion.div>
    </div>
  )
}

// Hook for pull-to-refresh logic
export function usePullToRefresh(onRefresh: () => Promise<void>) {
  const [isPulling, setIsPulling] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true)
    try {
      await onRefresh()
    } finally {
      setIsRefreshing(false)
    }
  }, [onRefresh])

  return {
    isPulling,
    setIsPulling,
    isRefreshing,
    handleRefresh,
  }
}
