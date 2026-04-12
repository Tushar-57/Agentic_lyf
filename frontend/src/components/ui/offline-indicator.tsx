import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { WifiOff, Wifi } from 'lucide-react'
import { cn } from '@/lib/utils'

export function OfflineIndicator() {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [showIndicator, setShowIndicator] = useState(false)

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
      // Keep indicator visible briefly to show reconnection
      setTimeout(() => setShowIndicator(false), 2000)
    }

    const handleOffline = () => {
      setIsOnline(false)
      setShowIndicator(true)
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    // Check initial state
    if (!navigator.onLine) {
      setShowIndicator(true)
    }

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  return (
    <AnimatePresence>
      {showIndicator && (
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          className={cn(
            "fixed top-0 left-0 right-0 z-[100] px-4 py-2 text-center text-sm font-medium",
            isOnline 
              ? "bg-green-500 text-white" 
              : "bg-destructive text-destructive-foreground"
          )}
        >
          <div className="flex items-center justify-center gap-2">
            {isOnline ? (
              <>
                <Wifi className="h-4 w-4" />
                <span>Back online</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4" />
                <span>You're offline. Reconnect to use all features.</span>
              </>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// Hook to check backend connectivity
export function useBackendStatus(pollingInterval = 30000) {
  const [isConnected, setIsConnected] = useState(true)
  const [lastChecked, setLastChecked] = useState<Date>(new Date())

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000)
        
        const response = await fetch('/api/health', {
          method: 'HEAD',
          signal: controller.signal
        })
        
        clearTimeout(timeoutId)
        setIsConnected(response.ok)
      } catch {
        setIsConnected(false)
      } finally {
        setLastChecked(new Date())
      }
    }

    // Check immediately
    checkConnection()

    // Then poll
    const interval = setInterval(checkConnection, pollingInterval)
    return () => clearInterval(interval)
  }, [pollingInterval])

  return { isConnected, lastChecked }
}

// Backend connectivity indicator
export function BackendStatusIndicator() {
  const { isConnected, lastChecked } = useBackendStatus()
  const [showDetails, setShowDetails] = useState(false)

  if (isConnected) return null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="fixed bottom-4 left-4 z-50"
    >
      <button
        onClick={() => setShowDetails(!showDetails)}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium",
          "bg-amber-100 text-amber-800 border border-amber-200",
          "dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-800"
        )}
      >
        <WifiOff className="h-3 w-3" />
        <span>Server connection lost</span>
      </button>
      
      {showDetails && (
        <div className="mt-2 p-3 rounded-lg bg-white dark:bg-slate-900 shadow-lg border border-border text-xs">
          <p className="text-muted-foreground">
            Checked {Math.round((Date.now() - lastChecked.getTime()) / 60000)} minutes ago
          </p>
          <p className="mt-1 text-muted-foreground">
            Some features are temporarily unavailable.
          </p>
        </div>
      )}
    </motion.div>
  )
}
