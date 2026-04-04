import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bell } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface FloatingApprovalProps {
  onOpenApprovalPanel: () => void
}

export const FloatingApprovalNotification = ({ onOpenApprovalPanel }: FloatingApprovalProps) => {
  const [pendingCount, setPendingCount] = useState(0)
  const [isVisible, setIsVisible] = useState(false)

  const fetchPendingCount = async () => {
    try {
      const response = await fetch('/api/approval/stats')
      if (response.ok) {
        const data = await response.json()
        const count = data.pending_interactions || 0
        setPendingCount(count)
        setIsVisible(count > 0)
      }
    } catch (error) {
      console.error('Failed to fetch pending count:', error)
    }
  }

  useEffect(() => {
    fetchPendingCount()
    // Poll every 5 seconds for updates
    const interval = setInterval(fetchPendingCount, 5000)
    return () => clearInterval(interval)
  }, [])

  if (!isVisible) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 50, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 50, scale: 0.9 }}
        className="fixed bottom-4 right-3 z-50 max-w-[calc(100vw-1rem)] sm:bottom-20 sm:right-4"
      >
        <Card className="border-2 border-yellow-400 bg-yellow-50 dark:border-yellow-600 dark:bg-yellow-950 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Bell className="h-6 w-6 text-yellow-600" />
                <Badge 
                  className="absolute -top-2 -right-2 h-5 w-5 p-0 text-xs bg-red-500 text-white flex items-center justify-center"
                >
                  {pendingCount}
                </Badge>
              </div>
              
              <div className="flex-1">
                <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  Approval Needed
                </p>
                <p className="text-xs text-yellow-600 dark:text-yellow-400">
                  {pendingCount} response{pendingCount !== 1 ? 's' : ''} waiting for review
                </p>
              </div>
              
              <Button
                size="sm"
                onClick={onOpenApprovalPanel}
                className="bg-yellow-600 hover:bg-yellow-700 text-white"
              >
                Review
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}