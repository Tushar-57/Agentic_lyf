import React, { useEffect, useMemo, useState } from 'react'
import { BellRing, Sparkles } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface AINotification {
  id: number
  notification_key: string
  kind: string
  severity: 'low' | 'medium' | 'high' | 'critical' | string
  status: 'active' | 'acknowledged' | 'resolved' | string
  title: string
  summary: string
  details?: string | null
  score?: number | null
  recommended_actions?: string[]
  last_seen_at?: string
  updated_at?: string
}

interface AINotificationEnvelope {
  persistence_enabled?: boolean
  notifications?: AINotification[]
}

const normalizeSeverity = (value: unknown): 'low' | 'medium' | 'high' | 'critical' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'low' || normalized === 'medium' || normalized === 'high' || normalized === 'critical') {
    return normalized
  }
  return 'medium'
}

const normalizeStatus = (value: unknown): 'active' | 'acknowledged' | 'resolved' => {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'active' || normalized === 'acknowledged' || normalized === 'resolved') {
    return normalized
  }
  return 'active'
}

const getTone = (severity: string) => {
  const normalized = normalizeSeverity(severity)
  if (normalized === 'critical') {
    return {
      wrapper: 'border-rose-300/80 bg-rose-50/70 dark:border-rose-900/70 dark:bg-rose-950/30',
      badge: 'bg-rose-100 text-rose-700 dark:bg-rose-900/60 dark:text-rose-200',
    }
  }

  if (normalized === 'high') {
    return {
      wrapper: 'border-amber-300/80 bg-amber-50/70 dark:border-amber-900/70 dark:bg-amber-950/30',
      badge: 'bg-amber-100 text-amber-700 dark:bg-amber-900/60 dark:text-amber-200',
    }
  }

  if (normalized === 'low') {
    return {
      wrapper: 'border-emerald-300/80 bg-emerald-50/70 dark:border-emerald-900/70 dark:bg-emerald-950/30',
      badge: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-200',
    }
  }

  return {
    wrapper: 'border-cyan-300/80 bg-cyan-50/70 dark:border-cyan-900/70 dark:bg-cyan-950/30',
    badge: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/60 dark:text-cyan-200',
  }
}

const formatTimestamp = (value: string | undefined) => {
  if (!value) {
    return 'n/a'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return 'n/a'
  }

  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export const AINotificationsCenter: React.FC = () => {
  const [notifications, setNotifications] = useState<AINotification[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [persistenceEnabled, setPersistenceEnabled] = useState(false)

  const loadNotifications = async (refresh: boolean) => {
    setError(null)
    setIsLoading(true)
    if (refresh) {
      setIsRefreshing(true)
    }

    try {
      const response = await fetch(
        refresh ? '/api/knowledge/notifications/refresh?limit=40' : '/api/knowledge/notifications?limit=40',
        { method: refresh ? 'POST' : 'GET' },
      )

      if (!response.ok) {
        throw new Error(`Notifications request failed with status ${response.status}`)
      }

      const payload = (await response.json()) as AINotificationEnvelope
      const list = Array.isArray(payload.notifications) ? payload.notifications : []

      setNotifications(
        list.map((entry) => ({
          ...entry,
          severity: normalizeSeverity(entry.severity),
          status: normalizeStatus(entry.status),
        })),
      )
      setPersistenceEnabled(Boolean(payload.persistence_enabled))
    } catch (loadError) {
      console.error('Failed to load AI notifications:', loadError)
      setError(loadError instanceof Error ? loadError.message : 'Unable to load notifications.')
    } finally {
      setIsLoading(false)
      if (refresh) {
        setIsRefreshing(false)
      }
    }
  }

  const setAcknowledged = async (notificationId: number, acknowledged: boolean) => {
    try {
      const response = await fetch(`/api/knowledge/notifications/${notificationId}/ack`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ acknowledged }),
      })

      if (!response.ok) {
        throw new Error(`Notification update failed with status ${response.status}`)
      }

      const updated = (await response.json()) as AINotification
      setNotifications((previous) => previous.map((entry) => (entry.id === updated.id ? updated : entry)))
    } catch (updateError) {
      console.error('Failed to update notification acknowledgement:', updateError)
      setError('Unable to update notification status right now.')
    }
  }

  useEffect(() => {
    void loadNotifications(true)
  }, [])

  const sortedNotifications = useMemo(() => {
    const severityRank: Record<string, number> = {
      critical: 0,
      high: 1,
      medium: 2,
      low: 3,
    }
    const statusRank: Record<string, number> = {
      active: 0,
      acknowledged: 1,
      resolved: 2,
    }

    return [...notifications].sort((left, right) => {
      const statusDiff = (statusRank[left.status] ?? 3) - (statusRank[right.status] ?? 3)
      if (statusDiff !== 0) {
        return statusDiff
      }

      const severityDiff = (severityRank[left.severity] ?? 9) - (severityRank[right.severity] ?? 9)
      if (severityDiff !== 0) {
        return severityDiff
      }

      const leftTime = new Date(left.last_seen_at || left.updated_at || 0).getTime()
      const rightTime = new Date(right.last_seen_at || right.updated_at || 0).getTime()
      return rightTime - leftTime
    })
  }, [notifications])

  const activeCount = sortedNotifications.filter((entry) => entry.status === 'active').length
  const highSignalCount = sortedNotifications.filter(
    (entry) => entry.status === 'active' && (entry.severity === 'high' || entry.severity === 'critical'),
  ).length

  return (
    <Card className="mx-auto w-full max-w-5xl border-border/70 bg-gradient-to-br from-cyan-50/75 via-white to-emerald-50/55 p-5 dark:from-cyan-950/20 dark:via-slate-950 dark:to-emerald-950/20">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="mb-1 inline-flex items-center gap-2 rounded-full border border-cyan-200/70 bg-white/80 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-cyan-700 dark:border-cyan-900/60 dark:bg-slate-900/70 dark:text-cyan-200">
            <BellRing className="h-3.5 w-3.5" />
            AI Notification Center
          </p>
          <h2 className="text-2xl font-bold">AI Notifications</h2>
          <p className="text-sm text-muted-foreground">
            Proactive alerts and goal-alignment scores generated from your checkups, priorities, and execution patterns.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="bg-white/75 text-[11px] dark:bg-slate-900/70">
            Active {activeCount}
          </Badge>
          <Badge variant="outline" className="bg-white/75 text-[11px] dark:bg-slate-900/70">
            High Signal {highSignalCount}
          </Badge>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => void loadNotifications(true)}
            disabled={isRefreshing}
            className="gap-2"
          >
            <Sparkles className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
            {isRefreshing ? 'Refreshing' : 'Refresh Signals'}
          </Button>
        </div>
      </div>

      <div className="mb-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>Persistence {persistenceEnabled ? 'enabled' : 'fallback mode'}</span>
        <span>{isLoading ? 'Syncing notifications...' : `${sortedNotifications.length} notifications`}</span>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
          {error}
        </div>
      )}

      {isLoading && sortedNotifications.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/70 bg-background/65 p-5 text-sm text-muted-foreground">
          Building your latest AI notification feed...
        </div>
      ) : sortedNotifications.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/70 bg-background/65 p-5 text-sm text-muted-foreground">
          No proactive alerts right now. Your system is currently balanced.
        </div>
      ) : (
        <div className="space-y-3">
          {sortedNotifications.map((notification) => {
            const tone = getTone(notification.severity)
            const actions = Array.isArray(notification.recommended_actions)
              ? notification.recommended_actions.slice(0, 2)
              : []

            return (
              <div
                key={notification.id}
                className={cn('rounded-xl border p-3.5', tone.wrapper)}
              >
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <Badge className={cn('text-[10px] uppercase tracking-wide', tone.badge)}>
                    {notification.severity}
                  </Badge>
                  <Badge variant="outline" className="bg-white/70 text-[10px] uppercase tracking-wide dark:bg-slate-900/70">
                    {notification.status}
                  </Badge>
                  <Badge variant="outline" className="bg-white/70 text-[10px] uppercase tracking-wide dark:bg-slate-900/70">
                    {notification.kind.replace(/[_-]+/g, ' ')}
                  </Badge>
                  {typeof notification.score === 'number' && (
                    <span className="text-xs font-semibold text-foreground">Score {Math.round(notification.score)}</span>
                  )}
                </div>

                <p className="text-sm font-semibold text-foreground">{notification.title}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{notification.summary}</p>

                {notification.details && (
                  <p className="mt-2 text-xs leading-relaxed text-foreground/80">{notification.details}</p>
                )}

                {actions.length > 0 && (
                  <ul className="mt-2 space-y-1 text-xs text-foreground/85">
                    {actions.map((action) => (
                      <li key={action}>- {action}</li>
                    ))}
                  </ul>
                )}

                <div className="mt-3 flex items-center justify-between">
                  <p className="text-[11px] text-muted-foreground">
                    Last signal: {formatTimestamp(notification.last_seen_at || notification.updated_at)}
                  </p>

                  {notification.id > 0 && (
                    <Button
                      type="button"
                      size="sm"
                      variant={notification.status === 'active' ? 'secondary' : 'outline'}
                      onClick={() => void setAcknowledged(notification.id, notification.status === 'active')}
                    >
                      {notification.status === 'active' ? 'Acknowledge' : 'Mark Active'}
                    </Button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </Card>
  )
}

export default AINotificationsCenter
