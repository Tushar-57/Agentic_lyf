import { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Button } from './button'
import { Database, FileX, Inbox, Search, Sparkles } from 'lucide-react'

interface EmptyStateProps {
  title: string
  description: string
  icon?: ReactNode
  action?: {
    label: string
    onClick: () => void
  }
  className?: string
}

function EmptyState({ 
  title, 
  description, 
  icon, 
  action,
  className 
}: EmptyStateProps) {
  return (
    <div className={cn(
      "flex flex-col items-center justify-center text-center p-8 rounded-2xl border border-dashed border-border/70 bg-background/65",
      className
    )}>
      {icon && (
        <div className="mb-4 text-muted-foreground">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-4">{description}</p>
      {action && (
        <Button onClick={action.onClick} variant="outline" size="sm">
          {action.label}
        </Button>
      )}
    </div>
  )
}

// Pre-configured empty states for common scenarios

function EmptyKnowledgeBase({ onCreate }: { onCreate?: () => void }) {
  return (
    <EmptyState
      title="No Knowledge Entries Yet"
      description="Your knowledge base is empty. Start by adding your preferences or completing onboarding to build your personal memory."
      icon={<Database className="h-12 w-12" />}
      action={onCreate ? { label: "Add First Entry", onClick: onCreate } : undefined}
    />
  )
}

function EmptyNotifications() {
  return (
    <EmptyState
      title="No Active Notifications"
      description="You're all caught up! New AI-generated insights will appear here when available."
      icon={<Inbox className="h-12 w-12" />}
    />
  )
}

function EmptySearchResults({ query, onClear }: { query: string; onClear: () => void }) {
  return (
    <EmptyState
      title="No Results Found"
      description={`We couldn't find anything matching "${query}". Try adjusting your search terms.`}
      icon={<Search className="h-12 w-12" />}
      action={{ label: "Clear Search", onClick: onClear }}
    />
  )
}

function EmptyChat({ onStart }: { onStart?: () => void }) {
  return (
    <EmptyState
      title="Start a Conversation"
      description="Ask me anything about productivity, health, finance, or scheduling. I'm here to help!"
      icon={<Sparkles className="h-12 w-12" />}
      action={onStart ? { label: "Send First Message", onClick: onStart } : undefined}
    />
  )
}

function ErrorState({ 
  title = "Something went wrong", 
  description = "An error occurred while loading this content. Please try again.",
  onRetry 
}: { 
  title?: string
  description?: string
  onRetry?: () => void 
}) {
  return (
    <EmptyState
      title={title}
      description={description}
      icon={<FileX className="h-12 w-12 text-destructive" />}
      action={onRetry ? { label: "Try Again", onClick: onRetry } : undefined}
    />
  )
}

export {
  EmptyState,
  EmptyKnowledgeBase,
  EmptyNotifications,
  EmptySearchResults,
  EmptyChat,
  ErrorState
}
