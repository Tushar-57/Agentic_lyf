import { cn } from "@/lib/utils"

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

// Chat message skeleton
function ChatMessageSkeleton({ isUser = false }: { isUser?: boolean }) {
  return (
    <div className={cn("flex gap-3 px-3 py-4 sm:px-4", isUser ? "flex-row-reverse" : "flex-row")}>
      <Skeleton className="h-8 w-8 rounded-full flex-shrink-0" />
      <div className={cn("flex min-w-0 flex-col max-w-[92%] sm:max-w-[80%]", isUser ? "items-end" : "items-start")}>
        <Skeleton className="h-4 w-24 mb-2" />
        <Skeleton className={cn("h-20 w-[280px] sm:w-[320px]", isUser ? "rounded-2xl" : "rounded-2xl")} />
      </div>
    </div>
  )
}

// Knowledge card skeleton
function KnowledgeCardSkeleton() {
  return (
    <div className="rounded-2xl border border-border/70 bg-white/70 p-4 dark:bg-slate-900/60 space-y-3">
      <div className="flex items-center justify-between">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-4 w-16" />
      </div>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <div className="flex gap-2 pt-2">
        <Skeleton className="h-6 w-16 rounded-full" />
        <Skeleton className="h-6 w-20 rounded-full" />
      </div>
    </div>
  )
}

// Analytics dashboard skeleton
function AnalyticsDashboardSkeleton() {
  return (
    <div className="space-y-6 p-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-2xl border border-border/70 bg-white/75 p-4 dark:bg-slate-900/60 space-y-3">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-8 w-16" />
            <Skeleton className="h-2 w-full" />
          </div>
        ))}
      </div>
      <Skeleton className="h-[300px] w-full rounded-2xl" />
    </div>
  )
}

// Notification item skeleton
function NotificationItemSkeleton() {
  return (
    <div className="rounded-xl border border-border/70 p-3.5 space-y-2">
      <div className="flex items-center gap-2">
        <Skeleton className="h-5 w-16 rounded-full" />
        <Skeleton className="h-5 w-20 rounded-full" />
      </div>
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-3 w-3/4" />
    </div>
  )
}

export { 
  Skeleton, 
  ChatMessageSkeleton, 
  KnowledgeCardSkeleton, 
  AnalyticsDashboardSkeleton,
  NotificationItemSkeleton 
}
