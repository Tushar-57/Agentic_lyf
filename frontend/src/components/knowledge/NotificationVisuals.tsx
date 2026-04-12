import React, { useMemo } from 'react'
import { cn } from '@/lib/utils'

// Sparkline Chart Component
interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  fillColor?: string
  strokeWidth?: number
  showDots?: boolean
  className?: string
}

export const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 120,
  height = 40,
  color = '#3b82f6',
  fillColor = 'rgba(59, 130, 246, 0.1)',
  strokeWidth = 2,
  showDots = false,
  className,
}) => {
  const pathD = useMemo(() => {
    if (data.length < 2) return ''
    
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1
    
    const points = data.map((value, index) => ({
      x: (index / (data.length - 1)) * width,
      y: height - ((value - min) / range) * height,
    }))
    
    // Create smooth curve using bezier
    let path = `M ${points[0].x} ${points[0].y}`
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1]
      const curr = points[i]
      const cpX = (prev.x + curr.x) / 2
      path += ` Q ${cpX} ${prev.y} ${curr.x} ${curr.y}`
    }
    
    return path
  }, [data, width, height])
  
  const areaPath = useMemo(() => {
    if (!pathD) return ''
    return `${pathD} L ${width} ${height} L 0 ${height} Z`
  }, [pathD, width, height])
  
  const trend = useMemo(() => {
    if (data.length < 2) return 'neutral'
    const first = data[0]
    const last = data[data.length - 1]
    if (last > first * 1.05) return 'up'
    if (last < first * 0.95) return 'down'
    return 'neutral'
  }, [data])
  
  const trendColor = trend === 'up' ? '#10b981' : trend === 'down' ? '#ef4444' : color
  
  return (
    <svg
      width={width}
      height={height}
      className={cn('overflow-visible', className)}
      viewBox={`0 0 ${width} ${height}`}
    >
      {/* Gradient definition */}
      <defs>
        <linearGradient id={`sparkline-gradient-${trend}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={trendColor} stopOpacity="0.3" />
          <stop offset="100%" stopColor={trendColor} stopOpacity="0" />
        </linearGradient>
      </defs>
      
      {/* Area fill */}
      <path
        d={areaPath}
        fill={`url(#sparkline-gradient-${trend})`}
        className="transition-all duration-500"
      />
      
      {/* Line */}
      <path
        d={pathD}
        fill="none"
        stroke={trendColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-all duration-500"
      />
      
      {/* Dots at data points */}
      {showDots && data.map((value, index) => {
        const min = Math.min(...data)
        const max = Math.max(...data)
        const range = max - min || 1
        const x = (index / (data.length - 1)) * width
        const y = height - ((value - min) / range) * height
        
        return (
          <circle
            key={index}
            cx={x}
            cy={y}
            r={3}
            fill={trendColor}
            className="transition-all duration-300"
          />
        )
      })}
      
      {/* Last value indicator */}
      {data.length > 0 && (
        <circle
          cx={width}
          cy={height - ((data[data.length - 1] - Math.min(...data)) / (Math.max(...data) - Math.min(...data) || 1)) * height}
          r={4}
          fill={trendColor}
          stroke="white"
          strokeWidth={2}
          className="animate-pulse"
        />
      )}
    </svg>
  )
}

// Circular Progress Ring
interface CircularProgressProps {
  value: number
  max?: number
  size?: number
  strokeWidth?: number
  color?: string
  bgColor?: string
  showValue?: boolean
  label?: string
  className?: string
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  value,
  max = 100,
  size = 60,
  strokeWidth = 6,
  color = '#3b82f6',
  bgColor = '#e5e7eb',
  showValue = true,
  label,
  className,
}) => {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)
  const strokeDashoffset = circumference - (percentage / 100) * circumference
  
  const colorClass = useMemo(() => {
    if (percentage >= 80) return '#10b981'
    if (percentage >= 60) return '#3b82f6'
    if (percentage >= 40) return '#f59e0b'
    return '#ef4444'
  }, [percentage])
  
  return (
    <div className={cn('relative inline-flex flex-col items-center', className)}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={bgColor}
          strokeWidth={strokeWidth}
          className="dark:stroke-slate-700"
        />
        
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={colorClass}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-700 ease-out"
        />
      </svg>
      
      {showValue && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold" style={{ color: colorClass }}>
            {Math.round(percentage)}
          </span>
        </div>
      )}
      
      {label && (
        <span className="mt-1 text-[10px] text-muted-foreground uppercase tracking-wide">
          {label}
        </span>
      )}
    </div>
  )
}

// Segmented Progress Bar
interface SegmentedProgressProps {
  segments: { label: string; value: number; color: string }[]
  total?: number
  height?: number
  showLabels?: boolean
  className?: string
}

export const SegmentedProgress: React.FC<SegmentedProgressProps> = ({
  segments,
  total,
  height = 24,
  showLabels = true,
  className,
}) => {
  const computedTotal = total || segments.reduce((sum, s) => sum + s.value, 0)
  
  return (
    <div className={cn('w-full', className)}>
      {/* Bar */}
      <div
        className="flex w-full overflow-hidden rounded-full"
        style={{ height }}
      >
        {segments.map((segment, index) => {
          const width = computedTotal > 0 ? (segment.value / computedTotal) * 100 : 0
          return (
            <div
              key={index}
              className="flex items-center justify-center transition-all duration-500 first:rounded-l-full last:rounded-r-full"
              style={{
                width: `${width}%`,
                backgroundColor: segment.color,
                minWidth: width > 5 ? 'auto' : '4px',
              }}
              title={`${segment.label}: ${segment.value}`}
            >
              {width > 15 && (
                <span className="text-[10px] font-medium text-white truncate px-1">
                  {Math.round(width)}%
                </span>
              )}
            </div>
          )
        })}
      </div>
      
      {/* Legend */}
      {showLabels && (
        <div className="mt-2 flex flex-wrap gap-3">
          {segments.map((segment, index) => (
            <div key={index} className="flex items-center gap-1.5">
              <div
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
              <span className="text-[10px] text-muted-foreground">
                {segment.label}: {segment.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Battery Level Indicator
interface BatteryIndicatorProps {
  level: number
  max?: number
  size?: 'sm' | 'md' | 'lg'
  colorScheme?: 'default' | 'traffic' | 'heatmap'
  label?: string
  className?: string
}

export const BatteryIndicator: React.FC<BatteryIndicatorProps> = ({
  level,
  max = 100,
  size = 'md',
  colorScheme = 'traffic',
  label,
  className,
}) => {
  const percentage = Math.min(Math.max((level / max) * 100, 0), 100)
  
  const sizeClasses = {
    sm: { container: 'h-4 w-8', cap: 'h-2 w-1', text: 'text-[8px]' },
    md: { container: 'h-6 w-12', cap: 'h-3 w-1.5', text: 'text-[10px]' },
    lg: { container: 'h-8 w-16', cap: 'h-4 w-2', text: 'text-xs' },
  }
  
  const color = useMemo(() => {
    if (colorScheme === 'traffic') {
      if (percentage >= 70) return '#10b981'
      if (percentage >= 40) return '#f59e0b'
      return '#ef4444'
    }
    if (colorScheme === 'heatmap') {
      // Heatmap gradient from blue (low) to red (high)
      if (percentage < 30) return '#3b82f6'
      if (percentage < 60) return '#8b5cf6'
      if (percentage < 80) return '#f59e0b'
      return '#ef4444'
    }
    return '#3b82f6'
  }, [percentage, colorScheme])
  
  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <div className="relative flex items-center">
        {/* Battery body */}
        <div
          className={cn(
            'rounded-sm border-2 border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 overflow-hidden',
            sizeClasses[size].container
          )}
        >
          <div
            className="h-full transition-all duration-500 ease-out"
            style={{
              width: `${percentage}%`,
              backgroundColor: color,
            }}
          />
        </div>
        
        {/* Battery cap */}
        <div
          className={cn(
            'rounded-r-sm bg-slate-300 dark:bg-slate-600',
            sizeClasses[size].cap
          )}
        />
      </div>
      
      <span className={cn('font-medium', sizeClasses[size].text)} style={{ color }}>
        {Math.round(percentage)}%
      </span>
      
      {label && (
        <span className={cn('text-muted-foreground', sizeClasses[size].text)}>
          {label}
        </span>
      )}
    </div>
  )
}

// Comparison Gauge (You vs Goal)
interface ComparisonGaugeProps {
  current: number
  target: number
  unit?: string
  label?: string
  className?: string
}

export const ComparisonGauge: React.FC<ComparisonGaugeProps> = ({
  current,
  target,
  unit = '',
  label,
  className,
}) => {
  const ratio = target > 0 ? Math.min((current / target) * 100, 150) : 0
  const isAhead = current >= target
  
  return (
    <div className={cn('space-y-2', className)}>
      {label && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">{label}</span>
          <span className={cn('font-medium', isAhead ? 'text-emerald-600' : 'text-amber-600')}>
            {isAhead ? '✓ On track' : '↓ Behind'}
          </span>
        </div>
      )}
      
      {/* Gauge bar */}
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        {/* Target marker */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-slate-500 dark:bg-slate-400 z-10"
          style={{ left: `${Math.min((target / Math.max(current, target)) * 100, 100)}%` }}
        />
        
        {/* Current progress */}
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500',
            isAhead ? 'bg-emerald-500' : 'bg-amber-500'
          )}
          style={{ width: `${Math.min(ratio, 100)}%` }}
        />
        
        {/* Excess indicator */}
        {ratio > 100 && (
          <div
            className="absolute h-full bg-emerald-300 dark:bg-emerald-700"
            style={{
              left: '100%',
              width: `${Math.min(ratio - 100, 50)}%`,
            }}
          />
        )}
      </div>
      
      {/* Values */}
      <div className="flex items-center justify-between text-xs">
        <span className={cn('font-medium', isAhead ? 'text-emerald-600' : 'text-amber-600')}>
          Current: {current.toFixed(1)}{unit}
        </span>
        <span className="text-muted-foreground">
          Target: {target.toFixed(1)}{unit}
        </span>
      </div>
    </div>
  )
}

// Metric Card with Trend
interface MetricCardProps {
  value: number | string
  label: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  icon?: React.ReactNode
  color?: string
  className?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  value,
  label,
  trend = 'neutral',
  trendValue,
  icon,
  color = '#3b82f6',
  className,
}) => {
  const trendColors = {
    up: '#10b981',
    down: '#ef4444',
    neutral: '#6b7280',
  }
  
  const trendIcons = {
    up: '↑',
    down: '↓',
    neutral: '→',
  }
  
  return (
    <div
      className={cn(
        'rounded-lg border bg-white/50 p-3 dark:bg-slate-900/50',
        className
      )}
      style={{ borderColor: `${color}30` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-2xl font-bold" style={{ color }}>
            {value}
          </p>
          <p className="text-xs text-muted-foreground">{label}</p>
        </div>
        {icon && (
          <div
            className="rounded-full p-1.5"
            style={{ backgroundColor: `${color}20` }}
          >
            {icon}
          </div>
        )}
      </div>
      
      {trendValue && (
        <div className="mt-2 flex items-center gap-1 text-xs">
          <span style={{ color: trendColors[trend] }}>
            {trendIcons[trend]} {trendValue}
          </span>
        </div>
      )}
    </div>
  )
}

// Heatmap Grid (for patterns)
interface HeatmapGridProps {
  data: number[]
  rows?: number
  colorScheme?: 'blue' | 'green' | 'purple' | 'orange'
  className?: string
}

export const HeatmapGrid: React.FC<HeatmapGridProps> = ({
  data,
  rows = 1,
  colorScheme = 'blue',
  className,
}) => {
  const colors = {
    blue: ['#dbeafe', '#93c5fd', '#60a5fa', '#3b82f6', '#2563eb'],
    green: ['#d1fae5', '#6ee7b7', '#34d399', '#10b981', '#059669'],
    purple: ['#f3e8ff', '#d8b4fe', '#c084fc', '#a855f7', '#7c3aed'],
    orange: ['#ffedd5', '#fdba74', '#fb923c', '#f97316', '#ea580c'],
  }
  
  const max = Math.max(...data, 1)
  const colorPalette = colors[colorScheme]
  
  return (
    <div
      className={cn('grid gap-1', className)}
      style={{ gridTemplateColumns: `repeat(${Math.ceil(data.length / rows)}, 1fr)` }}
    >
      {data.map((value, index) => {
        const intensity = Math.min(Math.floor((value / max) * colorPalette.length), colorPalette.length - 1)
        return (
          <div
            key={index}
            className="aspect-square rounded-sm transition-colors duration-300"
            style={{ backgroundColor: colorPalette[intensity] }}
            title={`${value}`}
          />
        )
      })}
    </div>
  )
}

// Animated Counter
interface AnimatedCounterProps {
  value: number
  duration?: number
  decimals?: number
  prefix?: string
  suffix?: string
  className?: string
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  duration = 1000,
  decimals = 0,
  prefix = '',
  suffix = '',
  className,
}) => {
  const [displayValue, setDisplayValue] = React.useState(0)
  const previousValue = React.useRef(value)
  
  React.useEffect(() => {
    const startValue = previousValue.current
    const endValue = value
    const diff = endValue - startValue
    const startTime = performance.now()
    
    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Easing function (ease-out)
      const easeOut = 1 - Math.pow(1 - progress, 3)
      const current = startValue + diff * easeOut
      
      setDisplayValue(current)
      
      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        previousValue.current = value
      }
    }
    
    requestAnimationFrame(animate)
  }, [value, duration])
  
  return (
    <span className={className}>
      {prefix}{displayValue.toFixed(decimals)}{suffix}
    </span>
  )
}

// Mini Calendar/Schedule View
interface MiniScheduleProps {
  blocks: { time: string; label: string; color: string; duration?: number }[]
  className?: string
}

export const MiniSchedule: React.FC<MiniScheduleProps> = ({
  blocks,
  className,
}) => {
  return (
    <div className={cn('space-y-1', className)}>
      {blocks.map((block, index) => (
        <div
          key={index}
          className="flex items-center gap-2 rounded px-2 py-1 text-xs"
          style={{ backgroundColor: `${block.color}20` }}
        >
          <span className="font-medium text-slate-600 dark:text-slate-400 w-12">
            {block.time}
          </span>
          <div
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: block.color }}
          />
          <span className="truncate">{block.label}</span>
          {block.duration && (
            <span className="ml-auto text-[10px] text-muted-foreground">
              {block.duration}m
            </span>
          )}
        </div>
      ))}
    </div>
  )
}
