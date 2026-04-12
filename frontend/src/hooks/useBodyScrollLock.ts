import { useEffect, useRef, useCallback } from 'react'

// Global state to track scroll lock count across components
let scrollLockCount = 0
let originalOverflow = ''

// Throttle utility for performance
function throttle<T extends (...args: any[]) => any>(func: T, limit: number): T {
  let inThrottle = false
  return ((...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }) as T
}

export function useBodyScrollLock(isLocked: boolean) {
  const isLockedRef = useRef(isLocked)
  const rafIdRef = useRef<number | null>(null)

  useEffect(() => {
    isLockedRef.current = isLocked

    if (typeof document === 'undefined') return

    // Cancel any pending animation frame
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }

    if (isLocked) {
      // Use requestAnimationFrame for performance
      rafIdRef.current = requestAnimationFrame(() => {
        // Only set overflow on first lock
        if (scrollLockCount === 0) {
          originalOverflow = document.body.style.overflow
          document.body.style.overflow = 'hidden'
        }
        scrollLockCount++
        rafIdRef.current = null
      })
    }

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }
      if (isLockedRef.current) {
        scrollLockCount--
        // Only restore overflow when all locks are released
        if (scrollLockCount === 0) {
          document.body.style.overflow = originalOverflow
        }
      }
    }
  }, [isLocked])
}

// Hook for throttled scroll handlers
export function useThrottledScrollHandler(
  callback: (scrollY: number) => void,
  throttleMs = 16 // ~60fps
) {
  const callbackRef = useRef(callback)
  const throttleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastScrollY = useRef(0)

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  const throttledHandler = useCallback(() => {
    const currentScrollY = window.scrollY

    // Skip if scroll hasn't changed meaningfully
    if (Math.abs(currentScrollY - lastScrollY.current) < 1) return
    lastScrollY.current = currentScrollY

    if (throttleTimeoutRef.current === null) {
      throttleTimeoutRef.current = setTimeout(() => {
        throttleTimeoutRef.current = null
        callbackRef.current(currentScrollY)
      }, throttleMs)
    }
  }, [throttleMs])

  useEffect(() => {
    const handleScroll = () => {
      // Use requestAnimationFrame for smooth performance
      requestAnimationFrame(throttledHandler)
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', handleScroll)
      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current)
        throttleTimeoutRef.current = null
      }
    }
  }, [throttledHandler])
}

// Hook for throttled resize handlers
export function useThrottledResizeHandler(
  callback: (width: number, height: number) => void,
  throttleMs = 100
) {
  const callbackRef = useRef(callback)
  const throttleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const dimensionsRef = useRef({ width: window.innerWidth, height: window.innerHeight })

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  const throttledHandler = useCallback(() => {
    const newWidth = window.innerWidth
    const newHeight = window.innerHeight

    // Skip if dimensions haven't changed meaningfully
    if (
      Math.abs(newWidth - dimensionsRef.current.width) < 1 &&
      Math.abs(newHeight - dimensionsRef.current.height) < 1
    ) {
      return
    }

    dimensionsRef.current = { width: newWidth, height: newHeight }

    if (throttleTimeoutRef.current === null) {
      throttleTimeoutRef.current = setTimeout(() => {
        throttleTimeoutRef.current = null
        callbackRef.current(newWidth, newHeight)
      }, throttleMs)
    }
  }, [throttleMs])

  useEffect(() => {
    const handleResize = () => {
      requestAnimationFrame(throttledHandler)
    }

    window.addEventListener('resize', handleResize, { passive: true })
    
    // Initial call
    callback(window.innerWidth, window.innerHeight)
    
    return () => {
      window.removeEventListener('resize', handleResize)
      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current)
        throttleTimeoutRef.current = null
      }
    }
  }, [throttledHandler])
}

export function useStableInterval(
  callback: () => void,
  delay: number | null,
  deps: React.DependencyList = []
) {
  const callbackRef = useRef(callback)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Keep callback ref up to date without triggering effect
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    // Clear previous interval when deps change
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    if (delay !== null) {
      intervalRef.current = setInterval(() => {
        callbackRef.current()
      }, delay)
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [delay, ...deps])
}

export default useBodyScrollLock
