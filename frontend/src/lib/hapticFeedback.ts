/**
 * Haptic Feedback Utilities
 * 
 * Provides haptic feedback for mobile interactions using the Vibration API.
 * Falls back gracefully on unsupported devices.
 */

// Haptic feedback types for different interactions
export type HapticType = 
  | 'light'      // Brief feedback for subtle interactions
  | 'medium'     // Standard feedback for buttons
  | 'heavy'      // Strong feedback for important actions
  | 'success'    // Success pattern
  | 'warning'    // Warning pattern
  | 'error'      // Error pattern
  | 'selection'  // Selection change

// Vibration patterns (in milliseconds)
const PATTERNS: Record<HapticType, number | number[]> = {
  light: 10,
  medium: 25,
  heavy: 50,
  success: [25, 50, 25],
  warning: [30, 100, 30],
  error: [50, 100, 50, 100, 50],
  selection: 15,
}

/**
 * Check if haptic feedback is supported
 */
export function isHapticSupported(): boolean {
  return typeof navigator !== 'undefined' && 'vibrate' in navigator
}

/**
 * Check if user has enabled reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Trigger haptic feedback
 */
export function haptic(type: HapticType = 'medium'): void {
  // Skip if not supported or user prefers reduced motion
  if (!isHapticSupported() || prefersReducedMotion()) {
    return
  }

  try {
    const pattern = PATTERNS[type]
    navigator.vibrate(pattern)
  } catch {
    // Silently fail if vibration API errors
  }
}

/**
 * Haptic feedback for button press
 */
export function hapticButtonPress(): void {
  haptic('medium')
}

/**
 * Haptic feedback for selection change
 */
export function hapticSelection(): void {
  haptic('selection')
}

/**
 * Haptic feedback for success action
 */
export function hapticSuccess(): void {
  haptic('success')
}

/**
 * Haptic feedback for error/warning
 */
export function hapticError(): void {
  haptic('error')
}

/**
 * Haptic feedback for long press
 */
export function hapticLongPress(): void {
  haptic('heavy')
}

/**
 * React hook for haptic feedback on touch devices
 */
export function useHaptic() {
  return {
    trigger: haptic,
    buttonPress: hapticButtonPress,
    selection: hapticSelection,
    success: hapticSuccess,
    error: hapticError,
    longPress: hapticLongPress,
    isSupported: isHapticSupported(),
  }
}

/**
 * Higher-order function to add haptic feedback to event handlers
 */
export function withHaptic<T extends (...args: any[]) => any>(
  handler: T,
  type: HapticType = 'medium'
): T {
  return ((...args: Parameters<T>) => {
    haptic(type)
    return handler(...args)
  }) as T
}
