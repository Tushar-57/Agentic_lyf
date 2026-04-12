/**
 * Design Tokens - Centralized Design System Configuration
 * 
 * This file contains all design tokens to ensure consistency across the application.
 * All hardcoded values should eventually reference these tokens.
 */

// ============================================================================
// BORDER RADIUS
// ============================================================================

export const borderRadius = {
  none: '0',
  sm: '0.375rem',    // 6px
  md: '0.5rem',      // 8px
  lg: '0.75rem',     // 12px
  xl: '1rem',        // 16px
  '2xl': '1.5rem',   // 24px
  '3xl': '2rem',     // 32px - replaces rounded-[28px] and rounded-[32px]
  full: '9999px',
} as const

// ============================================================================
// SHADOWS
// ============================================================================

export const shadows = {
  sm: '0 1px 2px 0 rgb(15 23 42 / 0.05)',
  md: '0 4px 6px -1px rgb(15 23 42 / 0.1), 0 2px 4px -2px rgb(15 23 42 / 0.1)',
  lg: '0 10px 15px -3px rgb(15 23 42 / 0.1), 0 4px 6px -4px rgb(15 23 42 / 0.1)',
  xl: '0 20px 25px -5px rgb(15 23 42 / 0.1), 0 8px 10px -6px rgb(15 23 42 / 0.1)',
  '2xl': '0 25px 50px -12px rgb(15 23 42 / 0.25)',
  
  // Custom panel shadows (replacing magic numbers)
  card: '0 4px 6px -1px rgba(15, 23, 42, 0.1), 0 2px 4px -2px rgba(15, 23, 42, 0.1)',
  panel: '0 24px 70px -50px rgba(15, 23, 42, 0.7)',
  elevated: '0 28px 80px -48px rgba(15, 23, 42, 0.65)',
  
  // Glow effects
  'glow-sm': '0 0 10px rgba(20, 184, 166, 0.3)',
  'glow-md': '0 0 20px rgba(20, 184, 166, 0.4)',
  'glow-lg': '0 0 30px rgba(20, 184, 166, 0.5)',
  
  // Colored glows
  'glow-cyan': '0 0 20px rgba(45, 212, 191, 0.4)',
  'glow-amber': '0 0 20px rgba(245, 158, 11, 0.4)',
} as const

// ============================================================================
// SPACING
// ============================================================================

export const spacing = {
  // Touch target minimums (44px for mobile accessibility)
  touchTarget: {
    sm: '2rem',      // 32px - minimum for desktop
    md: '2.75rem',   // 44px - minimum for mobile (WCAG 2.5.5)
    lg: '3rem',      // 48px - recommended for mobile
  },
  
  // Safe area insets (standardized)
  safeArea: {
    top: 'env(safe-area-inset-top)',
    bottom: 'env(safe-area-inset-bottom)',
    left: 'env(safe-area-inset-left)',
    right: 'env(safe-area-inset-right)',
  },
  
  // Content insets for mobile
  mobile: {
    bottomNav: '5.2rem',  // Height of mobile bottom nav + padding
    header: '3.5rem',     // Height of mobile header
  },
} as const

// ============================================================================
// BREAKPOINTS (Tailwind-aligned)
// ============================================================================

export const breakpoints = {
  sm: 640,   // Tailwind sm
  md: 768,   // Tailwind md  
  lg: 1024,  // Tailwind lg
  xl: 1280,  // Tailwind xl
  '2xl': 1536, // Tailwind 2xl
} as const

// Use these instead of hardcoded values
export const breakpointClasses = {
  mobile: 'md:hidden',      // Show only on mobile
  tablet: 'hidden md:block', // Show on tablet+
  desktop: 'hidden lg:block', // Show on desktop
} as const

// ============================================================================
// COLORS
// ============================================================================

// Semantic color roles (using CSS variables where possible)
export const colors = {
  // Brand gradient colors
  brand: {
    teal: '#14b8a6',
    cyan: '#06b6d4',
    amber: '#f59e0b',
  },
  
  // Status colors
  status: {
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
  },
  
  // Agent colors
  agents: {
    orchestrator: { from: '#0d9488', to: '#06b6d4' },
    productivity: { from: '#f59e0b', to: '#f97316' },
    health: { from: '#f43f5e', to: '#ef4444' },
    finance: { from: '#10b981', to: '#22c55e' },
    scheduling: { from: '#0ea5e9', to: '#3b82f6' },
    journal: { from: '#475569', to: '#64748b' },
  },
} as const

// ============================================================================
// ANIMATION
// ============================================================================

export const animation = {
  // Durations
  duration: {
    fast: 150,
    normal: 200,
    slow: 300,
    slower: 500,
  },
  
  // Easing
  easing: {
    default: [0.4, 0, 0.2, 1],
    spring: { type: 'spring', damping: 25, stiffness: 200 },
    bounce: { type: 'spring', damping: 10, stiffness: 100 },
  },
  
  // Framer Motion transition presets
  transitions: {
    default: { duration: 0.3, ease: [0.4, 0, 0.2, 1] },
    fast: { duration: 0.2, ease: 'easeOut' },
    spring: { type: 'spring', damping: 25, stiffness: 200 },
  },
  
  // Stagger children
  stagger: {
    fast: 0.05,
    normal: 0.1,
    slow: 0.15,
  },
} as const

// ============================================================================
// TYPOGRAPHY
// ============================================================================

export const typography = {
  fontFamily: {
    sans: ['Space Grotesk', 'Manrope', 'system-ui', 'sans-serif'],
    mono: ['IBM Plex Mono', 'JetBrains Mono', 'monospace'],
  },
  
  fontSize: {
    xs: ['0.75rem', { lineHeight: '1rem' }],
    sm: ['0.875rem', { lineHeight: '1.25rem' }],
    base: ['1rem', { lineHeight: '1.5rem' }],
    lg: ['1.125rem', { lineHeight: '1.75rem' }],
    xl: ['1.25rem', { lineHeight: '1.75rem' }],
    '2xl': ['1.5rem', { lineHeight: '2rem' }],
  },
} as const

// ============================================================================
// Z-INDEX SCALE
// ============================================================================

export const zIndex = {
  hide: -1,
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800,
} as const

// ============================================================================
// COMPONENT-SPECIFIC TOKENS
// ============================================================================

export const componentTokens = {
  // Sidebar
  sidebar: {
    width: {
      expanded: 280,
      collapsed: 80,
    },
    mobile: {
      width: 'min(88vw, 20rem)',
    },
  },
  
  // Chat
  chat: {
    bubble: {
      maxWidth: {
        mobile: '92%',
        desktop: '80%',
      },
    },
  },
  
  // Mobile navigation
  mobileNav: {
    height: '5.2rem',
    itemHeight: '2.75rem', // 44px touch target
  },
} as const

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Get shadow value by key
 */
export function getShadow(key: keyof typeof shadows): string {
  return shadows[key]
}

/**
 * Get border radius value by key
 */
export function getBorderRadius(key: keyof typeof borderRadius): string {
  return borderRadius[key]
}

/**
 * Get component token by path
 */
export function getComponentToken(
  component: keyof typeof componentTokens,
  path: string
): any {
  const tokens = componentTokens[component]
  return path.split('.').reduce((obj: any, key) => obj?.[key], tokens)
}

// Default export for convenience
export default {
  borderRadius,
  shadows,
  spacing,
  breakpoints,
  colors,
  animation,
  typography,
  zIndex,
  componentTokens,
}
