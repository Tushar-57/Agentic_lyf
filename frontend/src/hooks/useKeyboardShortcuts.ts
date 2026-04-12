import { useEffect, useCallback, useState } from 'react'
import { hapticButtonPress } from '@/lib/hapticFeedback'

export interface KeyboardShortcut {
  key: string
  modifier?: 'cmd' | 'ctrl' | 'alt' | 'shift'
  description: string
  action: () => void
  preventDefault?: boolean
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  const [showHelp, setShowHelp] = useState(false)

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Don't trigger shortcuts when typing in inputs
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement ||
      event.target instanceof HTMLSelectElement
    ) {
      return
    }

    for (const shortcut of shortcuts) {
      const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase()
      
      let modifierMatch = true
      if (shortcut.modifier) {
        switch (shortcut.modifier) {
          case 'cmd':
            modifierMatch = event.metaKey && !event.ctrlKey && !event.altKey
            break
          case 'ctrl':
            modifierMatch = event.ctrlKey && !event.metaKey && !event.altKey
            break
          case 'alt':
            modifierMatch = event.altKey && !event.metaKey && !event.ctrlKey
            break
          case 'shift':
            modifierMatch = event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey
            break
        }
      } else {
        // No modifier required - ensure no modifiers pressed
        modifierMatch = !event.metaKey && !event.ctrlKey && !event.altKey
      }

      if (keyMatch && modifierMatch) {
        if (shortcut.preventDefault !== false) {
          event.preventDefault()
        }
        
        // Haptic feedback on mobile
        hapticButtonPress()
        
        shortcut.action()
        return
      }
    }

    // Show help on Cmd/Ctrl + /
    if ((event.metaKey || event.ctrlKey) && event.key === '/') {
      event.preventDefault()
      setShowHelp(prev => !prev)
    }

    // Close help on Escape
    if (event.key === 'Escape' && showHelp) {
      setShowHelp(false)
    }
  }, [shortcuts, showHelp])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  return { showHelp, setShowHelp }
}

// Common shortcuts for the app
export function useAppKeyboardShortcuts(
  onViewChange: (view: string) => void,
  onToggleSettings: () => void,
  onToggleTheme?: () => void,
  isMac = navigator.platform.includes('Mac')
) {
  const shortcuts: KeyboardShortcut[] = [
    {
      key: 'k',
      modifier: isMac ? 'cmd' : 'ctrl',
      description: 'Open command palette',
      action: () => {
        // Dispatch custom event for command palette
        window.dispatchEvent(new CustomEvent('openCommandPalette'))
      },
    },
    {
      key: ',',
      modifier: isMac ? 'cmd' : 'ctrl',
      description: 'Open settings',
      action: onToggleSettings,
    },
    {
      key: '1',
      description: 'Go to Chat',
      action: () => onViewChange('chat'),
    },
    {
      key: '2',
      description: 'Go to Knowledge',
      action: () => onViewChange('knowledge'),
    },
    {
      key: '3',
      description: 'Go to Analytics',
      action: () => onViewChange('analytics'),
    },
    {
      key: '4',
      description: 'Go to Notifications',
      action: () => onViewChange('notifications'),
    },
    {
      key: '5',
      description: 'Go to Profile',
      action: () => onViewChange('profile'),
    },
    {
      key: 'd',
      modifier: isMac ? 'cmd' : 'ctrl',
      description: 'Toggle dark mode',
      action: () => onToggleTheme?.(),
      preventDefault: false,
    },
    {
      key: 'Escape',
      description: 'Close panels / Go back',
      action: () => {
        window.dispatchEvent(new CustomEvent('escapePressed'))
      },
    },
  ]

  return useKeyboardShortcuts(shortcuts)
}

// Hook for agent switching shortcuts
export function useAgentKeyboardShortcuts(
  onAgentChange: (agent: string) => void
) {
  const shortcuts: KeyboardShortcut[] = [
    {
      key: 'o',
      modifier: 'alt',
      description: 'Switch to Orchestrator',
      action: () => onAgentChange('orchestrator'),
    },
    {
      key: 'p',
      modifier: 'alt',
      description: 'Switch to Productivity',
      action: () => onAgentChange('productivity'),
    },
    {
      key: 'h',
      modifier: 'alt',
      description: 'Switch to Health',
      action: () => onAgentChange('health'),
    },
    {
      key: 'f',
      modifier: 'alt',
      description: 'Switch to Finance',
      action: () => onAgentChange('finance'),
    },
    {
      key: 's',
      modifier: 'alt',
      description: 'Switch to Scheduling',
      action: () => onAgentChange('scheduling'),
    },
    {
      key: 'j',
      modifier: 'alt',
      description: 'Switch to Journal',
      action: () => onAgentChange('journal'),
    },
  ]

  return useKeyboardShortcuts(shortcuts)
}

// KeyboardShortcut type is already exported above
