const BRIDGE_TOKEN_STORAGE_KEY = 'agentic.bridge.token'
const BRIDGE_USER_STORAGE_KEY = 'agentic.bridge.user'
const CONVERSATION_PREFIX = 'agentic.conversation'

function getStoredBridgeUserFallback(): string | null {
  try {
    return localStorage.getItem(BRIDGE_USER_STORAGE_KEY)
  } catch {
    return null
  }
}

function setStoredBridgeUserFallback(value: string): void {
  try {
    localStorage.setItem(BRIDGE_USER_STORAGE_KEY, value)
  } catch {
    // Ignore storage failures (privacy mode, disabled storage, etc.)
  }
}

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const segments = token.split('.')
    if (segments.length < 2) {
      return null
    }

    const base64Url = segments[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const padded = `${base64}${'='.repeat((4 - (base64.length % 4)) % 4)}`
    const payload = atob(padded)
    return JSON.parse(payload) as Record<string, unknown>
  } catch {
    return null
  }
}

function normalizeUserKey(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }

  const normalized = value
    .toString()
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '')

  if (!normalized) {
    return null
  }

  return normalized.slice(0, 128)
}

function deriveUserFromToken(token: string): string | null {
  const payload = decodeJwtPayload(token)
  if (!payload) {
    return null
  }

  const raw = payload.uid ?? payload.user_id ?? payload.sub ?? payload.email
  return normalizeUserKey(raw != null ? String(raw) : null)
}

function sanitizeBridgeParamsFromUrl(): void {
  const currentUrl = new URL(window.location.href)
  let updated = false

  if (currentUrl.searchParams.has('bridge_token')) {
    currentUrl.searchParams.delete('bridge_token')
    updated = true
  }

  if (currentUrl.searchParams.has('bridge_user')) {
    currentUrl.searchParams.delete('bridge_user')
    updated = true
  }

  const hashFragment = currentUrl.hash.startsWith('#') ? currentUrl.hash.slice(1) : currentUrl.hash
  if (hashFragment) {
    const hashParams = new URLSearchParams(hashFragment)
    if (hashParams.has('bridge_token')) {
      hashParams.delete('bridge_token')
      updated = true
    }
    if (hashParams.has('bridge_user')) {
      hashParams.delete('bridge_user')
      updated = true
    }

    const nextHash = hashParams.toString()
    currentUrl.hash = nextHash ? `#${nextHash}` : ''
  }

  if (updated) {
    window.history.replaceState({}, '', `${currentUrl.pathname}${currentUrl.search}${currentUrl.hash}`)
  }
}

export function bootstrapAgenticBridgeSession(): void {
  if (typeof window === 'undefined') {
    return
  }

  const currentUrl = new URL(window.location.href)
  const hashParams = new URLSearchParams(currentUrl.hash.startsWith('#') ? currentUrl.hash.slice(1) : currentUrl.hash)

  const bridgeTokenFromUrl = currentUrl.searchParams.get('bridge_token') || hashParams.get('bridge_token')
  const bridgeUserFromUrl = currentUrl.searchParams.get('bridge_user') || hashParams.get('bridge_user')

  if (bridgeTokenFromUrl) {
    sessionStorage.setItem(BRIDGE_TOKEN_STORAGE_KEY, bridgeTokenFromUrl)
  }

  const derivedUserKey = normalizeUserKey(bridgeUserFromUrl) || (bridgeTokenFromUrl ? deriveUserFromToken(bridgeTokenFromUrl) : null)
  if (derivedUserKey) {
    sessionStorage.setItem(BRIDGE_USER_STORAGE_KEY, derivedUserKey)
    setStoredBridgeUserFallback(derivedUserKey)
  }

  if (bridgeTokenFromUrl || bridgeUserFromUrl) {
    sanitizeBridgeParamsFromUrl()
  }
}

export function getAgenticBridgeToken(): string | null {
  if (typeof window === 'undefined') {
    return null
  }

  return sessionStorage.getItem(BRIDGE_TOKEN_STORAGE_KEY)
}

export function getAgenticBridgeUserKey(): string {
  if (typeof window === 'undefined') {
    return 'single_user'
  }

  const existing =
    normalizeUserKey(sessionStorage.getItem(BRIDGE_USER_STORAGE_KEY))
    || normalizeUserKey(getStoredBridgeUserFallback())

  if (existing) {
    sessionStorage.setItem(BRIDGE_USER_STORAGE_KEY, existing)
    return existing
  }

  const token = getAgenticBridgeToken()
  const derived = token ? deriveUserFromToken(token) : null
  if (derived) {
    sessionStorage.setItem(BRIDGE_USER_STORAGE_KEY, derived)
    setStoredBridgeUserFallback(derived)
    return derived
  }

  return 'single_user'
}

export function getOrCreateConversationId(): string {
  if (typeof window === 'undefined') {
    return 'single_user-conversation'
  }

  const userKey = getAgenticBridgeUserKey()
  const storageKey = `${CONVERSATION_PREFIX}.${userKey}`
  const existing = sessionStorage.getItem(storageKey)
  if (existing) {
    return existing
  }

  const generated = `${userKey}-${crypto.randomUUID()}`
  sessionStorage.setItem(storageKey, generated)
  return generated
}
