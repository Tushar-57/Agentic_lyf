const EMBEDDINGS_CACHE_KEY = 'agentic:embeddings:visualization:v1'
const EMBEDDINGS_CACHE_TTL_MS = 5 * 60 * 1000

export type EmbeddingVisualizationCacheEntry = {
  cachedAt: number
  data: any[]
}

export function readEmbeddingsCache(maxAgeMs: number = EMBEDDINGS_CACHE_TTL_MS): any[] | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    const raw = window.sessionStorage.getItem(EMBEDDINGS_CACHE_KEY)
    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as EmbeddingVisualizationCacheEntry
    if (!parsed || !Array.isArray(parsed.data) || typeof parsed.cachedAt !== 'number') {
      return null
    }

    if (Date.now() - parsed.cachedAt > maxAgeMs) {
      window.sessionStorage.removeItem(EMBEDDINGS_CACHE_KEY)
      return null
    }

    return parsed.data
  } catch {
    return null
  }
}

export function writeEmbeddingsCache(data: any[]): void {
  if (typeof window === 'undefined' || !Array.isArray(data)) {
    return
  }

  try {
    const payload: EmbeddingVisualizationCacheEntry = {
      cachedAt: Date.now(),
      data,
    }

    window.sessionStorage.setItem(EMBEDDINGS_CACHE_KEY, JSON.stringify(payload))
  } catch {
    // Ignore cache write failures and proceed without cache.
  }
}

export async function prefetchEmbeddingsVisualization(): Promise<any[] | null> {
  try {
    const response = await fetch('/api/knowledge/embeddings/visualization', {
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache',
      },
    })
    if (!response.ok) {
      return null
    }

    const data = (await response.json()) as any[]
    if (!Array.isArray(data)) {
      return null
    }

    writeEmbeddingsCache(data)
    return data
  } catch {
    return null
  }
}
