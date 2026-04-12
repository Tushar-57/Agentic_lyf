import { useRef, useCallback, useEffect } from 'react'

interface FetchState {
  abortController: AbortController | null
  isLoading: boolean
  pendingPromise: Promise<unknown> | null
}

interface UseSafeFetchOptions {
  deduplicate?: boolean
}

export function useSafeFetch(options: UseSafeFetchOptions = {}) {
  const { deduplicate = true } = options
  const stateRef = useRef<FetchState>({
    abortController: null,
    isLoading: false,
    pendingPromise: null,
  })

  const cancelPending = useCallback(() => {
    if (stateRef.current.abortController) {
      stateRef.current.abortController.abort()
      stateRef.current.abortController = null
    }
    stateRef.current.pendingPromise = null
  }, [])

  const safeFetch = useCallback(async <T>(
    url: string,
    fetchOptions: RequestInit = {},
    onSuccess: (data: T) => void,
    onError: (error: Error) => void,
    onFinally?: () => void
  ): Promise<void> => {
    // Cancel any pending request
    cancelPending()

    // Check if already loading (deduplication)
    if (deduplicate && stateRef.current.isLoading) {
      return
    }

    const abortController = new AbortController()
    stateRef.current.abortController = abortController
    stateRef.current.isLoading = true

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        signal: abortController.signal,
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json() as T
      
      // Only update state if not aborted
      if (!abortController.signal.aborted) {
        onSuccess(data)
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        // Request was cancelled, don't treat as error
        return
      }
      
      if (!abortController.signal.aborted) {
        onError(error instanceof Error ? error : new Error(String(error)))
      }
    } finally {
      if (stateRef.current.abortController === abortController) {
        stateRef.current.abortController = null
        stateRef.current.isLoading = false
      }
      onFinally?.()
    }
  }, [cancelPending, deduplicate])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelPending()
    }
  }, [cancelPending])

  return {
    safeFetch,
    cancelPending,
    get isLoading() {
      return stateRef.current.isLoading
    },
  }
}

export function useSequentialFetch() {
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map())

  const cancelAll = useCallback(() => {
    abortControllersRef.current.forEach((controller) => {
      controller.abort()
    })
    abortControllersRef.current.clear()
  }, [])

  const cancelRequest = useCallback((key: string) => {
    const controller = abortControllersRef.current.get(key)
    if (controller) {
      controller.abort()
      abortControllersRef.current.delete(key)
    }
  }, [])

  const fetchWithKey = useCallback(async <T>(
    key: string,
    url: string,
    fetchOptions: RequestInit = {}
  ): Promise<T | null> => {
    // Cancel existing request with same key
    cancelRequest(key)

    const abortController = new AbortController()
    abortControllersRef.current.set(key, abortController)

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        signal: abortController.signal,
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json() as T
      return data
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return null
      }
      throw error
    } finally {
      abortControllersRef.current.delete(key)
    }
  }, [cancelRequest])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAll()
    }
  }, [cancelAll])

  return {
    fetchWithKey,
    cancelRequest,
    cancelAll,
  }
}

export default useSafeFetch
