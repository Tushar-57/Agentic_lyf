import { getAgenticBridgeToken } from '@/lib/agenticBridgeSession'

const AGENTIC_LOCAL_ORIGIN_PATTERNS = [
  'http://127.0.0.1:8000',
];

const AGENTIC_API_ORIGIN = (import.meta.env.VITE_AGENTIC_API_ORIGIN as string | undefined)?.trim();
const AGENTIC_WS_ORIGIN = (import.meta.env.VITE_AGENTIC_WS_ORIGIN as string | undefined)?.trim();
const AGENTIC_PREFIX = (import.meta.env.VITE_AGENTIC_API_PREFIX as string) || '/agentic-api';

let fetchRewriteInstalled = false;

function joinOriginPath(origin: string, path: string): string {
  const normalizedOrigin = origin.replace(/\/+$/, '');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${normalizedOrigin}${normalizedPath}`;
}

function stripPrefix(path: string, prefix: string): string {
  if (!path.startsWith(prefix)) {
    return path;
  }

  const stripped = path.slice(prefix.length);
  return stripped.startsWith('/') ? stripped : `/${stripped}`;
}

function deriveWsOriginFromApi(apiOrigin?: string): string | undefined {
  if (!apiOrigin) {
    return undefined;
  }

  try {
    const parsed = new URL(apiOrigin);
    parsed.protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
    return parsed.origin;
  } catch {
    return undefined;
  }
}

function resolveWsOrigin(): string | undefined {
  if (AGENTIC_WS_ORIGIN) {
    return AGENTIC_WS_ORIGIN;
  }

  return deriveWsOriginFromApi(AGENTIC_API_ORIGIN);
}

function normalizeRuntimeUrl(url: string): string {
  if (typeof window === 'undefined') {
    return url;
  }

  try {
    const parsed = new URL(url, window.location.origin);
    if (parsed.origin === window.location.origin) {
      return `${parsed.pathname}${parsed.search}${parsed.hash}`;
    }
  } catch {
    return url;
  }

  return url;
}

function withPrefix(path: string): string {
  if (path.startsWith(AGENTIC_PREFIX)) {
    return path;
  }

  return `${AGENTIC_PREFIX}${path}`;
}

function rewriteUrl(url: string): string {
  const normalizedUrl = normalizeRuntimeUrl(url);

  if (AGENTIC_API_ORIGIN && normalizedUrl.startsWith(AGENTIC_PREFIX)) {
    return joinOriginPath(AGENTIC_API_ORIGIN, stripPrefix(normalizedUrl, AGENTIC_PREFIX));
  }

  for (const origin of AGENTIC_LOCAL_ORIGIN_PATTERNS) {
    if (normalizedUrl.startsWith(origin)) {
      if (AGENTIC_API_ORIGIN) {
        return joinOriginPath(AGENTIC_API_ORIGIN, normalizedUrl.slice(origin.length));
      }

      return withPrefix(normalizedUrl.slice(origin.length));
    }
  }

  if (normalizedUrl.startsWith('/api') || normalizedUrl === '/health') {
    if (AGENTIC_API_ORIGIN) {
      return joinOriginPath(AGENTIC_API_ORIGIN, normalizedUrl);
    }

    return withPrefix(normalizedUrl);
  }

  if (normalizedUrl.startsWith('/ws')) {
    const wsOrigin = resolveWsOrigin();
    if (wsOrigin) {
      return joinOriginPath(wsOrigin, normalizedUrl);
    }

    return withPrefix(normalizedUrl);
  }

  return normalizedUrl;
}

function isAgenticApiUrl(url: string): boolean {
  if (typeof window === 'undefined') {
    return url.startsWith('/api') || url.startsWith(AGENTIC_PREFIX);
  }

  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.pathname.startsWith('/api') || parsed.pathname.startsWith(AGENTIC_PREFIX);
  } catch {
    return url.startsWith('/api') || url.startsWith(AGENTIC_PREFIX);
  }
}

function buildRequestInit(rewrittenUrl: string, init?: RequestInit, fallbackHeaders?: HeadersInit): RequestInit {
  const headers = new Headers(init?.headers ?? fallbackHeaders ?? undefined)
  const bridgeToken = getAgenticBridgeToken()

  if (bridgeToken && isAgenticApiUrl(rewrittenUrl)) {
    headers.set('X-Agentic-Bridge-Token', bridgeToken)
  }

  return {
    ...init,
    headers,
  }
}

export function installApiRouting(): void {
  if (fetchRewriteInstalled || typeof window === 'undefined') {
    return;
  }

  fetchRewriteInstalled = true;
  const nativeFetch = window.fetch.bind(window);

  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    if (typeof input === 'string') {
      const rewritten = rewriteUrl(input)
      return nativeFetch(rewritten, buildRequestInit(rewritten, init));
    }

    if (input instanceof URL) {
      const rewritten = rewriteUrl(input.toString())
      return nativeFetch(rewritten, buildRequestInit(rewritten, init));
    }

    if (input instanceof Request) {
      const rewritten = rewriteUrl(input.url);
      const nextInit = buildRequestInit(rewritten, init, input.headers)
      if (rewritten === input.url) {
        return nativeFetch(input, nextInit);
      }

      const rewrittenRequest = new Request(rewritten, input);
      return nativeFetch(rewrittenRequest, nextInit);
    }

    return nativeFetch(input, init);
  };
}
