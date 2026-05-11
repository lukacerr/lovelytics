/**
 * Streaming client for the `POST /chat` SSE endpoint.
 *
 * The browser `EventSource` API only supports GET, so we hand-roll the SSE
 * parser over `fetch` + a `ReadableStream`. Frames follow the SSE shape:
 *
 *   event: <name>
 *   data: <json>
 *   \n
 *
 * Backed by the typed event taxonomy defined in `app/sse.py` / README §5.5.
 */

export type ChatMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export type Citation = {
  source: string
  header_path: string
  snippet: string
}

export type AgentEvent =
  | { kind: 'token'; delta: string }
  | { kind: 'tool_start'; name: string; args: Record<string, unknown> }
  | { kind: 'tool_end'; name: string; result: string }
  | { kind: 'subagent_start'; name: string; task: string }
  | { kind: 'subagent_end'; name: string; result: string }
  | { kind: 'citation'; citation: Citation }
  | { kind: 'final'; content: string }
  | { kind: 'error'; message: string; type: string }

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  'http://localhost:8000'

/**
 * Stream a chat turn. Yields one decoded `AgentEvent` per SSE frame.
 *
 * The caller is responsible for assembling tokens, threading tool/subagent
 * events into the UI timeline, etc. We just parse the wire format.
 */
export async function* streamChat(
  messages: ChatMessage[],
  signal?: AbortSignal,
): AsyncGenerator<AgentEvent> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  })

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '')
    throw new Error(
      `chat request failed (${response.status}): ${text || response.statusText}`,
    )
  }

  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += value

      // SSE frames are separated by a blank line. Process complete frames
      // and leave any partial frame in the buffer.
      let sep = buffer.indexOf('\n\n')
      while (sep !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        const event = parseFrame(frame)
        if (event) yield event
        sep = buffer.indexOf('\n\n')
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function parseFrame(frame: string): AgentEvent | null {
  let eventName = 'message'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }
  if (dataLines.length === 0) return null
  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
  } catch {
    return null
  }
  return toAgentEvent(eventName, payload)
}

function toAgentEvent(
  name: string,
  payload: Record<string, unknown>,
): AgentEvent | null {
  switch (name) {
    case 'token':
      return { kind: 'token', delta: String(payload.delta ?? '') }
    case 'tool_start':
      return {
        kind: 'tool_start',
        name: String(payload.name ?? ''),
        args: (payload.args as Record<string, unknown>) ?? {},
      }
    case 'tool_end':
      return {
        kind: 'tool_end',
        name: String(payload.name ?? ''),
        result: String(payload.result ?? ''),
      }
    case 'subagent_start':
      return {
        kind: 'subagent_start',
        name: String(payload.name ?? ''),
        task: String(payload.task ?? ''),
      }
    case 'subagent_end':
      return {
        kind: 'subagent_end',
        name: String(payload.name ?? ''),
        result: String(payload.result ?? ''),
      }
    case 'citation':
      return {
        kind: 'citation',
        citation: {
          source: String(payload.source ?? ''),
          header_path: String(payload.header_path ?? ''),
          snippet: String(payload.snippet ?? ''),
        },
      }
    case 'final':
      return { kind: 'final', content: String(payload.content ?? '') }
    case 'error':
      return {
        kind: 'error',
        message: String(payload.message ?? ''),
        type: String(payload.type ?? 'Error'),
      }
    default:
      return null
  }
}
