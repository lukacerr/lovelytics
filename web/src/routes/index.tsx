import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  type AgentEvent,
  type ChatMessage,
  type Citation,
  streamChat,
} from '#/lib/api'

export const Route = createFileRoute('/')({ component: ChatPage })

/**
 * One turn worth of agent activity. Tokens accumulate into `content`; tool
 * calls, subagent dispatches and citations are appended to `events` in
 * arrival order so the UI can render a faithful timeline of the run.
 */
type Turn = {
  id: string
  role: 'user' | 'assistant'
  content: string
  events: TimelineEvent[]
  citations: Citation[]
  status: 'streaming' | 'done' | 'error'
  error?: string
}

type TimelineEvent =
  | { kind: 'tool'; name: string; args: Record<string, unknown>; result?: string; status: 'running' | 'done' }
  | { kind: 'subagent'; name: string; task: string; result?: string; status: 'running' | 'done' }

const SUGGESTIONS = [
  'How many international transactions are in the dataset?',
  'What are the common indicators of credit card fraud?',
  'Predict if this transaction is fraudulent: $1,250 electronics, international, 3am, 2-month-old account.',
  'Which transactions look suspicious and why? Show me the top 5 with explanations.',
]

function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLTextAreaElement | null>(null)

  // Keep the view pinned to the latest message when streaming.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [turns])

  const history = useMemo<ChatMessage[]>(
    () =>
      turns
        .filter((t) => t.status !== 'error')
        .map((t) => ({ role: t.role, content: t.content })),
    [turns],
  )

  const send = useCallback(
    async (prompt: string) => {
      if (!prompt.trim() || streaming) return

      const userTurn: Turn = {
        id: crypto.randomUUID(),
        role: 'user',
        content: prompt.trim(),
        events: [],
        citations: [],
        status: 'done',
      }
      const assistantId = crypto.randomUUID()
      const assistantTurn: Turn = {
        id: assistantId,
        role: 'assistant',
        content: '',
        events: [],
        citations: [],
        status: 'streaming',
      }

      const nextMessages: ChatMessage[] = [
        ...history,
        { role: 'user', content: userTurn.content },
      ]
      setTurns((prev) => [...prev, userTurn, assistantTurn])
      setInput('')
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        for await (const event of streamChat(nextMessages, controller.signal)) {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId ? applyEvent(t, event) : t,
            ),
          )
        }
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId
                ? { ...t, status: 'done' }
                : t,
            ),
          )
        } else {
          const message = err instanceof Error ? err.message : String(err)
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId
                ? { ...t, status: 'error', error: message }
                : t,
            ),
          )
        }
      } finally {
        abortRef.current = null
        setStreaming(false)
        composerRef.current?.focus()
      }
    },
    [history, streaming],
  )

  const abort = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    void send(input)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send(input)
    }
  }

  return (
    <div className="flex h-dvh w-full flex-col bg-(--color-bg) text-(--color-ink)">
      <Header />

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 sm:px-6"
      >
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-8">
          {turns.length === 0 ? (
            <EmptyState onPick={(s) => void send(s)} disabled={streaming} />
          ) : (
            turns.map((t) => <TurnBubble key={t.id} turn={t} />)
          )}
        </div>
      </div>

      <Composer
        ref={composerRef}
        value={input}
        onChange={setInput}
        onSubmit={onSubmit}
        onKeyDown={onKeyDown}
        streaming={streaming}
        onAbort={abort}
      />
    </div>
  )
}

function applyEvent(turn: Turn, event: AgentEvent): Turn {
  switch (event.kind) {
    case 'token':
      return { ...turn, content: turn.content + event.delta }
    case 'tool_start':
      return {
        ...turn,
        events: [
          ...turn.events,
          {
            kind: 'tool',
            name: event.name,
            args: event.args,
            status: 'running',
          },
        ],
      }
    case 'tool_end': {
      // Match the most recent running tool with the same name.
      const events = [...turn.events]
      for (let i = events.length - 1; i >= 0; i--) {
        const e = events[i]
        if (e && e.kind === 'tool' && e.name === event.name && e.status === 'running') {
          events[i] = { ...e, status: 'done', result: event.result }
          break
        }
      }
      return { ...turn, events }
    }
    case 'subagent_start':
      return {
        ...turn,
        events: [
          ...turn.events,
          {
            kind: 'subagent',
            name: event.name,
            task: event.task,
            status: 'running',
          },
        ],
      }
    case 'subagent_end': {
      const events = [...turn.events]
      for (let i = events.length - 1; i >= 0; i--) {
        const e = events[i]
        if (
          e &&
          e.kind === 'subagent' &&
          e.name === event.name &&
          e.status === 'running'
        ) {
          events[i] = { ...e, status: 'done', result: event.result }
          break
        }
      }
      return { ...turn, events }
    }
    case 'citation':
      // De-dupe on source+header_path.
      if (
        turn.citations.some(
          (c) =>
            c.source === event.citation.source &&
            c.header_path === event.citation.header_path,
        )
      ) {
        return turn
      }
      return { ...turn, citations: [...turn.citations, event.citation] }
    case 'final':
      return {
        ...turn,
        content: event.content || turn.content,
        status: 'done',
      }
    case 'error':
      return {
        ...turn,
        status: 'error',
        error: `${event.type}: ${event.message}`,
      }
    default:
      return turn
  }
}

/* -------------------------------------------------------------------------- */
/* Layout pieces                                                              */
/* -------------------------------------------------------------------------- */

function Header() {
  return (
    <header className="flex items-center justify-between border-b border-(--color-line) px-4 py-3 sm:px-6">
      <div className="flex items-center gap-2.5">
        <div className="h-7 w-7 rounded-md bg-(--color-accent-soft) ring-1 ring-(--color-accent)/40 grid place-items-center">
          <span className="text-(--color-accent) font-mono text-sm font-semibold">
            L
          </span>
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight">
            Lovelytics
          </div>
          <div className="text-[11px] text-(--color-ink-muted)">
            Financial fraud analyst assistant
          </div>
        </div>
      </div>
      <a
        href="https://github.com/lukacerr/lovelytics"
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs text-(--color-ink-soft) hover:text-(--color-ink) transition"
      >
        source
      </a>
    </header>
  )
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (s: string) => void
  disabled: boolean
}) {
  return (
    <div className="rise-in flex flex-col items-center gap-6 pt-12 text-center sm:pt-20">
      <div className="h-12 w-12 rounded-full bg-(--color-accent-soft) ring-1 ring-(--color-accent)/40 grid place-items-center">
        <span className="text-(--color-accent) font-mono text-lg font-semibold">
          L
        </span>
      </div>
      <div className="max-w-xl space-y-2">
        <h1 className="text-xl font-semibold tracking-tight">
          Ask anything about the transactions, models, or knowledge base.
        </h1>
        <p className="text-sm text-(--color-ink-soft)">
          The agent can run pandas analytics, score fraud or purchase amounts
          with the ML models, and cite the financial-document KB. Try a
          starting question:
        </p>
      </div>
      <div className="grid w-full gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => onPick(s)}
            className="rounded-lg border border-(--color-line) bg-(--color-surface) px-4 py-3 text-left text-sm text-(--color-ink-soft) transition hover:border-(--color-accent)/40 hover:bg-(--color-surface-2) hover:text-(--color-ink) disabled:cursor-not-allowed disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

function TurnBubble({ turn }: { turn: Turn }) {
  if (turn.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm border border-(--color-accent)/35 bg-(--color-accent-soft) px-4 py-2.5 text-sm leading-relaxed text-(--color-ink)">
          {turn.content}
        </div>
      </div>
    )
  }

  return (
    <div className="rise-in flex flex-col gap-3">
      {turn.events.length > 0 && <Timeline events={turn.events} />}
      <div className="rounded-2xl rounded-bl-sm border border-(--color-line) bg-(--color-surface) px-4 py-3">
        {turn.content ? (
          <div className="prose-chat whitespace-pre-wrap text-sm leading-relaxed text-(--color-ink)">
            {turn.content}
          </div>
        ) : turn.status === 'streaming' ? (
          <Thinking />
        ) : null}
        {turn.status === 'error' && (
          <div className="mt-2 rounded-md border border-(--color-danger)/40 bg-(--color-danger)/10 px-3 py-2 text-xs text-(--color-danger)">
            {turn.error ?? 'Unknown error'}
          </div>
        )}
      </div>
      {turn.citations.length > 0 && <Citations items={turn.citations} />}
    </div>
  )
}

function Timeline({ events }: { events: TimelineEvent[] }) {
  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-(--color-line) bg-(--color-surface)/60 px-3 py-2.5">
      {events.map((e, i) => (
        <TimelineRow key={i} event={e} />
      ))}
    </div>
  )
}

function TimelineRow({ event }: { event: TimelineEvent }) {
  const running = event.status === 'running'
  const isSub = event.kind === 'subagent'
  const label = isSub ? `subagent · ${event.name}` : `tool · ${event.name}`
  const detail =
    isSub
      ? event.task
      : summariseArgs(event.args)

  return (
    <div className="flex items-start gap-2.5 text-[12px]">
      <span
        className={[
          'mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full',
          running ? 'bg-(--color-accent) pulse-dot' : 'bg-(--color-ink-muted)',
        ].join(' ')}
      />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2">
          <span className="font-mono text-(--color-accent-strong)">
            {label}
          </span>
          {running && (
            <span className="text-(--color-ink-muted)">running…</span>
          )}
        </div>
        {detail && (
          <div className="mt-0.5 truncate font-mono text-[11px] text-(--color-ink-soft)">
            {detail}
          </div>
        )}
      </div>
    </div>
  )
}

function summariseArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args)
  if (entries.length === 0) return ''
  const first = entries[0]
  if (!first) return ''
  const [key, value] = first
  const valueText =
    typeof value === 'string'
      ? value
      : (() => {
          try {
            return JSON.stringify(value)
          } catch {
            return String(value)
          }
        })()
  return `${key}: ${valueText}`
}

function Citations({ items }: { items: Citation[] }) {
  return (
    <details className="group rounded-xl border border-(--color-line) bg-(--color-surface)/60 px-3 py-2">
      <summary className="cursor-pointer list-none text-xs font-medium text-(--color-ink-soft) marker:hidden">
        <span className="text-(--color-accent-strong)">
          {items.length} citation{items.length === 1 ? '' : 's'}
        </span>
        <span className="ml-2 text-(--color-ink-muted) transition group-open:opacity-0">
          (click to expand)
        </span>
      </summary>
      <ul className="mt-2 flex flex-col gap-1.5">
        {items.map((c, i) => (
          <li key={`${c.source}-${c.header_path}-${i}`} className="text-[12px]">
            <span className="font-mono text-(--color-ink)">{c.source}</span>
            {c.header_path && (
              <span className="text-(--color-ink-muted)"> · {c.header_path}</span>
            )}
            {c.snippet && (
              <div className="mt-0.5 text-(--color-ink-soft)">{c.snippet}</div>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}

function Thinking() {
  return (
    <div className="flex items-center gap-2 text-(--color-ink-muted) text-sm">
      <span className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-(--color-accent)" />
      <span className="font-mono text-xs">thinking…</span>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Composer                                                                   */
/* -------------------------------------------------------------------------- */

type ComposerProps = {
  value: string
  onChange: (v: string) => void
  onSubmit: (e: React.FormEvent) => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  streaming: boolean
  onAbort: () => void
}

const Composer = ({
  ref,
  value,
  onChange,
  onSubmit,
  onKeyDown,
  streaming,
  onAbort,
}: ComposerProps & { ref?: React.Ref<HTMLTextAreaElement> }) => {
  return (
    <form
      onSubmit={onSubmit}
      className="border-t border-(--color-line) bg-(--color-bg) px-4 py-3 sm:px-6"
    >
      <div className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-2xl border border-(--color-line) bg-(--color-surface) px-3 py-2 focus-within:border-(--color-accent)/45 transition">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask the analyst…  (Enter to send, Shift+Enter for newline)"
          className="flex-1 resize-none bg-transparent px-1.5 py-1.5 text-sm leading-relaxed text-(--color-ink) placeholder:text-(--color-ink-muted) focus:outline-none"
          style={{ maxHeight: '12rem' }}
        />
        {streaming ? (
          <button
            type="button"
            onClick={onAbort}
            className="rounded-lg border border-(--color-line-strong) bg-(--color-surface-2) px-3 py-1.5 text-xs font-medium text-(--color-ink-soft) transition hover:border-(--color-danger)/40 hover:text-(--color-danger)"
          >
            stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            className="rounded-lg bg-(--color-accent) px-3 py-1.5 text-xs font-semibold text-(--color-bg) transition hover:bg-(--color-accent-strong) disabled:cursor-not-allowed disabled:opacity-40"
          >
            send
          </button>
        )}
      </div>
    </form>
  )
}
