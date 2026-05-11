import { createFileRoute } from '@tanstack/react-router'
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  type AgentEvent,
  type ChatMessage,
  type Citation,
  streamChat,
} from '#/lib/api'
import {
  type AssistantEntry,
  type Thread,
  type TimelineEvent,
  type Turn,
  useThreads,
} from '#/lib/threads'

export const Route = createFileRoute('/')({ component: ChatPage })

const SUGGESTIONS = [
  'How many international transactions are in the dataset?',
  'What are the common indicators of credit card fraud?',
  'Predict if this transaction is fraudulent: $1,250 electronics, international, 3am, 2-month-old account.',
  'Which transactions look suspicious and why? Show me the top 5 with explanations.',
]

const SIDEBAR_KEY = 'lovelytics.sidebar.open.v1'

function readSidebarOpen(): boolean {
  if (typeof window === 'undefined') return true
  try {
    const raw = window.localStorage.getItem(SIDEBAR_KEY)
    if (raw === null) return true
    return raw === '1'
  } catch {
    return true
  }
}

function ChatPage() {
  const {
    threads,
    active,
    activeId,
    createThread,
    selectThread,
    deleteThread,
    updateActiveTurns,
  } = useThreads()

  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() =>
    readSidebarOpen(),
  )
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? '1' : '0')
    } catch {
      // ignore
    }
  }, [sidebarOpen])

  const turns = active?.turns ?? []

  // Pin the scroll view to the latest entry while streaming.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [turns, activeId])

  const history = useMemo<ChatMessage[]>(
    () =>
      turns
        .filter((t) => t.status !== 'error')
        .map<ChatMessage>((t) =>
          t.role === 'user'
            ? { role: 'user', content: t.content }
            : { role: 'assistant', content: assistantPlainText(t) },
        ),
    [turns],
  )

  const send = useCallback(
    async (prompt: string) => {
      const trimmed = prompt.trim()
      if (!trimmed || streaming) return

      // Ensure there's an active thread; create one lazily on first send.
      let targetId = activeId
      if (!targetId) {
        targetId = createThread()
      }

      const userTurn: Turn = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        status: 'done',
      }
      const assistantId = crypto.randomUUID()
      const assistantTurn: Turn = {
        id: assistantId,
        role: 'assistant',
        entries: [],
        citations: [],
        status: 'streaming',
      }

      const nextMessages: ChatMessage[] = [
        ...history,
        { role: 'user', content: trimmed },
      ]
      updateActiveTurns((prev) => [...prev, userTurn, assistantTurn])
      setInput('')
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        for await (const event of streamChat(nextMessages, controller.signal)) {
          updateActiveTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId && t.role === 'assistant'
                ? applyEvent(t, event)
                : t,
            ),
          )
        }
        // Stream ended cleanly without a final event. Mark done.
        updateActiveTurns((prev) =>
          prev.map((t) =>
            t.id === assistantId && t.role === 'assistant' && t.status === 'streaming'
              ? { ...t, status: 'done' }
              : t,
          ),
        )
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          updateActiveTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId && t.role === 'assistant'
                ? { ...t, status: 'done' }
                : t,
            ),
          )
        } else {
          const message = err instanceof Error ? err.message : String(err)
          updateActiveTurns((prev) =>
            prev.map((t) =>
              t.id === assistantId && t.role === 'assistant'
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
    [history, streaming, activeId, createThread, updateActiveTurns],
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

  const onNewChat = useCallback(() => {
    if (streaming) return
    createThread()
    setInput('')
    composerRef.current?.focus()
  }, [createThread, streaming])

  return (
    <div className="flex h-dvh w-full bg-(--color-bg) text-(--color-ink)">
      <Sidebar
        open={sidebarOpen}
        threads={threads}
        activeId={activeId}
        onSelect={selectThread}
        onCreate={onNewChat}
        onDelete={deleteThread}
        streaming={streaming}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
        />

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 sm:px-6">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-8">
            {turns.length === 0 ? (
              <EmptyState
                onPick={(s) => void send(s)}
                disabled={streaming}
              />
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
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Event reducer — chronological assistant entries                            */
/* -------------------------------------------------------------------------- */

/**
 * Apply a streamed agent event to an assistant turn. Entries accumulate in
 * chronological order: incoming tokens extend (or open) the trailing text
 * segment; tool/subagent starts append a fresh event entry; ends mutate the
 * matching open event in place. This is what lets the rendered timeline
 * read top-to-bottom in the order things actually happened.
 */
function applyEvent(
  turn: Extract<Turn, { role: 'assistant' }>,
  event: AgentEvent,
): Turn {
  switch (event.kind) {
    case 'token': {
      const entries = [...turn.entries]
      const last = entries[entries.length - 1]
      if (last && last.kind === 'text') {
        entries[entries.length - 1] = {
          kind: 'text',
          content: last.content + event.delta,
        }
      } else {
        entries.push({ kind: 'text', content: event.delta })
      }
      return { ...turn, entries }
    }
    case 'tool_start':
      return {
        ...turn,
        entries: [
          ...turn.entries,
          {
            kind: 'event',
            event: {
              kind: 'tool',
              name: event.name,
              args: event.args,
              status: 'running',
            },
          },
        ],
      }
    case 'tool_end': {
      const entries = mutateMatchingEvent(turn.entries, (e) =>
        e.kind === 'tool' && e.name === event.name && e.status === 'running'
          ? { ...e, status: 'done', result: event.result }
          : null,
      )
      return { ...turn, entries }
    }
    case 'subagent_start':
      return {
        ...turn,
        entries: [
          ...turn.entries,
          {
            kind: 'event',
            event: {
              kind: 'subagent',
              name: event.name,
              task: event.task,
              status: 'running',
            },
          },
        ],
      }
    case 'subagent_end': {
      const entries = mutateMatchingEvent(turn.entries, (e) =>
        e.kind === 'subagent' && e.name === event.name && e.status === 'running'
          ? { ...e, status: 'done', result: event.result }
          : null,
      )
      return { ...turn, entries }
    }
    case 'citation':
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
    case 'final': {
      // Replace the final text segment with the canonical final content if
      // the model emitted one. Keeps tools/subagents in place above it.
      if (!event.content) return { ...turn, status: 'done' }
      const entries = [...turn.entries]
      const lastIdx = entries.length - 1
      if (lastIdx >= 0 && entries[lastIdx]?.kind === 'text') {
        entries[lastIdx] = { kind: 'text', content: event.content }
      } else {
        entries.push({ kind: 'text', content: event.content })
      }
      return { ...turn, entries, status: 'done' }
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

function mutateMatchingEvent(
  entries: AssistantEntry[],
  mutator: (e: TimelineEvent) => TimelineEvent | null,
): AssistantEntry[] {
  const next = [...entries]
  for (let i = next.length - 1; i >= 0; i--) {
    const entry = next[i]
    if (entry && entry.kind === 'event') {
      const updated = mutator(entry.event)
      if (updated) {
        next[i] = { kind: 'event', event: updated }
        break
      }
    }
  }
  return next
}

/**
 * Concatenate just the text segments of an assistant turn for use in the
 * history sent back to the API on the next request. The agent doesn't need
 * the tool transcripts replayed — it sees its own state.
 */
function assistantPlainText(turn: Turn): string {
  if (turn.role !== 'assistant') return ''
  return turn.entries
    .filter((e): e is { kind: 'text'; content: string } => e.kind === 'text')
    .map((e) => e.content)
    .join('')
}

/* -------------------------------------------------------------------------- */
/* Layout pieces                                                              */
/* -------------------------------------------------------------------------- */

function Header({
  sidebarOpen,
  onToggleSidebar,
}: {
  sidebarOpen: boolean
  onToggleSidebar: () => void
}) {
  return (
    <header className="flex items-center justify-between border-b border-(--color-line) px-4 py-3 sm:px-6">
      <div className="flex items-center gap-2.5">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
          className="grid h-7 w-7 place-items-center rounded-md border border-(--color-line) bg-(--color-surface) text-(--color-ink-soft) transition hover:border-(--color-accent)/40 hover:text-(--color-ink)"
        >
          <svg
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <div className="grid h-7 w-7 place-items-center rounded-md bg-(--color-accent-soft) ring-1 ring-(--color-accent)/40">
          <span className="font-mono text-sm font-semibold text-(--color-accent)">
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
        className="text-xs text-(--color-ink-soft) transition hover:text-(--color-ink)"
      >
        source
      </a>
    </header>
  )
}

function Sidebar({
  open,
  threads,
  activeId,
  onSelect,
  onCreate,
  onDelete,
  streaming,
}: {
  open: boolean
  threads: Thread[]
  activeId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
  streaming: boolean
}) {
  return (
    <aside
      className={[
        'flex h-dvh shrink-0 flex-col border-r border-(--color-line) bg-(--color-surface)/40 transition-[width] duration-200 ease-out',
        open ? 'w-64' : 'w-0',
      ].join(' ')}
      aria-hidden={!open}
    >
      <div className={['flex h-full flex-col overflow-hidden', open ? '' : 'invisible'].join(' ')}>
        <div className="flex items-center justify-between px-3 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-(--color-ink-muted)">
            Threads
          </div>
          <button
            type="button"
            onClick={onCreate}
            disabled={streaming}
            className="rounded-md border border-(--color-line) bg-(--color-surface) px-2 py-1 text-[11px] font-medium text-(--color-ink-soft) transition hover:border-(--color-accent)/40 hover:text-(--color-ink) disabled:cursor-not-allowed disabled:opacity-50"
          >
            + new
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-3">
          {threads.length === 0 ? (
            <div className="px-2 py-3 text-[11px] text-(--color-ink-muted)">
              No threads yet. Start one below.
            </div>
          ) : (
            <ul className="flex flex-col gap-0.5">
              {threads.map((t) => (
                <li key={t.id}>
                  <ThreadRow
                    thread={t}
                    active={t.id === activeId}
                    onSelect={() => onSelect(t.id)}
                    onDelete={() => onDelete(t.id)}
                  />
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </aside>
  )
}

function ThreadRow({
  thread,
  active,
  onSelect,
  onDelete,
}: {
  thread: Thread
  active: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  return (
    <div
      className={[
        'group flex items-center gap-1 rounded-md px-2 py-1.5 text-[12px] transition',
        active
          ? 'bg-(--color-accent-soft) text-(--color-ink)'
          : 'text-(--color-ink-soft) hover:bg-(--color-surface-2) hover:text-(--color-ink)',
      ].join(' ')}
    >
      <button
        type="button"
        onClick={onSelect}
        className="min-w-0 flex-1 truncate text-left"
        title={thread.title}
      >
        {thread.title}
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          if (window.confirm('Delete this thread?')) onDelete()
        }}
        aria-label="Delete thread"
        className="grid h-5 w-5 place-items-center rounded text-(--color-ink-muted) opacity-0 transition hover:bg-(--color-line) hover:text-(--color-danger) group-hover:opacity-100 focus:opacity-100"
      >
        <svg
          viewBox="0 0 24 24"
          width="12"
          height="12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="3 6 5 6 21 6" />
          <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
          <path d="M10 11v6M14 11v6" />
          <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
        </svg>
      </button>
    </div>
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
      <div className="grid h-12 w-12 place-items-center rounded-full bg-(--color-accent-soft) ring-1 ring-(--color-accent)/40">
        <span className="font-mono text-lg font-semibold text-(--color-accent)">
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

  const hasContent = turn.entries.length > 0
  return (
    <div className="rise-in flex flex-col gap-3">
      {hasContent || turn.status === 'streaming' ? (
        <div className="rounded-2xl rounded-bl-sm border border-(--color-line) bg-(--color-surface) px-4 py-3">
          {hasContent ? (
            <div className="flex flex-col gap-3">
              {turn.entries.map((entry, i) => (
                <EntryRenderer key={i} entry={entry} />
              ))}
              {turn.status === 'streaming' && <Thinking />}
            </div>
          ) : (
            <Thinking />
          )}
          {turn.status === 'error' && (
            <div className="mt-2 rounded-md border border-(--color-danger)/40 bg-(--color-danger)/10 px-3 py-2 text-xs text-(--color-danger)">
              {turn.error ?? 'Unknown error'}
            </div>
          )}
        </div>
      ) : null}
      {turn.citations.length > 0 && <Citations items={turn.citations} />}
    </div>
  )
}

function EntryRenderer({ entry }: { entry: AssistantEntry }) {
  if (entry.kind === 'text') {
    if (!entry.content) return null
    return (
      <div className="prose-chat min-w-0 text-sm leading-relaxed text-(--color-ink)">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            table: ({ node: _node, ...props }) => (
              <div className="table-wrap">
                <table {...props} />
              </div>
            ),
          }}
        >
          {entry.content}
        </ReactMarkdown>
      </div>
    )
  }
  return (
    <div className="rounded-lg border border-(--color-line) bg-(--color-surface-2)/60 px-3 py-2">
      <TimelineRow event={entry.event} />
    </div>
  )
}

function TimelineRow({ event }: { event: TimelineEvent }) {
  const running = event.status === 'running'
  const isSub = event.kind === 'subagent'
  const label = isSub ? `subagent · ${event.name}` : `tool · ${event.name}`
  const argsText = isSub
    ? event.task
    : (() => {
        try {
          return JSON.stringify(event.args, null, 2)
        } catch {
          return String(event.args)
        }
      })()
  const summary = isSub ? event.task : summariseArgs(event.args)
  const hasDetail = Boolean(argsText) || Boolean(event.result)

  return (
    <details className="group/row text-[12px]">
      <summary className="flex cursor-pointer list-none items-start gap-2.5 marker:hidden">
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
            {hasDetail && (
              <span className="ml-auto text-[10px] text-(--color-ink-muted) transition group-open/row:opacity-0">
                expand
              </span>
            )}
          </div>
          {summary && (
            <div className="mt-0.5 truncate font-mono text-[11px] text-(--color-ink-soft) group-open/row:hidden">
              {summary}
            </div>
          )}
        </div>
      </summary>
      {hasDetail && (
        <div className="ml-4 mt-1.5 flex flex-col gap-2">
          {argsText && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-(--color-ink-muted)">
                {isSub ? 'task' : 'arguments'}
              </div>
              <pre className="overflow-x-auto rounded-md border border-(--color-line) bg-(--color-surface-2) px-2.5 py-1.5 font-mono text-[11px] text-(--color-ink-soft) whitespace-pre-wrap break-words">
                {argsText}
              </pre>
            </div>
          )}
          {event.result && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-(--color-ink-muted)">
                result
              </div>
              <pre className="max-h-64 overflow-auto rounded-md border border-(--color-line) bg-(--color-surface-2) px-2.5 py-1.5 font-mono text-[11px] text-(--color-ink-soft) whitespace-pre-wrap break-words">
                {event.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </details>
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
    <div className="flex items-center gap-2 text-sm text-(--color-ink-muted)">
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
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-1.5">
        {streaming && <RunningIndicator />}
        <div className="flex items-end gap-2 rounded-2xl border border-(--color-line) bg-(--color-surface) px-3 py-2 transition focus-within:border-(--color-accent)/45">
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
      </div>
    </form>
  )
}

function RunningIndicator() {
  return (
    <div className="flex items-center gap-2 px-1 text-[11px] text-(--color-ink-soft)">
      <Spinner />
      <span className="font-mono">agent running…</span>
    </div>
  )
}

function Spinner() {
  return (
    <svg
      className="spinner"
      viewBox="0 0 24 24"
      width="12"
      height="12"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
    >
      <circle cx="12" cy="12" r="9" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" />
    </svg>
  )
}
