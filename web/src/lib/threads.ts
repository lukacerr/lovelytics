/**
 * LocalStorage-backed thread persistence for the chat UI.
 *
 * Threads are stored as a JSON array under a single key. We persist the
 * raw chronological turns (user + assistant with their inline timeline of
 * tool/subagent events and citations) so reopening a thread reproduces the
 * conversation exactly as it was rendered.
 *
 * The `useThreads` hook exposes a stable API for the route: list, active
 * thread id, switching, creating, deleting, and updating the currently
 * active thread's turns.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Citation } from '#/lib/api'

const STORAGE_KEY = 'lovelytics.threads.v1'
const ACTIVE_KEY = 'lovelytics.threads.active.v1'

export type TimelineEvent =
  | {
      kind: 'tool'
      name: string
      args: Record<string, unknown>
      result?: string
      status: 'running' | 'done'
    }
  | {
      kind: 'subagent'
      name: string
      task: string
      result?: string
      status: 'running' | 'done'
    }

/**
 * A single chronological entry inside an assistant turn. Either a chunk of
 * generated text or a tool/subagent invocation. Entries are stored in the
 * order they happened so the UI can render a faithful timeline.
 */
export type AssistantEntry =
  | { kind: 'text'; content: string }
  | { kind: 'event'; event: TimelineEvent }

export type Turn =
  | {
      id: string
      role: 'user'
      content: string
      status: 'done'
    }
  | {
      id: string
      role: 'assistant'
      entries: AssistantEntry[]
      citations: Citation[]
      status: 'streaming' | 'done' | 'error'
      error?: string
    }

export type Thread = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  turns: Turn[]
}

function readThreads(): Thread[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed as Thread[]
  } catch {
    return []
  }
}

function writeThreads(threads: Thread[]) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads))
  } catch {
    // Quota or private-mode failures are non-fatal; just lose persistence.
  }
}

function readActiveId(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage.getItem(ACTIVE_KEY)
  } catch {
    return null
  }
}

function writeActiveId(id: string | null) {
  try {
    if (id) window.localStorage.setItem(ACTIVE_KEY, id)
    else window.localStorage.removeItem(ACTIVE_KEY)
  } catch {
    // ignore
  }
}

/** Best-effort title from the first user turn. */
export function deriveTitle(turns: Turn[]): string {
  const firstUser = turns.find((t) => t.role === 'user')
  if (!firstUser) return 'New chat'
  const text = firstUser.content.replace(/\s+/g, ' ').trim()
  if (!text) return 'New chat'
  return text.length > 48 ? `${text.slice(0, 48)}…` : text
}

export function useThreads() {
  const [threads, setThreads] = useState<Thread[]>(() => readThreads())
  const [activeId, setActiveId] = useState<string | null>(() => readActiveId())

  // Persist on every change.
  useEffect(() => {
    writeThreads(threads)
  }, [threads])

  useEffect(() => {
    writeActiveId(activeId)
  }, [activeId])

  const active = useMemo(
    () => threads.find((t) => t.id === activeId) ?? null,
    [threads, activeId],
  )

  const sorted = useMemo(
    () => [...threads].sort((a, b) => b.updatedAt - a.updatedAt),
    [threads],
  )

  const createThread = useCallback((): string => {
    const id = crypto.randomUUID()
    const now = Date.now()
    const thread: Thread = {
      id,
      title: 'New chat',
      createdAt: now,
      updatedAt: now,
      turns: [],
    }
    setThreads((prev) => [thread, ...prev])
    setActiveId(id)
    return id
  }, [])

  const selectThread = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const deleteThread = useCallback(
    (id: string) => {
      setThreads((prev) => {
        const next = prev.filter((t) => t.id !== id)
        if (activeId === id) {
          const fallback = next[0]?.id ?? null
          setActiveId(fallback)
        }
        return next
      })
    },
    [activeId],
  )

  /**
   * Replace the active thread's turns. Used by the route on every state
   * mutation during a streaming run.
   */
  const updateActiveTurns = useCallback(
    (updater: (turns: Turn[]) => Turn[]) => {
      setThreads((prev) => {
        if (!activeId) return prev
        return prev.map((t) => {
          if (t.id !== activeId) return t
          const turns = updater(t.turns)
          return {
            ...t,
            turns,
            updatedAt: Date.now(),
            // Re-derive title once we have a first user turn.
            title: t.title === 'New chat' ? deriveTitle(turns) : t.title,
          }
        })
      })
    },
    [activeId],
  )

  return {
    threads: sorted,
    active,
    activeId,
    createThread,
    selectThread,
    deleteThread,
    updateActiveTurns,
  }
}
