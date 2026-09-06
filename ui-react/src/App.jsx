import { useEffect, useRef, useState } from 'react'

// ---------------------------------------------------------------------------
// API helper
// ---------------------------------------------------------------------------

async function request(path, options) {
  const res = await fetch(path, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
  return data
}

const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),
  del: (path) => request(path, { method: 'DELETE' }),
  upload: (path, formData) => request(path, { method: 'POST', body: formData }),
}

// ---------------------------------------------------------------------------
// Icons (small hand-drawn inline SVGs, no icon library dependency)
// ---------------------------------------------------------------------------

const iconProps = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }

const IconGrid = () => (
  <svg {...iconProps}><rect x="3" y="3" width="8" height="8" rx="1.5" /><rect x="13" y="3" width="8" height="8" rx="1.5" /><rect x="3" y="13" width="8" height="8" rx="1.5" /><rect x="13" y="13" width="8" height="8" rx="1.5" /></svg>
)
const IconFile = () => (
  <svg {...iconProps}><path d="M6 2.5h8l4 4V21a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1Z" /><path d="M9 12h6M9 16h6" /></svg>
)
const IconSearch = () => (
  <svg {...iconProps}><circle cx="10.5" cy="10.5" r="6.5" /><path d="m20 20-4.4-4.4" /></svg>
)
const IconChat = () => (
  <svg {...iconProps}><path d="M4 5h16a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H9l-4 4v-4H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z" /><path d="M8 9h8M8 12.5h5" /></svg>
)
const IconPulse = () => (
  <svg {...iconProps}><path d="M3 12h4l2-7 4 14 2-7h6" /></svg>
)
const IconUpload = () => (
  <svg {...iconProps} width="28" height="28"><path d="M12 16V4M8 8l4-4 4 4" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></svg>
)
const IconRefresh = () => (
  <svg {...iconProps} width="15" height="15"><path d="M4 4v5h5" /><path d="M20 20v-5h-5" /><path d="M5.5 8.5A7 7 0 0 1 19 9M18.5 15.5A7 7 0 0 1 5 15" /></svg>
)
const IconX = () => (
  <svg {...iconProps} width="14" height="14"><path d="M5 5l14 14M19 5 5 19" /></svg>
)

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { key: 'dashboard', label: 'Dashboard', icon: IconGrid },
  { key: 'documents', label: 'Documents', icon: IconFile },
  { key: 'retrieve', label: 'Retrieve', icon: IconSearch },
  { key: 'ask', label: 'Ask', icon: IconChat },
  { key: 'traces', label: 'Error analysis', icon: IconPulse },
]

const PAGE_COPY = {
  dashboard: { title: 'Support knowledge assistant', tagline: 'Ingest documents, inspect evidence, answer safely, and review failures.' },
  documents: { title: 'Documents & ingestion', tagline: 'Add source files and see what is currently indexed.' },
  retrieve: { title: 'Retrieval inspector', tagline: 'See exactly which chunks a question matches, before any answer is generated.' },
  ask: { title: 'Ask a question', tagline: 'A weak match is refused before the LLM is ever called.' },
  traces: { title: 'Error analysis', tagline: 'Review past questions, label failures, and track what needs fixing.' },
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [toast, setToast] = useState(null)
  // Which provider to try first -- the other is always the automatic
  // fallback if this one hits a rate limit (429). Lives here, not inside
  // the Ask page, because the top bar showing it is visible everywhere.
  const [provider, setProvider] = useState('gemini')

  const notifyError = (message) => setToast({ message })

  const refresh = async () => {
    try {
      const [h, d] = await Promise.all([api.get('/health'), api.get('/documents')])
      setHealth(h)
      setDocuments(d)
    } catch (e) {
      notifyError(e.message)
    }
  }

  useEffect(() => { refresh() }, [])

  const copy = PAGE_COPY[page]

  return (
    <div className="shell">
      <div className="topbar-global">
        <span className="topbar-brand">Ask My Tickets</span>
        <div className="topbar-global-right">
          <ProviderToggle provider={provider} setProvider={setProvider} />
          <HealthChip health={health} />
        </div>
      </div>

      <div className="shell-body">
        <Sidebar page={page} setPage={setPage} />

        <main className="main">
          <div className="topbar">
            <div>
              <p className="eyebrow">{page === 'traces' ? 'WEEK 5 · EVALS & ERROR ANALYSIS' : 'RAG WORKBENCH'}</p>
              <h1>{copy.title}</h1>
              <p>{copy.tagline}</p>
            </div>
          </div>

          {toast && (
            <div className="toast">
              <span>{toast.message}</span>
              <button onClick={() => setToast(null)} aria-label="Dismiss"><IconX /></button>
            </div>
          )}

          <div className={page === 'ask' ? 'page page-full' : 'page'}>
            {page === 'dashboard' && <Dashboard health={health} documents={documents} setPage={setPage} />}
            {page === 'documents' && <Documents documents={documents} refresh={refresh} notifyError={notifyError} />}
            {page === 'retrieve' && <Retrieve notifyError={notifyError} />}
            {page === 'ask' && <Ask notifyError={notifyError} provider={provider} />}
            {page === 'traces' && <Traces notifyError={notifyError} />}
          </div>
        </main>
      </div>
    </div>
  )
}

function Sidebar({ page, setPage }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="eyebrow">Ask My Tickets</span>
        <h2>Knowledge hub</h2>
        <span>Weeks 1&ndash;5</span>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
          <button key={key} className={`nav-item${page === key ? ' active' : ''}`} onClick={() => setPage(key)}>
            <Icon />
            {label}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <b>Current flow</b>
        Ingest &rarr; retrieve &rarr; answer &rarr; analyse failures
      </div>
    </aside>
  )
}

function HealthChip({ health }) {
  if (!health) return null
  const ok = health.status === 'ok'
  return (
    <span className={`health-chip ${ok ? 'ok' : 'degraded'}`}>
      <span className="health-dot" />
      {ok ? 'All systems ok' : 'Degraded'}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

function Dashboard({ health, documents, setPage }) {
  const chunkCount = documents.reduce((sum, doc) => sum + doc.chunk_count, 0)

  const steps = [
    { number: '1', title: 'Documents', copy: 'Add and index source files', target: 'documents' },
    { number: '2', title: 'Retrieve', copy: 'See the chunks found for a question', target: 'retrieve' },
    { number: '3', title: 'Ask', copy: 'Generate a cited answer, or a refusal', target: 'ask' },
    { number: '4', title: 'Analyse', copy: 'Read traces and label failures', target: 'traces' },
  ]

  return (
    <>
      <div className="metrics">
        <div className="metric"><span>Indexed documents</span><strong>{documents.length}</strong><small>Ready for retrieval</small></div>
        <div className="metric"><span>Knowledge chunks</span><strong>{chunkCount}</strong><small>Sentence-aware segments</small></div>
        <div className="metric"><span>App health</span><strong>{health?.status ?? '...'}</strong><small>Embedding, Qdrant, provider</small></div>
      </div>

      <div className="card">
        <p className="eyebrow">Your workflow</p>
        <h2>From support documents to evidence-backed answers</h2>
        <p style={{ marginTop: '0.4rem' }}>Each screen exposes one step of the system, so you can inspect what happens before trusting an answer.</p>
        <div className="workflow-steps">
          {steps.map((step) => (
            <button key={step.title} className="step-btn" onClick={() => setPage(step.target)}>
              <span className="step-number">{step.number}</span>
              <b>{step.title}</b>
              <small>{step.copy}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Quick start</h2>
        <ol className="quick-start">
          <li>Upload or review the support-ticket source documents.</li>
          <li>Use the retrieval inspector to validate the evidence returned.</li>
          <li>Ask a customer question and confirm the source citations.</li>
          <li>For Week 5, sample recent traces and label each failure before grouping them.</li>
        </ol>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Documents & upload
// ---------------------------------------------------------------------------

const ACCEPTED_EXTENSIONS = ['.txt', '.md', '.docx', '.pdf', '.xlsx']

function Documents({ documents, refresh, notifyError }) {
  const [pendingFile, setPendingFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState(null) // { type: 'busy' | 'success', message }

  const acceptFile = (file) => {
    if (!file) return
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      notifyError(`Unsupported file type "${ext}". Supported: ${ACCEPTED_EXTENSIONS.join(', ')}`)
      return
    }
    setPendingFile(file)
    setStatus(null)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    acceptFile(e.dataTransfer.files?.[0])
  }

  const upload = async () => {
    if (!pendingFile) return
    setStatus({ type: 'busy', message: 'Uploading and rebuilding the index...' })
    try {
      const formData = new FormData()
      formData.append('file', pendingFile)
      const result = await api.upload('/upload', formData)
      await refresh()
      setStatus({ type: 'success', message: result.message ?? 'Uploaded and ingested.' })
      setPendingFile(null)
    } catch (e) {
      setStatus(null)
      notifyError(e.message)
    }
  }

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h2>Add a document</h2>
          <p>Uploading rebuilds the whole index, so the new file is searchable immediately.</p>
        </div>
      </div>

      <label
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <IconUpload />
        <b>Drag & drop a file here, or click to browse</b>
        <small>One file at a time. It's saved into the documents folder and indexed right away.</small>
        <div className="format-chips">
          {ACCEPTED_EXTENSIONS.map((ext) => <span key={ext}>{ext}</span>)}
        </div>
        <input
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          style={{ display: 'none' }}
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
      </label>

      {pendingFile && (
        <div className="pending-upload">
          <div className="file-info">
            <span className="file-badge">{pendingFile.name.split('.').pop().toUpperCase()}</span>
            <span>{pendingFile.name}</span>
          </div>
          <div className="actions">
            <button className="btn secondary small" onClick={() => setPendingFile(null)}>Cancel</button>
            <button className="btn small" onClick={upload} disabled={status?.type === 'busy'}>
              {status?.type === 'busy' ? 'Uploading...' : 'Upload & ingest'}
            </button>
          </div>
        </div>
      )}

      {status?.type === 'success' && <p className="upload-status success">{status.message}</p>}

      <h2 style={{ marginTop: '1.75rem' }}>Indexed documents ({documents.length})</h2>
      {documents.length === 0 ? (
        <p className="doc-empty">No documents indexed yet &mdash; add one above.</p>
      ) : (
        <div className="doc-grid">
          {documents.map((doc) => (
            <div key={doc.filename} className="doc-card">
              <div className="doc-card-head">
                <span className="doc-card-ext">{doc.filename.split('.').pop().toUpperCase()}</span>
                <b className="doc-card-name" title={doc.filename}>{doc.filename}</b>
              </div>
              <span className="file-badge doc-card-count">{doc.chunk_count} {doc.chunk_count === 1 ? 'chunk' : 'chunks'}</span>
              <p>{doc.preview}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Retrieval inspector
// ---------------------------------------------------------------------------

function Retrieve({ notifyError }) {
  const [question, setQuestion] = useState('')
  const [chunks, setChunks] = useState([])
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      setChunks(await api.post('/retrieve', { question, top_k: 3 }))
    } catch (e) {
      notifyError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>Retrieval inspector</h2>
      <p>Semantic search only &mdash; inspect the ticket chunks before anything gets generated.</p>
      <form onSubmit={submit} style={{ marginTop: '1rem' }}>
        <div className="field">
          <label>Question</label>
          <textarea required value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. why can't I log in after too many wrong passwords?" />
        </div>
        <button className="btn" disabled={busy}>{busy ? 'Searching...' : 'Inspect retrieval'}</button>
      </form>
      <Evidence chunks={chunks} scoreKey="distance" />
    </div>
  )
}

function Evidence({ chunks, scoreKey }) {
  if (!chunks || chunks.length === 0) return <p className="evidence-empty">No results yet.</p>
  return (
    <ol className="evidence">
      {chunks.map((c) => (
        <li key={`${c.source}-${c.chunk_index}`}>
          <div className="evidence-head">
            <b>{c.source} &middot; chunk {c.chunk_index}{c.page != null ? ` · page ${c.page}` : ''}</b>
            <span className="score-badge">{scoreKey} {Number(c[scoreKey]).toFixed(3)}</span>
          </div>
          <p>{c.text}</p>
        </li>
      ))}
    </ol>
  )
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

function Ask({ notifyError, provider }) {
  const [sessions, setSessions] = useState([])
  const [currentId, setCurrentId] = useState(null)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [loadingSessions, setLoadingSessions] = useState(true)
  // Generation settings -- how many chunks retrieval hands the LLM, and how
  // random/deterministic its wording is. Kept per-question (sent with every
  // /ask call), not saved with the session, so changing them mid-conversation
  // only affects the next question asked.
  const [topK, setTopK] = useState(3)
  const [temperature, setTemperature] = useState(1)
  // Guards the initial load against React StrictMode's dev-mode double
  // effect invocation, which would otherwise create two "New chat"
  // sessions on a fresh page load.
  const didInit = useRef(false)

  const loadSessions = async () => {
    try {
      return await api.get('/sessions')
    } catch (e) {
      notifyError(e.message)
      return []
    }
  }

  const openSession = async (id) => {
    try {
      const session = await api.get(`/sessions/${id}`)
      setCurrentId(session.id)
      setMessages(session.messages)
    } catch (e) {
      notifyError(e.message)
    }
  }

  const startNewChat = async () => {
    try {
      const session = await api.post('/sessions', {})
      setSessions((prev) => [
        { id: session.id, title: session.title, created_at: session.created_at, updated_at: session.updated_at, message_count: 0 },
        ...prev,
      ])
      setCurrentId(session.id)
      setMessages([])
    } catch (e) {
      notifyError(e.message)
    }
  }

  // On first load: open the most recent conversation, or start one if
  // there's nothing saved yet -- so the chat is never empty of a session.
  useEffect(() => {
    if (didInit.current) return
    didInit.current = true
    ;(async () => {
      setLoadingSessions(true)
      const list = await loadSessions()
      setSessions(list)
      if (list.length > 0) await openSession(list[0].id)
      else await startNewChat()
      setLoadingSessions(false)
    })()
  }, [])

  const removeSession = async (id, e) => {
    e.stopPropagation()
    try {
      await api.del(`/sessions/${id}`)
    } catch (e) {
      notifyError(e.message)
      return
    }
    const remaining = sessions.filter((s) => s.id !== id)
    setSessions(remaining)
    if (id === currentId) {
      if (remaining.length > 0) await openSession(remaining[0].id)
      else await startNewChat()
    }
  }

  const submit = async (e) => {
    e.preventDefault()
    if (!question.trim() || !currentId || busy) return
    const asked = question
    setQuestion('')
    setMessages((prev) => [...prev, { role: 'user', content: asked }])
    setBusy(true)
    try {
      const result = await api.post('/ask', { question: asked, top_k: topK, provider, temperature, session_id: currentId })
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: result.answer,
        sources: result.sources,
        provider: result.provider,
        trace_id: result.trace_id,
        skipped_llm: result.skipped_llm,
      }])
      setSessions(await loadSessions()) // pick up the updated title/order
    } catch (err) {
      notifyError(err.message)
      setMessages((prev) => prev.slice(0, -1)) // roll back the optimistic user bubble
      setQuestion(asked)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="chat-shell">
      <aside className="chat-sidebar">
        <button className="btn new-chat-btn" onClick={startNewChat}>+ New chat</button>
        <div className="chat-session-list">
          {loadingSessions && <p className="empty-state">Loading chats...</p>}
          {!loadingSessions && sessions.length === 0 && <p className="empty-state">No chats yet.</p>}
          {sessions.map((s) => (
            <button
              key={s.id}
              className={`chat-session-row${s.id === currentId ? ' active' : ''}`}
              onClick={() => openSession(s.id)}
            >
              <span className="chat-session-title">{s.title}</span>
              <span className="chat-session-remove" onClick={(e) => removeSession(s.id, e)} title="Delete chat">
                <IconX />
              </span>
            </button>
          ))}
        </div>

        <div className="chat-settings">
          <p className="chat-settings-title">Generation settings</p>
          <SliderField
            label="Top K"
            value={topK}
            onChange={setTopK}
            min={1}
            max={10}
            step={1}
            hint="How many retrieved chunks are handed to the LLM."
          />
          <SliderField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0}
            max={2}
            step={0.1}
            hint="Lower = more deterministic wording, higher = more varied."
          />
        </div>
      </aside>

      <div className="chat-main">
        <div className="chat-messages">
          {messages.length === 0 && (
            <p className="chat-empty">Ask a customer question below. A weak match is refused before the LLM is ever called.</p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`chat-bubble ${m.role}`}>
              {m.role === 'user' ? (
                <p>{m.content}</p>
              ) : (
                <div className={`answer-card ${m.skipped_llm ? 'refused' : 'grounded'}`}>
                  <span className={`answer-status ${m.skipped_llm ? 'refused' : 'grounded'}`}>
                    {m.skipped_llm ? 'No relevant evidence found' : 'Grounded answer'}
                  </span>
                  <p className="answer-text">{m.content}</p>
                  <div className="answer-sources">
                    <span className="label">Sources:</span>
                    {m.sources && m.sources.length > 0
                      ? m.sources.map((s) => <span key={s} className="pill">{s}</span>)
                      : <span className="pill">none</span>}
                  </div>
                  {m.provider && <div className="trace-ref">answered by: {m.provider}</div>}
                  {m.trace_id && <div className="trace-ref">trace: {m.trace_id}</div>}
                </div>
              )}
            </div>
          ))}
          {busy && (
            <div className="chat-bubble assistant">
              <div className="answer-card thinking">Thinking...</div>
            </div>
          )}
        </div>

        <form className="chat-input-row" onSubmit={submit}>
          <textarea
            required
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. what happens to my data if I cancel my subscription?"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit(e)
              }
            }}
          />
          <button className="btn" disabled={busy || !question.trim()}>{busy ? 'Thinking...' : 'Send'}</button>
        </form>
      </div>
    </div>
  )
}

function SliderField({ label, value, onChange, min, max, step, hint }) {
  // A slider + a synced numeric input, so a value can be dragged roughly or
  // typed exactly -- same pattern as most LLM playground "temperature"
  // controls.
  const clamp = (n) => Math.min(max, Math.max(min, n))

  return (
    <div className="slider-field" title={hint}>
      <label>{label}</label>
      <div className="slider-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <input
          type="number"
          className="slider-number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => {
            const raw = e.target.value
            if (raw === '') return
            const parsed = Number(raw)
            if (!Number.isNaN(parsed)) onChange(clamp(parsed))
          }}
        />
      </div>
    </div>
  )
}

function ProviderToggle({ provider, setProvider }) {
  const isGemini = provider === 'gemini'
  return (
    <div className="provider-toggle" title="Which provider to try first -- the other is the automatic fallback on a rate limit">
      <span className={`provider-toggle-label${isGemini ? ' active' : ''}`}>Gemini</span>
      <button
        type="button"
        role="switch"
        aria-checked={isGemini}
        className={`switch${isGemini ? '' : ' off'}`}
        onClick={() => setProvider(isGemini ? 'groq' : 'gemini')}
      >
        <span className="switch-knob" />
      </button>
      <span className={`provider-toggle-label${!isGemini ? ' active' : ''}`}>Groq</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Traces & error analysis
// ---------------------------------------------------------------------------

const OUTCOME_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'answer', label: 'Answered' },
  { key: 'refusal', label: 'Refused' },
  { key: 'error', label: 'Error' },
]

const PROBLEM_TYPES = [
  { value: '', label: 'Not set' },
  { value: 'retrieval_failure', label: 'Retrieval failure (wrong document fetched)' },
  { value: 'generation_failure', label: 'Generation failure (right document, wrong answer)' },
  { value: 'refusal_misfire', label: 'Refusal misfire (refused despite good evidence)' },
  { value: 'other', label: 'Other' },
]

function Traces({ notifyError }) {
  const [traces, setTraces] = useState([])
  const [selected, setSelected] = useState(null)
  const [filter, setFilter] = useState('all')
  const [unreviewedOnly, setUnreviewedOnly] = useState(false)
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      setTraces(await api.get('/traces?limit=50'))
    } catch (e) {
      notifyError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const open = async (traceId) => {
    try {
      setSelected(await api.get(`/traces/${traceId}`))
    } catch (e) {
      notifyError(e.message)
    }
  }

  const onSaved = (traceId, annotation) => {
    setTraces((prev) => prev.map((t) => (
      t.trace_id === traceId ? { ...t, annotation: { ...t.annotation, ...annotation } } : t
    )))
  }

  const visible = traces.filter((t) => {
    if (filter !== 'all' && t.outcome !== filter) return false
    if (unreviewedOnly && t.annotation?.reviewed) return false
    return true
  })

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h2>Trace review</h2>
          <p>Open a trace, read the evidence, then label what kind of failure (if any) it was.</p>
        </div>
        <button className="btn secondary small" onClick={load}><IconRefresh /> Refresh</button>
      </div>

      <div className="trace-toolbar">
        <div className="filter-tabs">
          {OUTCOME_FILTERS.map((f) => (
            <button key={f.key} className={filter === f.key ? 'active' : ''} onClick={() => setFilter(f.key)}>{f.label}</button>
          ))}
        </div>
        <label className="checkbox-inline">
          <input type="checkbox" checked={unreviewedOnly} onChange={(e) => setUnreviewedOnly(e.target.checked)} />
          Unreviewed only
        </label>
      </div>

      <div className="trace-layout">
        <div className="trace-list">
          {loading && <p className="empty-state">Loading traces...</p>}
          {!loading && visible.length === 0 && <p className="empty-state">No traces match this filter.</p>}
          {visible.map((t) => (
            <button
              key={t.trace_id}
              className={`trace-row${selected?.trace_id === t.trace_id ? ' selected' : ''}`}
              onClick={() => open(t.trace_id)}
            >
              <div className="trace-row-top">
                <b>{t.question}</b>
              </div>
              <div className="trace-row-meta">
                <OutcomeBadge outcome={t.outcome} />
                {t.annotation?.reviewed && <span className="reviewed-dot" title="Reviewed" />}
              </div>
            </button>
          ))}
        </div>

        {selected && (
          <TraceDetail trace={selected} onSaved={onSaved} notifyError={notifyError} />
        )}
      </div>
    </div>
  )
}

function OutcomeBadge({ outcome }) {
  const label = outcome === 'answer' ? 'Answered' : outcome === 'refusal' ? 'Refused' : 'Error'
  const cls = outcome === 'answer' ? 'answer' : outcome === 'refusal' ? 'refusal' : 'error'
  return <span className={`status-badge ${cls}`}>{label}</span>
}

function TraceDetail({ trace, onSaved, notifyError }) {
  const retrievalStage = trace.stages?.find((s) => s.name === 'retrieval')
  const chunks = retrievalStage?.data?.chunks ?? []

  const [reviewed, setReviewed] = useState(Boolean(trace.annotation?.reviewed))
  const [problemType, setProblemType] = useState(trace.annotation?.problem_type ?? '')
  const [severity, setSeverity] = useState(trace.annotation?.severity ?? null)
  const [note, setNote] = useState(trace.annotation?.failure_note ?? '')
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setReviewed(Boolean(trace.annotation?.reviewed))
    setProblemType(trace.annotation?.problem_type ?? '')
    setSeverity(trace.annotation?.severity ?? null)
    setNote(trace.annotation?.failure_note ?? '')
    setSaved(false)
  }, [trace.trace_id])

  // The backend refuses to save a problem_type without an accompanying
  // failure note (an "open-coded" note has to exist before you can label
  // it) -- checked here too so the button disables instead of the save
  // silently failing.
  const needsNoteFirst = Boolean(problemType) && !note.trim()

  const save = async () => {
    setSaving(true)
    setSaved(false)
    try {
      const annotation = { reviewed, problem_type: problemType || null, severity, failure_note: note || null }
      await api.post(`/traces/${trace.trace_id}/annotation`, annotation)
      onSaved(trace.trace_id, annotation)
      setSaved(true)
    } catch (e) {
      notifyError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="trace-detail">
      <div className="trace-detail-head">
        <h3>{trace.question}</h3>
        <div className="trace-detail-meta">
          <OutcomeBadge outcome={trace.outcome} />
          <span className="pill">top_k {trace.config?.top_k ?? 3}</span>
          <span className="pill">{trace.config?.model ?? ''}</span>
        </div>
      </div>

      <div className="trace-answer-block">{trace.answer || trace.error || 'No answer text recorded.'}</div>

      <div>
        <h4 style={{ fontSize: '0.9rem', marginBottom: '0.6rem' }}>Retrieved evidence</h4>
        <Evidence chunks={chunks} scoreKey="distance" />
      </div>

      <div className="annotation-panel">
        <h4>Label this trace</h4>

        <label className="checkbox-inline" style={{ marginBottom: '0.9rem' }}>
          <input type="checkbox" checked={reviewed} onChange={(e) => setReviewed(e.target.checked)} />
          Mark as reviewed
        </label>

        <div className="field">
          <label>Failure type</label>
          <select value={problemType} onChange={(e) => setProblemType(e.target.value)}>
            {PROBLEM_TYPES.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
        </div>

        <div className="field">
          <label>Severity</label>
          <div className="severity-row">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                className={`severity-btn${severity === n ? ' active' : ''}`}
                onClick={() => setSeverity(severity === n ? null : n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Notes {problemType && <span style={{ color: 'var(--warn)', fontWeight: 600 }}>(required to set a failure type)</span>}</label>
          <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="What went wrong, and why?" />
        </div>

        <button className="btn" onClick={save} disabled={saving || needsNoteFirst}>{saving ? 'Saving...' : 'Save annotation'}</button>
        {needsNoteFirst && <p style={{ color: 'var(--warn)', fontSize: '0.82rem', marginTop: '0.5rem' }}>Write a note above before assigning a failure type.</p>}
        {saved && <p className="annotation-saved">Saved.</p>}
      </div>
    </div>
  )
}
