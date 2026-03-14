import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CanvasBBox, BBox, getClassColor } from '../components/CanvasBBox'

// ─── Types ────────────────────────────────────────────────────────────────────

type ImageItem = {
  id: number
  path: string
  thumbnailPath?: string
  displayPath?: string
  width: number
  height: number
  status: 'unannotated' | 'ai_pending' | 'annotated'
  labeled: boolean
}

type ClassItem  = { id: number; name: string }
type Annotation = {
  id: number; imageId: number; classId: number
  x: number; y: number; w: number; h: number
  source: 'manual' | 'ai'
}

interface AiDefaultConfig {
  engine: 'glm4v' | 'model'
  // glm4v
  classId?: number
  prompt?: string
  // model
  artifactId?: number
  confidenceThreshold?: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

const THUMB_SIZE = 20

const STATUS_BG: Record<string, string> = {
  ai_pending: 'rgba(250,173,20,0.85)',
  annotated:  'rgba(82,196,26,0.85)',
}
const STATUS_LBL: Record<string, string> = {
  ai_pending: 'AI待审',
  annotated:  '已标注',
}

// ─── Component ────────────────────────────────────────────────────────────────

export const Annotator: React.FC<{ projectId: number; onBack: () => void }> = ({
  projectId,
  onBack,
}) => {
  // Image strip state
  const [thumbPage, setThumbPage]   = useState(1)
  const [thumbImages, setThumbImages] = useState<ImageItem[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [loadingThumb, setLoadingThumb] = useState(false)

  // Current image & annotations
  const [current, setCurrent]   = useState<ImageItem | null>(null)
  const [anns, setAnns]         = useState<Annotation[]>([])
  const [classes, setClasses]   = useState<ClassItem[]>([])
  const [classId, setClassId]   = useState<number | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  // Filter
  const [filter, setFilter] = useState<string>('all')

  // Undo / redo
  const undoStack = useRef<any[]>([])
  const redoStack = useRef<any[]>([])

  // Right-click context menu
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null)
  const ctxMenuRef = useRef<HTMLDivElement>(null)

  // AI quick job state
  const [aiRunning, setAiRunning] = useState(false)
  const aiPollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Helpers ────────────────────────────────────────────────────────────────

  const buildFilterParams = (f = filter) => {
    if (f === 'labeled')   return '&labeled=true'
    if (f === 'unlabeled') return '&labeled=false'
    if (f !== 'all')       return `&status=${f}`
    return ''
  }

  // ── Data loading ────────────────────────────────────────────────────────────

  const fetchThumbPage = useCallback(async (
    page: number,
    f = filter,
  ): Promise<ImageItem[]> => {
    setLoadingThumb(true)
    try {
      const fp = f === 'labeled'   ? '&labeled=true'
               : f === 'unlabeled' ? '&labeled=false'
               : f !== 'all'       ? `&status=${f}`
               : ''
      const res  = await fetch(
        `/api/projects/${projectId}/images?page=${page}&page_size=${THUMB_SIZE}${fp}`
      )
      const data = await res.json()
      const items: ImageItem[] = data.items || []
      setThumbImages(items)
      setTotalImages(data.total || 0)
      setThumbPage(page)
      // Auto-select first image when page 1 loads and nothing is selected
      if (page === 1 && !current && items.length > 0) {
        setCurrent(items[0])
      }
      return items
    } finally {
      setLoadingThumb(false)
    }
  }, [projectId, filter, current])

  const loadClasses = async () => {
    const res  = await fetch(`/api/projects/${projectId}/classes`)
    const data = await res.json()
    setClasses(data)
    if (data.length > 0 && classId == null) setClassId(data[0].id)
  }

  const loadAnns = async (imageId: number) => {
    const res = await fetch(`/api/images/${imageId}/annotations`)
    setAnns(await res.json())
  }

  useEffect(() => { loadClasses() }, [])
  useEffect(() => { fetchThumbPage(1, filter) }, [filter, projectId])
  useEffect(() => { if (current) loadAnns(current.id) }, [current?.id])
  useEffect(() => { setSelectedId(null) }, [current?.id])

  // ── Image URL ───────────────────────────────────────────────────────────────

  const imageUrl = useMemo(() => {
    if (!current) return ''
    const p = (current.displayPath || current.path).replace(/\\/g, '/')
    return p.startsWith('http://') || p.startsWith('https://') ? p : `/files/${p}`
  }, [current])

  const thumbUrl = (img: ImageItem) => {
    const p = (img.thumbnailPath || img.path).replace(/\\/g, '/')
    return p.startsWith('http://') || p.startsWith('https://') ? p : `/files/${p}`
  }

  // ── Annotation CRUD ─────────────────────────────────────────────────────────

  const onCreate = async (box: { x: number; y: number; w: number; h: number; classId: number; source?: 'manual' | 'ai' }) => {
    if (!current) return
    const res = await fetch('/api/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageId: current.id, ...box }),
    })
    if (!res.ok) return
    const created = await res.json()
    undoStack.current.push({ type: 'create', after: created })
    redoStack.current = []
    await loadAnns(current.id)
    if (current.status === 'unannotated') {
      const updated = { ...current, status: 'annotated' as const }
      setCurrent(updated)
      setThumbImages(prev => prev.map(img => img.id === current.id ? updated : img))
    }
  }

  const onDelete = async (id: number) => {
    const before = anns.find(a => a.id === id)
    await fetch(`/api/annotations/${id}`, { method: 'DELETE' })
    if (current) await loadAnns(current.id)
    if (before) undoStack.current.push({ type: 'delete', before })
    redoStack.current = []
  }

  const onUpdate = async (id: number, patch: Partial<BBox>) => {
    const before = anns.find(a => a.id === id)
    const res = await fetch(`/api/annotations/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    if (!res.ok) return
    const after = await res.json()
    undoStack.current.push({ type: 'update', before, after })
    redoStack.current = []
    if (current) await loadAnns(current.id)
  }

  // ── Undo / Redo ─────────────────────────────────────────────────────────────

  const performUndo = async () => {
    if (undoStack.current.length === 0) return
    const op = undoStack.current.pop()!
    if (op.type === 'create') {
      await fetch(`/api/annotations/${op.after.id}`, { method: 'DELETE' })
    } else if (op.type === 'delete') {
      await fetch('/api/annotations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(op.before) })
    } else if (op.type === 'update') {
      await fetch(`/api/annotations/${op.after.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: op.before.x, y: op.before.y, w: op.before.w, h: op.before.h, classId: op.before.classId }) })
    }
    redoStack.current.push(op)
    if (current) await loadAnns(current.id)
  }

  const performRedo = async () => {
    if (redoStack.current.length === 0) return
    const op = redoStack.current.pop()!
    if (op.type === 'create') {
      await fetch('/api/annotations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(op.after) })
    } else if (op.type === 'delete') {
      await fetch(`/api/annotations/${op.before.id}`, { method: 'DELETE' })
    } else if (op.type === 'update') {
      await fetch(`/api/annotations/${op.after.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: op.after.x, y: op.after.y, w: op.after.w, h: op.after.h, classId: op.after.classId }) })
    }
    undoStack.current.push(op)
    if (current) await loadAnns(current.id)
  }

  // ── Mark labeled ────────────────────────────────────────────────────────────

  const markCurrentLabeled = async () => {
    if (!current || current.labeled) return
    const res = await fetch(`/api/images/${current.id}/labeled`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labeled: true }),
    })
    if (!res.ok) return
    const updated = { ...current, labeled: true }
    setCurrent(updated)
    setThumbImages(prev => prev.map(img => img.id === current.id ? updated : img))
    // Jump to next unlabeled in strip
    const idx  = thumbImages.findIndex(img => img.id === current.id)
    const next = thumbImages.find((img, i) => i > idx && !img.labeled)
    if (next) setCurrent(next)
  }

  // ── Right-click context menu ────────────────────────────────────────────────

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    if (!current) return
    setCtxMenu({ x: e.clientX, y: e.clientY })
  }

  // Close context menu when clicking outside
  useEffect(() => {
    if (!ctxMenu) return
    const close = () => setCtxMenu(null)
    document.addEventListener('click', close, { once: true })
    return () => document.removeEventListener('click', close)
  }, [ctxMenu])

  // Delete current image
  const handleDeleteImage = async () => {
    setCtxMenu(null)
    if (!current) return
    if (!window.confirm('确认删除当前图片？将同时删除其关联标注，不可恢复！')) return
    const res = await fetch(`/api/projects/${projectId}/images`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: [current.id] }),
    })
    if (!res.ok) { alert('删除失败'); return }
    const idx  = thumbImages.findIndex(img => img.id === current.id)
    const next = thumbImages[idx + 1] ?? thumbImages[idx - 1] ?? null
    setThumbImages(prev => prev.filter(img => img.id !== current.id))
    setTotalImages(prev => prev - 1)
    setCurrent(next)
    setAnns([])
  }

  // Clear all annotations for current image
  const handleClearAnnotations = async () => {
    setCtxMenu(null)
    if (!current || anns.length === 0) return
    if (!window.confirm(`确认清空该图片的全部 ${anns.length} 个标注？`)) return
    for (const ann of anns) {
      await fetch(`/api/annotations/${ann.id}`, { method: 'DELETE' })
    }
    setAnns([])
    undoStack.current = []
    redoStack.current = []
  }

  // Quick AI recognition using saved default config
  const handleQuickAi = async () => {
    setCtxMenu(null)
    if (!current) return
    const raw = localStorage.getItem('annotator_ai_default_config')
    if (!raw) {
      alert('暂无默认 AI 识别配置，请前往「AI识别」页面保存默认配置后再使用。')
      return
    }
    let cfg: AiDefaultConfig
    try { cfg = JSON.parse(raw) } catch { alert('配置读取失败，请重新保存'); return }

    setAiRunning(true)
    try {
      let url  = ''
      let body: Record<string, unknown> = { projectId, imageIds: [current.id] }

      if (cfg.engine === 'glm4v') {
        url  = '/api/autolabel/jobs'
        body = { ...body, classId: cfg.classId, prompt: cfg.prompt ?? '', threshold: 0.3 }
      } else {
        url  = '/api/autolabel/model-jobs'
        body = { ...body, artifactId: cfg.artifactId, confidenceThreshold: cfg.confidenceThreshold ?? 0.25, iouThreshold: 0.45 }
      }

      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'AI识别启动失败')
      }
      const job = await res.json()
      // Poll until done, then reload annotations
      const poll = async () => {
        const r  = await fetch(`/api/autolabel/jobs/${job.id}`)
        if (!r.ok) { setAiRunning(false); return }
        const j  = await r.json()
        if (j.status === 'succeeded') {
          setAiRunning(false)
          await loadAnns(current.id)
          // Refresh thumb status
          await fetchThumbPage(thumbPage)
        } else if (j.status === 'failed' || j.status === 'canceled') {
          setAiRunning(false)
          alert('AI识别失败：' + (j.status === 'failed' ? '请检查配置' : '已取消'))
        } else {
          aiPollRef.current = setTimeout(poll, 1500)
        }
      }
      poll()
    } catch (e: unknown) {
      setAiRunning(false)
      alert('AI识别失败：' + (e instanceof Error ? e.message : String(e)))
    }
  }

  useEffect(() => {
    return () => { if (aiPollRef.current) clearTimeout(aiPollRef.current) }
  }, [])

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // 1-9: Select class
      if (e.key >= '1' && e.key <= '9') {
        const idx = Number(e.key) - 1
        if (classes[idx]) setClassId(classes[idx].id)
      }
      if (e.key === 'Escape') setSelectedId(null)
      if ((e.key === 'Backspace' || e.key === 'Delete') && selectedId != null) {
        e.preventDefault()
        onDelete(selectedId)
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) performRedo(); else performUndo()
      }
      // Arrow keys: move selected box or navigate images
      if (selectedId != null && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault()
        const step = e.shiftKey ? 5 : 1
        const a = anns.find(x => x.id === selectedId)
        if (!a) return
        let { x, y } = a
        if (e.key === 'ArrowUp')    y -= step
        if (e.key === 'ArrowDown')  y += step
        if (e.key === 'ArrowLeft')  x -= step
        if (e.key === 'ArrowRight') x += step
        x = Math.max(0, Math.min((current?.width ?? 1) - a.w, x))
        y = Math.max(0, Math.min((current?.height ?? 1) - a.h, y))
        onUpdate(a.id, { x, y })
      }
      if (selectedId === null && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        e.preventDefault()
        const idx        = thumbImages.findIndex(img => img.id === current?.id)
        const totalPages = Math.ceil(totalImages / THUMB_SIZE)

        if (e.key === 'ArrowRight') {
          if (idx < thumbImages.length - 1) {
            // 当前页还有下一张
            setCurrent(thumbImages[idx + 1])
          } else if (thumbPage < totalPages) {
            // 到达当前页末尾 → 自动翻到下一页，选第一张
            fetchThumbPage(thumbPage + 1).then(items => {
              if (items.length > 0) setCurrent(items[0])
            })
          }
        }
        if (e.key === 'ArrowLeft') {
          if (idx > 0) {
            // 当前页还有上一张
            setCurrent(thumbImages[idx - 1])
          } else if (thumbPage > 1) {
            // 到达当前页开头 → 自动翻到上一页，选最后一张
            fetchThumbPage(thumbPage - 1).then(items => {
              if (items.length > 0) setCurrent(items[items.length - 1])
            })
          }
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, selectedId, anns, current, thumbImages, thumbPage, totalImages, fetchThumbPage])

  // ── Render ──────────────────────────────────────────────────────────────────

  const totalPages  = Math.ceil(totalImages / THUMB_SIZE)
  const currentIdx  = thumbImages.findIndex(img => img.id === current?.id)

  return (
    <div style={{
      display: 'grid',
      gridTemplateRows: '48px auto 1fr',
      height: 'calc(100vh - 64px)',   /* 64px = ProjectDetail 顶部 Header 高度 */
      overflow: 'hidden',
      background: '#f5f7fa',
    }}>

      {/* ══════════ Row 1: Header ══════════ */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '0 16px',
        background: '#fff',
        borderBottom: '1px solid #eee',
        flexShrink: 0,
      }}>
        <button
          onClick={onBack}
          style={{
            padding: '5px 12px',
            border: '1px solid #d9d9d9',
            borderRadius: 6,
            cursor: 'pointer',
            background: '#fff',
            fontSize: 13,
          }}
        >
          ← 返回
        </button>

        <strong style={{ fontSize: 14, color: '#333' }}>标注工作台</strong>

        <div style={{ marginLeft: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 13, color: '#666' }}>筛选：</span>
          <select
            value={filter}
            onChange={e => setFilter(e.target.value)}
            style={{ fontSize: 13, padding: '3px 8px', border: '1px solid #d9d9d9', borderRadius: 4 }}
          >
            <option value="all">全部</option>
            <option value="unlabeled">未完成</option>
            <option value="labeled">已完成</option>
            <option value="unannotated">未标注</option>
            <option value="ai_pending">AI待审</option>
            <option value="annotated">已标注</option>
          </select>
        </div>

        <div style={{ marginLeft: 'auto', fontSize: 12, color: '#888' }}>
          第 {thumbPage}/{totalPages || 1} 页 · 共 {totalImages} 张
          {currentIdx >= 0 && ` · 当前第 ${(thumbPage - 1) * THUMB_SIZE + currentIdx + 1} 张`}
        </div>

        {aiRunning && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontSize: 12, color: '#722ed1', padding: '4px 10px',
            background: '#f9f0ff', borderRadius: 12, border: '1px solid #d3adf7',
          }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#722ed1', animation: 'pulse 1s infinite' }} />
            AI识别中...
          </div>
        )}
      </header>

      {/* ══════════ Row 2: Thumbnail Strip ══════════ */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 12px',
        background: '#fff',
        borderBottom: '1px solid #eee',
        flexShrink: 0,
      }}>
        {/* Left arrow */}
        <button
          onClick={() => fetchThumbPage(thumbPage - 1)}
          disabled={thumbPage <= 1 || loadingThumb}
          style={{
            width: 28, height: 28, flexShrink: 0,
            border: '1px solid #d9d9d9', borderRadius: 4,
            cursor: thumbPage <= 1 ? 'not-allowed' : 'pointer',
            background: thumbPage <= 1 ? '#f5f5f5' : '#fff',
            color: thumbPage <= 1 ? '#ccc' : '#333',
            fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          ‹
        </button>

        {/* Thumbnail grid */}
        <div style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: 'repeat(20, 1fr)',
          gap: 4,
          minWidth: 0,
        }}>
          {loadingThumb ? (
            // Loading placeholders
            Array.from({ length: THUMB_SIZE }).map((_, i) => (
              <div key={i} style={{ aspectRatio: '1/1', background: '#f0f0f0', borderRadius: 3 }} />
            ))
          ) : (
            thumbImages.map(img => {
              const isSelected = current?.id === img.id
              const bg = STATUS_BG[img.status]
              const lbl = STATUS_LBL[img.status]
              return (
                <div
                  key={img.id}
                  onClick={() => setCurrent(img)}
                  style={{
                    position: 'relative',
                    borderRadius: 3,
                    overflow: 'hidden',
                    cursor: 'pointer',
                    border: isSelected ? '2px solid #1677ff' : '2px solid transparent',
                    boxShadow: isSelected ? '0 0 0 1px rgba(22,119,255,0.2)' : 'none',
                    background: '#f0f0f0',
                    flexShrink: 0,
                  }}
                >
                  <img
                    src={thumbUrl(img)}
                    alt=""
                    loading="lazy"
                    style={{ width: '100%', aspectRatio: '1/1', objectFit: 'cover', display: 'block' }}
                  />
                  {/* Status badge */}
                  {lbl && (
                    <div style={{
                      position: 'absolute', bottom: 0, left: 0, right: 0,
                      background: bg, color: '#fff', fontSize: 9,
                      textAlign: 'center', lineHeight: '13px',
                    }}>
                      {lbl}
                    </div>
                  )}
                  {/* Labeled checkmark */}
                  {img.labeled && (
                    <div style={{
                      position: 'absolute', top: 2, right: 2,
                      width: 13, height: 13, borderRadius: '50%',
                      background: '#52c41a', color: '#fff',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 8, fontWeight: 700, lineHeight: 1,
                    }}>
                      ✓
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Right arrow */}
        <button
          onClick={() => fetchThumbPage(thumbPage + 1)}
          disabled={thumbPage >= totalPages || loadingThumb}
          style={{
            width: 28, height: 28, flexShrink: 0,
            border: '1px solid #d9d9d9', borderRadius: 4,
            cursor: thumbPage >= totalPages ? 'not-allowed' : 'pointer',
            background: thumbPage >= totalPages ? '#f5f5f5' : '#fff',
            color: thumbPage >= totalPages ? '#ccc' : '#333',
            fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          ›
        </button>
      </div>

      {/* ══════════ Row 3: Canvas + Right Panel ══════════ */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 220px',
        overflow: 'hidden',
      }}>

        {/* Canvas area (left) */}
        <div
          style={{ position: 'relative', overflow: 'hidden', background: '#f0f0f0' }}
          onContextMenu={handleContextMenu}
        >
          {current ? (
            <CanvasBBox
              imageUrl={imageUrl}
              naturalWidth={current.width}
              naturalHeight={current.height}
              boxes={anns.map(a => ({
                id: a.id, x: a.x, y: a.y, w: a.w, h: a.h,
                classId: a.classId, source: a.source,
              }))}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onCreate={onCreate}
              onDelete={onDelete}
              onUpdate={onUpdate}
              onSelectClassId={classId}
            />
          ) : (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              height: '100%', color: '#aaa', fontSize: 14,
            }}>
              请在上方选择一张图片开始标注
            </div>
          )}

          {/* ── Right-click context menu ── */}
          {ctxMenu && (
            <div
              ref={ctxMenuRef}
              style={{
                position: 'fixed',
                top: ctxMenu.y,
                left: ctxMenu.x,
                zIndex: 9999,
                background: '#fff',
                border: '1px solid #e8e8e8',
                borderRadius: 8,
                boxShadow: '0 6px 20px rgba(0,0,0,0.12)',
                minWidth: 170,
                overflow: 'hidden',
                fontSize: 13,
              }}
            >
              {[
                { label: '🗑️  删除当前图片', action: handleDeleteImage, danger: true },
                { label: '🧹  清空当前标注', action: handleClearAnnotations, danger: false },
                null, // divider
                { label: '⚡  AI识别（默认配置）', action: handleQuickAi, danger: false },
              ].map((item, i) =>
                item === null ? (
                  <div key={i} style={{ height: 1, background: '#f0f0f0', margin: '2px 0' }} />
                ) : (
                  <button
                    key={i}
                    onClick={item.action}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      padding: '9px 14px', border: 'none', background: 'transparent',
                      cursor: 'pointer', color: item.danger ? '#ff4d4f' : '#333',
                      fontSize: 13,
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = '#f5f5f5')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    {item.label}
                  </button>
                )
              )}
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <aside style={{
          borderLeft: '1px solid #e8e8e8',
          background: '#fff',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: '10px 10px',
          gap: 0,
        }}>

          {/* Class list */}
          <div style={{ flexShrink: 0 }}>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: '#333' }}>
              类别
              <span style={{ fontWeight: 400, color: '#aaa', fontSize: 11, marginLeft: 4 }}>
                （按 1-9 快选）
              </span>
            </div>
            {classes.map(c => {
              const color = getClassColor(c.id)
              const cnt   = anns.filter(a => a.classId === c.id).length
              return (
                <div
                  key={c.id}
                  onClick={() => setClassId(c.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 7,
                    padding: '6px 8px', marginBottom: 3, borderRadius: 5,
                    cursor: 'pointer',
                    background: classId === c.id ? `${color}18` : 'transparent',
                    border: `2px solid ${classId === c.id ? color : 'transparent'}`,
                    transition: 'background 0.1s',
                  }}
                >
                  <div style={{
                    width: 13, height: 13, borderRadius: 3,
                    backgroundColor: color, flexShrink: 0,
                  }} />
                  <span style={{ fontSize: 13, flex: 1, fontWeight: classId === c.id ? 600 : 400 }}>
                    {c.name}
                  </span>
                  {cnt > 0 && (
                    <span style={{
                      fontSize: 11, color: '#fff', background: color,
                      borderRadius: 10, padding: '1px 6px', lineHeight: 1.5,
                    }}>
                      {cnt}
                    </span>
                  )}
                </div>
              )
            })}
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: '#f0f0f0', margin: '10px 0', flexShrink: 0 }} />

          {/* Annotation list */}
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginBottom: 6, flexShrink: 0,
            }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>
                标注（{anns.length}）
              </span>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  onClick={performUndo}
                  title="Ctrl+Z"
                  style={{ fontSize: 12, padding: '2px 8px', border: '1px solid #d9d9d9', borderRadius: 4, cursor: 'pointer', background: '#fff' }}
                >
                  ↩ 撤销
                </button>
                <button
                  onClick={performRedo}
                  title="Ctrl+Shift+Z"
                  style={{ fontSize: 12, padding: '2px 8px', border: '1px solid #d9d9d9', borderRadius: 4, cursor: 'pointer', background: '#fff' }}
                >
                  ↪ 重做
                </button>
              </div>
            </div>

            {anns.length === 0 && (
              <div style={{ fontSize: 12, color: '#bbb', textAlign: 'center', padding: '16px 0' }}>
                暂无标注
              </div>
            )}

            {anns.map(a => {
              const color     = getClassColor(a.classId)
              const className = classes.find(c => c.id === a.classId)?.name ?? `类别${a.classId}`
              return (
                <div
                  key={a.id}
                  onClick={() => setSelectedId(a.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '5px 8px', marginBottom: 3, borderRadius: 4,
                    borderLeft: `3px solid ${color}`,
                    background: selectedId === a.id ? `${color}15` : '#fafafa',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ width: 9, height: 9, borderRadius: 2, backgroundColor: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 12, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {className}
                  </span>
                  {a.source === 'ai' && (
                    <span style={{ fontSize: 10, color: '#722ed1', background: '#f9f0ff', padding: '1px 4px', borderRadius: 3 }}>AI</span>
                  )}
                  <select
                    value={a.classId}
                    onChange={e => onUpdate(a.id, { classId: Number(e.target.value) })}
                    onClick={e => e.stopPropagation()}
                    style={{ fontSize: 11, padding: '1px 2px', border: '1px solid #e8e8e8', borderRadius: 3, maxWidth: 56 }}
                  >
                    {classes.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                  <button
                    onClick={e => { e.stopPropagation(); onDelete(a.id) }}
                    style={{ fontSize: 12, padding: '1px 5px', border: '1px solid #ffccc7', borderRadius: 3, color: '#ff4d4f', background: '#fff5f5', cursor: 'pointer' }}
                  >
                    ×
                  </button>
                </div>
              )
            })}
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: '#f0f0f0', margin: '10px 0', flexShrink: 0 }} />

          {/* Actions */}
          <div style={{ flexShrink: 0 }}>
            {/* Labeled status + button */}
            {current && (
              <div style={{
                padding: '8px 10px',
                background: current.labeled ? '#f6ffed' : '#fff7e6',
                borderRadius: 6,
                border: `1px solid ${current.labeled ? '#b7eb8f' : '#ffd591'}`,
                marginBottom: 8,
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <span style={{ fontSize: 13, color: current.labeled ? '#52c41a' : '#fa8c16', fontWeight: 500 }}>
                  {current.labeled ? '✓ 已完成' : '未完成'}
                </span>
                <button
                  onClick={markCurrentLabeled}
                  disabled={current.labeled}
                  style={{
                    fontSize: 12, padding: '4px 12px', borderRadius: 4, border: 'none',
                    background: current.labeled ? '#f0f0f0' : '#52c41a',
                    color: current.labeled ? '#aaa' : '#fff',
                    cursor: current.labeled ? 'not-allowed' : 'pointer',
                    fontWeight: 500,
                  }}
                >
                  {current.labeled ? '已完成' : '标记完成'}
                </button>
              </div>
            )}

            {/* Tips */}
            <div style={{ fontSize: 11, color: '#bbb', lineHeight: 1.6 }}>
              拖拽创建框 · 点击选中 · Delete删除<br />
              方向键微调 · Shift加速 · Ctrl+Z撤销
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
