import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CanvasBBox, BBox, getClassColor } from '../components/CanvasBBox'

type ImageItem = {
  id: number;
  path: string;
  thumbnailPath?: string;
  displayPath?: string;
  width: number;
  height: number;
  status: 'unannotated'|'ai_pending'|'annotated';
  labeled: boolean;
}
type ClassItem = { id: number; name: string }
type Annotation = { id: number; imageId: number; classId: number; x: number; y: number; w: number; h: number; source: 'manual'|'ai' }

const PAGE_SIZE = 50

export const Annotator: React.FC<{ projectId: number; onBack: () => void }> = ({ projectId, onBack }) => {
  const [images, setImages] = useState<ImageItem[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [page, setPage] = useState(1)
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [current, setCurrent] = useState<ImageItem | null>(null)
  const [anns, setAnns] = useState<Annotation[]>([])
  const [classId, setClassId] = useState<number | null>(null)
  const [filter, setFilter] = useState<'all'|'unannotated'|'ai_pending'|'annotated'|'labeled'|'unlabeled'>('all')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const undoStack = useRef<any[]>([])
  const redoStack = useRef<any[]>([])
  const imageListRef = useRef<HTMLUListElement>(null)

  // 图片多选和删除相关状态
  const [selectMode, setSelectMode] = useState(false)
  const [selectedImages, setSelectedImages] = useState<Set<number>>(new Set())
  const [deletingImages, setDeletingImages] = useState(false)

  // AI 自动标注面板状态
  const [aiPanelOpen, setAiPanelOpen] = useState(false)
  const [aiClassId, setAiClassId] = useState<number | null>(null)
  const [aiPrompt, setAiPrompt] = useState('')
  const [aiScope, setAiScope] = useState<'unlabeled'|'selected'>('unlabeled')
  const [aiJobId, setAiJobId] = useState<number | null>(null)
  const [aiJobStatus, setAiJobStatus] = useState<any>(null)
  const [aiStarting, setAiStarting] = useState(false)
  const aiPollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // 加载第一页图片
  const loadImages = async (resetPage = true) => {
    const currentPage = resetPage ? 1 : page
    let statusParam = ''
    let labeledParam = ''
    if (filter === 'labeled') {
      labeledParam = '&labeled=true'
    } else if (filter === 'unlabeled') {
      labeledParam = '&labeled=false'
    } else if (filter !== 'all') {
      statusParam = `&status=${filter}`
    }
    const res = await fetch(`/api/projects/${projectId}/images?page=${currentPage}&page_size=${PAGE_SIZE}${statusParam}${labeledParam}`)
    const data = await res.json()

    if (resetPage) {
      setImages(data.items)
      setPage(1)
      if (!current && data.items.length > 0) setCurrent(data.items[0])
    } else {
      setImages(prev => [...prev, ...data.items])
    }

    setTotalImages(data.total)
    setHasMore(data.has_more)
  }

  // 加载更多图片
  const loadMoreImages = useCallback(async () => {
    if (loadingMore || !hasMore) return
    setLoadingMore(true)
    try {
      const nextPage = page + 1
      let statusParam = ''
      let labeledParam = ''
      if (filter === 'labeled') {
        labeledParam = '&labeled=true'
      } else if (filter === 'unlabeled') {
        labeledParam = '&labeled=false'
      } else if (filter !== 'all') {
        statusParam = `&status=${filter}`
      }
      const res = await fetch(`/api/projects/${projectId}/images?page=${nextPage}&page_size=${PAGE_SIZE}${statusParam}${labeledParam}`)
      const data = await res.json()

      setImages(prev => [...prev, ...data.items])
      setPage(nextPage)
      setHasMore(data.has_more)
    } finally {
      setLoadingMore(false)
    }
  }, [page, filter, projectId, loadingMore, hasMore])

  // 滚动加载检测
  const handleScroll = useCallback((e: React.UIEvent<HTMLUListElement>) => {
    const target = e.target as HTMLUListElement
    const scrollBottom = target.scrollHeight - target.scrollTop - target.clientHeight
    if (scrollBottom < 100 && hasMore && !loadingMore) {
      loadMoreImages()
    }
  }, [hasMore, loadingMore, loadMoreImages])

  const loadClasses = async () => {
    const res = await fetch(`/api/projects/${projectId}/classes`)
    const data = await res.json()
    setClasses(data)
    if (data.length > 0 && classId == null) setClassId(data[0].id)
    if (data.length > 0 && aiClassId == null) setAiClassId(data[0].id)
  }

  const loadAnns = async (imageId: number) => {
    const res = await fetch(`/api/images/${imageId}/annotations`)
    setAnns(await res.json())
  }

  useEffect(() => { loadClasses() }, [])
  useEffect(() => { loadImages() }, [filter])
  useEffect(() => { if (current) loadAnns(current.id) }, [current?.id])
  useEffect(() => { setSelectedId(null) }, [current?.id])

  // 切换图片选择
  const toggleImageSelection = (imageId: number) => {
    setSelectedImages(prev => {
      const next = new Set(prev)
      if (next.has(imageId)) {
        next.delete(imageId)
      } else {
        next.add(imageId)
      }
      return next
    })
  }

  // 全选/取消全选当前页
  const toggleSelectAll = () => {
    if (selectedImages.size === images.length) {
      setSelectedImages(new Set())
    } else {
      setSelectedImages(new Set(images.map(img => img.id)))
    }
  }

  // 退出选择模式
  const exitSelectMode = () => {
    setSelectMode(false)
    setSelectedImages(new Set())
  }

  // 删除选中的图片
  const deleteSelectedImages = async () => {
    if (selectedImages.size === 0) return
    if (!confirm(`确认删除选中的 ${selectedImages.size} 张图片吗？将同时删除其关联的标注数据，此操作不可恢复！`)) return

    setDeletingImages(true)
    try {
      const res = await fetch(`/api/projects/${projectId}/images`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_ids: Array.from(selectedImages) })
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg)
      }
      const result = await res.json()
      alert(`删除成功：${result.deleted_images} 张图片，${result.deleted_annotations} 条标注`)

      if (current && selectedImages.has(current.id)) {
        setCurrent(null)
        setAnns([])
      }

      exitSelectMode()
      loadImages()
    } catch (e: any) {
      alert('删除失败：' + (e.message || e))
    } finally {
      setDeletingImages(false)
    }
  }

  // 批量标记完成
  const bulkMarkLabeled = async (labeled: boolean) => {
    if (selectedImages.size === 0) return
    try {
      const res = await fetch(`/api/projects/${projectId}/images/mark-labeled`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_ids: Array.from(selectedImages), labeled })
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      alert(`已${labeled ? '标记' : '取消标记'} ${result.updated} 张图片`)

      // 更新本地状态
      setImages(prev => prev.map(img =>
        selectedImages.has(img.id) ? { ...img, labeled } : img
      ))
      if (current && selectedImages.has(current.id)) {
        setCurrent(prev => prev ? { ...prev, labeled } : prev)
      }
    } catch (e: any) {
      alert('操作失败：' + (e.message || e))
    }
  }

  // 标记当前图片完成，并跳转下一张未完成
  const markCurrentLabeled = async () => {
    if (!current) return
    try {
      const res = await fetch(`/api/images/${current.id}/labeled`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labeled: true })
      })
      if (!res.ok) throw new Error(await res.text())

      // 更新本地状态
      const updated = { ...current, labeled: true }
      setCurrent(updated)
      setImages(prev => prev.map(img => img.id === current.id ? { ...img, labeled: true } : img))

      // 跳转到下一张未完成的图片
      const currentIdx = images.findIndex(img => img.id === current.id)
      const nextUnlabeled = images.find((img, idx) => idx > currentIdx && !img.labeled)
      if (nextUnlabeled) {
        setCurrent(nextUnlabeled)
      }
    } catch (e: any) {
      alert('标记失败：' + (e.message || e))
    }
  }

  // 优先使用标注用图(displayPath)，其次使用原图(path)；统一将反斜杠替换为正斜杠
  const imageUrl = useMemo(() => {
    if (!current) return ''
    return `/files/${(current.displayPath || current.path).replace(/\\/g, '/')}`
  }, [current])

  const onCreate = async (box: { x: number; y: number; w: number; h: number; classId: number; source?: 'manual'|'ai' }) => {
    if (!current) return
    const res = await fetch('/api/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageId: current.id, ...box })
    })
    if (!res.ok) return
    const created = await res.json()
    undoStack.current.push({ type: 'create', after: created })
    redoStack.current = []
    await loadAnns(current.id)
    if (current.status === 'unannotated') {
      setImages(prev => prev.map(img => img.id === current.id ? { ...img, status: 'annotated' } : img))
      setCurrent(prev => prev ? { ...prev, status: 'annotated' } : prev)
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
    const res = await fetch(`/api/annotations/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) })
    if (!res.ok) return
    const after = await res.json()
    undoStack.current.push({ type: 'update', before, after })
    redoStack.current = []
    if (current) await loadAnns(current.id)
  }

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

  // AI 自动标注：启动 job
  const startAiJob = async () => {
    if (!aiClassId) { alert('请选择目标类别'); return }
    if (!aiPrompt.trim()) { alert('请输入检测提示词'); return }
    setAiStarting(true)
    try {
      const body: any = {
        projectId,
        classId: aiClassId,
        prompt: aiPrompt.trim(),
        threshold: 0.3,
      }
      if (aiScope === 'selected' && selectedImages.size > 0) {
        body.imageIds = Array.from(selectedImages)
      }

      const res = await fetch('/api/autolabel/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || '启动失败')
      }
      const job = await res.json()
      setAiJobId(job.id)
      setAiJobStatus(job)
      pollAiJob(job.id)
    } catch (e: any) {
      alert('AI 标注启动失败：' + (e.message || e))
    } finally {
      setAiStarting(false)
    }
  }

  const pollAiJob = (jobId: number) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/autolabel/jobs/${jobId}`)
        if (!res.ok) return
        const job = await res.json()
        setAiJobStatus(job)
        if (job.status === 'succeeded' || job.status === 'failed') {
          if (job.status === 'succeeded') loadImages()
          return
        }
        aiPollRef.current = setTimeout(poll, 2000)
      } catch {
        aiPollRef.current = setTimeout(poll, 3000)
      }
    }
    poll()
  }

  useEffect(() => {
    return () => {
      if (aiPollRef.current) clearTimeout(aiPollRef.current)
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
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
      if (selectedId != null && ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) {
        e.preventDefault()
        const step = e.shiftKey ? 5 : 1
        const a = anns.find(x => x.id === selectedId)
        if (!a) return
        let { x, y } = a
        if (e.key === 'ArrowUp') y -= step
        if (e.key === 'ArrowDown') y += step
        if (e.key === 'ArrowLeft') x -= step
        if (e.key === 'ArrowRight') x += step
        x = Math.max(0, Math.min((current?.width ?? 1) - a.w, x))
        y = Math.max(0, Math.min((current?.height ?? 1) - a.h, y))
        onUpdate(a.id, { x, y })
      }
      if (selectedId === null && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        e.preventDefault()
        const idx = images.findIndex(img => img.id === current?.id)
        if (e.key === 'ArrowDown' && idx < images.length - 1) setCurrent(images[idx + 1])
        if (e.key === 'ArrowUp' && idx > 0) setCurrent(images[idx - 1])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, selectedId, anns, current, images])

  return (
    <div style={{ display: 'grid', gridTemplateRows: '48px 1fr', height: '100vh' }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px', borderBottom: '1px solid #eee' }}>
        <button onClick={onBack} style={{ padding: '6px 10px' }}>← 返回</button>
        <strong>项目 {projectId} 标注工作台</strong>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
            <option value='all'>全部</option>
            <option value='unlabeled'>未完成</option>
            <option value='labeled'>已完成</option>
            <option value='unannotated'>未标注</option>
            <option value='ai_pending'>AI待审</option>
            <option value='annotated'>已标注</option>
          </select>
        </div>
      </header>
      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 280px', gap: 12, padding: 12, overflow: 'hidden' }}>
        {/* 左侧：图片列表 */}
        <aside style={{ borderRight: '1px solid #eee', paddingRight: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8, flexShrink: 0 }}>
            <div style={{ fontWeight: 600 }}>图片列表（{images.length}/{totalImages}）</div>
            {!selectMode ? (
              <button
                onClick={() => setSelectMode(true)}
                style={{ fontSize: 12, padding: '4px 8px', color: '#666', border: '1px solid #d9d9d9', borderRadius: 4, cursor: 'pointer' }}
              >
                选择
              </button>
            ) : (
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  onClick={toggleSelectAll}
                  style={{ fontSize: 12, padding: '4px 8px', color: '#1890ff', border: '1px solid #1890ff', borderRadius: 4, cursor: 'pointer' }}
                >
                  {selectedImages.size === images.length ? '取消全选' : '全选'}
                </button>
                <button
                  onClick={exitSelectMode}
                  style={{ fontSize: 12, padding: '4px 8px', color: '#666', border: '1px solid #d9d9d9', borderRadius: 4, cursor: 'pointer' }}
                >
                  取消
                </button>
              </div>
            )}
          </div>
          {selectMode && selectedImages.size > 0 && (
            <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: '#666' }}>已选 {selectedImages.size} 张</span>
              <button
                onClick={() => bulkMarkLabeled(true)}
                style={{ fontSize: 12, padding: '4px 10px', color: 'white', backgroundColor: '#52c41a', border: 'none', borderRadius: 4, cursor: 'pointer' }}
              >
                批量完成
              </button>
              <button
                onClick={deleteSelectedImages}
                disabled={deletingImages}
                style={{ fontSize: 12, padding: '4px 10px', color: 'white', backgroundColor: deletingImages ? '#ccc' : '#ff4d4f', border: 'none', borderRadius: 4, cursor: deletingImages ? 'not-allowed' : 'pointer' }}
              >
                {deletingImages ? '删除中...' : '删除选中'}
              </button>
            </div>
          )}
          <ul
            ref={imageListRef}
            onScroll={handleScroll}
            style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 8, overflow: 'auto', flex: 1 }}
          >
            {images.map(img => (
              <li
                key={img.id}
                onClick={() => selectMode ? toggleImageSelection(img.id) : setCurrent(img)}
                style={{
                  padding: 8,
                  border: selectedImages.has(img.id) ? '2px solid #1890ff' : '1px solid #eee',
                  borderRadius: 6,
                  cursor: 'pointer',
                  background: selectedImages.has(img.id) ? '#e6f7ff' : (current?.id === img.id ? '#f5f9ff' : '#fff'),
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8,
                  position: 'relative',
                }}
              >
                {selectMode && (
                  <input
                    type="checkbox"
                    checked={selectedImages.has(img.id)}
                    onChange={() => toggleImageSelection(img.id)}
                    onClick={e => e.stopPropagation()}
                    style={{ marginTop: 2 }}
                  />
                )}
                {/* 缩略图 */}
                <img
                  src={`/files/${(img.thumbnailPath || img.path).replace(/\\/g, '/')}`}
                  alt=""
                  style={{ width: 60, height: 60, objectFit: 'cover', borderRadius: 4, backgroundColor: '#f0f0f0', flexShrink: 0 }}
                  loading="lazy"
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: '#333', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {img.path.split('/').slice(-1)[0]}
                  </div>
                  <div style={{ fontSize: 11, color: '#999' }}>状态：{img.status}</div>
                </div>
                {/* 已完成标记 */}
                {img.labeled && (
                  <span
                    title="已完成"
                    style={{
                      position: 'absolute', top: 4, right: 4,
                      width: 18, height: 18, borderRadius: '50%',
                      backgroundColor: '#52c41a', color: 'white',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 11, fontWeight: 700, lineHeight: 1,
                    }}
                  >
                    ✓
                  </span>
                )}
              </li>
            ))}
            {loadingMore && (
              <li style={{ padding: 8, textAlign: 'center', color: '#999', fontSize: 12 }}>加载中...</li>
            )}
            {!hasMore && images.length > 0 && (
              <li style={{ padding: 8, textAlign: 'center', color: '#ccc', fontSize: 12 }}>已加载全部</li>
            )}
          </ul>
        </aside>

        {/* 中间：画布 */}
        <main style={{ border: '1px solid #eee', borderRadius: 8, padding: 8, overflow: 'hidden' }}>
          {current ? (
            <CanvasBBox
              imageUrl={imageUrl}
              naturalWidth={current.width}
              naturalHeight={current.height}
              boxes={anns.map(a => ({ id: a.id, x: a.x, y: a.y, w: a.w, h: a.h, classId: a.classId, source: a.source }))}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onCreate={onCreate}
              onDelete={onDelete}
              onUpdate={onUpdate}
              onSelectClassId={classId}
            />
          ) : (
            <div style={{ padding: 24, color: '#666' }}>请选择一张图片开始标注</div>
          )}
        </main>

        {/* 右侧：类别 + 标注列表 + 操作 + AI面板 */}
        <aside style={{ borderLeft: '1px solid #eee', paddingLeft: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: 1, overflow: 'auto' }}>

            {/* 标记完成按钮 */}
            {current && (
              <div style={{ marginBottom: 12, padding: 10, backgroundColor: current.labeled ? '#f6ffed' : '#fff7e6', borderRadius: 6, border: `1px solid ${current.labeled ? '#b7eb8f' : '#ffd591'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 13, color: current.labeled ? '#52c41a' : '#fa8c16', fontWeight: 500 }}>
                    {current.labeled ? '✓ 已完成' : '未完成'}
                  </span>
                  <button
                    onClick={markCurrentLabeled}
                    disabled={current.labeled}
                    style={{
                      fontSize: 12,
                      padding: '4px 12px',
                      backgroundColor: current.labeled ? '#f0f0f0' : '#52c41a',
                      color: current.labeled ? '#999' : 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: current.labeled ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {current.labeled ? '已完成' : '标记完成'}
                  </button>
                </div>
              </div>
            )}

            <div style={{ fontWeight: 600, marginBottom: 8 }}>类别选择</div>
            <ul style={{ listStyle: 'none', padding: 0, display: 'grid', gap: 6 }}>
              {classes.map(c => (
                <li key={c.id}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '4px 6px', borderRadius: 4, background: classId === c.id ? `${getClassColor(c.id)}20` : 'transparent', border: classId === c.id ? `2px solid ${getClassColor(c.id)}` : '2px solid transparent' }}>
                    <span style={{ width: 16, height: 16, borderRadius: 3, backgroundColor: getClassColor(c.id), flexShrink: 0 }} />
                    <input type="radio" name="cls" checked={classId === c.id} onChange={() => setClassId(c.id)} style={{ display: 'none' }} />
                    <span style={{ fontWeight: classId === c.id ? 600 : 400 }}>{c.name}</span>
                  </label>
                </li>
              ))}
            </ul>

            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>当前标注（{anns.length}）</div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button onClick={performUndo} title="撤销 (Ctrl+Z)" style={{ fontSize: 12 }}>撤销</button>
                  <button onClick={performRedo} title="重做 (Ctrl+Shift+Z)" style={{ fontSize: 12 }}>重做</button>
                </div>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, display: 'grid', gap: 4 }}>
                {anns.map(a => {
                  const annColor = getClassColor(a.classId)
                  const className = classes.find(c => c.id === a.classId)?.name || `类别${a.classId}`
                  return (
                    <li key={a.id} style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: selectedId===a.id ? `${annColor}20` : 'transparent', padding: '4px 6px', borderRadius: 4, borderLeft: `3px solid ${annColor}` }}>
                      <span onClick={() => setSelectedId(a.id)} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: annColor, flexShrink: 0 }} />
                        <span>{className}</span>
                      </span>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <select value={a.classId} onChange={(e)=> onUpdate(a.id, { classId: Number(e.target.value) })} style={{ fontSize: 12 }}>
                          {classes.map(c => (<option key={c.id} value={c.id}>{c.name}</option>))}
                        </select>
                        <button onClick={() => onDelete(a.id)} style={{ fontSize: 12 }}>删除</button>
                      </div>
                    </li>
                  )
                })}
              </ul>
              <div style={{ marginTop: 12, color: '#999', fontSize: 12 }}>
                操作提示：
                1) 选择类别后，在画布上拖拽创建矩形框；
                2) 点击框以选中，拖动移动；拖拽红色小方块可缩放；
                3) Delete 删除，方向键微调（Shift加速），Ctrl+Z 撤销；
                4) 右侧列表可快速切换类别。
              </div>
            </div>

            {/* AI 自动标注面板 */}
            <div style={{ marginTop: 16, borderTop: '1px solid #eee', paddingTop: 12 }}>
              <button
                onClick={() => setAiPanelOpen(v => !v)}
                style={{ fontSize: 13, fontWeight: 600, background: 'none', border: 'none', cursor: 'pointer', color: '#722ed1', padding: 0, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                {aiPanelOpen ? '▼' : '▶'} AI 自动标注（GLM-4V）
              </button>

              {aiPanelOpen && (
                <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
                  {/* 目标类别 */}
                  <div>
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>目标类别</div>
                    <select
                      value={aiClassId ?? ''}
                      onChange={e => setAiClassId(Number(e.target.value))}
                      style={{ width: '100%', fontSize: 13, padding: '4px 6px' }}
                    >
                      {classes.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                    </select>
                  </div>

                  {/* 检测提示词 */}
                  <div>
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>检测提示词</div>
                    <input
                      value={aiPrompt}
                      onChange={e => setAiPrompt(e.target.value)}
                      placeholder="例如：识别图中所有的人"
                      style={{ width: '100%', fontSize: 13, padding: '4px 6px', boxSizing: 'border-box' }}
                    />
                  </div>

                  {/* 处理范围 */}
                  <div>
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>处理范围</div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 4 }}>
                        <input type="radio" name="aiScope" checked={aiScope === 'unlabeled'} onChange={() => setAiScope('unlabeled')} />
                        所有未完成
                      </label>
                      <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 4 }}>
                        <input type="radio" name="aiScope" checked={aiScope === 'selected'} onChange={() => setAiScope('selected')} />
                        仅选中（{selectedImages.size}张）
                      </label>
                    </div>
                  </div>

                  {/* 启动按钮 */}
                  <button
                    onClick={startAiJob}
                    disabled={aiStarting || (aiJobStatus?.status === 'running')}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: (aiStarting || aiJobStatus?.status === 'running') ? '#d9d9d9' : '#722ed1',
                      color: 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: (aiStarting || aiJobStatus?.status === 'running') ? 'not-allowed' : 'pointer',
                      fontSize: 13,
                    }}
                  >
                    {aiStarting ? '提交中...' : (aiJobStatus?.status === 'running' ? '标注中...' : '开始 AI 标注')}
                  </button>

                  {/* 进度显示 */}
                  {aiJobStatus && (
                    <div style={{ padding: 8, backgroundColor: '#f9f0ff', borderRadius: 4, border: '1px solid #d3adf7', fontSize: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span style={{ color: '#722ed1', fontWeight: 500 }}>
                          {aiJobStatus.status === 'succeeded' ? '标注完成' :
                           aiJobStatus.status === 'failed' ? '标注失败' :
                           aiJobStatus.status === 'running' ? '标注中...' : '等待中...'}
                        </span>
                        <span style={{ color: '#999' }}>
                          {aiJobStatus.processedCount}/{aiJobStatus.imagesCount} 张
                        </span>
                      </div>
                      {/* 进度条 */}
                      <div style={{ width: '100%', height: 6, backgroundColor: '#e6d5f7', borderRadius: 3, overflow: 'hidden', marginBottom: 4 }}>
                        <div style={{
                          width: `${aiJobStatus.imagesCount > 0 ? Math.round(aiJobStatus.processedCount / aiJobStatus.imagesCount * 100) : 0}%`,
                          height: '100%',
                          backgroundColor: aiJobStatus.status === 'failed' ? '#ff4d4f' :
                                          aiJobStatus.status === 'succeeded' ? '#52c41a' : '#722ed1',
                          transition: 'width 0.3s ease',
                          borderRadius: 3,
                        }} />
                      </div>
                      {aiJobStatus.status === 'succeeded' && (
                        <div style={{ color: '#52c41a' }}>共生成 {aiJobStatus.boxesCount} 个标注框</div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

          </div>
        </aside>
      </div>
    </div>
  )
}
