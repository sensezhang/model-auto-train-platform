import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CanvasBBox, BBox, getClassColor } from '../components/CanvasBBox'

type ImageItem = {
  id: number;
  path: string;
  thumbnailPath?: string;
  displayPath?: string;
  width: number;
  height: number;
  status: 'unannotated'|'ai_pending'|'annotated'
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
  const [filter, setFilter] = useState<'all'|'unannotated'|'ai_pending'|'annotated'>('all')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const undoStack = useRef<any[]>([])
  const redoStack = useRef<any[]>([])
  const imageListRef = useRef<HTMLUListElement>(null)

  // 图片多选和删除相关状态
  const [selectMode, setSelectMode] = useState(false)
  const [selectedImages, setSelectedImages] = useState<Set<number>>(new Set())
  const [deletingImages, setDeletingImages] = useState(false)

  // 加载第一页图片
  const loadImages = async (resetPage = true) => {
    const currentPage = resetPage ? 1 : page
    const statusParam = filter === 'all' ? '' : `&status=${filter}`
    const res = await fetch(`/api/projects/${projectId}/images?page=${currentPage}&page_size=${PAGE_SIZE}${statusParam}`)
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
      const statusParam = filter === 'all' ? '' : `&status=${filter}`
      const res = await fetch(`/api/projects/${projectId}/images?page=${nextPage}&page_size=${PAGE_SIZE}${statusParam}`)
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

      // 如果当前图片被删除，清除选中
      if (current && selectedImages.has(current.id)) {
        setCurrent(null)
        setAnns([])
      }

      // 退出选择模式并刷新列表
      exitSelectMode()
      loadImages()
    } catch (e: any) {
      alert('删除失败：' + (e.message || e))
    } finally {
      setDeletingImages(false)
    }
  }

  // 优先使用标注用图(displayPath)，其次使用原图(path)
  const imageUrl = useMemo(() => {
    if (!current) return ''
    // 使用标注用图（较小，加载快）或原图
    return `/files/${current.displayPath || current.path}`
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
    // 更新当前图片的状态（如果从未标注变为已标注）
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // 数字键选择类别（1-9）
      if (e.key >= '1' && e.key <= '9') {
        const idx = Number(e.key) - 1
        if (classes[idx]) setClassId(classes[idx].id)
      }
      if (e.key === 'Escape') setSelectedId(null)
      if ((e.key === 'Backspace' || e.key === 'Delete') && selectedId != null) {
        e.preventDefault()
        onDelete(selectedId)
      }
      // 撤销/重做
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) performRedo(); else performUndo()
      }
      // 方向键微调
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
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [classes, selectedId, anns, current])

  return (
    <div style={{ display: 'grid', gridTemplateRows: '48px 1fr', height: '100vh' }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px', borderBottom: '1px solid #eee' }}>
        <button onClick={onBack} style={{ padding: '6px 10px' }}>← 返回</button>
        <strong>项目 {projectId} 标注工作台</strong>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
            <option value='all'>全部</option>
            <option value='unannotated'>未标注</option>
            <option value='ai_pending'>AI待审</option>
            <option value='annotated'>已标注</option>
          </select>
        </div>
      </header>
      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 280px', gap: 12, padding: 12, overflow: 'hidden' }}>
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
            <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
              <span style={{ fontSize: 12, color: '#666' }}>已选 {selectedImages.size} 张</span>
              <button
                onClick={deleteSelectedImages}
                disabled={deletingImages}
                style={{
                  fontSize: 12,
                  padding: '4px 12px',
                  color: 'white',
                  backgroundColor: deletingImages ? '#ccc' : '#ff4d4f',
                  border: 'none',
                  borderRadius: 4,
                  cursor: deletingImages ? 'not-allowed' : 'pointer'
                }}
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
                  gap: 8
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
                  src={`/files/${img.thumbnailPath || img.path}`}
                  alt=""
                  style={{
                    width: 60,
                    height: 60,
                    objectFit: 'cover',
                    borderRadius: 4,
                    backgroundColor: '#f0f0f0',
                    flexShrink: 0
                  }}
                  loading="lazy"
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: '#333', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.path.split('/').slice(-1)[0]}</div>
                  <div style={{ fontSize: 11, color: '#999' }}>状态：{img.status}</div>
                </div>
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
        <aside style={{ borderLeft: '1px solid #eee', paddingLeft: 8, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ flex: 1, overflow: 'auto' }}>
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
              3) Delete 删除，方向键微调（Shift加速），Ctrl+Z 撤销，Ctrl+Shift+Z 重做；
              4) 右侧列表可快速切换类别。
            </div>
          </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
