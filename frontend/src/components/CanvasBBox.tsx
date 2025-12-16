import React, { useEffect, useMemo, useRef, useState } from 'react'

export type BBox = { id?: number; x: number; y: number; w: number; h: number; classId: number; source?: 'manual'|'ai' }

type Props = {
  imageUrl: string
  naturalWidth: number
  naturalHeight: number
  boxes: BBox[]
  selectedId: number | null
  onSelect: (id: number | null) => void
  onCreate: (box: Omit<BBox, 'id'>) => void
  onDelete: (id: number) => void
  onUpdate: (id: number, patch: Partial<BBox>) => void
  onSelectClassId: number | null
  classColors?: Record<number, string>  // 类别ID到颜色的映射
}

type DragMode = 'draw' | 'move' | 'resize' | 'pan'
type Handle = 'nw'|'n'|'ne'|'e'|'se'|'s'|'sw'|'w'

const ZOOM_LEVELS = [1, 1.5, 2, 3, 4]

// 预定义的类别颜色（高对比度、易区分）
export const CLASS_COLORS = [
  '#FF6B6B',  // 红
  '#4ECDC4',  // 青
  '#45B7D1',  // 蓝
  '#96CEB4',  // 绿
  '#FFE66D',  // 黄
  '#DDA0DD',  // 紫
  '#FF8C42',  // 橙
  '#98D8C8',  // 薄荷
  '#F7DC6F',  // 金
  '#BB8FCE',  // 淡紫
  '#85C1E9',  // 天蓝
  '#F1948A',  // 粉红
]

// 根据类别ID获取颜色
export const getClassColor = (classId: number, customColors?: Record<number, string>): string => {
  if (customColors && customColors[classId]) {
    return customColors[classId]
  }
  return CLASS_COLORS[classId % CLASS_COLORS.length]
}

export const CanvasBBox: React.FC<Props> = ({ imageUrl, naturalWidth, naturalHeight, boxes, selectedId, onSelect, onCreate, onDelete, onUpdate, onSelectClassId, classColors }) => {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const [baseSize, setBaseSize] = useState<{ w: number; h: number }>({ w: naturalWidth, h: naturalHeight })
  const [zoomIndex, setZoomIndex] = useState(0)
  const [hoverId, setHoverId] = useState<number | null>(null)
  const [drag, setDrag] = useState<null | { mode: DragMode; startX: number; startY: number; init?: { x: number; y: number; w: number; h: number }; handle?: Handle; scrollStart?: { x: number; y: number } }>(null)

  const zoomLevel = ZOOM_LEVELS[zoomIndex]
  const renderSize = useMemo(() => ({
    w: Math.round(baseSize.w * zoomLevel),
    h: Math.round(baseSize.h * zoomLevel)
  }), [baseSize, zoomLevel])

  useEffect(() => {
    const updateSize = () => {
      const maxW = containerRef.current?.clientWidth || naturalWidth
      const baseScale = Math.min(1, maxW / naturalWidth)
      setBaseSize({ w: Math.round(naturalWidth * baseScale), h: Math.round(naturalHeight * baseScale) })
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [naturalWidth, naturalHeight])

  // 重置缩放当图片改变时
  useEffect(() => {
    setZoomIndex(0)
  }, [imageUrl])

  const scale = useMemo(() => ({ x: renderSize.w / naturalWidth, y: renderSize.h / naturalHeight }), [renderSize, naturalWidth, naturalHeight])

  const toLocal = (clientX: number, clientY: number) => {
    const scrollEl = scrollContainerRef.current
    const el = containerRef.current!
    const rect = el.getBoundingClientRect()
    const scrollX = scrollEl?.scrollLeft || 0
    const scrollY = scrollEl?.scrollTop || 0
    const lx = clientX - rect.left + scrollX
    const ly = clientY - rect.top + scrollY
    return {
      x: Math.max(0, Math.min(renderSize.w, lx)),
      y: Math.max(0, Math.min(renderSize.h, ly)),
    }
  }
  const toImage = (lx: number, ly: number) => ({ x: lx / scale.x, y: ly / scale.y })

  const clampBox = (b: { x: number; y: number; w: number; h: number }) => {
    let { x, y, w, h } = b
    if (w < 1) w = 1
    if (h < 1) h = 1
    if (x < 0) x = 0
    if (y < 0) y = 0
    if (x + w > naturalWidth) x = naturalWidth - w
    if (y + h > naturalHeight) y = naturalHeight - h
    return { x, y, w, h }
  }

  const hitHandleSize = 6
  const getHandles = (b: BBox) => {
    const rx = b.x * scale.x, ry = b.y * scale.y, rw = b.w * scale.x, rh = b.h * scale.y
    const cx = rx + rw / 2, cy = ry + rh / 2
    return [
      { k: 'nw', x: rx, y: ry }, { k: 'n', x: cx, y: ry }, { k: 'ne', x: rx + rw, y: ry },
      { k: 'e', x: rx + rw, y: cy }, { k: 'se', x: rx + rw, y: ry + rh }, { k: 's', x: cx, y: ry + rh },
      { k: 'sw', x: rx, y: ry + rh }, { k: 'w', x: rx, y: cy },
    ] as { k: Handle, x: number, y: number }[]
  }

  // 双击切换缩放
  const onDoubleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    const scrollEl = scrollContainerRef.current
    if (!scrollEl) return

    // 获取双击位置相对于图片的比例
    const rect = containerRef.current!.getBoundingClientRect()
    const clickX = e.clientX - rect.left + scrollEl.scrollLeft
    const clickY = e.clientY - rect.top + scrollEl.scrollTop
    const ratioX = clickX / renderSize.w
    const ratioY = clickY / renderSize.h

    // 切换缩放级别
    const nextIndex = (zoomIndex + 1) % ZOOM_LEVELS.length
    setZoomIndex(nextIndex)

    // 计算新的滚动位置以保持双击点居中
    const newZoom = ZOOM_LEVELS[nextIndex]
    const newW = baseSize.w * newZoom
    const newH = baseSize.h * newZoom

    // 延迟设置滚动位置，等待渲染完成
    setTimeout(() => {
      if (scrollEl) {
        const targetX = ratioX * newW - scrollEl.clientWidth / 2
        const targetY = ratioY * newH - scrollEl.clientHeight / 2
        scrollEl.scrollLeft = Math.max(0, targetX)
        scrollEl.scrollTop = Math.max(0, targetY)
      }
    }, 0)
  }

  const onSurfaceDown = (e: React.MouseEvent) => {
    // 如果放大状态下按住空格或中键，进入平移模式
    if (zoomLevel > 1 && (e.button === 1 || e.altKey)) {
      e.preventDefault()
      const scrollEl = scrollContainerRef.current
      if (scrollEl) {
        setDrag({
          mode: 'pan',
          startX: e.clientX,
          startY: e.clientY,
          scrollStart: { x: scrollEl.scrollLeft, y: scrollEl.scrollTop }
        })
      }
      return
    }

    const { x: lx, y: ly } = toLocal(e.clientX, e.clientY)
    // 如果点在已选框上，进入move
    const sel = boxes.find(b => b.id === selectedId)
    if (sel) {
      const rx = sel.x * scale.x, ry = sel.y * scale.y, rw = sel.w * scale.x, rh = sel.h * scale.y
      const inBox = lx >= rx && ly >= ry && lx <= rx + rw && ly <= ry + rh
      if (inBox) {
        setDrag({ mode: 'move', startX: lx, startY: ly, init: { x: sel.x, y: sel.y, w: sel.w, h: sel.h } })
        return
      }
      // 点在句柄上？
      for (const h of getHandles(sel)) {
        if (Math.abs(lx - h.x) <= hitHandleSize && Math.abs(ly - h.y) <= hitHandleSize) {
          setDrag({ mode: 'resize', startX: lx, startY: ly, init: { x: sel.x, y: sel.y, w: sel.w, h: sel.h }, handle: h.k })
          return
        }
      }
    }
    // 未命中则开始画框
    if (!onSelectClassId) {
      // 仅选中清空
      onSelect(null)
      return
    }
    // 画框时统一使用“图像坐标系”（未缩放的自然像素）
    const sx = lx / scale.x
    const sy = ly / scale.y
    setDrag({ mode: 'draw', startX: sx, startY: sy })
    onSelect(null)
  }

  const onSurfaceMove = (e: React.MouseEvent) => {
    if (!drag) return

    // 平移模式
    if (drag.mode === 'pan' && drag.scrollStart) {
      const scrollEl = scrollContainerRef.current
      if (scrollEl) {
        scrollEl.scrollLeft = drag.scrollStart.x - (e.clientX - drag.startX)
        scrollEl.scrollTop = drag.scrollStart.y - (e.clientY - drag.startY)
      }
      return
    }

    const { x: lx, y: ly } = toLocal(e.clientX, e.clientY)
    if (drag.mode === 'draw') {
      // 将鼠标位置转换到图像坐标，再计算矩形
      const ix = lx / scale.x
      const iy = ly / scale.y
      setDrag(prev => prev ? { ...prev, init: { x: Math.min(prev.startX, ix), y: Math.min(prev.startY, iy), w: Math.abs(ix - prev.startX), h: Math.abs(iy - prev.startY) } } : prev)
    } else if (drag.mode === 'move' && drag.init) {
      const dx = (lx - drag.startX) / scale.x
      const dy = (ly - drag.startY) / scale.y
      const nb = clampBox({ x: drag.init.x + dx, y: drag.init.y + dy, w: drag.init.w, h: drag.init.h })
      setDrag({ ...drag, init: nb })
    } else if (drag.mode === 'resize' && drag.init && drag.handle) {
      let { x, y, w, h } = drag.init
      const ix = lx / scale.x
      const iy = ly / scale.y
      switch (drag.handle) {
        case 'nw': w = (x + w) - ix; h = (y + h) - iy; x = ix; y = iy; break
        case 'n':  h = (y + h) - iy; y = iy; break
        case 'ne': w = ix - x; h = (y + h) - iy; y = iy; break
        case 'e':  w = ix - x; break
        case 'se': w = ix - x; h = iy - y; break
        case 's':  h = iy - y; break
        case 'sw': w = (x + w) - ix; x = ix; h = iy - y; break
        case 'w':  w = (x + w) - ix; x = ix; break
      }
      if (w < 1) w = 1; if (h < 1) h = 1
      const nb = clampBox({ x, y, w, h })
      setDrag({ ...drag, init: nb })
    }
  }

  const onSurfaceUp = () => {
    if (!drag) return
    if (drag.mode === 'draw' && drag.init && onSelectClassId) {
      const nb = clampBox(drag.init)
      onCreate({ x: nb.x, y: nb.y, w: nb.w, h: nb.h, classId: onSelectClassId, source: 'manual' })
    } else if ((drag.mode === 'move' || drag.mode === 'resize') && drag.init && selectedId) {
      const nb = clampBox(drag.init)
      onUpdate(selectedId, { x: nb.x, y: nb.y, w: nb.w, h: nb.h })
    }
    setDrag(null)
  }

  const renderBoxes = () => (
    <svg width={renderSize.w} height={renderSize.h} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      {boxes.map(b => {
        const rx = Math.round(b.x * scale.x)
        const ry = Math.round(b.y * scale.y)
        const rw = Math.round(b.w * scale.x)
        const rh = Math.round(b.h * scale.y)
        const active = b.id === selectedId
        const hovered = hoverId === b.id
        const baseColor = getClassColor(b.classId, classColors)
        // 选中时用白色边框突出，悬停时加亮
        const strokeColor = active ? '#FFFFFF' : (hovered ? '#FFFFFF' : baseColor)
        const fillColor = active ? `${baseColor}40` : (hovered ? `${baseColor}20` : 'transparent')
        return (
          <g key={b.id ?? `${b.x}-${b.y}-${b.w}-${b.h}`} style={{ pointerEvents: 'auto' }} onMouseEnter={() => setHoverId(b.id ?? 0)} onMouseLeave={() => setHoverId(null)} onMouseDown={(e) => { e.stopPropagation(); if (b.id) onSelect(b.id) }}>
            {/* 底层颜色边框 */}
            <rect x={rx} y={ry} width={rw} height={rh} fill={fillColor} stroke={baseColor} strokeWidth={active ? 3 : 2} />
            {/* 选中或悬停时的外边框 */}
            {(active || hovered) && (
              <rect x={rx - 1} y={ry - 1} width={rw + 2} height={rh + 2} fill="transparent" stroke={strokeColor} strokeWidth={1} strokeDasharray={active ? 'none' : '4 2'} />
            )}
            {/* 类别标签 */}
            <rect x={rx} y={ry - 18} width={24} height={16} fill={baseColor} rx={2} />
            <text x={rx + 12} y={ry - 6} fill="white" fontSize={11} fontWeight="bold" textAnchor="middle">{b.classId}</text>
            {/* 调整手柄 */}
            {active && getHandles(b).map(h => (
              <rect key={h.k} x={h.x - hitHandleSize} y={h.y - hitHandleSize} width={hitHandleSize*2} height={hitHandleSize*2} fill={baseColor} stroke="#FFFFFF" strokeWidth={1} rx={2}
                onMouseDown={(e) => { e.stopPropagation(); setDrag({ mode: 'resize', startX: h.x, startY: h.y, init: { x: b.x, y: b.y, w: b.w, h: b.h }, handle: h.k }) }} />
            ))}
          </g>
        )
      })}
      {drag && drag.mode === 'draw' && drag.init && onSelectClassId && (
        <rect x={drag.init.x * scale.x} y={drag.init.y * scale.y} width={drag.init.w * scale.x} height={drag.init.h * scale.y} fill={`${getClassColor(onSelectClassId, classColors)}30`} stroke={getClassColor(onSelectClassId, classColors)} strokeWidth={2} strokeDasharray="4 4" />
      )}
    </svg>
  )

  const getCursor = () => {
    if (drag?.mode === 'pan') return 'grabbing'
    if (drag) return 'crosshair'
    if (zoomLevel > 1) return 'grab'
    return 'default'
  }

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 缩放指示器 */}
      {zoomLevel > 1 && (
        <div style={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 10,
          backgroundColor: 'rgba(0,0,0,0.6)',
          color: 'white',
          padding: '4px 8px',
          borderRadius: 4,
          fontSize: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 8
        }}>
          <span>{Math.round(zoomLevel * 100)}%</span>
          <button
            onClick={() => setZoomIndex(0)}
            style={{
              background: 'transparent',
              border: '1px solid white',
              color: 'white',
              padding: '2px 6px',
              borderRadius: 3,
              cursor: 'pointer',
              fontSize: 11
            }}
          >
            重置
          </button>
        </div>
      )}

      {/* 可滚动容器 */}
      <div
        ref={scrollContainerRef}
        style={{
          flex: 1,
          overflow: zoomLevel > 1 ? 'auto' : 'hidden',
          position: 'relative'
        }}
      >
        <div
          style={{
            position: 'relative',
            width: renderSize.w,
            height: renderSize.h,
            userSelect: 'none',
            cursor: getCursor()
          }}
          onMouseDown={onSurfaceDown}
          onMouseMove={onSurfaceMove}
          onMouseUp={onSurfaceUp}
          onMouseLeave={onSurfaceUp}
          onDoubleClick={onDoubleClick}
        >
          <img src={imageUrl} alt="img" draggable={false} style={{ width: renderSize.w, height: renderSize.h, display: 'block' }} />
          {renderBoxes()}
        </div>
      </div>

      {/* 操作提示 */}
      <div style={{ fontSize: 11, color: '#999', padding: '4px 0', textAlign: 'center' }}>
        双击放大/缩小 | {zoomLevel > 1 ? 'Alt+拖动平移 | ' : ''}当前 {Math.round(zoomLevel * 100)}%
      </div>
    </div>
  )
}
