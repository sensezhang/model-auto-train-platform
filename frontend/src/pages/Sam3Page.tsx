import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Card, Button, Slider, Input, Space, Typography, Row, Col,
  Table, Tag, Progress, Alert, Divider, message, Spin,
} from 'antd'
import {
  FolderOpenOutlined, FileImageOutlined, DeleteOutlined,
  ScissorOutlined, PlayCircleOutlined, LeftOutlined, RightOutlined,
} from '@ant-design/icons'

const { Text, Title } = Typography

interface Segment {
  class_name: string
  polygon: number[]   // flat [x1,y1,x2,y2,...]
  bbox: [number, number, number, number]  // [x,y,w,h]
  confidence: number
}

interface SAM3Result {
  image_width: number
  image_height: number
  segments: Segment[]
}

interface UploadedImage {
  file: File
  dataUrl: string
}

// 与 CanvasBBox 保持一致的调色板
const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFE66D', '#DDA0DD', '#FF8C42', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E9', '#F1948A',
]

interface Props {
  projectId?: number
}

export const Sam3Page: React.FC<Props> = () => {
  // 配置
  const [classInput, setClassInput] = useState('')
  const [conf, setConf] = useState(0.4)
  const [iou, setIou] = useState(0.9)

  // 图片列表
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)

  // 结果：Map<imageIndex, SAM3Result>
  const [results, setResults] = useState<Map<number, SAM3Result>>(new Map())

  // 运行状态
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })

  // 悬浮高亮的类别名，null 表示无悬浮
  const [hoveredClass, setHoveredClass] = useState<string | null>(null)

  // 文件输入
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  // 追踪图片实际渲染尺寸，用于精确对齐 SVG 遮罩
  const imgRef = useRef<HTMLImageElement>(null)
  const [renderSize, setRenderSize] = useState({ w: 0, h: 0 })

  const syncImgSize = useCallback(() => {
    const el = imgRef.current
    if (el && el.offsetWidth > 0) {
      setRenderSize({ w: el.offsetWidth, h: el.offsetHeight })
    }
  }, [])

  // 当前图片切换时重置，并重新绑定 ResizeObserver
  useEffect(() => {
    setRenderSize({ w: 0, h: 0 })
    const el = imgRef.current
    if (!el) return
    const obs = new ResizeObserver(syncImgSize)
    obs.observe(el)
    return () => obs.disconnect()
  }, [currentIndex, syncImgSize])

  // 读取文件为 dataUrl
  const readFile = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const imageFiles = Array.from(files).filter(f =>
      /\.(jpg|jpeg|png|webp|bmp)$/i.test(f.name)
    )
    if (!imageFiles.length) {
      message.warning('未找到支持的图片文件（jpg/png/webp/bmp）')
      return
    }
    const items: UploadedImage[] = await Promise.all(
      imageFiles.map(async f => ({ file: f, dataUrl: await readFile(f) }))
    )
    setUploadedImages(prev => [...prev, ...items])
    setCurrentIndex(0)
    setResults(new Map())
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) handleFiles(e.target.files)
    e.target.value = ''
  }

  const clearAll = () => {
    setUploadedImages([])
    setCurrentIndex(0)
    setResults(new Map())
    setProgress({ current: 0, total: 0 })
  }

  // 解析类别
  const getLabels = () =>
    classInput.split(',').map(s => s.trim()).filter(Boolean)

  // 获取类别对应颜色（固定顺序）
  const labelColorMap = (labels: string[]): Record<string, string> => {
    const map: Record<string, string> = {}
    labels.forEach((l, i) => { map[l] = COLORS[i % COLORS.length] })
    // 无文字提示时 "object" 用第一个颜色
    map['object'] = COLORS[0]
    return map
  }

  // 执行单张或全部推理
  const runInference = async (indices: number[]) => {
    if (!uploadedImages.length) {
      message.warning('请先上传图片')
      return
    }
    const labels = getLabels()
    setRunning(true)
    setProgress({ current: 0, total: indices.length })

    for (let i = 0; i < indices.length; i++) {
      const img = uploadedImages[indices[i]]
      try {
        const res = await fetch('/api/inference/sam3', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_data: img.dataUrl,
            text_labels: labels,
            conf,
            iou,
            imgsz: 512,
          }),
        })
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error(err.detail || `HTTP ${res.status}`)
        }
        const data: SAM3Result = await res.json()
        setResults(prev => new Map(prev).set(indices[i], data))
        // 切换到最后处理的图片
        setCurrentIndex(indices[i])
      } catch (e: any) {
        message.error(`图片 ${img.file.name} 推理失败：${e.message}`)
      }
      setProgress(p => ({ ...p, current: i + 1 }))
    }

    setRunning(false)
    message.success(`分割完成，共处理 ${indices.length} 张图片`)
  }

  const current = uploadedImages[currentIndex] ?? null
  const result = results.get(currentIndex) ?? null
  const labels = getLabels()
  const colorMap = labelColorMap(labels.length ? labels : ['object'])

  // viewBox 单位 → 屏幕像素的缩放比，用于让文字/描边保持合适的屏幕尺寸
  const svgScale = result && renderSize.w > 0 ? result.image_width / renderSize.w : 1

  // 统计当前图片结果
  const classCounts: Record<string, number> = {}
  result?.segments.forEach(s => {
    classCounts[s.class_name] = (classCounts[s.class_name] || 0) + 1
  })

  // 结果表格列
  const columns = [
    { title: '#', dataIndex: 'idx', width: 50 },
    {
      title: '类别', dataIndex: 'class_name',
      render: (name: string) => (
        <Tag
          color={colorMap[name] ?? COLORS[0]}
          style={{ cursor: 'pointer' }}
          onMouseEnter={() => setHoveredClass(name)}
          onMouseLeave={() => setHoveredClass(null)}
        >
          {name}
        </Tag>
      ),
    },
    {
      title: '置信度', dataIndex: 'confidence',
      render: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
    {
      title: '位置 (x, y)', key: 'pos',
      render: (_: any, row: any) =>
        `(${Math.round(row.bbox[0])}, ${Math.round(row.bbox[1])})`,
    },
    {
      title: '尺寸 (w × h)', key: 'size',
      render: (_: any, row: any) =>
        `${Math.round(row.bbox[2])} × ${Math.round(row.bbox[3])}`,
    },
  ]

  const tableData = result?.segments.map((s, i) => ({
    key: i, idx: i + 1, ...s,
  })) ?? []

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      {/* 隐藏的文件输入 */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.bmp"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <input
        ref={folderInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.bmp"
        multiple
        {...{ webkitdirectory: '', directory: '' } as any}
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* ── 卡片 1：配置 ── */}
      <Card title={<Space><ScissorOutlined />SAM3 分割配置</Space>}>
        <Row gutter={[24, 16]}>
          <Col xs={24} md={14}>
            <Text strong style={{ display: 'block', marginBottom: 6, fontSize: 13 }}>
              识别类别
            </Text>
            <Input
              value={classInput}
              onChange={e => setClassInput(e.target.value)}
              placeholder="car, person, bicycle（多个类别用逗号分隔）"
              allowClear
            />
            <Text type="secondary" style={{ fontSize: 12, marginTop: 4, display: 'block' }}>
              留空则分割图片中的所有对象
            </Text>
          </Col>
          <Col xs={24} md={10}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <Text style={{ fontSize: 13 }}>置信度阈值</Text>
                <Text type="secondary">{conf.toFixed(2)}</Text>
              </div>
              <Slider
                min={0.1} max={0.9} step={0.05}
                value={conf} onChange={setConf}
                tooltip={{ formatter: v => v?.toFixed(2) }}
              />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <Text style={{ fontSize: 13 }}>IOU 阈值</Text>
                <Text type="secondary">{iou.toFixed(2)}</Text>
              </div>
              <Slider
                min={0.1} max={0.95} step={0.05}
                value={iou} onChange={setIou}
                tooltip={{ formatter: v => v?.toFixed(2) }}
              />
            </div>
          </Col>
        </Row>
      </Card>

      {/* ── 卡片 2：图片上传与分割 ── */}
      <Card title="图片上传与分割">
        {/* 工具栏 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 8 }}>
          <Space wrap>
            <Button icon={<FileImageOutlined />} onClick={() => fileInputRef.current?.click()}>
              选择文件
            </Button>
            <Button icon={<FolderOpenOutlined />} onClick={() => folderInputRef.current?.click()}>
              选择文件夹
            </Button>
            {uploadedImages.length > 0 && (
              <>
                <Button danger icon={<DeleteOutlined />} onClick={clearAll}>
                  清空
                </Button>
                <Text type="secondary">{uploadedImages.length} 张图片</Text>
              </>
            )}
          </Space>
          <Space wrap>
            {uploadedImages.length > 0 && (
              <>
                <Button
                  size="small"
                  icon={<LeftOutlined />}
                  disabled={currentIndex === 0}
                  onClick={() => setCurrentIndex(i => i - 1)}
                />
                <Text type="secondary">{currentIndex + 1} / {uploadedImages.length}</Text>
                <Button
                  size="small"
                  icon={<RightOutlined />}
                  disabled={currentIndex >= uploadedImages.length - 1}
                  onClick={() => setCurrentIndex(i => i + 1)}
                />
              </>
            )}
            {uploadedImages.length > 0 && (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                loading={running}
                onClick={() => runInference([currentIndex])}
              >
                执行分割
              </Button>
            )}
            {uploadedImages.length > 1 && (
              <Button
                icon={<PlayCircleOutlined />}
                loading={running}
                onClick={() => runInference(uploadedImages.map((_, i) => i))}
              >
                全部分割
              </Button>
            )}
          </Space>
        </div>

        {/* 进度条（批量时显示） */}
        {running && progress.total > 1 && (
          <div style={{ marginBottom: 12 }}>
            <Progress
              percent={Math.round(progress.current / progress.total * 100)}
              status="active"
              format={() => `${progress.current} / ${progress.total}`}
            />
          </div>
        )}

        {/* 主图区域 */}
        {uploadedImages.length === 0 ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            style={{
              minHeight: 280,
              background: '#fafafa',
              border: '2px dashed #d9d9d9',
              borderRadius: 8,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              gap: 12,
            }}
          >
            <ScissorOutlined style={{ fontSize: 52, color: '#bbb' }} />
            <Text type="secondary">点击或选择图片 / 文件夹开始分割（支持 jpg / png / webp）</Text>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 100px', gap: 12 }}>
            {/* 主图 + SVG 遮盖层 */}
            <div
              style={{
                background: '#f7f8fa',
                borderRadius: 8,
                border: '1px solid #e8e8e8',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 360,
                overflow: 'hidden',
              }}
            >
              {current && (
                <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                  <img
                    ref={imgRef}
                    src={current.dataUrl}
                    alt={current.file.name}
                    onLoad={syncImgSize}
                    style={{ maxWidth: '100%', maxHeight: 560, display: 'block', borderRadius: 4 }}
                  />
                  {running && !results.has(currentIndex) && (
                    <div style={{
                      position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                      background: 'rgba(0,0,0,0.45)', borderRadius: 4,
                      display: 'flex', flexDirection: 'column',
                      alignItems: 'center', justifyContent: 'center', gap: 12,
                    }}>
                      <Spin size="large" />
                      <span style={{ color: '#fff', fontSize: 14 }}>SAM3 分割中，请稍候...</span>
                    </div>
                  )}
                  {result && result.segments.length > 0 && (
                    <svg
                      style={{
                        position: 'absolute',
                        top: 0, left: 0,
                        // 使用图片实际渲染像素尺寸，确保 SVG 与图片完全对齐
                        width: renderSize.w > 0 ? renderSize.w : '100%',
                        height: renderSize.h > 0 ? renderSize.h : '100%',
                        pointerEvents: 'none',
                      }}
                      viewBox={`0 0 ${result.image_width} ${result.image_height}`}
                      preserveAspectRatio="none"
                    >
                      {result.segments.map((seg, i) => {
                        const pts: string[] = []
                        for (let j = 0; j < seg.polygon.length; j += 2) {
                          pts.push(`${seg.polygon[j]},${seg.polygon[j + 1]}`)
                        }
                        const color = colorMap[seg.class_name] ?? COLORS[i % COLORS.length]
                        const isHovered = hoveredClass === seg.class_name
                        const isDimmed  = hoveredClass !== null && !isHovered
                        // fill / stroke 随悬浮状态变化
                        const fillAlpha   = isDimmed ? '18' : isHovered ? 'aa' : '50'
                        const strokeW     = isDimmed ? 1.5  : isHovered ? 5    : 3
                        // 文字大小 & 描边：以屏幕像素为单位乘以 svgScale 换算回 viewBox 坐标
                        const fontSize    = Math.round(13 * svgScale)
                        const textStroke  = Math.round(3  * svgScale)
                        return (
                          <g key={i}>
                            <polygon
                              points={pts.join(' ')}
                              fill={color + fillAlpha}
                              stroke={color}
                              strokeWidth={strokeW}
                              // non-scaling-stroke 让描边宽度以屏幕像素为单位，不随 viewBox 缩放
                              style={{ vectorEffect: 'non-scaling-stroke' } as any}
                            />
                            <text
                              x={seg.bbox[0] + 4 * svgScale}
                              y={seg.bbox[1] + fontSize + 2 * svgScale}
                              fontSize={fontSize}
                              fill="#fff"
                              style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: textStroke } as any}
                            >
                              {seg.class_name} {Math.round(seg.confidence * 100)}%
                            </text>
                          </g>
                        )
                      })}
                    </svg>
                  )}
                  {result && result.segments.length === 0 && (
                    <div style={{
                      position: 'absolute', top: 8, left: 8,
                      background: 'rgba(0,0,0,0.55)', color: '#fff',
                      borderRadius: 4, padding: '4px 10px', fontSize: 13,
                    }}>
                      未检测到目标
                    </div>
                  )}
                  {results.has(currentIndex) && (
                    <div style={{
                      position: 'absolute', top: 8, right: 8,
                      background: 'rgba(22,119,255,0.85)', color: '#fff',
                      borderRadius: 4, padding: '2px 8px', fontSize: 12,
                    }}>
                      ✓ 已分割
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 缩略图竖列 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflowY: 'auto', maxHeight: 560 }}>
              {uploadedImages.map((img, idx) => (
                <div
                  key={idx}
                  onClick={() => setCurrentIndex(idx)}
                  style={{
                    width: 90, height: 90,
                    borderRadius: 6,
                    overflow: 'hidden',
                    flexShrink: 0,
                    cursor: 'pointer',
                    border: idx === currentIndex ? '2px solid #1677ff' : '2px solid transparent',
                    position: 'relative',
                  }}
                >
                  <img src={img.dataUrl} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="" />
                  {results.has(idx) && (
                    <div style={{
                      position: 'absolute', bottom: 0, right: 0,
                      background: 'rgba(22,119,255,0.8)', color: '#fff',
                      fontSize: 10, padding: '1px 4px', borderTopLeftRadius: 4,
                    }}>
                      ✓ {results.get(idx)?.segments.length}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {current && (
          <div style={{ marginTop: 8, fontSize: 12 }}>
            <Text type="secondary">文件：{current.file.name}</Text>
            {result && (
              <Text type="secondary" style={{ marginLeft: 16 }}>
                尺寸：{result.image_width} × {result.image_height} px
              </Text>
            )}
          </div>
        )}
      </Card>

      {/* ── 卡片 3：分割结果 ── */}
      {result && (
        <Card
          title="分割结果"
          extra={
            <Text type="secondary" style={{ fontSize: 12 }}>
              {result.segments.length} 个目标
            </Text>
          }
        >
          {result.segments.length === 0 ? (
            <Alert type="info" message="未检测到目标，请尝试调低置信度阈值或修改类别描述" showIcon />
          ) : (
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              {/* 汇总 */}
              <div>
                <Text>
                  共检测到 <strong>{result.segments.length}</strong> 个目标，涉及{' '}
                  <strong>{Object.keys(classCounts).length}</strong> 个类别
                </Text>
              </div>

              {/* 类别分布 chips —— 悬浮时高亮对应掩膜 */}
              <Space wrap>
                {Object.entries(classCounts).map(([name, count]) => (
                  <Tag
                    key={name}
                    color={colorMap[name] ?? COLORS[0]}
                    style={{
                      fontSize: 13,
                      cursor: 'pointer',
                      transition: 'opacity 0.15s',
                      opacity: hoveredClass && hoveredClass !== name ? 0.45 : 1,
                    }}
                    onMouseEnter={() => setHoveredClass(name)}
                    onMouseLeave={() => setHoveredClass(null)}
                  >
                    {name} × {count}
                  </Tag>
                ))}
              </Space>

              <Divider style={{ margin: '4px 0' }} />

              {/* 明细表格 */}
              <Table
                columns={columns}
                dataSource={tableData}
                size="small"
                pagination={{ pageSize: 10, showSizeChanger: false }}
                scroll={{ x: true }}
              />
            </Space>
          )}
        </Card>
      )}
    </Space>
  )
}
