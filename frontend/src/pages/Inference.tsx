import React, { useEffect, useState, useRef, useCallback } from 'react'
import {
  Card, Button, Select, Slider, Tag, Space, Typography, message,
  Empty, Table, Divider, Row, Col, Statistic,
} from 'antd'
import {
  FolderOpenOutlined, FileImageOutlined, DeleteOutlined,
  PlayCircleOutlined, SwapOutlined, InboxOutlined, RobotOutlined,
} from '@ant-design/icons'

const { Text } = Typography

// ─── Types ────────────────────────────────────────────────────────────────────

type AvailableModel = {
  artifact_id: number
  job_id: number
  project_id: number
  project_name: string
  framework: string
  model_variant: string
  format: string
  path: string
  map50: number | null
  map50_95: number | null
  created_at: string
}

type Detection = {
  class_id: number
  class_name: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
}

type VisualizeResult = {
  detections: Detection[]
  image_width: number
  image_height: number
  inference_time_ms: number
  image_data: string
  model_info: { framework: string; model_variant: string; artifact_id: number; job_id: number }
}

const COLORS = [
  '#FF0000', '#00C851', '#0055FF', '#FFD600',
  '#FF00FF', '#00FFFF', '#FF8000', '#8000FF',
  '#0080FF', '#FF0080', '#80FF00', '#00FF80',
]

// ─── Component ────────────────────────────────────────────────────────────────

export const Inference: React.FC<{ onBack: () => void; defaultProjectId?: number }> = ({ onBack, defaultProjectId }) => {
  const [models, setModels]               = useState<AvailableModel[]>([])
  const [selectedModel, setSelectedModel] = useState<number | null>(null)
  const [loading, setLoading]             = useState(false)
  const [inferring, setInferring]         = useState(false)

  const [uploadedImages, setUploadedImages]     = useState<{ file: File; dataUrl: string }[]>([])
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [result, setResult]               = useState<VisualizeResult | null>(null)
  const [showOriginal, setShowOriginal]   = useState(false)

  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [iouThreshold, setIouThreshold]               = useState(0.45)
  const [imgsz, setImgsz]                             = useState(640)

  const fileInputRef   = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  // ── 加载模型 ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    setLoading(true)
    fetch('/api/inference/models')
      .then(r => r.json())
      .then(data => {
        setModels(data)
        const projectModels = defaultProjectId
          ? data.filter((m: AvailableModel) => m.project_id === defaultProjectId)
          : data
        const candidates = projectModels.length > 0 ? projectModels : data
        if (candidates.length > 0) setSelectedModel(candidates[0].artifact_id)
      })
      .catch(() => setModels([]))
      .finally(() => setLoading(false))
  }, [])

  // ── 处理文件选择 ──────────────────────────────────────────────────────────────
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const imageFiles = Array.from(files).filter(file => {
      const ext = file.name.toLowerCase().split('.').pop()
      return ['jpg', 'jpeg', 'png', 'webp'].includes(ext || '')
    })

    if (imageFiles.length === 0) {
      message.warning('未找到有效的图片文件（支持 jpg/jpeg/png/webp）')
      return
    }

    const promises = imageFiles.map(file =>
      new Promise<{ file: File; dataUrl: string }>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve({ file, dataUrl: reader.result as string })
        reader.onerror = reject
        reader.readAsDataURL(file)
      })
    )

    Promise.all(promises).then(results => {
      setUploadedImages(results)
      setCurrentImageIndex(0)
      setResult(null)
    })

    e.target.value = ''
  }, [])

  // ── 执行推理 ──────────────────────────────────────────────────────────────────
  const runInference = useCallback(async () => {
    if (!selectedModel || uploadedImages.length === 0) return
    const currentImage = uploadedImages[currentImageIndex]
    if (!currentImage) return

    setInferring(true)
    setResult(null)
    try {
      const response = await fetch('/api/inference/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          artifact_id: selectedModel,
          image_data: currentImage.dataUrl,
          confidence_threshold: confidenceThreshold,
          iou_threshold: iouThreshold,
          imgsz,
        }),
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || '推理失败')
      }
      setResult(await response.json())
    } catch (e: any) {
      message.error(e.message || '推理失败，请检查模型是否正常')
    } finally {
      setInferring(false)
    }
  }, [selectedModel, uploadedImages, currentImageIndex, confidenceThreshold, iouThreshold, imgsz])

  const goToPrevImage = useCallback(() => {
    if (currentImageIndex > 0) { setCurrentImageIndex(currentImageIndex - 1); setResult(null) }
  }, [currentImageIndex])

  const goToNextImage = useCallback(() => {
    if (currentImageIndex < uploadedImages.length - 1) { setCurrentImageIndex(currentImageIndex + 1); setResult(null) }
  }, [currentImageIndex, uploadedImages.length])

  const clearImages = useCallback(() => {
    setUploadedImages([]); setCurrentImageIndex(0); setResult(null)
  }, [])

  const selectedModelInfo = models.find(m => m.artifact_id === selectedModel)

  // ── 检测结果表格列 ────────────────────────────────────────────────────────────
  const detectionColumns = [
    {
      title: '#', width: 48,
      render: (_: any, __: any, idx: number) => <Text type="secondary">{idx + 1}</Text>,
    },
    {
      title: '类别', dataIndex: 'class_name',
      render: (name: string, det: Detection) => (
        <Space size={6}>
          <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 2, backgroundColor: COLORS[det.class_id % COLORS.length] }} />
          <Text>{name}</Text>
        </Space>
      ),
    },
    {
      title: '置信度', dataIndex: 'confidence',
      render: (conf: number) => (
        <Tag color={conf > 0.7 ? 'success' : conf > 0.4 ? 'warning' : 'error'}>
          {(conf * 100).toFixed(1)}%
        </Tag>
      ),
    },
    {
      title: '位置',
      render: (det: Detection) => (
        <Text type="secondary" style={{ fontFamily: 'monospace', fontSize: 12 }}>
          ({det.x.toFixed(0)}, {det.y.toFixed(0)})
        </Text>
      ),
    },
    {
      title: '尺寸',
      render: (det: Detection) => (
        <Text type="secondary" style={{ fontFamily: 'monospace', fontSize: 12 }}>
          {det.width.toFixed(0)} × {det.height.toFixed(0)}
        </Text>
      ),
    },
  ]

  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>

      {/* ═══ 标题（独立模式） ═══ */}
      {!defaultProjectId && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Button onClick={onBack}>← 返回</Button>
          <Text strong style={{ fontSize: 18 }}>模型推理验证</Text>
        </div>
      )}

      {/* ═══ 模型配置（选择模型 + 推理参数） ═══ */}
      <Card title={<Space><RobotOutlined style={{ color: '#1677ff' }} /><span>模型配置</span></Space>}>
        <Row gutter={[32, 16]}>

          {/* 左：模型选择 + 摘要 */}
          <Col xs={24} md={14}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>选择模型</div>
            {loading ? (
              <Text type="secondary">加载模型中...</Text>
            ) : models.length === 0 ? (
              <Empty description="暂无可用模型，请先完成训练" image={Empty.PRESENTED_IMAGE_SIMPLE} style={{ padding: '12px 0' }} />
            ) : (
              <>
                <Select
                  style={{ width: '100%' }}
                  value={selectedModel}
                  onChange={v => { setSelectedModel(v); setResult(null) }}
                  options={models.map(m => ({
                    value: m.artifact_id,
                    label: `[${m.framework.toUpperCase()}] ${m.project_name} · ${m.model_variant}${m.map50_95 ? ` (mAP: ${(m.map50_95 * 100).toFixed(1)}%)` : ''}`,
                  }))}
                />
                {selectedModelInfo && (
                  <Row gutter={[12, 8]} style={{ marginTop: 14 }}>
                    {[
                      { label: '框架',     value: selectedModelInfo.framework.toUpperCase() },
                      { label: '模型',     value: selectedModelInfo.model_variant           },
                      { label: '所属项目', value: selectedModelInfo.project_name            },
                      ...(selectedModelInfo.map50     ? [{ label: 'mAP50',    value: `${(selectedModelInfo.map50 * 100).toFixed(2)}%`    }] : []),
                      ...(selectedModelInfo.map50_95  ? [{ label: 'mAP50-95', value: `${(selectedModelInfo.map50_95 * 100).toFixed(2)}%` }] : []),
                    ].map(item => (
                      <Col key={item.label} span={6}>
                        <Statistic
                          title={<Text type="secondary" style={{ fontSize: 11 }}>{item.label}</Text>}
                          value={item.value}
                          valueStyle={{ fontSize: 14, fontWeight: 600 }}
                        />
                      </Col>
                    ))}
                  </Row>
                )}
              </>
            )}
          </Col>

          {/* 右：推理参数 */}
          <Col xs={24} md={10}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <Text style={{ fontWeight: 500, fontSize: 13 }}>置信度阈值</Text>
                  <Tag color="blue">{confidenceThreshold.toFixed(2)}</Tag>
                </div>
                <Slider
                  min={0.01} max={0.99} step={0.01}
                  value={confidenceThreshold}
                  onChange={setConfidenceThreshold}
                  tooltip={{ formatter: v => v?.toFixed(2) }}
                />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <Text style={{ fontWeight: 500, fontSize: 13 }}>IoU 阈值</Text>
                  <Tag color="blue">{iouThreshold.toFixed(2)}</Tag>
                </div>
                <Slider
                  min={0.1} max={0.9} step={0.05}
                  value={iouThreshold}
                  onChange={setIouThreshold}
                  tooltip={{ formatter: v => v?.toFixed(2) }}
                />
              </div>
              <div>
                <Text style={{ fontWeight: 500, fontSize: 13, display: 'block', marginBottom: 6 }}>图像尺寸</Text>
                <Select
                  style={{ width: '100%' }}
                  value={imgsz}
                  onChange={setImgsz}
                  options={[320, 416, 512, 640, 800, 1024, 1280].map(v => ({ value: v, label: String(v) }))}
                />
              </div>
            </div>
          </Col>
        </Row>
      </Card>

      {/* ═══ 图片检测（上传 + 预览） ═══ */}
      <Card title="图片检测">

        {/* 工具栏 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <Space wrap>
            <Button icon={<FileImageOutlined />} onClick={() => fileInputRef.current?.click()}>
              选择文件
            </Button>
            <Button icon={<FolderOpenOutlined />} onClick={() => folderInputRef.current?.click()}>
              选择文件夹
            </Button>
            {uploadedImages.length > 0 && (
              <>
                <Button danger icon={<DeleteOutlined />} onClick={clearImages}>清空</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>共 {uploadedImages.length} 张</Text>
              </>
            )}
          </Space>

          <Space wrap>
            {uploadedImages.length > 0 && (
              <>
                <Button size="small" disabled={currentImageIndex === 0} onClick={goToPrevImage}>‹ 上一张</Button>
                <Text type="secondary" style={{ fontSize: 13 }}>{currentImageIndex + 1} / {uploadedImages.length}</Text>
                <Button size="small" disabled={currentImageIndex >= uploadedImages.length - 1} onClick={goToNextImage}>下一张 ›</Button>
              </>
            )}
            {result && (
              <Button icon={<SwapOutlined />} onClick={() => setShowOriginal(!showOriginal)}>
                {showOriginal ? '查看结果' : '查看原图'}
              </Button>
            )}
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              loading={inferring}
              disabled={!selectedModel || uploadedImages.length === 0}
              onClick={runInference}
            >
              {inferring ? '推理中...' : '执行推理'}
            </Button>
          </Space>
        </div>

        {/* 隐藏文件输入 */}
        <input ref={fileInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple style={{ display: 'none' }} onChange={handleFileSelect} />
        <input ref={folderInputRef} type="file" accept=".jpg,.jpeg,.png,.webp" multiple {...{ webkitdirectory: '', directory: '' } as any} style={{ display: 'none' }} onChange={handleFileSelect} />

        {/* 主体：空状态 or 图片预览 */}
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
              transition: 'border-color 0.2s',
            }}
          >
            <InboxOutlined style={{ fontSize: 52, color: '#bbb' }} />
            <Text type="secondary">点击或选择图片开始推理（支持 jpg / png / webp）</Text>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 100px', gap: 12 }}>
            {/* 主图预览 */}
            <div style={{
              background: '#f7f8fa',
              borderRadius: 8,
              border: '1px solid #e8e8e8',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 360,
            }}>
              <img
                src={showOriginal || !result ? uploadedImages[currentImageIndex]?.dataUrl : result.image_data}
                alt="推理结果"
                style={{ maxWidth: '100%', maxHeight: 560, objectFit: 'contain', borderRadius: 4 }}
              />
            </div>

            {/* 缩略图竖向列 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflowY: 'auto', maxHeight: 560 }}>
              {uploadedImages.map((img, idx) => (
                <div
                  key={idx}
                  onClick={() => { setCurrentImageIndex(idx); setResult(null) }}
                  style={{
                    width: 90,
                    height: 90,
                    borderRadius: 6,
                    overflow: 'hidden',
                    flexShrink: 0,
                    cursor: 'pointer',
                    border: idx === currentImageIndex ? '2px solid #1677ff' : '2px solid transparent',
                    boxShadow: idx === currentImageIndex ? '0 0 0 2px #bae0ff' : 'none',
                    transition: 'all 0.15s',
                  }}
                >
                  <img src={img.dataUrl} alt={img.file.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 文件信息 */}
        {uploadedImages.length > 0 && (
          <div style={{ marginTop: 8, fontSize: 12 }}>
            <Text type="secondary">文件：{uploadedImages[currentImageIndex]?.file.name}</Text>
            {result && (
              <Text type="secondary" style={{ marginLeft: 16 }}>
                尺寸：{result.image_width} × {result.image_height} px
              </Text>
            )}
          </div>
        )}
      </Card>

      {/* ═══ 检测结果 ═══ */}
      {result && (
        <Card
          title="检测结果"
          extra={
            <Text type="secondary" style={{ fontSize: 12 }}>
              推理耗时：<Text strong>{result.inference_time_ms.toFixed(1)} ms</Text>
            </Text>
          }
        >
          {result.detections.length === 0 ? (
            <Empty description="未检测到目标，可以尝试降低置信度阈值" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          ) : (
            <>
              {/* 汇总 Banner */}
              <div style={{
                display: 'flex', gap: 32, padding: '10px 16px',
                background: '#f6ffed', borderRadius: 6, border: '1px solid #b7eb8f',
                marginBottom: 14, fontSize: 13,
              }}>
                <span><Text strong>检测总数：</Text>{result.detections.length} 个</span>
                <span>
                  <Text strong>类别：</Text>
                  {[...new Set(result.detections.map(d => d.class_name))].join('、')}
                </span>
              </div>

              {/* 类别统计 chips */}
              <div style={{ marginBottom: 14 }}>
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>各类别数量</Text>
                <Space wrap>
                  {Object.entries(
                    result.detections.reduce((acc, det) => {
                      acc[det.class_name] = (acc[det.class_name] || 0) + 1
                      return acc
                    }, {} as Record<string, number>)
                  ).map(([className, count]) => {
                    const det = result.detections.find(d => d.class_name === className)
                    return (
                      <div key={className} style={{
                        display: 'inline-flex', alignItems: 'center', gap: 6,
                        padding: '4px 12px', background: '#f5f5f5', borderRadius: 4, fontSize: 13,
                      }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, display: 'inline-block', backgroundColor: COLORS[(det?.class_id || 0) % COLORS.length] }} />
                        <span>{className}</span>
                        <Text strong>{count}</Text>
                      </div>
                    )
                  })}
                </Space>
              </div>

              <Divider style={{ margin: '10px 0' }} />

              {/* 检测明细表 */}
              <Table
                dataSource={result.detections.map((d, i) => ({ ...d, key: i }))}
                columns={detectionColumns}
                size="small"
                pagination={false}
                scroll={{ y: 280 }}
              />
            </>
          )}
        </Card>
      )}
    </Space>
  )
}
