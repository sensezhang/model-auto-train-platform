import React, { useEffect, useRef, useState } from 'react'
import {
  Card, Select, Input, Radio, Button, Progress, Alert, Space, Typography,
  Tag, Statistic, Row, Col, message, Divider, DatePicker, Pagination,
  Slider, Spin, Badge, Empty,
} from 'antd'
import {
  RobotOutlined, PlayCircleOutlined, ThunderboltOutlined,
  FilterOutlined, ReloadOutlined, CheckOutlined, SaveOutlined,
} from '@ant-design/icons'
import { Project } from '../../pages/App'
import dayjs, { Dayjs } from 'dayjs'

const { Text, Paragraph } = Typography
const { TextArea } = Input
const { RangePicker } = DatePicker

// ─── Types ────────────────────────────────────────────────────────────────────

interface ImageItem {
  id: number
  path: string
  thumbnailPath?: string
  displayPath?: string
  width?: number
  height?: number
  status: 'unannotated' | 'ai_pending' | 'annotated'
  labeled: boolean
  createdAt: string
}

interface AvailableModel {
  artifact_id: number
  job_id: number
  project_id: number
  project_name: string
  framework: string
  model_variant: string
  format: string
  map50?: number
  map50_95?: number
  created_at: string
}

interface AiJobStatus {
  id: number
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'
  imagesCount: number
  processedCount: number
  boxesCount: number
}

interface AiDefaultConfig {
  engine: 'glm4v' | 'model'
  classId?: number
  prompt?: string
  artifactId?: number
  confidenceThreshold?: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

const PAGE_SIZE = 60  // 20 × 3 grid

const getImgUrl = (img: ImageItem) => {
  const p = (img.thumbnailPath || img.path).replace(/\\/g, '/')
  return p.startsWith('http://') || p.startsWith('https://') ? p : `/files/${p}`
}

const STATUS_BADGE: Record<string, { bg: string; label: string }> = {
  unannotated: { bg: '',                          label: '' },
  ai_pending:  { bg: 'rgba(250,173,20,0.85)',     label: 'AI待审' },
  annotated:   { bg: 'rgba(82,196,26,0.85)',      label: '已标注' },
}

const JOB_STATUS_TAG: Record<string, { color: string; label: string }> = {
  pending:   { color: 'gold',    label: '等待中' },
  running:   { color: 'blue',    label: '标注中' },
  succeeded: { color: 'green',   label: '已完成' },
  failed:    { color: 'red',     label: '失败'   },
  canceled:  { color: 'default', label: '已取消' },
}

// ─── Component ────────────────────────────────────────────────────────────────

export const AutoLabel: React.FC<{ project: Project }> = ({ project }) => {

  // ── Engine ──────────────────────────────────────────────────────────────
  const [engine, setEngine] = useState<'glm4v' | 'model'>('glm4v')

  // ── GLM-4V config ────────────────────────────────────────────────────────
  const [classId, setClassId] = useState<number | null>(
    project.classes.length > 0 ? project.classes[0].id : null
  )
  const [prompt, setPrompt] = useState('')

  // ── Trained model config ─────────────────────────────────────────────────
  const [models, setModels]               = useState<AvailableModel[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [artifactId, setArtifactId]       = useState<number | null>(null)
  const [confThreshold, setConfThreshold] = useState(0.25)

  // ── Image filter ─────────────────────────────────────────────────────────
  const [filterStatus, setFilterStatus] = useState('all')
  const [dateRange, setDateRange]       = useState<[Dayjs, Dayjs] | null>(null)

  // ── Thumbnail grid ───────────────────────────────────────────────────────
  const [images, setImages]           = useState<ImageItem[]>([])
  const [imageTotal, setImageTotal]   = useState(0)
  const [imagePage, setImagePage]     = useState(1)
  const [loadingImages, setLoadingImages] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  // ── Scope fallback (when no thumbnail selected) ──────────────────────────
  const [scope, setScope] = useState<'unlabeled' | 'all'>('unlabeled')

  // ── Saved default AI config ──────────────────────────────────────────────
  const [savedDefault, setSavedDefault] = useState<AiDefaultConfig | null>(() => {
    try {
      const raw = localStorage.getItem('annotator_ai_default_config')
      return raw ? (JSON.parse(raw) as AiDefaultConfig) : null
    } catch { return null }
  })

  // ── Job ──────────────────────────────────────────────────────────────────
  const [starting, setStarting]       = useState(false)
  const [jobStatus, setJobStatus]     = useState<AiJobStatus | null>(null)
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => { if (pollRef.current) clearTimeout(pollRef.current) }
  }, [])

  // ── Load trained models ──────────────────────────────────────────────────
  const loadModels = async () => {
    setLoadingModels(true)
    try {
      const res  = await fetch('/api/inference/models')
      const data = await res.json()
      const list = (Array.isArray(data) ? data : []) as AvailableModel[]
      const projectModels = list.filter(m => m.project_id === project.id)
      setModels(projectModels)
      if (projectModels.length > 0 && !artifactId) {
        setArtifactId(projectModels[0].artifact_id)
      }
    } catch {
      setModels([])
    } finally {
      setLoadingModels(false)
    }
  }

  useEffect(() => { loadModels() }, [project.id])

  // ── Load thumbnail images ────────────────────────────────────────────────
  const loadImages = async (
    page: number,
    status = filterStatus,
    range: [Dayjs, Dayjs] | null = dateRange,
  ) => {
    setLoadingImages(true)
    try {
      const params = new URLSearchParams({
        page:      String(page),
        page_size: String(PAGE_SIZE),
      })
      if (status !== 'all') params.set('status', status)
      if (range?.[0]) params.set('date_from', range[0].startOf('day').toISOString())
      if (range?.[1]) params.set('date_to',   range[1].endOf('day').toISOString())

      const res  = await fetch(`/api/projects/${project.id}/images?${params}`)
      const data = await res.json()
      setImages(data.items  || [])
      setImageTotal(data.total || 0)
      setImagePage(page)
      if (page === 1) setSelectedIds(new Set())
    } catch {
      setImages([])
    } finally {
      setLoadingImages(false)
    }
  }

  useEffect(() => { loadImages(1) }, [project.id])

  // ── Thumbnail selection ──────────────────────────────────────────────────
  const toggleSelect = (id: number) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAllPage  = () =>
    setSelectedIds(prev => { const n = new Set(prev); images.forEach(i => n.add(i.id)); return n })

  const clearSelection = () => setSelectedIds(new Set())

  // ── Poll job status ──────────────────────────────────────────────────────
  const pollJob = (jobId: number) => {
    const poll = async () => {
      try {
        const res = await fetch(`/api/autolabel/jobs/${jobId}`)
        if (!res.ok) return
        const job = await res.json() as AiJobStatus
        setJobStatus(job)
        if (!['succeeded', 'failed', 'canceled'].includes(job.status)) {
          pollRef.current = setTimeout(poll, 2000)
        } else if (job.status === 'succeeded') {
          message.success(`标注完成，共生成 ${job.boxesCount} 个标注框`)
        } else if (job.status === 'failed') {
          message.error('标注任务失败，请检查配置')
        }
      } catch {
        pollRef.current = setTimeout(poll, 3000)
      }
    }
    poll()
  }

  // ── Save as default config for Annotator quick-AI ───────────────────────
  const saveAsDefault = () => {
    let cfg: AiDefaultConfig
    if (engine === 'glm4v') {
      if (!classId)       { message.warning('请先选择目标类别');       return }
      if (!prompt.trim()) { message.warning('请先填写检测提示词');     return }
      cfg = { engine: 'glm4v', classId, prompt: prompt.trim() }
    } else {
      if (!artifactId) { message.warning('请先选择要使用的模型'); return }
      cfg = { engine: 'model', artifactId, confidenceThreshold: confThreshold }
    }
    localStorage.setItem('annotator_ai_default_config', JSON.stringify(cfg))
    setSavedDefault(cfg)
    message.success('已保存为标注工作台默认配置，可在标注工作台右键菜单中使用')
  }

  // ── Start job ────────────────────────────────────────────────────────────
  const handleStart = async () => {
    const imageIds = selectedIds.size > 0 ? [...selectedIds] : null

    if (engine === 'glm4v') {
      if (!classId)       { message.warning('请选择目标类别');   return }
      if (!prompt.trim()) { message.warning('请输入检测提示词'); return }

      setStarting(true)
      try {
        const body: Record<string, unknown> = {
          projectId: project.id,
          classId,
          prompt: prompt.trim(),
          threshold: 0.3,
          scope,
        }
        if (imageIds) body.imageIds = imageIds

        const res = await fetch('/api/autolabel/jobs', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(body),
        })
        if (!res.ok) throw new Error((await res.json()).detail || '启动失败')
        const job = await res.json() as AiJobStatus
        setJobStatus(job)
        message.success('GLM-4V 标注任务已启动')
        pollJob(job.id)
      } catch (e: unknown) {
        message.error('启动失败：' + (e instanceof Error ? e.message : String(e)))
      } finally {
        setStarting(false)
      }

    } else {
      if (!artifactId) { message.warning('请选择要使用的模型'); return }

      setStarting(true)
      try {
        const body: Record<string, unknown> = {
          projectId:           project.id,
          artifactId,
          confidenceThreshold: confThreshold,
          iouThreshold:        0.45,
          scope,
        }
        if (imageIds) body.imageIds = imageIds

        const res = await fetch('/api/autolabel/model-jobs', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(body),
        })
        if (!res.ok) throw new Error((await res.json()).detail || '启动失败')
        const job = await res.json() as AiJobStatus
        setJobStatus(job)
        message.success('模型标注任务已启动')
        pollJob(job.id)
      } catch (e: unknown) {
        message.error('启动失败：' + (e instanceof Error ? e.message : String(e)))
      } finally {
        setStarting(false)
      }
    }
  }

  // ── Computed values ──────────────────────────────────────────────────────
  const isRunning = jobStatus?.status === 'running' || jobStatus?.status === 'pending'

  const progress = jobStatus && jobStatus.imagesCount > 0
    ? Math.round(jobStatus.processedCount / jobStatus.imagesCount * 100)
    : 0

  const startLabel = () => {
    if (isRunning) return '标注进行中...'
    if (selectedIds.size > 0) return `开始标注（已选 ${selectedIds.size} 张）`
    return `开始标注（${scope === 'unlabeled' ? '仅未标注' : '全部图片'}）`
  }

  const canStart =
    !isRunning && !starting &&
    (engine === 'glm4v' ? project.classes.length > 0 : models.length > 0)

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>

      {/* ═══════════════ 标注引擎 ═══════════════ */}
      <Card title={<Space><RobotOutlined /><span>标注引擎</span></Space>}>

        {/* Engine toggle */}
        <Radio.Group
          value={engine}
          onChange={e => setEngine(e.target.value)}
          optionType="button"
          buttonStyle="solid"
          style={{ marginBottom: 20 }}
        >
          <Radio.Button value="glm4v">
            <Space size={4}><RobotOutlined />GLM-4V 大模型</Space>
          </Radio.Button>
          <Radio.Button value="model">
            <Space size={4}><ThunderboltOutlined />已训练模型</Space>
          </Radio.Button>
        </Radio.Group>

        <Divider style={{ margin: '0 0 16px' }} />

        {/* GLM-4V config panel */}
        {engine === 'glm4v' && (
          <Space direction="vertical" style={{ width: '100%' }} size={14}>
            <Paragraph type="secondary" style={{ margin: 0 }}>
              使用 GLM-4V 视觉大模型检测图片目标，生成标注框。
              需在后端 .env 中配置 <Text code>AUTOLABEL_BASE_URL</Text> 和
              <Text code> AUTOLABEL_API_KEY</Text>。
            </Paragraph>

            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>目标类别</Text>
              {project.classes.length === 0
                ? <Alert message="请先在「类别」菜单添加标注类别" type="warning" showIcon />
                : (
                  <Select
                    value={classId}
                    onChange={setClassId}
                    style={{ width: 240 }}
                    options={project.classes.map(c => ({ value: c.id, label: c.name }))}
                  />
                )
              }
            </div>

            <div>
              <Text strong style={{ display: 'block', marginBottom: 6 }}>检测提示词</Text>
              <TextArea
                rows={3}
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="例如：识别图中所有的猫，用边界框标注出每只猫的位置"
                style={{ maxWidth: 480 }}
              />
            </div>
          </Space>
        )}

        {/* Trained model config panel */}
        {engine === 'model' && (
          <Space direction="vertical" style={{ width: '100%' }} size={14}>
            <Paragraph type="secondary" style={{ margin: 0 }}>
              使用项目已训练的 YOLO / RF-DETR 模型批量推理，自动生成标注框。
            </Paragraph>

            {models.length === 0 ? (
              <Alert
                message="暂无可用模型"
                description="请先在「训练」菜单完成模型训练，然后点击「刷新」"
                type="warning"
                showIcon
                action={
                  <Button size="small" onClick={loadModels} loading={loadingModels}>
                    刷新
                  </Button>
                }
              />
            ) : (
              <>
                <div>
                  <Text strong style={{ display: 'block', marginBottom: 6 }}>选择模型</Text>
                  <Select
                    value={artifactId}
                    onChange={setArtifactId}
                    style={{ width: '100%', maxWidth: 480 }}
                    loading={loadingModels}
                    options={models.map(m => ({
                      value: m.artifact_id,
                      label: (
                        <Space>
                          <Tag
                            color={m.framework === 'yolo' ? 'blue' : 'green'}
                            style={{ margin: 0 }}
                          >
                            {m.framework.toUpperCase()}
                          </Tag>
                          <span>{m.model_variant}</span>
                          {m.map50 != null && (
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              mAP50: {(m.map50 * 100).toFixed(1)}%
                            </Text>
                          )}
                        </Space>
                      ),
                    }))}
                  />
                </div>

                <div>
                  <Text strong style={{ display: 'block', marginBottom: 10 }}>
                    置信度阈值：
                    <Text style={{ fontWeight: 'normal', color: '#1677ff' }}>
                      {confThreshold.toFixed(2)}
                    </Text>
                  </Text>
                  <Slider
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    value={confThreshold}
                    onChange={setConfThreshold}
                    style={{ maxWidth: 400 }}
                    marks={{ 0.05: '0.05', 0.25: '0.25', 0.5: '0.50', 0.75: '0.75', 0.95: '0.95' }}
                    tooltip={{ formatter: v => v?.toFixed(2) }}
                  />
                </div>
              </>
            )}
          </Space>
        )}

        {/* ── 保存为标注工作台默认配置 ── */}
        <Divider style={{ margin: '16px 0 10px' }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {savedDefault
              ? `当前默认：${savedDefault.engine === 'glm4v' ? 'GLM-4V 大模型' : '已训练模型'}`
              : '暂无默认配置（未保存）'}
          </Text>
          <Button
            size="small"
            type="primary"
            ghost
            icon={<SaveOutlined />}
            onClick={saveAsDefault}
          >
            保存为标注工作台默认配置
          </Button>
        </div>
      </Card>

      {/* ═══════════════ 图片选择 ═══════════════ */}
      <Card
        title={
          <Space>
            <FilterOutlined />
            <span>图片选择</span>
            {selectedIds.size > 0 && (
              <Badge count={selectedIds.size} style={{ backgroundColor: '#1677ff' }} />
            )}
          </Space>
        }
        extra={
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => loadImages(imagePage)}
            loading={loadingImages}
          >
            刷新
          </Button>
        }
      >
        {/* Filter bar */}
        <Row gutter={[12, 8]} align="middle" style={{ marginBottom: 12 }}>
          <Col>
            <Text type="secondary" style={{ fontSize: 13 }}>图片状态</Text>
          </Col>
          <Col>
            <Select
              value={filterStatus}
              onChange={v => { setFilterStatus(v); loadImages(1, v, dateRange) }}
              size="small"
              style={{ width: 110 }}
              options={[
                { value: 'all',         label: '全部图片' },
                { value: 'unannotated', label: '未标注'   },
                { value: 'ai_pending',  label: 'AI待审'   },
                { value: 'annotated',   label: '已标注'   },
              ]}
            />
          </Col>
          <Col>
            <Text type="secondary" style={{ fontSize: 13 }}>上传时间</Text>
          </Col>
          <Col>
            <RangePicker
              size="small"
              value={dateRange}
              onChange={v => {
                const r = (v ?? null) as [Dayjs, Dayjs] | null
                setDateRange(r)
                loadImages(1, filterStatus, r)
              }}
              allowClear
              placeholder={['开始日期', '结束日期']}
            />
          </Col>
        </Row>

        {/* Selection toolbar */}
        <div style={{
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'space-between',
          padding:        '6px 10px',
          marginBottom:   10,
          background:     '#fafafa',
          border:         '1px solid #f0f0f0',
          borderRadius:   6,
        }}>
          <Space size={6}>
            <Text type="secondary" style={{ fontSize: 13 }}>
              共 {imageTotal} 张 · 当前页 {images.length} 张
            </Text>
            {selectedIds.size > 0 && (
              <Tag color="blue">已选 {selectedIds.size} 张</Tag>
            )}
          </Space>
          <Space>
            <Button size="small" onClick={selectAllPage}>全选当前页</Button>
            <Button
              size="small"
              onClick={clearSelection}
              disabled={selectedIds.size === 0}
            >
              清空选择
            </Button>
          </Space>
        </div>

        {/* Thumbnail grid */}
        <Spin spinning={loadingImages} tip="加载中...">
          {images.length === 0 ? (
            <Empty
              description="暂无图片，请先在「数据管理」中导入图片"
              style={{ padding: '32px 0' }}
            />
          ) : (
            <div style={{
              display:             'grid',
              gridTemplateColumns: 'repeat(20, 1fr)',   /* 固定 20 列，撑满整行 */
              gap:                 4,
            }}>
              {images.map(img => {
                const sel   = selectedIds.has(img.id)
                const badge = STATUS_BADGE[img.status]

                return (
                  <div
                    key={img.id}
                    onClick={() => toggleSelect(img.id)}
                    style={{
                      position:   'relative',
                      borderRadius: 4,
                      overflow:   'hidden',
                      cursor:     'pointer',
                      border:     sel ? '2px solid #1677ff' : '2px solid transparent',
                      boxShadow:  sel ? '0 0 0 2px rgba(22,119,255,0.18)' : 'none',
                      background: '#f5f5f5',
                      transition: 'border-color 0.12s, box-shadow 0.12s',
                    }}
                  >
                    {/* Thumbnail image */}
                    <img
                      src={getImgUrl(img)}
                      alt=""
                      loading="lazy"
                      style={{
                        width:       '100%',
                        aspectRatio: '1 / 1',
                        objectFit:   'cover',
                        display:     'block',
                      }}
                    />

                    {/* Checkbox overlay (top-left) */}
                    <div style={{
                      position:       'absolute',
                      top:            3,
                      left:           3,
                      width:          16,
                      height:         16,
                      borderRadius:   3,
                      background:     sel ? '#1677ff' : 'rgba(255,255,255,0.88)',
                      border:         sel ? 'none' : '1px solid #bbb',
                      display:        'flex',
                      alignItems:     'center',
                      justifyContent: 'center',
                      transition:     'background 0.12s',
                      pointerEvents:  'none',
                    }}>
                      {sel && <CheckOutlined style={{ fontSize: 9, color: '#fff' }} />}
                    </div>

                    {/* Status badge (bottom) */}
                    {badge.label && (
                      <div style={{
                        position:   'absolute',
                        bottom:     0,
                        left:       0,
                        right:      0,
                        background: badge.bg,
                        color:      '#fff',
                        fontSize:   10,
                        textAlign:  'center',
                        lineHeight: '15px',
                        pointerEvents: 'none',
                      }}>
                        {badge.label}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </Spin>

        {/* Pagination */}
        {imageTotal > PAGE_SIZE && (
          <div style={{ marginTop: 14, textAlign: 'center' }}>
            <Pagination
              current={imagePage}
              pageSize={PAGE_SIZE}
              total={imageTotal}
              showSizeChanger={false}
              showTotal={t => `共 ${t} 张`}
              onChange={p => loadImages(p)}
              size="small"
            />
          </div>
        )}
      </Card>

      {/* ═══════════════ 启动标注 ═══════════════ */}
      <Card>
        {selectedIds.size > 0 ? (
          <Alert
            type="info"
            showIcon
            message={`已手动选择 ${selectedIds.size} 张图片`}
            description="点击「开始标注」将仅对选中图片进行处理"
            style={{ marginBottom: 16 }}
          />
        ) : (
          <div style={{ marginBottom: 16 }}>
            <Text strong style={{ display: 'block', marginBottom: 8 }}>
              未手动选图时的处理范围
            </Text>
            <Radio.Group value={scope} onChange={e => setScope(e.target.value)}>
              <Radio value="unlabeled">仅处理未完成标注的图片</Radio>
              <Radio value="all">处理项目全部图片</Radio>
            </Radio.Group>
          </div>
        )}

        <Button
          type="primary"
          size="large"
          icon={<PlayCircleOutlined />}
          loading={starting}
          disabled={!canStart}
          onClick={handleStart}
          style={{ minWidth: 220 }}
        >
          {startLabel()}
        </Button>
      </Card>

      {/* ═══════════════ 任务进度 ═══════════════ */}
      {jobStatus && (
        <Card title="任务进度">
          <Space direction="vertical" style={{ width: '100%' }} size={16}>

            <Space>
              <Tag color={JOB_STATUS_TAG[jobStatus.status]?.color}>
                {JOB_STATUS_TAG[jobStatus.status]?.label ?? jobStatus.status}
              </Tag>
              <Text type="secondary">任务 #{jobStatus.id}</Text>
            </Space>

            <Progress
              percent={progress}
              status={
                jobStatus.status === 'failed'    ? 'exception' :
                jobStatus.status === 'succeeded' ? 'success'   : 'active'
              }
            />

            <Row gutter={24}>
              <Col span={8}>
                <Statistic title="待处理图片" value={jobStatus.imagesCount} />
              </Col>
              <Col span={8}>
                <Statistic title="已处理" value={jobStatus.processedCount} />
              </Col>
              <Col span={8}>
                <Statistic title="生成标注框" value={jobStatus.boxesCount} />
              </Col>
            </Row>

            {jobStatus.status === 'succeeded' && (
              <Alert
                type="success"
                showIcon
                message={`标注完成！共生成 ${jobStatus.boxesCount} 个标注框`}
                description="请前往「标注」菜单，审核并确认 AI 生成的标注结果"
              />
            )}

            {jobStatus.status === 'failed' && (
              <Alert
                type="error"
                showIcon
                message="标注任务失败"
                description={
                  engine === 'glm4v'
                    ? '请检查后端 .env 中 AUTOLABEL_BASE_URL / AUTOLABEL_API_KEY 配置是否正确'
                    : '请检查模型文件是否存在且完整，或尝试重新训练'
                }
              />
            )}
          </Space>
        </Card>
      )}
    </Space>
  )
}
