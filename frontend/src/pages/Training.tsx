import React, { useEffect, useState } from 'react'
import {
  Card, Button, Select, InputNumber, Space, Switch, Tag, Typography,
  message, Modal, Progress, Empty, Row, Col, Divider, Statistic, Alert, Spin,
} from 'antd'
import {
  PlayCircleOutlined, StopOutlined, DownloadOutlined, ExportOutlined,
  ReloadOutlined, ThunderboltOutlined, CheckCircleOutlined, DeleteOutlined,
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const { Text } = Typography

// ─── Types ────────────────────────────────────────────────────────────────────

type Project = { id: number; name: string }

type TrainingJob = {
  id: number
  projectId: number
  framework: string
  modelVariant: string
  epochs: number
  imgsz: number
  batch: number | null
  seed: number
  status: string
  logsRef: string | null
  map50: number | null
  map50_95: number | null
  precision: number | null
  recall: number | null
  startedAt: string | null
  finishedAt: string | null
}

type ModelArtifact = {
  id: number
  trainingJobId: number
  format: string
  path: string
  size: number | null
  createdAt: string
}

type GPU = {
  id: number
  name: string
  memory_total: number
  memory_free: number
  memory_used: number
  compute_capability: string
}

type TrainingMetric = {
  epoch: number
  loss: number | null
  mAP50: number | null
  'mAP50-95': number | null
  mAP: number | null
  timestamp: string
}

type ResultImage = { name: string; path: string }

// ─── Constants ────────────────────────────────────────────────────────────────

const STATUS_CFG: Record<string, { color: string; label: string }> = {
  succeeded: { color: 'success',    label: '已完成' },
  failed:    { color: 'error',      label: '失败'   },
  running:   { color: 'processing', label: '训练中' },
  canceled:  { color: 'default',    label: '已取消' },
  pending:   { color: 'warning',    label: '等待中' },
}

const YOLO_MODELS = [
  { value: 'yolov11n', label: 'YOLOv11n（最小）' },
  { value: 'yolov11s', label: 'YOLOv11s（小）'   },
  { value: 'yolov11m', label: 'YOLOv11m（中）'   },
  { value: 'yolov11l', label: 'YOLOv11l（大）'   },
  { value: 'yolov11x', label: 'YOLOv11x（超大）' },
]

const RFDETR_MODELS = [
  { value: 'rfdetr-small',  label: 'RF-DETR Small'  },
  { value: 'rfdetr-medium', label: 'RF-DETR Medium' },
  { value: 'rfdetr-large',  label: 'RF-DETR Large'  },
]

// ─── Component ────────────────────────────────────────────────────────────────

export const Training: React.FC<{ onBack: () => void; defaultProjectId?: number }> = ({ onBack, defaultProjectId }) => {
  const [projects, setProjects]               = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<number | null>(defaultProjectId ?? null)
  const [framework, setFramework]             = useState<'yolo' | 'rfdetr'>('yolo')
  const [modelVariant, setModelVariant]       = useState('yolov11n')
  const [epochs, setEpochs]                   = useState(50)
  const [imgsz, setImgsz]                     = useState(640)
  const [batch, setBatch]                     = useState<number | null>(null)
  const [seed, setSeed]                       = useState(42)
  const [creating, setCreating]               = useState(false)
  const [jobs, setJobs]                       = useState<TrainingJob[]>([])
  const [selectedJob, setSelectedJob]         = useState<number | null>(null)
  const [logs, setLogs]                       = useState<string[]>([])
  const [artifacts, setArtifacts]             = useState<ModelArtifact[]>([])
  const [autoRefresh, setAutoRefresh]         = useState(true)
  const [exporting, setExporting]             = useState<number | null>(null)
  const [availableGpus, setAvailableGpus]     = useState<GPU[]>([])
  const [loadingGpus, setLoadingGpus]         = useState(true)
  const [selectedGpus, setSelectedGpus]       = useState<number[]>([])
  const [metrics, setMetrics]                 = useState<TrainingMetric[]>([])
  const [resultImages, setResultImages]       = useState<ResultImage[]>([])

  // ── GPU ─────────────────────────────────────────────────────────────────────
  const loadGpuInfo = () => {
    setLoadingGpus(true)
    fetch('/api/system/gpus')
      .then(r => r.json())
      .then((gpus: GPU[]) => {
        setAvailableGpus(gpus)
        if (gpus.length > 0 && selectedGpus.length === 0) setSelectedGpus([gpus[0].id])
      })
      .catch(() => setAvailableGpus([]))
      .finally(() => setLoadingGpus(false))
  }

  useEffect(() => {
    fetch('/api/projects').then(r => r.json()).then(d => setProjects(Array.isArray(d) ? d : [])).catch(() => {})
    loadGpuInfo()
  }, [])

  useEffect(() => {
    const t = setInterval(loadGpuInfo, 5000)
    return () => clearInterval(t)
  }, [selectedGpus])

  // ── Jobs ─────────────────────────────────────────────────────────────────────
  const loadJobs = () => {
    if (!selectedProject) return
    fetch('/api/training/jobs')
      .then(r => r.json())
      .then(d => {
        if (Array.isArray(d)) {
          setJobs(d.filter((j: TrainingJob) => j.projectId === selectedProject).sort((a: TrainingJob, b: TrainingJob) => b.id - a.id))
        }
      })
      .catch(() => {})
  }

  useEffect(() => { loadJobs() }, [selectedProject])
  useEffect(() => {
    if (!autoRefresh || !selectedProject) return
    const t = setInterval(loadJobs, 3000)
    return () => clearInterval(t)
  }, [autoRefresh, selectedProject])

  // ── Job detail ───────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedJob) { setLogs([]); setArtifacts([]); setMetrics([]); setResultImages([]); return }

    fetch(`/api/training/jobs/${selectedJob}/artifacts`).then(r => r.json()).then(setArtifacts).catch(() => setArtifacts([]))
    fetch(`/api/training/jobs/${selectedJob}/result-images`).then(r => r.json()).then(setResultImages).catch(() => setResultImages([]))

    const loadMetrics = () =>
      fetch(`/api/training/jobs/${selectedJob}/metrics?last_n=100`).then(r => r.json()).then(setMetrics).catch(() => {})
    loadMetrics()
    const mi = setInterval(loadMetrics, 5000)

    const job = jobs.find(j => j.id === selectedJob)
    if (!job?.logsRef) return () => clearInterval(mi)

    const es = new EventSource(`/api/training/jobs/${selectedJob}/logs/stream`)
    const buf: string[] = []
    es.onmessage = (e) => { if (e.data && e.data !== '[waiting logs]') { buf.push(e.data); setLogs([...buf]) } }
    es.onerror = () => es.close()
    return () => { clearInterval(mi); es.close() }
  }, [selectedJob, jobs])

  useEffect(() => {
    setModelVariant(framework === 'yolo' ? 'yolov11n' : 'rfdetr-medium')
  }, [framework])

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const handleCreateJob = async () => {
    if (!selectedProject) { message.warning('请选择项目'); return }
    setCreating(true)
    try {
      const res = await fetch('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedProject, framework, modelVariant, epochs, imgsz,
          batch: batch || null, seed,
          gpuIds: selectedGpus.length > 0 ? selectedGpus.join(',') : null,
        }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || '创建失败')
      const job = await res.json()
      setJobs([job, ...jobs])
      setSelectedJob(job.id)
      message.success('训练任务已创建并开始执行')
    } catch (e: any) {
      message.error(e.message || '创建训练任务失败')
    } finally {
      setCreating(false)
    }
  }

  const handleCancelJob = (jobId: number) => {
    Modal.confirm({
      title: '确认取消训练任务',
      okText: '确认取消', okButtonProps: { danger: true }, cancelText: '返回',
      onOk: async () => {
        try {
          const res = await fetch(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' })
          if (!res.ok) throw new Error()
          message.success('训练任务已取消'); loadJobs()
        } catch { message.error('取消失败') }
      },
    })
  }

  const handleDeleteJob = (jobId: number, e: React.MouseEvent) => {
    e.stopPropagation()
    Modal.confirm({
      title: '删除训练任务',
      content: '确认删除该训练任务及其所有记录？此操作不可恢复。',
      okText: '确认删除', okButtonProps: { danger: true }, cancelText: '取消',
      onOk: async () => {
        try {
          const res = await fetch(`/api/training/jobs/${jobId}`, { method: 'DELETE' })
          if (!res.ok) throw new Error((await res.json()).detail || '删除失败')
          message.success('任务已删除')
          setJobs(prev => prev.filter(j => j.id !== jobId))
          if (selectedJob === jobId) {
            setSelectedJob(null)
            setLogs([]); setArtifacts([]); setMetrics([]); setResultImages([])
          }
        } catch (e: any) {
          message.error(e.message || '删除失败')
        }
      },
    })
  }

  const handleExportOnnx = (artifactId: number, jobId: number) => {
    Modal.confirm({
      title: '导出 ONNX',
      content: '确认将该模型导出为 ONNX 格式？',
      okText: '确认导出', cancelText: '取消',
      onOk: async () => {
        setExporting(artifactId)
        try {
          const res = await fetch(`/api/training/jobs/${jobId}/export-onnx?artifact_id=${artifactId}&simplify=false`, { method: 'POST' })
          if (!res.ok) throw new Error((await res.json()).detail || '导出失败')
          const data = await res.json()
          message.success(data.message || '导出成功！')
          if (selectedJob) fetch(`/api/training/jobs/${selectedJob}/artifacts`).then(r => r.json()).then(setArtifacts).catch(() => {})
        } catch (e: any) {
          message.error(e.message || '导出失败')
        } finally { setExporting(null) }
      },
    })
  }

  const currentJobDetail = jobs.find(j => j.id === selectedJob)
  const currentModels    = framework === 'yolo' ? YOLO_MODELS : RFDETR_MODELS

  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>

      {/* ═══ 标题（独立模式） ═══ */}
      {!defaultProjectId && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Button onClick={onBack}>← 返回</Button>
          <Text strong style={{ fontSize: 18 }}>训练中心</Text>
        </div>
      )}

      {/* ═══ 创建训练任务 ═══ */}
      <Card title={<Space><ThunderboltOutlined style={{ color: '#1677ff' }} /><span>创建训练任务</span></Space>}>

        {/* ── 参数行 ── */}
        <Row gutter={[16, 16]} align="bottom">

          {/* 项目选择（非内嵌模式）*/}
          {!defaultProjectId && (
            <Col xs={24} sm={12} md={5} lg={4}>
              <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>项目</div>
              <Select
                style={{ width: '100%' }}
                placeholder="请选择"
                value={selectedProject}
                onChange={setSelectedProject}
                options={projects.map(p => ({ value: p.id, label: p.name }))}
              />
            </Col>
          )}

          {/* 训练框架 - 下拉选择 */}
          <Col xs={12} sm={8} md={defaultProjectId ? 4 : 4} lg={defaultProjectId ? 3 : 3}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>训练框架</div>
            <Select
              style={{ width: '100%' }}
              value={framework}
              onChange={v => setFramework(v as 'yolo' | 'rfdetr')}
              options={[
                { value: 'yolo',   label: 'YOLO'    },
                { value: 'rfdetr', label: 'RF-DETR' },
              ]}
            />
          </Col>

          {/* 模型变体 */}
          <Col xs={12} sm={8} md={defaultProjectId ? 5 : 4} lg={defaultProjectId ? 5 : 4}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>模型变体</div>
            <Select style={{ width: '100%' }} value={modelVariant} onChange={setModelVariant} options={currentModels} />
          </Col>

          {/* 训练轮数 */}
          <Col xs={12} sm={6} md={3} lg={2}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>轮数</div>
            <InputNumber style={{ width: '100%' }} min={1} max={1000} value={epochs} onChange={v => setEpochs(v || 50)} />
          </Col>

          {/* 图像尺寸 */}
          <Col xs={12} sm={6} md={3} lg={2}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>图像尺寸</div>
            <InputNumber style={{ width: '100%' }} min={320} max={1280} step={32} value={imgsz} onChange={v => setImgsz(v || 640)} />
          </Col>

          {/* Batch */}
          <Col xs={12} sm={6} md={3} lg={2}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>
              Batch<Text type="secondary" style={{ fontSize: 11, marginLeft: 2 }}>(空=自动)</Text>
            </div>
            <InputNumber style={{ width: '100%' }} min={1} max={128} placeholder="自动" value={batch ?? undefined} onChange={v => setBatch(v ?? null)} />
          </Col>

          {/* 随机种子 */}
          <Col xs={12} sm={6} md={3} lg={2}>
            <div style={{ fontWeight: 500, marginBottom: 6, fontSize: 13 }}>随机种子</div>
            <InputNumber style={{ width: '100%' }} min={0} value={seed} onChange={v => setSeed(v || 42)} />
          </Col>

          {/* 开始训练按钮 - 右对齐 */}
          <Col xs={24} sm={24} md={24} lg={4} style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              loading={creating}
              disabled={!selectedProject}
              onClick={handleCreateJob}
              size="large"
            >
              {creating ? '创建中...' : '开始训练'}
            </Button>
          </Col>
        </Row>

        {/* ── GPU 选择（始终显示） ── */}
        <Divider style={{ margin: '16px 0 12px' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <Text strong style={{ fontSize: 13 }}>GPU 选择</Text>
          <Space size={6}>
            <Text type="secondary" style={{ fontSize: 12 }}>每 5 秒自动刷新</Text>
            <Button size="small" icon={<ReloadOutlined />} onClick={loadGpuInfo} loading={loadingGpus} />
          </Space>
        </div>
        {loadingGpus ? (
          <div style={{ padding: '16px 0', textAlign: 'center' }}>
            <Spin tip="获取 GPU 信息中..." />
          </div>
        ) : availableGpus.length === 0 ? (
          <Alert type="warning" showIcon message="未检测到 GPU，将使用 CPU 训练" style={{ marginBottom: 4 }} />
        ) : (
          <Row gutter={[10, 10]}>
            {availableGpus.map(gpu => {
              const used = (gpu.memory_total - gpu.memory_free) / gpu.memory_total
              const sel  = selectedGpus.includes(gpu.id)
              return (
                <Col key={gpu.id} xs={24} sm={12} md={8} lg={6}>
                  <div
                    onClick={() => setSelectedGpus(sel ? selectedGpus.filter(id => id !== gpu.id) : [...selectedGpus, gpu.id])}
                    style={{
                      padding: '10px 14px',
                      border: sel ? '2px solid #1677ff' : '1px solid #e8e8e8',
                      borderRadius: 8,
                      cursor: 'pointer',
                      background: sel ? '#e6f4ff' : '#fafafa',
                      transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <Space size={6}>
                        {sel && <CheckCircleOutlined style={{ color: '#1677ff', fontSize: 13 }} />}
                        <Text strong style={{ fontSize: 13 }}>GPU {gpu.id}: {gpu.name}</Text>
                      </Space>
                      {gpu.compute_capability && <Text type="secondary" style={{ fontSize: 11 }}>CC {gpu.compute_capability}</Text>}
                    </div>
                    <Progress
                      percent={Math.round(used * 100)}
                      size="small"
                      strokeColor={used > 0.9 ? '#ff4d4f' : used > 0.7 ? '#faad14' : '#52c41a'}
                      showInfo={false}
                      style={{ marginBottom: 4 }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                      <Text type="secondary">已用 {((gpu.memory_total - gpu.memory_free) / 1024).toFixed(1)} GB</Text>
                      <Text type="secondary">共 {(gpu.memory_total / 1024).toFixed(1)} GB</Text>
                    </div>
                  </div>
                </Col>
              )
            })}
          </Row>
        )}
        {selectedGpus.length > 1 && (
          <div style={{ marginTop: 10, padding: '8px 12px', background: '#fffbe6', borderRadius: 6, fontSize: 12, color: '#ad8b00', border: '1px solid #ffe58f' }}>
            ⚡ 多 GPU 训练：将使用 {selectedGpus.length} 张 GPU 进行分布式训练
          </div>
        )}
      </Card>

      {/* ═══ 训练任务列表 ═══ */}
      <Card
        title="训练任务"
        extra={
          <Space size={8}>
            <Text type="secondary" style={{ fontSize: 12 }}>自动刷新</Text>
            <Switch size="small" checked={autoRefresh} onChange={setAutoRefresh} />
            <Button size="small" icon={<ReloadOutlined />} onClick={loadJobs} />
          </Space>
        }
      >
        {!selectedProject ? (
          <Empty description="请先选择项目" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : jobs.length === 0 ? (
          <Empty description="暂无训练任务，创建后将在此显示" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : (
          <Row gutter={[12, 12]}>
            {jobs.map(job => {
              const cfg        = STATUS_CFG[job.status] || { color: 'default', label: job.status }
              const isSelected = selectedJob === job.id
              const canDelete  = ['succeeded', 'failed', 'canceled'].includes(job.status)
              return (
                <Col key={job.id} xs={12} sm={8} md={6} lg={4} xl={3}>
                  <div
                    onClick={() => setSelectedJob(job.id)}
                    style={{
                      position: 'relative',
                      padding: '12px 14px',
                      paddingRight: canDelete ? 32 : 14,
                      border: isSelected ? '2px solid #1677ff' : '1px solid #e8e8e8',
                      borderRadius: 8,
                      cursor: 'pointer',
                      background: isSelected ? '#e6f4ff' : '#fff',
                      transition: 'all 0.15s',
                      height: '100%',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <Text strong style={{ fontSize: 13 }}>#{job.id}</Text>
                      <Tag color={cfg.color} style={{ margin: 0, fontSize: 11 }}>{cfg.label}</Tag>
                    </div>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
                      {job.framework.toUpperCase()} · {job.modelVariant}
                    </Text>
                    {job.map50_95 !== null && (
                      <Text style={{ fontSize: 12, color: '#52c41a', fontWeight: 500 }}>
                        mAP: {(job.map50_95 * 100).toFixed(1)}%
                      </Text>
                    )}
                    {canDelete && (
                      <Button
                        size="small"
                        danger
                        type="text"
                        icon={<DeleteOutlined />}
                        onClick={(e) => handleDeleteJob(job.id, e)}
                        style={{ position: 'absolute', top: 6, right: 4, padding: '0 4px' }}
                      />
                    )}
                  </div>
                </Col>
              )
            })}
          </Row>
        )}
      </Card>

      {/* ═══ 任务详情（选中时显示） ═══ */}
      {selectedJob && currentJobDetail && (
        <Card
          title={`任务 #${selectedJob} 详情`}
          extra={
            currentJobDetail.status === 'running' && (
              <Button danger size="small" icon={<StopOutlined />} onClick={() => handleCancelJob(selectedJob)}>
                取消训练
              </Button>
            )
          }
        >
          <Row gutter={[16, 16]}>
            {[
              { label: '框架',     value: currentJobDetail.framework.toUpperCase() },
              { label: '模型',     value: currentJobDetail.modelVariant },
              { label: '训练轮数', value: currentJobDetail.epochs },
              { label: '图像尺寸', value: currentJobDetail.imgsz },
              { label: 'Batch',    value: currentJobDetail.batch || '自动' },
              { label: '随机种子', value: currentJobDetail.seed },
              ...(currentJobDetail.map50     !== null ? [{ label: 'mAP50',     value: `${(currentJobDetail.map50 * 100).toFixed(2)}%`    }] : []),
              ...(currentJobDetail.map50_95  !== null ? [{ label: 'mAP50-95',  value: `${(currentJobDetail.map50_95 * 100).toFixed(2)}%` }] : []),
              ...(currentJobDetail.precision !== null ? [{ label: 'Precision', value: `${(currentJobDetail.precision * 100).toFixed(2)}%` }] : []),
              ...(currentJobDetail.recall    !== null ? [{ label: 'Recall',    value: `${(currentJobDetail.recall * 100).toFixed(2)}%`    }] : []),
            ].map(item => (
              <Col key={item.label} xs={12} sm={8} md={6} lg={4} xl={3}>
                <Statistic
                  title={<Text type="secondary" style={{ fontSize: 12 }}>{item.label}</Text>}
                  value={item.value}
                  valueStyle={{ fontSize: 16, fontWeight: 600 }}
                />
              </Col>
            ))}
          </Row>

          {artifacts.length > 0 && (
            <>
              <Divider style={{ margin: '16px 0 12px' }} />
              <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 10 }}>模型文件</Text>
              <Row gutter={[10, 10]}>
                {artifacts.map(art => (
                  <Col key={art.id} xs={24} sm={12} md={8}>
                    <div style={{ padding: '10px 14px', background: '#f7f8fa', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Space size={8}>
                        <Tag color="blue" style={{ margin: 0 }}>{art.format.toUpperCase()}</Tag>
                        <div>
                          <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>{art.path.split(/[\\/]/).pop()}</Text>
                          {art.size && <Text type="secondary" style={{ fontSize: 11 }}>{(art.size / 1024 / 1024).toFixed(2)} MB</Text>}
                        </div>
                      </Space>
                      <Space size={6}>
                        <a href={`/api/training/artifacts/${art.id}/download`} download>
                          <Button size="small" icon={<DownloadOutlined />}>下载</Button>
                        </a>
                        {(art.format === 'pt' || art.format === 'pth') && (
                          <Button
                            size="small" type="primary" ghost
                            icon={<ExportOutlined />}
                            loading={exporting === art.id}
                            onClick={() => handleExportOnnx(art.id, selectedJob!)}
                          >
                            导出 ONNX
                          </Button>
                        )}
                      </Space>
                    </div>
                  </Col>
                ))}
              </Row>
            </>
          )}
        </Card>
      )}

      {/* ═══ 训练指标图表 ═══ */}
      {selectedJob && metrics.length > 0 && (
        <Card title="训练指标">
          <Row gutter={[24, 0]}>
            {metrics.some(m => m.loss !== null) && (
              <Col xs={24} md={12}>
                <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 8 }}>损失曲线（Loss）</Text>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#ff4d4f" name="Loss" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Col>
            )}
            {(metrics.some(m => m.mAP50 !== null) || metrics.some(m => m.mAP !== null)) && (
              <Col xs={24} md={12}>
                <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 8 }}>平均精度（mAP）</Text>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    {metrics.some(m => m.mAP50 !== null) && (
                      <Line type="monotone" dataKey="mAP50" stroke="#1677ff" name="mAP50" dot={false} strokeWidth={2} />
                    )}
                    {metrics.some(m => m['mAP50-95'] !== null) && (
                      <Line type="monotone" dataKey="mAP50-95" stroke="#52c41a" name="mAP50-95" dot={false} strokeWidth={2} />
                    )}
                    {metrics.some(m => m.mAP !== null) && (
                      <Line type="monotone" dataKey="mAP" stroke="#722ed1" name="mAP" dot={false} strokeWidth={2} />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </Col>
            )}
          </Row>
          <Text type="secondary" style={{ fontSize: 11, display: 'block', textAlign: 'center', marginTop: 8 }}>
            实时更新 · 显示最近 {metrics.length} 个 Epoch
          </Text>
        </Card>
      )}

      {/* ═══ 训练日志 ═══ */}
      {selectedJob && (
        <Card title="训练日志">
          <div style={{
            backgroundColor: '#1e1e1e',
            color: '#d4d4d4',
            padding: '12px 16px',
            borderRadius: 6,
            fontFamily: 'Consolas, Monaco, monospace',
            fontSize: 12,
            height: 400,
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            lineHeight: 1.6,
          }}>
            {logs.length === 0
              ? <span style={{ color: '#666' }}>等待日志输出...</span>
              : logs.map((line, i) => <div key={i}>{line}</div>)
            }
          </div>
        </Card>
      )}

      {/* ═══ 训练结果图片 ═══ */}
      {(() => {
        const job = jobs.find(j => j.id === selectedJob)
        if (!job || job.status !== 'succeeded' || resultImages.length === 0) return null
        const priority = ['metrics_plot.png', 'results.png']
        const main     = resultImages.find(img => priority.includes(img.name))
        const rest     = resultImages.filter(img => !priority.includes(img.name))
        return (
          <Card title={<span style={{ color: '#52c41a' }}>🎉 训练结果</span>}>
            {main && (
              <div style={{ marginBottom: rest.length > 0 ? 16 : 0 }}>
                <img src={main.path} alt={main.name} style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid #e8e8e8', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }} />
              </div>
            )}
            {rest.length > 0 && (
              <>
                <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 8 }}>其他图片</Text>
                <Space wrap>
                  {rest.map((img, i) => (
                    <a key={i} href={img.path} target="_blank" rel="noopener noreferrer">
                      <Button size="small">{img.name}</Button>
                    </a>
                  ))}
                </Space>
              </>
            )}
          </Card>
        )
      })()}

    </Space>
  )
}
