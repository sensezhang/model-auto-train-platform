import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

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

type ResultImage = {
  name: string
  path: string
}

export const Training: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<number | null>(null)
  const [framework, setFramework] = useState<'yolo' | 'rfdetr'>('yolo')
  const [modelVariant, setModelVariant] = useState('yolov11n')
  const [epochs, setEpochs] = useState(50)
  const [imgsz, setImgsz] = useState(640)
  const [batch, setBatch] = useState<number | null>(null)
  const [seed, setSeed] = useState(42)
  const [creating, setCreating] = useState(false)
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [selectedJob, setSelectedJob] = useState<number | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([])
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [exporting, setExporting] = useState<number | null>(null)
  const [availableGpus, setAvailableGpus] = useState<GPU[]>([])
  const [selectedGpus, setSelectedGpus] = useState<number[]>([])
  const [metrics, setMetrics] = useState<TrainingMetric[]>([])
  const [resultImages, setResultImages] = useState<ResultImage[]>([])

  // 加载GPU信息
  const loadGpuInfo = () => {
    fetch('/api/system/gpus')
      .then(r => r.json())
      .then((gpus: GPU[]) => {
        setAvailableGpus(gpus)
        // 首次加载时默认选中第一张GPU
        if (gpus.length > 0 && selectedGpus.length === 0) {
          setSelectedGpus([gpus[0].id])
        }
      })
      .catch(() => setAvailableGpus([]))
  }

  // 加载项目列表和GPU信息
  useEffect(() => {
    fetch('/api/projects')
      .then(r => r.json())
      .then(setProjects)
      .catch(() => setProjects([]))

    loadGpuInfo()
  }, [])

  // 定时刷新GPU信息（每5秒）
  useEffect(() => {
    const timer = setInterval(loadGpuInfo, 5000)
    return () => clearInterval(timer)
  }, [selectedGpus])

  // 加载训练任务列表
  const loadJobs = () => {
    if (!selectedProject) return
    // 获取所有训练任务，然后过滤当前项目的
    fetch(`/api/training/jobs`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data)) {
          // 过滤当前项目的任务
          const filtered = data.filter((job: TrainingJob) => job.projectId === selectedProject)
          setJobs(filtered.sort((a: TrainingJob, b: TrainingJob) => b.id - a.id))
        }
      })
      .catch(() => {})
  }

  useEffect(() => {
    loadJobs()
  }, [selectedProject])

  // 自动刷新任务状态
  useEffect(() => {
    if (!autoRefresh || !selectedProject) return
    const timer = setInterval(loadJobs, 3000)
    return () => clearInterval(timer)
  }, [autoRefresh, selectedProject])

  // 加载任务详情
  useEffect(() => {
    if (!selectedJob) {
      setLogs([])
      setArtifacts([])
      setMetrics([])
      setResultImages([])
      return
    }

    // 加载artifacts
    fetch(`/api/training/jobs/${selectedJob}/artifacts`)
      .then(r => r.json())
      .then(setArtifacts)
      .catch(() => setArtifacts([]))

    // 加载结果图片列表
    fetch(`/api/training/jobs/${selectedJob}/result-images`)
      .then(r => r.json())
      .then(setResultImages)
      .catch(() => setResultImages([]))

    // 加载训练指标
    const loadMetrics = () => {
      fetch(`/api/training/jobs/${selectedJob}/metrics?last_n=100`)
        .then(r => r.json())
        .then(setMetrics)
        .catch(() => setMetrics([]))
    }

    loadMetrics()

    // 定期更新指标（每5秒）
    const metricsInterval = setInterval(loadMetrics, 5000)

    // 清理定时器
    const cleanup = () => {
      clearInterval(metricsInterval)
    }

    // 加载日志（通过EventSource流式加载）
    const job = jobs.find(j => j.id === selectedJob)
    if (!job || !job.logsRef) {
      return () => {
        clearInterval(metricsInterval)
      }
    }

    const eventSource = new EventSource(`/api/training/jobs/${selectedJob}/logs/stream`)
    const newLogs: string[] = []

    eventSource.onmessage = (event) => {
      const line = event.data
      if (line && line !== '[waiting logs]') {
        newLogs.push(line)
        setLogs([...newLogs])
      }
    }

    eventSource.onerror = () => {
      eventSource.close()
    }

    return () => {
      clearInterval(metricsInterval)
      eventSource.close()
    }
  }, [selectedJob, jobs])

  // 创建训练任务
  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedProject) {
      alert('请选择项目')
      return
    }

    setCreating(true)
    try {
      const res = await fetch('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedProject,
          framework,
          modelVariant,
          epochs,
          imgsz,
          batch: batch || null,
          seed,
          gpuIds: selectedGpus.length > 0 ? selectedGpus.join(',') : null,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || '创建训练任务失败')
      }

      const job = await res.json()
      setJobs([job, ...jobs])
      setSelectedJob(job.id)
      alert('训练任务已创建并开始执行')
    } catch (e: any) {
      alert(e.message || '创建训练任务失败')
    } finally {
      setCreating(false)
    }
  }

  // 取消训练任务
  const handleCancelJob = async (jobId: number) => {
    if (!confirm('确认取消该训练任务吗？')) return

    try {
      const res = await fetch(`/api/training/jobs/${jobId}/cancel`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('取消失败')
      loadJobs()
    } catch (e) {
      alert('取消失败')
    }
  }

  // 导出ONNX
  const handleExportOnnx = async (artifactId: number, jobId: number) => {
    if (!confirm('确认导出该模型为ONNX格式吗？')) return

    setExporting(artifactId)
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/export-onnx?artifact_id=${artifactId}&simplify=false`, {
        method: 'POST',
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || '导出失败')
      }
      const data = await res.json()
      alert(data.message || '导出成功！')

      // 重新加载artifacts
      if (selectedJob) {
        fetch(`/api/training/jobs/${selectedJob}/artifacts`)
          .then(r => r.json())
          .then(setArtifacts)
          .catch(() => {})
      }
    } catch (e: any) {
      alert(e.message || '导出失败')
    } finally {
      setExporting(null)
    }
  }

  // YOLO模型选项
  const yoloModels = [
    { value: 'yolov11n', label: 'YOLOv11n (最小)' },
    { value: 'yolov11s', label: 'YOLOv11s (小)' },
    { value: 'yolov11m', label: 'YOLOv11m (中)' },
    { value: 'yolov11l', label: 'YOLOv11l (大)' },
    { value: 'yolov11x', label: 'YOLOv11x (超大)' },
  ]

  // RF-DETR模型选项
  const rfdetrModels = [
    { value: 'rfdetr-small', label: 'RF-DETR Small' },
    { value: 'rfdetr-medium', label: 'RF-DETR Medium' },
    { value: 'rfdetr-large', label: 'RF-DETR Large' },
  ]

  const currentModels = framework === 'yolo' ? yoloModels : rfdetrModels

  // 当框架改变时，重置模型选择
  useEffect(() => {
    if (framework === 'yolo') {
      setModelVariant('yolov11n')
    } else {
      setModelVariant('rfdetr-medium')
    }
  }, [framework])

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 16, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <button onClick={onBack} style={{ padding: '8px 16px' }}>← 返回</button>
        <h2 style={{ margin: 0 }}>训练中心</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '400px 1fr', gap: 24 }}>
        {/* 左侧：创建训练任务 */}
        <div>
          <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginBottom: 16 }}>
            <h3 style={{ marginTop: 0 }}>创建训练任务</h3>
            <form onSubmit={handleCreateJob} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {/* 项目选择 */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>项目</label>
                <select
                  value={selectedProject || ''}
                  onChange={e => setSelectedProject(Number(e.target.value))}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                  required
                >
                  <option value="">请选择项目</option>
                  {projects.map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>

              {/* 训练框架选择 */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>训练框架</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  <label style={{
                    flex: 1,
                    padding: 12,
                    border: framework === 'yolo' ? '2px solid #1890ff' : '1px solid #d9d9d9',
                    borderRadius: 6,
                    cursor: 'pointer',
                    backgroundColor: framework === 'yolo' ? '#e6f7ff' : 'white',
                    textAlign: 'center'
                  }}>
                    <input
                      type="radio"
                      name="framework"
                      checked={framework === 'yolo'}
                      onChange={() => setFramework('yolo')}
                      style={{ marginRight: 8 }}
                    />
                    YOLO
                  </label>
                  <label style={{
                    flex: 1,
                    padding: 12,
                    border: framework === 'rfdetr' ? '2px solid #1890ff' : '1px solid #d9d9d9',
                    borderRadius: 6,
                    cursor: 'pointer',
                    backgroundColor: framework === 'rfdetr' ? '#e6f7ff' : 'white',
                    textAlign: 'center'
                  }}>
                    <input
                      type="radio"
                      name="framework"
                      checked={framework === 'rfdetr'}
                      onChange={() => setFramework('rfdetr')}
                      style={{ marginRight: 8 }}
                    />
                    RF-DETR
                  </label>
                </div>
              </div>

              {/* 模型变体 */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>模型变体</label>
                <select
                  value={modelVariant}
                  onChange={e => setModelVariant(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                >
                  {currentModels.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </div>

              {/* Epochs */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>训练轮数 (Epochs)</label>
                <input
                  type="number"
                  value={epochs}
                  onChange={e => setEpochs(Number(e.target.value))}
                  min={1}
                  max={1000}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                />
              </div>

              {/* 图像尺寸 */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>图像尺寸</label>
                <input
                  type="number"
                  value={imgsz}
                  onChange={e => setImgsz(Number(e.target.value))}
                  min={320}
                  max={1280}
                  step={32}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                />
              </div>

              {/* Batch Size */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>
                  Batch Size (留空自动)
                </label>
                <input
                  type="number"
                  value={batch || ''}
                  onChange={e => setBatch(e.target.value ? Number(e.target.value) : null)}
                  min={1}
                  max={128}
                  placeholder="自动"
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                />
              </div>

              {/* 随机种子 */}
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 500 }}>随机种子</label>
                <input
                  type="number"
                  value={seed}
                  onChange={e => setSeed(Number(e.target.value))}
                  min={0}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                />
              </div>

              {/* GPU选择 */}
              {availableGpus.length > 0 && (
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
                    选择GPU (可多选)
                  </label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 200, overflowY: 'auto' }}>
                    {availableGpus.map(gpu => (
                      <label key={gpu.id} style={{
                        padding: 10,
                        border: selectedGpus.includes(gpu.id) ? '2px solid #1890ff' : '1px solid #d9d9d9',
                        borderRadius: 6,
                        cursor: 'pointer',
                        backgroundColor: selectedGpus.includes(gpu.id) ? '#e6f7ff' : 'white',
                        transition: 'all 0.2s'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                          <input
                            type="checkbox"
                            checked={selectedGpus.includes(gpu.id)}
                            onChange={e => {
                              if (e.target.checked) {
                                setSelectedGpus([...selectedGpus, gpu.id])
                              } else {
                                setSelectedGpus(selectedGpus.filter(id => id !== gpu.id))
                              }
                            }}
                            style={{ marginRight: 8, marginTop: 4 }}
                          />
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 500, marginBottom: 4 }}>
                              GPU {gpu.id}: {gpu.name}
                              {gpu.compute_capability && <span style={{ fontSize: 11, color: '#999', marginLeft: 8 }}>CC {gpu.compute_capability}</span>}
                            </div>
                            {/* 显存使用进度条 */}
                            <div style={{ marginBottom: 4 }}>
                              <div style={{
                                height: 8,
                                backgroundColor: '#f0f0f0',
                                borderRadius: 4,
                                overflow: 'hidden',
                                position: 'relative'
                              }}>
                                <div style={{
                                  height: '100%',
                                  width: `${((gpu.memory_total - gpu.memory_free) / gpu.memory_total) * 100}%`,
                                  backgroundColor: ((gpu.memory_total - gpu.memory_free) / gpu.memory_total) > 0.9 ? '#ff4d4f' :
                                                   ((gpu.memory_total - gpu.memory_free) / gpu.memory_total) > 0.7 ? '#faad14' : '#52c41a',
                                  borderRadius: 4,
                                  transition: 'width 0.3s ease'
                                }} />
                              </div>
                            </div>
                            <div style={{ fontSize: 11, color: '#666', display: 'flex', justifyContent: 'space-between' }}>
                              <span>已用: {((gpu.memory_total - gpu.memory_free) / 1024).toFixed(1)}GB</span>
                              <span>可用: {(gpu.memory_free / 1024).toFixed(1)}GB / {(gpu.memory_total / 1024).toFixed(1)}GB</span>
                            </div>
                          </div>
                        </div>
                      </label>
                    ))}
                  </div>
                  {selectedGpus.length > 1 && (
                    <div style={{ marginTop: 8, padding: 8, backgroundColor: '#fffbe6', borderRadius: 4, fontSize: 12, color: '#ad8b00' }}>
                      ⚡ 多GPU训练：将使用 {selectedGpus.length} 张GPU进行分布式训练
                    </div>
                  )}
                </div>
              )}

              <button
                type="submit"
                disabled={creating || !selectedProject}
                style={{
                  padding: '12px 16px',
                  backgroundColor: creating || !selectedProject ? '#d9d9d9' : '#1890ff',
                  color: 'white',
                  border: 'none',
                  borderRadius: 6,
                  cursor: creating || !selectedProject ? 'not-allowed' : 'pointer',
                  fontWeight: 500
                }}
              >
                {creating ? '创建中...' : '开始训练'}
              </button>
            </form>
          </section>

          {/* 训练任务列表 */}
          <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ margin: 0 }}>训练任务</h3>
              <label style={{ fontSize: 14, display: 'flex', alignItems: 'center', gap: 4 }}>
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={e => setAutoRefresh(e.target.checked)}
                />
                自动刷新
              </label>
            </div>
            {!selectedProject ? (
              <p style={{ color: '#999', fontSize: 14 }}>请先选择项目</p>
            ) : jobs.length === 0 ? (
              <p style={{ color: '#999', fontSize: 14 }}>暂无训练任务</p>
            ) : (
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
                {jobs.map(job => (
                  <li
                    key={job.id}
                    onClick={() => setSelectedJob(job.id)}
                    style={{
                      padding: 12,
                      border: selectedJob === job.id ? '2px solid #1890ff' : '1px solid #eee',
                      borderRadius: 6,
                      cursor: 'pointer',
                      backgroundColor: selectedJob === job.id ? '#e6f7ff' : 'white'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ fontWeight: 500 }}>#{job.id}</span>
                      <span style={{
                        fontSize: 12,
                        padding: '2px 8px',
                        borderRadius: 4,
                        backgroundColor:
                          job.status === 'succeeded' ? '#f6ffed' :
                          job.status === 'failed' ? '#fff1f0' :
                          job.status === 'running' ? '#e6f7ff' :
                          job.status === 'canceled' ? '#fafafa' : '#fffbe6',
                        color:
                          job.status === 'succeeded' ? '#52c41a' :
                          job.status === 'failed' ? '#ff4d4f' :
                          job.status === 'running' ? '#1890ff' :
                          job.status === 'canceled' ? '#999' : '#faad14'
                      }}>
                        {job.status}
                      </span>
                    </div>
                    <div style={{ fontSize: 13, color: '#666' }}>
                      {job.framework.toUpperCase()} - {job.modelVariant}
                    </div>
                    {job.map50_95 !== null && (
                      <div style={{ fontSize: 12, color: '#52c41a', marginTop: 4 }}>
                        mAP50-95: {(job.map50_95 * 100).toFixed(2)}%
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        {/* 右侧：任务详情和日志 */}
        <div>
          {!selectedJob ? (
            <div style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, textAlign: 'center', color: '#999' }}>
              <p>请选择一个训练任务查看详情</p>
            </div>
          ) : (
            <>
              {/* 任务详情 */}
              <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <h3 style={{ margin: 0 }}>任务 #{selectedJob} 详情</h3>
                  {jobs.find(j => j.id === selectedJob)?.status === 'running' && (
                    <button
                      onClick={() => handleCancelJob(selectedJob)}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: '#ff4d4f',
                        color: 'white',
                        border: 'none',
                        borderRadius: 4,
                        cursor: 'pointer'
                      }}
                    >
                      取消训练
                    </button>
                  )}
                </div>
                {(() => {
                  const job = jobs.find(j => j.id === selectedJob)
                  if (!job) return null
                  return (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 14 }}>
                      <div><strong>框架:</strong> {job.framework.toUpperCase()}</div>
                      <div><strong>模型:</strong> {job.modelVariant}</div>
                      <div><strong>Epochs:</strong> {job.epochs}</div>
                      <div><strong>图像尺寸:</strong> {job.imgsz}</div>
                      <div><strong>Batch Size:</strong> {job.batch || '自动'}</div>
                      <div><strong>种子:</strong> {job.seed}</div>
                      {job.map50 !== null && <div><strong>mAP50:</strong> {(job.map50 * 100).toFixed(2)}%</div>}
                      {job.map50_95 !== null && <div><strong>mAP50-95:</strong> {(job.map50_95 * 100).toFixed(2)}%</div>}
                      {job.precision !== null && <div><strong>Precision:</strong> {(job.precision * 100).toFixed(2)}%</div>}
                      {job.recall !== null && <div><strong>Recall:</strong> {(job.recall * 100).toFixed(2)}%</div>}
                    </div>
                  )
                })()}

                {/* 模型文件 */}
                {artifacts.length > 0 && (
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee' }}>
                    <h4 style={{ margin: '0 0 8px 0' }}>模型文件</h4>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {artifacts.map(art => (
                        <li key={art.id} style={{
                          fontSize: 13,
                          padding: 8,
                          backgroundColor: '#f5f5f5',
                          borderRadius: 4,
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}>
                          <div>
                            <strong style={{ color: '#1890ff' }}>{art.format.toUpperCase()}:</strong>
                            <span style={{ marginLeft: 8, color: '#666' }}>{art.path}</span>
                            {art.size && (
                              <span style={{ marginLeft: 8, color: '#999' }}>
                                ({(art.size / 1024 / 1024).toFixed(2)} MB)
                              </span>
                            )}
                          </div>
                          <div style={{ display: 'flex', gap: 8 }}>
                            {/* 下载按钮 */}
                            <a
                              href={`/api/training/artifacts/${art.id}/download`}
                              download
                              style={{
                                padding: '4px 12px',
                                fontSize: 12,
                                backgroundColor: '#1890ff',
                                color: 'white',
                                border: 'none',
                                borderRadius: 4,
                                textDecoration: 'none',
                                cursor: 'pointer'
                              }}
                            >
                              下载
                            </a>
                            {/* 导出ONNX按钮 */}
                            {(art.format === 'pt' || art.format === 'pth') && (
                              <button
                                onClick={() => handleExportOnnx(art.id, selectedJob!)}
                                disabled={exporting === art.id}
                                style={{
                                  padding: '4px 12px',
                                  fontSize: 12,
                                  backgroundColor: exporting === art.id ? '#d9d9d9' : '#52c41a',
                                  color: 'white',
                                  border: 'none',
                                  borderRadius: 4,
                                  cursor: exporting === art.id ? 'not-allowed' : 'pointer'
                                }}
                              >
                                {exporting === art.id ? '导出中...' : '导出ONNX'}
                              </button>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>

              {/* 训练指标图表 */}
              {metrics.length > 0 && (
                <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginBottom: 16 }}>
                  <h3 style={{ marginTop: 0 }}>训练指标</h3>

                  {/* Loss曲线 */}
                  {metrics.some(m => m.loss !== null) && (
                    <div style={{ marginBottom: 24 }}>
                      <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#666' }}>损失 (Loss)</h4>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={metrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="loss" stroke="#ff4d4f" name="Loss" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* mAP曲线 */}
                  {(metrics.some(m => m.mAP50 !== null) || metrics.some(m => m.mAP !== null)) && (
                    <div>
                      <h4 style={{ margin: '0 0 12px 0', fontSize: 14, color: '#666' }}>平均精度 (mAP)</h4>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={metrics}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                          <YAxis label={{ value: 'mAP', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          {metrics.some(m => m.mAP50 !== null) && (
                            <Line type="monotone" dataKey="mAP50" stroke="#1890ff" name="mAP50" dot={false} />
                          )}
                          {metrics.some(m => m['mAP50-95'] !== null) && (
                            <Line type="monotone" dataKey="mAP50-95" stroke="#52c41a" name="mAP50-95" dot={false} />
                          )}
                          {metrics.some(m => m.mAP !== null) && (
                            <Line type="monotone" dataKey="mAP" stroke="#722ed1" name="mAP" dot={false} />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  <div style={{ marginTop: 12, fontSize: 12, color: '#999', textAlign: 'center' }}>
                    实时更新 • 显示最近 {metrics.length} 个epoch
                  </div>
                </section>
              )}

              {/* 训练日志 */}
              <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
                <h3 style={{ marginTop: 0 }}>训练日志</h3>
                <div
                  style={{
                    backgroundColor: '#1e1e1e',
                    color: '#d4d4d4',
                    padding: 16,
                    borderRadius: 4,
                    fontFamily: 'Consolas, Monaco, monospace',
                    fontSize: 12,
                    maxHeight: 500,
                    overflowY: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}
                >
                  {logs.length === 0 ? (
                    <div style={{ color: '#999' }}>等待日志...</div>
                  ) : (
                    logs.map((line, i) => (
                      <div key={i}>{line}</div>
                    ))
                  )}
                </div>
              </section>

              {/* 训练结果图片 - 仅在训练成功后显示 */}
              {(() => {
                const job = jobs.find(j => j.id === selectedJob)
                if (!job || job.status !== 'succeeded' || resultImages.length === 0) return null

                // 优先显示 metrics_plot.png 或 results.png
                const priorityImages = ['metrics_plot.png', 'results.png']
                const mainImage = resultImages.find(img => priorityImages.includes(img.name))

                return (
                  <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginTop: 16 }}>
                    <h3 style={{ marginTop: 0, color: '#52c41a' }}>训练结果</h3>
                    {mainImage && (
                      <div style={{ marginBottom: 16 }}>
                        <img
                          src={mainImage.path}
                          alt={mainImage.name}
                          style={{
                            maxWidth: '100%',
                            borderRadius: 8,
                            border: '1px solid #eee',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                          }}
                        />
                      </div>
                    )}
                    {resultImages.length > 1 && (
                      <div>
                        <h4 style={{ margin: '16px 0 8px 0', fontSize: 14, color: '#666' }}>其他图片</h4>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                          {resultImages
                            .filter(img => !priorityImages.includes(img.name))
                            .map((img, i) => (
                              <a
                                key={i}
                                href={img.path}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{
                                  padding: '6px 12px',
                                  backgroundColor: '#f5f5f5',
                                  borderRadius: 4,
                                  fontSize: 12,
                                  color: '#1890ff',
                                  textDecoration: 'none'
                                }}
                              >
                                {img.name}
                              </a>
                            ))
                          }
                        </div>
                      </div>
                    )}
                  </section>
                )
              })()}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
