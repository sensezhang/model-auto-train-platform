import React, { useEffect, useState } from 'react'
import {
  Card, Table, Tag, Button, Space, Typography, Select, Statistic, Row, Col,
  message, Tooltip, Empty, Alert, Spin,
} from 'antd'
import {
  DownloadOutlined, ExportOutlined, DatabaseOutlined,
  CheckCircleOutlined, CloseCircleOutlined, SyncOutlined,
} from '@ant-design/icons'
import { Project } from '../../pages/App'

const { Text, Title } = Typography

interface TrainingJob {
  id: number
  projectId: number
  framework: string
  modelVariant: string
  epochs: number
  status: string
  map50: number | null
  map50_95: number | null
  precision: number | null
  recall: number | null
  startedAt: string | null
  finishedAt: string | null
}

interface ModelArtifact {
  id: number
  trainingJobId: number
  format: string
  path: string
  size: number | null
  createdAt: string
}

const formatSize = (bytes: number | null) => {
  if (!bytes) return '-'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(1) + ' MB'
}

const statusTag = (status: string) => {
  const map: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
    pending:   { color: 'gold',    icon: <SyncOutlined spin />,     label: '等待中' },
    running:   { color: 'blue',    icon: <SyncOutlined spin />,     label: '训练中' },
    succeeded: { color: 'green',   icon: <CheckCircleOutlined />,   label: '已完成' },
    failed:    { color: 'red',     icon: <CloseCircleOutlined />,   label: '失败' },
    canceled:  { color: 'default', icon: <CloseCircleOutlined />,   label: '已取消' },
  }
  const cfg = map[status] || { color: 'default', icon: null, label: status }
  return <Tag color={cfg.color} icon={cfg.icon}>{cfg.label}</Tag>
}

export const ModelManagement: React.FC<{ project: Project }> = ({ project }) => {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null)
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([])
  const [loadingArtifacts, setLoadingArtifacts] = useState(false)
  const [exporting, setExporting] = useState<number | null>(null)

  const loadJobs = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/training/jobs')
      const data = await res.json()
      const filtered = Array.isArray(data)
        ? data.filter((j: TrainingJob) => j.projectId === project.id)
        : []
      setJobs(filtered)
      // 默认选最新完成的任务
      const finished = filtered.filter((j: TrainingJob) => j.status === 'succeeded')
      if (finished.length > 0 && !selectedJobId) {
        setSelectedJobId(finished[0].id)
      }
    } catch { setJobs([]) }
    finally { setLoading(false) }
  }

  const loadArtifacts = async (jobId: number) => {
    setLoadingArtifacts(true)
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/artifacts`)
      const data = await res.json()
      setArtifacts(Array.isArray(data) ? data : [])
    } catch { setArtifacts([]) }
    finally { setLoadingArtifacts(false) }
  }

  useEffect(() => { loadJobs() }, [project.id])
  useEffect(() => { if (selectedJobId) loadArtifacts(selectedJobId) }, [selectedJobId])

  const handleExportOnnx = async (artifactId: number, jobId: number) => {
    setExporting(artifactId)
    try {
      const res = await fetch(`/api/training/jobs/${jobId}/export-onnx?artifact_id=${artifactId}&simplify=false`, {
        method: 'POST',
      })
      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.detail || '导出失败')
      }
      const d = await res.json()
      message.success(d.message || 'ONNX 导出成功')
      loadArtifacts(jobId)
    } catch (e: any) {
      message.error(e.message || '导出失败')
    } finally {
      setExporting(null)
    }
  }

  const selectedJob = jobs.find(j => j.id === selectedJobId)

  const artifactColumns = [
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      width: 80,
      render: (fmt: string) => <Tag color={fmt === 'onnx' ? 'purple' : 'blue'}>{fmt.toUpperCase()}</Tag>,
    },
    {
      title: '文件大小',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (size: number | null) => <Text type="secondary">{formatSize(size)}</Text>,
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (t: string) => <Text type="secondary">{new Date(t).toLocaleString('zh-CN')}</Text>,
    },
    {
      title: '操作',
      key: 'action',
      width: 160,
      render: (_: any, record: ModelArtifact) => (
        <Space>
          <Tooltip title="下载模型文件">
            <Button
              type="link"
              icon={<DownloadOutlined />}
              href={`/api/training/artifacts/${record.id}/download`}
              target="_blank"
            >
              下载
            </Button>
          </Tooltip>
          {record.format === 'pt' && (
            <Tooltip title="将 .pt 模型转换为 ONNX 格式">
              <Button
                type="link"
                icon={<ExportOutlined />}
                loading={exporting === record.id}
                onClick={() => handleExportOnnx(record.id, record.trainingJobId)}
              >
                导出 ONNX
              </Button>
            </Tooltip>
          )}
        </Space>
      ),
    },
  ]

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      {/* 训练任务选择 */}
      <Card
        title={<Space><DatabaseOutlined /><span>模型管理</span></Space>}
        extra={
          <Button onClick={loadJobs} loading={loading} size="small">刷新</Button>
        }
      >
        {jobs.length === 0 ? (
          <Empty description="暂无训练任务，请先在「训练」菜单创建任务" />
        ) : (
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>选择训练任务</Text>
              <Select
                value={selectedJobId}
                onChange={setSelectedJobId}
                style={{ width: '100%', maxWidth: 480 }}
                options={jobs.map(j => ({
                  value: j.id,
                  label: (
                    <Space>
                      {statusTag(j.status)}
                      <span>#{j.id} {j.framework === 'yolo' ? 'YOLOv11' : 'RF-DETR'} {j.modelVariant}</span>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {j.startedAt ? new Date(j.startedAt).toLocaleDateString('zh-CN') : ''}
                      </Text>
                    </Space>
                  ),
                }))}
              />
            </div>

            {/* 训练指标 */}
            {selectedJob && selectedJob.status === 'succeeded' && (
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="mAP50"
                    value={selectedJob.map50 ? (selectedJob.map50 * 100).toFixed(1) : '-'}
                    suffix={selectedJob.map50 ? '%' : ''}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="mAP50-95"
                    value={selectedJob.map50_95 ? (selectedJob.map50_95 * 100).toFixed(1) : '-'}
                    suffix={selectedJob.map50_95 ? '%' : ''}
                    valueStyle={{ color: '#1677ff' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Precision"
                    value={selectedJob.precision ? (selectedJob.precision * 100).toFixed(1) : '-'}
                    suffix={selectedJob.precision ? '%' : ''}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Recall"
                    value={selectedJob.recall ? (selectedJob.recall * 100).toFixed(1) : '-'}
                    suffix={selectedJob.recall ? '%' : ''}
                  />
                </Col>
              </Row>
            )}
          </Space>
        )}
      </Card>

      {/* 模型产物列表 */}
      {selectedJobId && (
        <Card title={`任务 #${selectedJobId} 模型产物`}>
          <Spin spinning={loadingArtifacts}>
            {artifacts.length === 0 ? (
              <Empty description="暂无模型产物（训练完成后自动生成）" />
            ) : (
              <Table
                dataSource={artifacts}
                columns={artifactColumns}
                rowKey="id"
                pagination={false}
                size="middle"
              />
            )}
          </Spin>
        </Card>
      )}
    </Space>
  )
}
