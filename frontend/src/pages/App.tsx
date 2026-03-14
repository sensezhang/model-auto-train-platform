import React, { useEffect, useState } from 'react'
import {
  ConfigProvider,
  Layout,
  Button,
  Card,
  Row,
  Col,
  Tag,
  Modal,
  Form,
  Input,
  Empty,
  Badge,
  Space,
  Typography,
  Popconfirm,
  message,
  Tooltip,
} from 'antd'
import {
  PlusOutlined,
  FolderOutlined,
  PictureOutlined,
  DeleteOutlined,
  ArrowRightOutlined,
} from '@ant-design/icons'
import { ProjectDetail } from './ProjectDetail'

const { Header, Content } = Layout
const { Title, Text } = Typography

export type ProjectClass = { id: number; name: string; color?: string }
export type Project = {
  id: number
  name: string
  description?: string
  classes: ProjectClass[]
  imageCount: number
}

// 类别颜色池
const CLASS_COLORS = [
  '#1677ff', '#52c41a', '#fa8c16', '#722ed1', '#eb2f96',
  '#13c2c2', '#fadb14', '#f5222d', '#2f54eb', '#a0d911',
]
const getTagColor = (index: number) => CLASS_COLORS[index % CLASS_COLORS.length]

export const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState<'ok' | 'error' | 'unknown'>('unknown')
  const [createOpen, setCreateOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const [currentProject, setCurrentProject] = useState<Project | null>(null)
  const [form] = Form.useForm()

  const refresh = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/projects')
      const data = await res.json()
      setProjects(Array.isArray(data) ? data : [])
    } catch {
      setProjects([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => setHealth(d.status === 'ok' ? 'ok' : 'error'))
      .catch(() => setHealth('error'))
    refresh()
  }, [])

  const handleCreate = async () => {
    try {
      const values = await form.validateFields()
      setCreating(true)
      const body = {
        name: values.name.trim(),
        description: '',
        classes: values.classes
          ? values.classes.split(',').map((s: string) => s.trim()).filter(Boolean)
          : [],
      }
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error('创建失败')
      message.success('项目创建成功')
      form.resetFields()
      setCreateOpen(false)
      refresh()
    } catch (e: any) {
      if (e?.errorFields) return // 表单校验失败
      message.error(e.message || '创建项目失败')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (projectId: number) => {
    try {
      const res = await fetch(`/api/projects/${projectId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('删除失败')
      message.success('项目已删除')
      refresh()
    } catch {
      message.error('删除失败')
    }
  }

  // 进入项目详情
  if (currentProject) {
    return (
      <ProjectDetail
        project={currentProject}
        onBack={() => {
          setCurrentProject(null)
          refresh()
        }}
        onProjectUpdated={(updated) => {
          setCurrentProject(updated)
          setProjects(prev => prev.map(p => p.id === updated.id ? updated : p))
        }}
      />
    )
  }

  return (
    <ConfigProvider theme={{ token: { colorPrimary: '#1677ff', borderRadius: 8 } }}>
      <Layout style={{ minHeight: '100vh', background: '#f5f5f5' }}>
        {/* 顶部导航 */}
        <Header
          style={{
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            position: 'sticky',
            top: 0,
            zIndex: 100,
          }}
        >
          <Space align="center" size={12}>
            <FolderOutlined style={{ fontSize: 22, color: '#1677ff' }} />
            <Title level={4} style={{ margin: 0, color: '#262626' }}>
              标注与训练平台
            </Title>
            <Badge
              status={health === 'ok' ? 'success' : health === 'error' ? 'error' : 'default'}
              text={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {health === 'ok' ? '服务正常' : health === 'error' ? '服务异常' : '检测中'}
                </Text>
              }
            />
          </Space>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateOpen(true)}
          >
            新建项目
          </Button>
        </Header>

        {/* 内容区 */}
        <Content style={{ padding: '24px', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
          {projects.length === 0 && !loading ? (
            <Card style={{ textAlign: 'center', padding: '48px 0' }}>
              <Empty
                description="暂无项目，点击右上角新建项目开始"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
                  新建第一个项目
                </Button>
              </Empty>
            </Card>
          ) : (
            <Row gutter={[16, 16]}>
              {projects.map((p) => (
                <Col key={p.id} xs={24} sm={12} lg={8}>
                  <Card
                    hoverable
                    style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
                    bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column' }}
                    actions={[
                      <Button
                        key="open"
                        type="primary"
                        icon={<ArrowRightOutlined />}
                        block
                        style={{ margin: '0 12px', width: 'calc(100% - 24px)' }}
                        onClick={() => setCurrentProject(p)}
                      >
                        进入项目
                      </Button>,
                    ]}
                  >
                    {/* 项目标题 */}
                    <div style={{ marginBottom: 12 }}>
                      <Text strong style={{ fontSize: 16 }}>{p.name}</Text>
                      {p.description && (
                        <Text type="secondary" style={{ display: 'block', fontSize: 12, marginTop: 2 }}>
                          {p.description}
                        </Text>
                      )}
                    </div>

                    {/* 类别标签 */}
                    <div style={{ marginBottom: 12, flex: 1 }}>
                      {p.classes.length > 0 ? (
                        <Space wrap size={4}>
                          {p.classes.slice(0, 6).map((c, i) => (
                            <Tag key={c.id} color={getTagColor(i)} style={{ margin: 0 }}>
                              {c.name}
                            </Tag>
                          ))}
                          {p.classes.length > 6 && (
                            <Tag color="default">+{p.classes.length - 6}</Tag>
                          )}
                        </Space>
                      ) : (
                        <Text type="secondary" style={{ fontSize: 12 }}>暂无类别</Text>
                      )}
                    </div>

                    {/* 图片数量 + 删除 */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 8 }}>
                      <Space size={4}>
                        <PictureOutlined style={{ color: '#8c8c8c' }} />
                        <Text type="secondary" style={{ fontSize: 13 }}>
                          {p.imageCount} 张图片
                        </Text>
                      </Space>
                      <Popconfirm
                        title="删除项目"
                        description="将同时删除所有图片和标注，此操作不可恢复！"
                        onConfirm={() => handleDelete(p.id)}
                        okText="确认删除"
                        cancelText="取消"
                        okButtonProps={{ danger: true }}
                      >
                        <Tooltip title="删除项目">
                          <Button
                            type="text"
                            danger
                            icon={<DeleteOutlined />}
                            size="small"
                          />
                        </Tooltip>
                      </Popconfirm>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          )}
        </Content>

        {/* 新建项目 Modal */}
        <Modal
          title="新建项目"
          open={createOpen}
          onOk={handleCreate}
          onCancel={() => { setCreateOpen(false); form.resetFields() }}
          confirmLoading={creating}
          okText="创建"
          cancelText="取消"
          width={480}
        >
          <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
            <Form.Item
              name="name"
              label="项目名称"
              rules={[{ required: true, message: '请输入项目名称' }]}
            >
              <Input placeholder="例如：车牌检测项目" autoFocus />
            </Form.Item>
            <Form.Item
              name="classes"
              label="检测类别（可选）"
              extra="多个类别请用英文逗号分隔，例如：cat, dog, person"
            >
              <Input placeholder="cat, dog, person" />
            </Form.Item>
          </Form>
        </Modal>
      </Layout>
    </ConfigProvider>
  )
}
