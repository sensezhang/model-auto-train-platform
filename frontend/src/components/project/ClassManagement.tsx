import React, { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Input,
  Space,
  Tag,
  Typography,
  Popconfirm,
  message,
  Form,
  Empty,
} from 'antd'
import { PlusOutlined, DeleteOutlined, TagsOutlined } from '@ant-design/icons'
import { Project, ProjectClass } from '../../pages/App'

const { Title, Text } = Typography

const CLASS_COLORS = [
  '#1677ff', '#52c41a', '#fa8c16', '#722ed1', '#eb2f96',
  '#13c2c2', '#fadb14', '#f5222d', '#2f54eb', '#a0d911',
]

interface Props {
  project: Project
  onProjectUpdated: (updated: Project) => void
}

export const ClassManagement: React.FC<Props> = ({ project, onProjectUpdated }) => {
  const [classes, setClasses] = useState<ProjectClass[]>(project.classes || [])
  const [newName, setNewName] = useState('')
  const [adding, setAdding] = useState(false)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  // 同步父组件传来的 classes
  useEffect(() => {
    setClasses(project.classes || [])
  }, [project.id])

  const refreshClasses = async () => {
    try {
      const res = await fetch(`/api/projects/${project.id}/classes`)
      const data = await res.json()
      setClasses(Array.isArray(data) ? data : [])
    } catch {}
  }

  const handleAdd = async () => {
    const name = newName.trim()
    if (!name) { message.warning('请输入类别名称'); return }
    if (classes.some(c => c.name.toLowerCase() === name.toLowerCase())) {
      message.warning('类别名称已存在')
      return
    }
    setAdding(true)
    try {
      const res = await fetch(`/api/projects/${project.id}/classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
      if (!res.ok) throw new Error('添加失败')
      const cls = await res.json()
      const updated = [...classes, cls]
      setClasses(updated)
      setNewName('')
      message.success(`类别「${name}」已添加`)
      // 通知父组件更新
      onProjectUpdated({ ...project, classes: updated })
    } catch {
      message.error('添加类别失败')
    } finally {
      setAdding(false)
    }
  }

  const handleDelete = async (classId: number, className: string) => {
    setDeletingId(classId)
    try {
      // 目前后端没有 DELETE /classes/{id} 接口，暂时只提示
      message.info('类别删除功能暂不支持（会影响已有标注数据）')
    } finally {
      setDeletingId(null)
    }
  }

  const columns = [
    {
      title: '序号',
      key: 'index',
      width: 60,
      render: (_: any, __: any, index: number) => (
        <Text type="secondary">{index + 1}</Text>
      ),
    },
    {
      title: '类别名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, _: any, index: number) => (
        <Space>
          <Tag color={CLASS_COLORS[index % CLASS_COLORS.length]}>{name}</Tag>
        </Space>
      ),
    },
    {
      title: '快捷键',
      key: 'hotkey',
      width: 100,
      render: (_: any, __: any, index: number) => (
        <Text type="secondary" keyboard>{index + 1 <= 9 ? String(index + 1) : '-'}</Text>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_: any, record: ProjectClass) => (
        <Popconfirm
          title="删除类别"
          description="删除后不影响已有标注数据，但后续无法新增该类别标注"
          onConfirm={() => handleDelete(record.id, record.name)}
          okText="删除"
          cancelText="取消"
          okButtonProps={{ danger: true }}
        >
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            size="small"
            loading={deletingId === record.id}
          />
        </Popconfirm>
      ),
    },
  ]

  return (
    <Card
      title={
        <Space>
          <TagsOutlined />
          <span>类别管理</span>
          <Text type="secondary" style={{ fontSize: 13, fontWeight: 400 }}>
            共 {classes.length} 个类别
          </Text>
        </Space>
      }
      style={{ minHeight: 400 }}
    >
      {/* 添加类别 */}
      <Space.Compact style={{ marginBottom: 20, width: '100%', maxWidth: 400 }}>
        <Input
          placeholder="输入新类别名称"
          value={newName}
          onChange={e => setNewName(e.target.value)}
          onPressEnter={handleAdd}
          maxLength={50}
        />
        <Button
          type="primary"
          icon={<PlusOutlined />}
          loading={adding}
          onClick={handleAdd}
        >
          添加
        </Button>
      </Space.Compact>

      {/* 类别列表 */}
      {classes.length === 0 ? (
        <Empty
          description="暂无类别，请在上方添加"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <Table
          dataSource={classes}
          columns={columns}
          rowKey="id"
          pagination={false}
          size="middle"
          footer={() => (
            <Text type="secondary" style={{ fontSize: 12 }}>
              💡 在标注工作台中，可按数字键 1-9 快速切换类别
            </Text>
          )}
        />
      )}
    </Card>
  )
}
