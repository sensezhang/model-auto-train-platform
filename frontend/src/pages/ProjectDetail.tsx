import React, { useState } from 'react'
import {
  ConfigProvider,
  Layout,
  Menu,
  Breadcrumb,
  Typography,
  Button,
  Space,
} from 'antd'
import {
  TagsOutlined,
  FolderOpenOutlined,
  RobotOutlined,
  EditOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  ScissorOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons'
import { Project } from './App'
import { ClassManagement } from '../components/project/ClassManagement'
import { DataManagement } from '../components/project/DataManagement'
import { AutoLabel } from '../components/project/AutoLabel'
import { AnnotatorPage } from '../components/project/AnnotatorPage'
import { TrainingPage } from '../components/project/TrainingPage'
import { ModelManagement } from '../components/project/ModelManagement'
import { InferencePage } from '../components/project/InferencePage'
import { Sam3Wrapper } from '../components/project/Sam3Wrapper'

const { Sider, Content, Header } = Layout
const { Title, Text } = Typography

type MenuKey = 'classes' | 'data' | 'autolabel' | 'annotate' | 'training' | 'models' | 'inference' | 'sam3'

const menuItems = [
  { key: 'classes',   label: '类别',   icon: <TagsOutlined /> },
  { key: 'data',      label: '数据管理', icon: <FolderOpenOutlined /> },
  { key: 'autolabel', label: 'AI识别', icon: <RobotOutlined /> },
  { key: 'annotate',  label: '标注',   icon: <EditOutlined /> },
  { key: 'training',  label: '训练',   icon: <ThunderboltOutlined /> },
  { key: 'models',    label: '模型管理', icon: <DatabaseOutlined /> },
  { key: 'inference', label: '推理',   icon: <ExperimentOutlined /> },
  { key: 'sam3',      label: 'SAM3分割', icon: <ScissorOutlined /> },
]

export interface ProjectDetailProps {
  project: Project
  onBack: () => void
  onProjectUpdated: (updated: Project) => void
}

export const ProjectDetail: React.FC<ProjectDetailProps> = ({ project, onBack, onProjectUpdated }) => {
  const [activeMenu, setActiveMenu] = useState<MenuKey>('classes')

  const renderContent = () => {
    switch (activeMenu) {
      case 'classes':
        return (
          <ClassManagement
            project={project}
            onProjectUpdated={onProjectUpdated}
          />
        )
      case 'data':
        return <DataManagement project={project} />
      case 'autolabel':
        return <AutoLabel project={project} />
      case 'annotate':
        return (
          <AnnotatorPage
            project={project}
            onBack={() => setActiveMenu('classes')}
          />
        )
      case 'training':
        return <TrainingPage project={project} />
      case 'models':
        return <ModelManagement project={project} />
      case 'inference':
        return <InferencePage project={project} />
      case 'sam3':
        return <Sam3Wrapper project={project} />
      default:
        return null
    }
  }

  return (
    <ConfigProvider theme={{ token: { colorPrimary: '#1677ff', borderRadius: 8 } }}>
      <Layout style={{ minHeight: '100vh' }}>
        {/* 顶部 Header */}
        <Header
          style={{
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            padding: '0 24px',
            position: 'sticky',
            top: 0,
            zIndex: 100,
          }}
        >
          <Space align="center" size={16}>
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={onBack}
              style={{ color: '#595959' }}
            >
              项目列表
            </Button>
            <Breadcrumb
              items={[
                { title: '项目列表', onClick: onBack, className: 'cursor-pointer' },
                { title: <Text strong>{project.name}</Text> },
              ]}
            />
          </Space>
        </Header>

        <Layout>
          {/* 左侧菜单 */}
          <Sider
            width={200}
            style={{
              background: '#fff',
              borderRight: '1px solid #f0f0f0',
              minHeight: 'calc(100vh - 64px)',
            }}
          >
            <Menu
              mode="inline"
              selectedKeys={[activeMenu]}
              onClick={({ key }) => setActiveMenu(key as MenuKey)}
              style={{ border: 'none', paddingTop: 12 }}
              items={menuItems}
            />
          </Sider>

          {/* 右侧内容 */}
          <Content
            style={{
              background: activeMenu === 'annotate' ? '#f5f7fa' : '#f5f5f5',
              padding: activeMenu === 'annotate' ? 0 : 24,
              minHeight: 'calc(100vh - 64px)',
              height: activeMenu === 'annotate' ? 'calc(100vh - 64px)' : undefined,
              overflow: activeMenu === 'annotate' ? 'hidden' : undefined,
            }}
          >
            {renderContent()}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  )
}
