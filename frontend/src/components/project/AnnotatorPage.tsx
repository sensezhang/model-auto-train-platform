import React from 'react'
import { Annotator } from '../../pages/Annotator'
import { Project } from '../../pages/App'

interface Props {
  project: Project
  onBack: () => void
}

/**
 * 标注页面包装器：将 Annotator 嵌入项目详情页的「标注」菜单
 * Annotator 占满整个视口（height: 100vh），所以此处直接渲染
 */
export const AnnotatorPage: React.FC<Props> = ({ project, onBack }) => {
  return (
    <Annotator
      projectId={project.id}
      onBack={onBack}
    />
  )
}
