import React from 'react'
import { Inference } from '../../pages/Inference'
import { Project } from '../../pages/App'

interface Props {
  project: Project
}

/**
 * 推理页面包装器：将 Inference 嵌入项目详情页的「推理」菜单
 * Inference 使用 onBack 回调返回，项目详情内不需要跳转，传空函数
 */
export const InferencePage: React.FC<Props> = ({ project }) => {
  return (
    <Inference
      onBack={() => {}}
      defaultProjectId={project.id}
    />
  )
}
