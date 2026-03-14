import React from 'react'
import { Training } from '../../pages/Training'
import { Project } from '../../pages/App'

interface Props {
  project: Project
}

/**
 * 训练页面包装器：将 Training 嵌入项目详情页的「训练」菜单
 * 传入 defaultProjectId 以自动选中当前项目
 */
export const TrainingPage: React.FC<Props> = ({ project }) => {
  return (
    <Training
      onBack={() => {}}          // 项目详情内不需要返回，传空函数
      defaultProjectId={project.id}
    />
  )
}
