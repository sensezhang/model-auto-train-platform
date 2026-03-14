import React from 'react'
import { Sam3Page } from '../../pages/Sam3Page'
import { Project } from '../../pages/App'

interface Props {
  project: Project
}

export const Sam3Wrapper: React.FC<Props> = ({ project }) => {
  return <Sam3Page projectId={project.id} />
}
