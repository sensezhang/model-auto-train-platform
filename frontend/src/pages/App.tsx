import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Annotator } from './Annotator'
import { ExportDialog } from '../components/ExportDialog'

type Project = { id: number; name: string }

type ImportResult = {
  total: number
  imported: number
  duplicates: number
  errors: number
  annotations_imported?: number
  annotations_skipped?: number
  class_mapping_found?: boolean
}

type ImportProgress = {
  jobId: number
  projectId: number
  status: string
  total: number
  current: number
  progress: number
  message: string
  imported?: number
  duplicates?: number
  errors?: number
  annotations_imported?: number
  annotations_skipped?: number
}

export const App: React.FC = () => {
  const [health, setHealth] = useState('unknown')
  const [projects, setProjects] = useState<Project[]>([])
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newClasses, setNewClasses] = useState('')
  const [importing, setImporting] = useState<number | null>(null)
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null)
  const [importResult, setImportResult] = useState<Record<number, ImportResult>>({})
  const [deleting, setDeleting] = useState<number | null>(null)
  const [exporting, setExporting] = useState<number | null>(null)
  const [exportDialogProject, setExportDialogProject] = useState<Project | null>(null)
  const [importDialogProject, setImportDialogProject] = useState<Project | null>(null)

  const refresh = () => {
    fetch('/api/projects').then(r => r.json()).then(setProjects).catch(() => setProjects([]))
  }

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(d => setHealth(d.status)).catch(() => setHealth('error'))
    refresh()
  }, [])

  const onCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    try {
      const body = {
        name: newName.trim(),
        description: '',
        classes: newClasses.split(',').map(s => s.trim()).filter(Boolean),
      }
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error('创建项目失败')
      setNewName('')
      setNewClasses('')
      refresh()
    } catch (e) {
      console.error(e)
      alert('创建项目失败')
    } finally {
      setCreating(false)
    }
  }

  const onImport = async (projectId: number, files: File[] | null, format: 'images' | 'yolo' | 'single' | 'folder' = 'images') => {
    if (!files || files.length === 0) return

    // 单张图片上传
    if (format === 'single') {
      const file = files[0]
      const ext = file.name.toLowerCase().split('.').pop()
      if (!['jpg', 'jpeg', 'png'].includes(ext || '')) {
        alert('请上传 JPG 或 PNG 图片')
        return
      }
      setImporting(projectId)
      setImportDialogProject(null)

      try {
        const fd = new FormData()
        fd.append('file', file)
        const res = await fetch(`/api/projects/${projectId}/import/image`, {
          method: 'POST',
          body: fd,
        })
        const data = await res.json()
        if (data.success) {
          alert('导入成功')
        } else if (data.duplicate) {
          alert('图片已存在（重复）')
        } else {
          alert('导入失败: ' + data.message)
        }
        refresh()
      } catch (e) {
        console.error(e)
        alert('导入失败')
      } finally {
        setImporting(null)
      }
      return
    }

    // 文件夹批量上传
    if (format === 'folder') {
      // 过滤出图片文件
      const imageFiles = files.filter(file => {
        const ext = file.name.toLowerCase().split('.').pop()
        return ['jpg', 'jpeg', 'png'].includes(ext || '')
      })

      if (imageFiles.length === 0) {
        alert('所选文件夹中没有找到 JPG/PNG 图片')
        return
      }

      setImporting(projectId)
      setImportDialogProject(null)
      setImportProgress({
        jobId: 0,
        projectId,
        status: 'running',
        total: imageFiles.length,
        current: 0,
        progress: 0,
        message: `正在导入 0/${imageFiles.length}...`
      })

      let imported = 0
      let duplicates = 0
      let errors = 0

      try {
        for (let i = 0; i < imageFiles.length; i++) {
          const file = imageFiles[i]
          try {
            const fd = new FormData()
            fd.append('file', file)
            const res = await fetch(`/api/projects/${projectId}/import/image`, {
              method: 'POST',
              body: fd,
            })
            const data = await res.json()
            if (data.success) {
              imported++
            } else if (data.duplicate) {
              duplicates++
            } else {
              errors++
            }
          } catch {
            errors++
          }

          // 更新进度
          const current = i + 1
          setImportProgress({
            jobId: 0,
            projectId,
            status: 'running',
            total: imageFiles.length,
            current,
            progress: Math.round(current / imageFiles.length * 100),
            message: `正在导入 ${current}/${imageFiles.length}...`,
            imported,
            duplicates,
            errors,
          })
        }

        // 完成
        setImportProgress({
          jobId: 0,
          projectId,
          status: 'succeeded',
          total: imageFiles.length,
          current: imageFiles.length,
          progress: 100,
          message: '导入完成',
          imported,
          duplicates,
          errors,
        })

        setImportResult(prev => ({
          ...prev,
          [projectId]: {
            total: imageFiles.length,
            imported,
            duplicates,
            errors,
          }
        }))

        // 3秒后清除进度显示
        setTimeout(() => setImportProgress(null), 3000)
        refresh()
      } catch (e) {
        console.error(e)
        alert('导入失败')
        setImportProgress(null)
      } finally {
        setImporting(null)
      }
      return
    }

    // ZIP 文件上传
    const file = files[0]
    if (!file.name.toLowerCase().endsWith('.zip')) {
      alert('请上传zip文件')
      return
    }
    setImporting(projectId)
    setImportDialogProject(null)
    setImportProgress(null)

    try {
      const fd = new FormData()
      fd.append('file', file)

      if (format === 'yolo') {
        // YOLO格式使用异步导入，返回job_id后轮询进度
        const res = await fetch(`/api/projects/${projectId}/import/yolo?import_annotations=true`, {
          method: 'POST',
          body: fd,
        })
        if (!res.ok) throw new Error('导入失败')
        const data = await res.json()
        const jobId = data.job_id

        // 开始轮询进度
        setImportProgress({
          jobId,
          projectId,
          status: 'pending',
          total: 0,
          current: 0,
          progress: 0,
          message: '准备导入...'
        })

        // 轮询直到完成
        const pollProgress = async () => {
          try {
            const progressRes = await fetch(`/api/projects/${projectId}/import/jobs/${jobId}`)
            if (!progressRes.ok) return

            const progressData = await progressRes.json()
            setImportProgress({
              jobId,
              projectId,
              status: progressData.status,
              total: progressData.total,
              current: progressData.current,
              progress: progressData.progress,
              message: progressData.message,
              imported: progressData.imported,
              duplicates: progressData.duplicates,
              errors: progressData.errors,
              annotations_imported: progressData.annotations_imported,
              annotations_skipped: progressData.annotations_skipped,
            })

            if (progressData.status === 'succeeded' || progressData.status === 'failed') {
              // 导入完成
              setImporting(null)
              if (progressData.status === 'succeeded') {
                setImportResult(prev => ({
                  ...prev,
                  [projectId]: {
                    total: progressData.total,
                    imported: progressData.imported,
                    duplicates: progressData.duplicates,
                    errors: progressData.errors,
                    annotations_imported: progressData.annotations_imported,
                    annotations_skipped: progressData.annotations_skipped,
                  }
                }))
              } else {
                alert('导入失败: ' + progressData.message)
              }
              // 3秒后清除进度显示
              setTimeout(() => setImportProgress(null), 3000)
              refresh()
              return
            }

            // 继续轮询
            setTimeout(pollProgress, 1000)
          } catch (e) {
            console.error('轮询进度失败:', e)
            setTimeout(pollProgress, 2000)
          }
        }

        pollProgress()
      } else {
        // 普通图片导入（同步）
        const res = await fetch(`/api/projects/${projectId}/import`, {
          method: 'POST',
          body: fd,
        })
        if (!res.ok) throw new Error('导入失败')
        const data = await res.json()
        setImportResult(prev => ({ ...prev, [projectId]: data }))
        setImporting(null)
        refresh()
      }
    } catch (e) {
      console.error(e)
      alert('导入失败')
      setImporting(null)
      setImportProgress(null)
    }
  }

  const onDeleteProject = async (projectId: number) => {
    if (!confirm('确认删除该项目吗？将同时删除所有图片和标注数据，此操作不可恢复！')) return
    setDeleting(projectId)
    try {
      const res = await fetch(`/api/projects/${projectId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('删除失败')
      refresh()
    } catch (e) {
      console.error(e)
      alert('删除失败')
    } finally {
      setDeleting(null)
    }
  }

  const onExport = (project: Project) => {
    setExportDialogProject(project)
  }

  if ((window as any).__view === 'annotator') {
    // 仅兼容性：不使用真实路由器。保留。
  }

  const [view, setView] = useState<'home'|'annotator'>('home')
  const [annotatorProjectId, setAnnotatorProjectId] = useState<number | null>(null)

  if (view === 'annotator' && annotatorProjectId != null) {
    return <Annotator projectId={annotatorProjectId} onBack={() => setView('home')} />
  }

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 16, maxWidth: 900, margin: '0 auto' }}>
      <h2>标注与训练平台</h2>
      <p>后端健康状态: {health}</p>

      <section style={{ marginTop: 24, padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
        <h3>创建项目</h3>
        <form onSubmit={onCreate} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            placeholder="项目名称"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            style={{ padding: 8, flex: 1 }}
          />
          <input
            placeholder="类别(以逗号分隔)"
            value={newClasses}
            onChange={e => setNewClasses(e.target.value)}
            style={{ padding: 8, flex: 1 }}
          />
          <button disabled={creating} type="submit" style={{ padding: '8px 12px' }}>
            {creating ? '创建中...' : '创建'}
          </button>
        </form>
      </section>

      <section style={{ marginTop: 24 }}>
        <h3>项目列表</h3>
        {projects.length === 0 ? (
          <p>暂无项目，请先创建。</p>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0, display: 'grid', gap: 12 }}>
            {projects.map(p => (
              <li key={p.id} style={{ border: '1px solid #eee', borderRadius: 8, padding: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <strong>{p.name}</strong>
                    <div style={{ color: '#666', fontSize: 12 }}>ID: {p.id}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                    <button onClick={() => { setAnnotatorProjectId(p.id); setView('annotator') }} style={{ padding: '6px 10px' }}>打开标注工作台</button>
                    <button
                      onClick={() => onExport(p)}
                      disabled={exporting === p.id}
                      style={{ padding: '6px 10px', backgroundColor: '#52c41a', color: 'white', border: '1px solid #52c41a', borderRadius: 6, cursor: exporting === p.id ? 'not-allowed' : 'pointer' }}
                    >
                      {exporting === p.id ? '导出中...' : '导出数据集'}
                    </button>
                    <button
                      onClick={() => setImportDialogProject(p)}
                      disabled={importing === p.id}
                      style={{ padding: '6px 10px', backgroundColor: '#1890ff', color: 'white', border: '1px solid #1890ff', borderRadius: 6, cursor: importing === p.id ? 'not-allowed' : 'pointer' }}
                    >
                      {importing === p.id ? '导入中...' : '导入数据'}
                    </button>
                    <button onClick={() => onDeleteProject(p.id)} style={{ padding: '6px 10px', color: '#ff4d4f', borderColor: '#ff4d4f' }}>
                      {deleting === p.id ? '删除中...' : '删除项目'}
                    </button>
                  </div>
                </div>
                {/* 导入进度条 */}
                {importProgress && importProgress.projectId === p.id && (
                  <div style={{ marginTop: 8, padding: 12, backgroundColor: '#e6f7ff', borderRadius: 4, border: '1px solid #91d5ff' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <span style={{ fontWeight: 500, color: '#1890ff' }}>
                        {importProgress.status === 'succeeded' ? '导入完成' :
                         importProgress.status === 'failed' ? '导入失败' : '正在导入...'}
                      </span>
                      <span style={{ color: '#666', fontSize: 13 }}>
                        {importProgress.current} / {importProgress.total} ({importProgress.progress}%)
                      </span>
                    </div>
                    {/* 进度条 */}
                    <div style={{ width: '100%', height: 8, backgroundColor: '#d9d9d9', borderRadius: 4, overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${importProgress.progress}%`,
                          height: '100%',
                          backgroundColor: importProgress.status === 'failed' ? '#ff4d4f' :
                                          importProgress.status === 'succeeded' ? '#52c41a' : '#1890ff',
                          transition: 'width 0.3s ease',
                          borderRadius: 4,
                        }}
                      />
                    </div>
                    <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>
                      {importProgress.message}
                    </div>
                    {importProgress.status === 'succeeded' && (
                      <div style={{ marginTop: 6, fontSize: 12, color: '#52c41a' }}>
                        新增 {importProgress.imported}，重复 {importProgress.duplicates}，错误 {importProgress.errors}
                        {importProgress.annotations_imported !== undefined && (
                          <span>，标注导入 {importProgress.annotations_imported}</span>
                        )}
                      </div>
                    )}
                  </div>
                )}
                {importResult[p.id] && !importProgress && (
                  <div style={{ marginTop: 8, fontSize: 14, backgroundColor: '#f6ffed', padding: 8, borderRadius: 4 }}>
                    导入结果：总数 {importResult[p.id].total}，新增 {importResult[p.id].imported}，
                    重复 {importResult[p.id].duplicates}，错误 {importResult[p.id].errors}
                    {importResult[p.id].annotations_imported !== undefined && (
                      <span>，标注导入 {importResult[p.id].annotations_imported}，标注跳过 {importResult[p.id].annotations_skipped}</span>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* 导出对话框 */}
      {exportDialogProject && (
        <ExportDialog
          projectId={exportDialogProject.id}
          projectName={exportDialogProject.name}
          onClose={() => setExportDialogProject(null)}
          onExportStart={() => setExporting(exportDialogProject.id)}
          onExportEnd={(success, message) => {
            setExporting(null)
            alert(message)
          }}
        />
      )}

      {/* 导入对话框 */}
      {importDialogProject && (
        <ImportDialog
          projectId={importDialogProject.id}
          projectName={importDialogProject.name}
          onClose={() => setImportDialogProject(null)}
          onImport={(files, format) => onImport(importDialogProject.id, files, format)}
        />
      )}
    </div>
  )
}

// 导入对话框组件
const ImportDialog: React.FC<{
  projectId: number
  projectName: string
  onClose: () => void
  onImport: (files: File[], format: 'images' | 'yolo' | 'single' | 'folder') => void
}> = ({ projectId, projectName, onClose, onImport }) => {
  const [format, setFormat] = useState<'images' | 'yolo' | 'single' | 'folder'>('folder')
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const folderInputRef = React.useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onImport(Array.from(files), format)
    }
  }

  const handleButtonClick = () => {
    if (format === 'folder') {
      folderInputRef.current?.click()
    } else {
      fileInputRef.current?.click()
    }
  }

  const getAcceptTypes = () => {
    if (format === 'single' || format === 'folder') {
      return '.jpg,.jpeg,.png'
    }
    return '.zip'
  }

  const getButtonText = () => {
    switch (format) {
      case 'single': return '选择图片'
      case 'folder': return '选择文件夹'
      default: return '选择ZIP文件'
    }
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: 8,
        padding: 24,
        width: 450,
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
      }}>
        <h3 style={{ marginTop: 0, marginBottom: 16 }}>导入数据 - {projectName}</h3>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>选择导入格式</div>

          <label style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            padding: 12,
            border: format === 'folder' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            borderRadius: 6,
            cursor: 'pointer',
            marginBottom: 8,
            backgroundColor: format === 'folder' ? '#e6f7ff' : 'white'
          }}>
            <input
              type="radio"
              name="format"
              checked={format === 'folder'}
              onChange={() => setFormat('folder')}
              style={{ marginTop: 2 }}
            />
            <div>
              <div style={{ fontWeight: 500 }}>选择文件夹</div>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                选择包含 JPG/PNG 图片的文件夹，批量导入所有图片。
              </div>
            </div>
          </label>

          <label style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            padding: 12,
            border: format === 'single' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            borderRadius: 6,
            cursor: 'pointer',
            marginBottom: 8,
            backgroundColor: format === 'single' ? '#e6f7ff' : 'white'
          }}>
            <input
              type="radio"
              name="format"
              checked={format === 'single'}
              onChange={() => setFormat('single')}
              style={{ marginTop: 2 }}
            />
            <div>
              <div style={{ fontWeight: 500 }}>单张图片</div>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                上传单张 JPG/PNG 图片，导入后需要手动标注。
              </div>
            </div>
          </label>

          <label style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            padding: 12,
            border: format === 'yolo' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            borderRadius: 6,
            cursor: 'pointer',
            marginBottom: 8,
            backgroundColor: format === 'yolo' ? '#e6f7ff' : 'white'
          }}>
            <input
              type="radio"
              name="format"
              checked={format === 'yolo'}
              onChange={() => setFormat('yolo')}
              style={{ marginTop: 2 }}
            />
            <div>
              <div style={{ fontWeight: 500 }}>YOLO格式数据集</div>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                包含 train/valid/test 文件夹、images 和 labels 子目录、data.yaml 类别定义。
                将自动导入图片和标注，类别按名称匹配。
              </div>
            </div>
          </label>

          <label style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            padding: 12,
            border: format === 'images' ? '2px solid #1890ff' : '1px solid #d9d9d9',
            borderRadius: 6,
            cursor: 'pointer',
            backgroundColor: format === 'images' ? '#e6f7ff' : 'white'
          }}>
            <input
              type="radio"
              name="format"
              checked={format === 'images'}
              onChange={() => setFormat('images')}
              style={{ marginTop: 2 }}
            />
            <div>
              <div style={{ fontWeight: 500 }}>纯图片ZIP</div>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                仅包含图片文件的ZIP压缩包，不包含标注。
                导入后需要手动标注。
              </div>
            </div>
          </label>
        </div>

        {/* 普通文件选择 */}
        <input
          ref={fileInputRef}
          type="file"
          accept={getAcceptTypes()}
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />

        {/* 文件夹选择 */}
        <input
          ref={folderInputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          multiple
          {...{ webkitdirectory: '', directory: '' } as any}
          style={{ display: 'none' }}
          onChange={handleFileSelect}
        />

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              backgroundColor: 'white',
              cursor: 'pointer'
            }}
          >
            取消
          </button>
          <button
            onClick={handleButtonClick}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: 4,
              backgroundColor: '#1890ff',
              color: 'white',
              cursor: 'pointer'
            }}
          >
            {getButtonText()}
          </button>
        </div>
      </div>
    </div>
  )
}
