import React, { useState, useRef } from 'react'
import {
  Card, Tabs, Radio, Button, Progress, Alert, Space, Typography, Divider,
  Select, InputNumber, Switch, Slider, message, Row, Col, Tag,
} from 'antd'
import {
  ImportOutlined, ExportOutlined, VideoCameraOutlined,
  FileZipOutlined, PictureOutlined, FolderOpenOutlined,
} from '@ant-design/icons'
import { Project } from '../../pages/App'

const { Text, Title } = Typography
const { TabPane } = Tabs

type ImportFormat = 'folder' | 'single' | 'yolo' | 'coco' | 'images' | 'video'

interface ImportProgress {
  status: 'running' | 'succeeded' | 'failed'
  total: number
  current: number
  progress: number
  message: string
  imported?: number
  duplicates?: number
  errors?: number
  annotations_imported?: number
}

interface AugConfig {
  enabled: boolean
  count: number
  brightness: { enabled: boolean; min: number; max: number }
  contrast: { enabled: boolean; min: number; max: number }
  saturation: { enabled: boolean; min: number; max: number }
  noise: { enabled: boolean; type: 'gaussian' | 'salt_pepper'; intensity: number }
  blur: { enabled: boolean; radius: number }
  resize: { enabled: boolean; width: number | null; height: number | null; keepAspect: boolean }
}

const defaultAugConfig: AugConfig = {
  enabled: false,
  count: 1,
  brightness: { enabled: false, min: 0.7, max: 1.3 },
  contrast: { enabled: false, min: 0.7, max: 1.3 },
  saturation: { enabled: false, min: 0.7, max: 1.3 },
  noise: { enabled: false, type: 'gaussian', intensity: 0.02 },
  blur: { enabled: false, radius: 1.0 },
  resize: { enabled: false, width: null, height: null, keepAspect: true },
}

const FORMAT_OPTIONS = [
  {
    key: 'folder' as ImportFormat,
    label: '选择文件夹',
    icon: <FolderOpenOutlined />,
    desc: '选择包含 JPG/PNG 图片的本地文件夹，批量导入所有图片',
  },
  {
    key: 'single' as ImportFormat,
    label: '单张图片',
    icon: <PictureOutlined />,
    desc: '上传单张 JPG/PNG 图片',
  },
  {
    key: 'yolo' as ImportFormat,
    label: 'YOLO 格式数据集',
    icon: <FileZipOutlined />,
    desc: '包含 train/valid/test 目录、images+labels 子目录和 data.yaml 类别定义的 ZIP 包',
  },
  {
    key: 'coco' as ImportFormat,
    label: 'COCO 格式数据集',
    icon: <FileZipOutlined />,
    desc: '包含 train/valid/test 目录和 _annotations.coco.json 文件的 ZIP 包',
  },
  {
    key: 'images' as ImportFormat,
    label: '纯图片 ZIP',
    icon: <FileZipOutlined />,
    desc: '仅包含图片文件的 ZIP 压缩包，不含标注，导入后需手动标注',
  },
  {
    key: 'video' as ImportFormat,
    label: '视频文件（自动抽帧）',
    icon: <VideoCameraOutlined />,
    desc: '支持 mp4/avi/mov/mkv/wmv/flv，按 1fps 自动抽取帧图片',
  },
]

export const DataManagement: React.FC<{ project: Project }> = ({ project }) => {
  // ---- 导入状态 ----
  const [importFormat, setImportFormat] = useState<ImportFormat>('folder')
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)

  // ---- 导出状态 ----
  const [exportFormat, setExportFormat] = useState<'yolo' | 'coco'>('yolo')
  const [seed, setSeed] = useState(42)
  const [augConfig, setAugConfig] = useState<AugConfig>(defaultAugConfig)
  const [exporting, setExporting] = useState(false)
  const [localExporting, setLocalExporting] = useState(false)
  const [localResult, setLocalResult] = useState<any>(null)

  // ---- 缩略图 ----
  const [generatingThumbs, setGeneratingThumbs] = useState(false)

  // ---- 工具函数 ----
  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  const handleImport = async (files: File[]) => {
    if (!files.length) return
    setImporting(true)
    setImportProgress(null)

    try {
      if (importFormat === 'single') {
        const file = files[0]
        const base64 = await fileToBase64(file)
        const res = await fetch(`/api/projects/${project.id}/import/image-base64`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, data: base64 }),
        })
        const data = await res.json()
        if (data.success) message.success('图片导入成功')
        else if (data.duplicate) message.warning('图片已存在（重复）')
        else message.error('导入失败：' + data.message)
        return
      }

      if (importFormat === 'folder') {
        const imageFiles = files.filter(f => /\.(jpg|jpeg|png)$/i.test(f.name))
        if (!imageFiles.length) { message.error('所选文件夹中没有 JPG/PNG 图片'); return }

        let imported = 0, duplicates = 0, errors = 0
        setImportProgress({ status: 'running', total: imageFiles.length, current: 0, progress: 0, message: '准备导入...' })

        for (let i = 0; i < imageFiles.length; i++) {
          try {
            const base64 = await fileToBase64(imageFiles[i])
            const res = await fetch(`/api/projects/${project.id}/import/image-base64`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ filename: imageFiles[i].name, data: base64 }),
            })
            const data = await res.json()
            if (data.success) imported++
            else if (data.duplicate) duplicates++
            else errors++
          } catch { errors++ }

          setImportProgress({
            status: 'running',
            total: imageFiles.length,
            current: i + 1,
            progress: Math.round((i + 1) / imageFiles.length * 100),
            message: `正在导入 ${i + 1}/${imageFiles.length}...`,
            imported, duplicates, errors,
          })
        }
        setImportProgress({ status: 'succeeded', total: imageFiles.length, current: imageFiles.length, progress: 100, message: '导入完成', imported, duplicates, errors })
        message.success(`导入完成：${imported} 张新增，${duplicates} 张重复，${errors} 张错误`)
        return
      }

      if (importFormat === 'video') {
        const file = files[0]
        const fd = new FormData()
        fd.append('file', file)
        const res = await fetch(`/api/projects/${project.id}/import/video`, { method: 'POST', body: fd })
        if (!res.ok) throw new Error('视频上传失败')
        const data = await res.json()
        const jobId = data.jobId
        setImportProgress({ status: 'running', total: 0, current: 0, progress: 0, message: '视频上传成功，开始抽帧...' })
        const poll = async () => {
          const r = await fetch(`/api/projects/${project.id}/import/jobs/${jobId}`)
          const d = await r.json()
          setImportProgress({ status: d.status === 'succeeded' ? 'succeeded' : d.status === 'failed' ? 'failed' : 'running', total: d.total, current: d.current, progress: d.total > 0 ? Math.round(d.current / d.total * 100) : 0, message: d.message, imported: d.imported, duplicates: d.duplicates, errors: d.errors })
          if (d.status !== 'succeeded' && d.status !== 'failed') setTimeout(poll, 1000)
          else if (d.status === 'failed') message.error('抽帧失败：' + d.message)
          else message.success('视频抽帧完成')
        }
        poll()
        return
      }

      // YOLO / COCO / images ZIP — 使用 XHR 发送原始二进制流（绕过 multipart，支持上传进度显示）
      const file = files[0]
      let endpoint = ''
      if (importFormat === 'yolo') endpoint = `/api/projects/${project.id}/import/yolo?import_annotations=true&filename=${encodeURIComponent(file.name)}`
      else if (importFormat === 'coco') endpoint = `/api/projects/${project.id}/import/coco?filename=${encodeURIComponent(file.name)}`
      else endpoint = `/api/projects/${project.id}/import?filename=${encodeURIComponent(file.name)}`

      setImportProgress({ status: 'running', total: 0, current: 0, progress: 0, message: '准备上传...' })

      // 使用 XHR 以支持 upload.onprogress（fetch 不支持上传进度）
      const jobId: string = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', endpoint)
        xhr.setRequestHeader('Content-Type', 'application/octet-stream')

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            const uploadedMB = (e.loaded / 1024 / 1024).toFixed(0)
            const totalMB = (e.total / 1024 / 1024).toFixed(0)
            const pct = Math.round(e.loaded / e.total * 100)
            setImportProgress({
              status: 'running',
              total: e.total,
              current: e.loaded,
              progress: pct,
              message: `正在上传... ${uploadedMB} MB / ${totalMB} MB`,
            })
          }
        }

        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText)
              resolve(data.job_id)
            } catch { reject(new Error('响应解析失败')) }
          } else {
            try {
              const err = JSON.parse(xhr.responseText)
              reject(new Error(err.detail || `导入失败 (${xhr.status})`))
            } catch { reject(new Error(`导入失败 (${xhr.status})`)) }
          }
        }

        xhr.onerror = () => reject(new Error('网络错误，请检查连接'))
        xhr.send(file)
      })

      setImportProgress({ status: 'running', total: 0, current: 0, progress: 0, message: '文件上传完成，正在导入...' })
      const poll = async () => {
        const r = await fetch(`/api/projects/${project.id}/import/jobs/${jobId}`)
        const d = await r.json()
        const progress = d.total > 0 ? Math.round(d.current / d.total * 100) : (d.progress ?? 0)
        setImportProgress({
          status: d.status === 'succeeded' ? 'succeeded' : d.status === 'failed' ? 'failed' : 'running',
          total: d.total,
          current: d.current,
          progress,
          message: d.message,
          imported: d.imported,
          duplicates: d.duplicates,
          errors: d.errors,
          annotations_imported: d.annotations_imported,
        })
        if (d.status !== 'succeeded' && d.status !== 'failed') setTimeout(poll, 1000)
        else if (d.status === 'failed') message.error('导入失败：' + d.message)
        else if (importFormat === 'images') message.success(`导入完成：${d.imported} 张新增，${d.duplicates} 张重复`)
        else message.success(`导入完成：${d.imported} 张新增，${d.annotations_imported ?? 0} 条标注`)
      }
      poll()
    } catch (e: any) {
      message.error(e.message || '导入失败')
    } finally {
      setImporting(false)
    }
  }

  const handleSelectFile = () => {
    if (importFormat === 'folder') folderInputRef.current?.click()
    else if (importFormat === 'video') videoInputRef.current?.click()
    else fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length) handleImport(files)
    e.target.value = ''
  }

  const handleGenerateThumbnails = async () => {
    setGeneratingThumbs(true)
    try {
      const res = await fetch(`/api/projects/${project.id}/generate-thumbnails`, { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') message.info(`开始生成缩略图：${data.total} 张，请稍等...`)
      else message.success(data.message)
    } catch { message.error('生成缩略图失败') }
    finally { setGeneratingThumbs(false) }
  }

  // ---- 导出 ----
  const updateAug = (path: string, val: any) => {
    setAugConfig(prev => {
      const cfg = { ...prev }
      const keys = path.split('.')
      let cur: any = cfg
      for (let i = 0; i < keys.length - 1; i++) { cur[keys[i]] = { ...cur[keys[i]] }; cur = cur[keys[i]] }
      cur[keys[keys.length - 1]] = val
      return cfg
    })
  }

  const buildExportBody = () => ({
    format: exportFormat,
    seed,
    train_ratio: 0.8,
    val_ratio: 0.1,
    test_ratio: 0.1,
    augmentation: (augConfig.enabled || augConfig.resize.enabled) ? augConfig : null,
  })

  const handleDownload = async () => {
    setExporting(true)
    try {
      const res = await fetch(`/api/projects/${project.id}/export/download`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildExportBody()),
      })
      if (!res.ok) throw new Error(await res.text())
      const cd = res.headers.get('content-disposition')
      let filename = `${project.name}_${exportFormat}.zip`
      if (cd) { const m = cd.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/) ; if (m) filename = m[1].replace(/['"]/g, '').trim() }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = filename
      document.body.appendChild(a); a.click(); URL.revokeObjectURL(url); document.body.removeChild(a)
      message.success('数据集下载成功')
    } catch (e: any) { message.error('下载失败：' + e.message) }
    finally { setExporting(false) }
  }

  const handleLocalExport = async () => {
    setLocalExporting(true)
    setLocalResult(null)
    try {
      const res = await fetch(`/api/projects/${project.id}/export/local`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildExportBody()),
      })
      if (!res.ok) throw new Error(await res.text())
      setLocalResult(await res.json())
      message.success('服务器压缩包生成成功')
    } catch (e: any) { message.error('生成失败：' + e.message) }
    finally { setLocalExporting(false) }
  }

  return (
    <Card title="数据管理" style={{ minHeight: 500 }}>
      <Tabs defaultActiveKey="import">
        {/* ======= 导入 ======= */}
        <TabPane tab={<span><ImportOutlined /> 导入数据</span>} key="import">
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {/* 格式选择 */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 12 }}>选择导入格式</Text>
              <Radio.Group
                value={importFormat}
                onChange={e => setImportFormat(e.target.value)}
                style={{ display: 'flex', flexDirection: 'column', gap: 8 }}
              >
                {FORMAT_OPTIONS.map(opt => (
                  <Radio.Button
                    key={opt.key}
                    value={opt.key}
                    style={{
                      height: 'auto',
                      padding: '10px 16px',
                      textAlign: 'left',
                      borderRadius: 8,
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 10,
                    }}
                  >
                    <Space align="start">
                      <span style={{ fontSize: 16, lineHeight: '22px' }}>{opt.icon}</span>
                      <div>
                        <div style={{ fontWeight: 500, lineHeight: '22px' }}>{opt.label}</div>
                        <div style={{ fontSize: 12, color: '#8c8c8c', fontWeight: 400, marginTop: 2 }}>{opt.desc}</div>
                      </div>
                    </Space>
                  </Radio.Button>
                ))}
              </Radio.Group>
            </div>

            {/* 隐藏 file inputs */}
            <input ref={fileInputRef} type="file" accept={importFormat === 'single' ? '.jpg,.jpeg,.png' : '.zip'} style={{ display: 'none' }} onChange={handleFileChange} />
            <input ref={folderInputRef} type="file" accept=".jpg,.jpeg,.png" multiple {...{ webkitdirectory: '', directory: '' } as any} style={{ display: 'none' }} onChange={handleFileChange} />
            <input ref={videoInputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.wmv,.flv" style={{ display: 'none' }} onChange={handleFileChange} />

            <Button type="primary" icon={<ImportOutlined />} loading={importing} onClick={handleSelectFile} size="large">
              {importFormat === 'folder' ? '选择文件夹' : importFormat === 'video' ? '选择视频文件' : importFormat === 'single' ? '选择图片' : '选择 ZIP 文件'}
            </Button>

            {/* 进度条 */}
            {importProgress && (
              <Alert
                type={importProgress.status === 'succeeded' ? 'success' : importProgress.status === 'failed' ? 'error' : 'info'}
                message={importProgress.message}
                description={
                  <>
                    <Progress percent={importProgress.progress} status={importProgress.status === 'failed' ? 'exception' : importProgress.status === 'succeeded' ? 'success' : 'active'} />
                    {importProgress.status === 'succeeded' && (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        新增 {importProgress.imported}，重复 {importProgress.duplicates}，错误 {importProgress.errors}
                        {importProgress.annotations_imported ? `，标注导入 ${importProgress.annotations_imported}` : ''}
                      </Text>
                    )}
                  </>
                }
              />
            )}

            <Divider />

            {/* 生成缩略图 */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>生成缩略图</Text>
              <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 12 }}>
                为项目中尚未生成缩略图的图片批量生成缩略图，加速标注页面的图片列表加载
              </Text>
              <Button loading={generatingThumbs} onClick={handleGenerateThumbnails}>
                生成缩略图
              </Button>
            </div>
          </Space>
        </TabPane>

        {/* ======= 导出 ======= */}
        <TabPane tab={<span><ExportOutlined /> 导出数据集</span>} key="export">
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {/* 格式 */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>导出格式</Text>
              <Radio.Group value={exportFormat} onChange={e => setExportFormat(e.target.value)}>
                <Radio.Button value="yolo">
                  <Space direction="vertical" size={0} style={{ textAlign: 'left' }}>
                    <span style={{ fontWeight: 500 }}>YOLO</span>
                    <Text type="secondary" style={{ fontSize: 11 }}>适合 Ultralytics 训练</Text>
                  </Space>
                </Radio.Button>
                <Radio.Button value="coco">
                  <Space direction="vertical" size={0} style={{ textAlign: 'left' }}>
                    <span style={{ fontWeight: 500 }}>COCO</span>
                    <Text type="secondary" style={{ fontSize: 11 }}>JSON 标注格式</Text>
                  </Space>
                </Radio.Button>
              </Radio.Group>
            </div>

            {/* 随机种子 */}
            <div>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>随机种子</Text>
              <InputNumber value={seed} onChange={v => setSeed(v ?? 42)} min={0} />
            </div>

            {/* 数据增强 */}
            <Card size="small" title={
              <Space>
                <Switch checked={augConfig.enabled} onChange={v => updateAug('enabled', v)} size="small" />
                <span>数据增强</span>
                <Tag color="blue">仅训练集</Tag>
              </Space>
            }>
              {augConfig.enabled && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Space align="center">
                    <Text>每张增强数量：</Text>
                    <InputNumber min={1} max={10} value={augConfig.count} onChange={v => updateAug('count', v ?? 1)} />
                    <Text type="secondary">张</Text>
                  </Space>
                  <Space wrap>
                    {[
                      { key: 'brightness', label: '亮度' },
                      { key: 'contrast', label: '对比度' },
                      { key: 'saturation', label: '饱和度' },
                    ].map(({ key, label }) => (
                      <Space key={key}>
                        <Switch size="small" checked={(augConfig as any)[key].enabled} onChange={v => updateAug(`${key}.enabled`, v)} />
                        <Text>{label}</Text>
                        {(augConfig as any)[key].enabled && (
                          <Space size={4}>
                            <InputNumber size="small" step={0.1} min={0.1} max={2} value={(augConfig as any)[key].min} onChange={(v: any) => updateAug(`${key}.min`, v)} style={{ width: 60 }} />
                            <Text>~</Text>
                            <InputNumber size="small" step={0.1} min={0.1} max={2} value={(augConfig as any)[key].max} onChange={(v: any) => updateAug(`${key}.max`, v)} style={{ width: 60 }} />
                          </Space>
                        )}
                      </Space>
                    ))}
                  </Space>
                </Space>
              )}
            </Card>

            {/* 图片尺寸调整 */}
            <Card size="small" title={
              <Space>
                <Switch checked={augConfig.resize.enabled} onChange={v => updateAug('resize.enabled', v)} size="small" />
                <span>调整图片尺寸</span>
              </Space>
            }>
              {augConfig.resize.enabled && (
                <Space wrap>
                  <Space>
                    <Text>宽：</Text>
                    <InputNumber placeholder="原始" value={augConfig.resize.width ?? undefined} onChange={v => updateAug('resize.width', v ?? null)} min={32} max={4096} />
                    <Text>px</Text>
                  </Space>
                  <Space>
                    <Text>高：</Text>
                    <InputNumber placeholder="原始" value={augConfig.resize.height ?? undefined} onChange={v => updateAug('resize.height', v ?? null)} min={32} max={4096} />
                    <Text>px</Text>
                  </Space>
                  <Space>
                    <Switch size="small" checked={augConfig.resize.keepAspect} onChange={v => updateAug('resize.keepAspect', v)} />
                    <Text>保持宽高比</Text>
                  </Space>
                </Space>
              )}
            </Card>

            {/* 本地导出结果 */}
            {localResult && (
              <Alert
                type="success"
                message="服务器压缩包已生成"
                description={
                  <Space direction="vertical" size={2}>
                    <Text>文件名：{localResult.filename}</Text>
                    <Text>大小：{localResult.size_human}</Text>
                    <Text>图片数：{localResult.total_images}</Text>
                    <Text>路径：<Text code>{localResult.path}</Text></Text>
                  </Space>
                }
              />
            )}

            {/* 导出按钮 */}
            <Space>
              <Button loading={localExporting} onClick={handleLocalExport} icon={<ExportOutlined />}>
                生成服务器压缩包
              </Button>
              <Button type="primary" loading={exporting} onClick={handleDownload} icon={<ExportOutlined />}>
                下载到本地
              </Button>
            </Space>
          </Space>
        </TabPane>
      </Tabs>
    </Card>
  )
}
