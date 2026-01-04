import React, { useState } from 'react'

interface AugmentationConfig {
  enabled: boolean
  brightness: { enabled: boolean; min: number; max: number }
  contrast: { enabled: boolean; min: number; max: number }
  saturation: { enabled: boolean; min: number; max: number }
  noise: { enabled: boolean; type: 'gaussian' | 'salt_pepper'; intensity: number }
  blur: { enabled: boolean; radius: number }
  resize: { enabled: boolean; width: number | null; height: number | null; keepAspect: boolean }
  count: number
}

interface ExportDialogProps {
  projectId: number
  projectName: string
  onClose: () => void
  onExportStart: () => void
  onExportEnd: (success: boolean, message: string) => void
}

const defaultAugConfig: AugmentationConfig = {
  enabled: false,
  brightness: { enabled: false, min: 0.7, max: 1.3 },
  contrast: { enabled: false, min: 0.7, max: 1.3 },
  saturation: { enabled: false, min: 0.7, max: 1.3 },
  noise: { enabled: false, type: 'gaussian', intensity: 0.02 },
  blur: { enabled: false, radius: 1.0 },
  resize: { enabled: false, width: null, height: null, keepAspect: true },
  count: 1
}

export const ExportDialog: React.FC<ExportDialogProps> = ({
  projectId,
  projectName,
  onClose,
  onExportStart,
  onExportEnd
}) => {
  const [augConfig, setAugConfig] = useState<AugmentationConfig>(defaultAugConfig)
  const [exporting, setExporting] = useState(false)
  const [localExporting, setLocalExporting] = useState(false)
  const [localResult, setLocalResult] = useState<{
    success: boolean
    path: string
    filename: string
    size_human: string
    total_images: number
  } | null>(null)
  const [seed, setSeed] = useState(42)
  const [format, setFormat] = useState<'coco' | 'yolo'>('yolo')

  const updateAugConfig = (path: string, value: any) => {
    setAugConfig(prev => {
      const newConfig = { ...prev }
      const keys = path.split('.')
      let current: any = newConfig
      for (let i = 0; i < keys.length - 1; i++) {
        current[keys[i]] = { ...current[keys[i]] }
        current = current[keys[i]]
      }
      current[keys[keys.length - 1]] = value
      return newConfig
    })
  }

  const handleLocalExport = async () => {
    setLocalExporting(true)
    setLocalResult(null)

    try {
      // 如果启用了数据增强或调整图片尺寸，发送 augmentation 配置
      const shouldSendAugmentation = augConfig.enabled || augConfig.resize.enabled
      const body = {
        format,
        seed,
        train_ratio: 0.8,
        val_ratio: 0.1,
        test_ratio: 0.1,
        augmentation: shouldSendAugmentation ? augConfig : null
      }

      const res = await fetch(`/api/projects/${projectId}/export/local`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(errorText || '导出失败')
      }

      const result = await res.json()
      setLocalResult(result)
    } catch (e) {
      console.error(e)
      onExportEnd(false, '生成本地压缩包失败：' + (e as Error).message)
    } finally {
      setLocalExporting(false)
    }
  }

  const handleExport = async () => {
    setExporting(true)
    onExportStart()

    try {
      // 如果启用了数据增强或调整图片尺寸，发送 augmentation 配置
      const shouldSendAugmentation = augConfig.enabled || augConfig.resize.enabled
      const body = {
        format,
        seed,
        train_ratio: 0.8,
        val_ratio: 0.1,
        test_ratio: 0.1,
        augmentation: shouldSendAugmentation ? augConfig : null
      }

      const res = await fetch(`/api/projects/${projectId}/export/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (!res.ok) {
        const errorText = await res.text()
        throw new Error(errorText || '导出失败')
      }

      // 获取文件名
      const contentDisposition = res.headers.get('content-disposition')
      let filename = `${projectName}_${format}_export.zip`
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '').trim()
        }
      }

      // 下载文件
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      onExportEnd(true, '导出成功！')
      onClose()
    } catch (e) {
      console.error(e)
      onExportEnd(false, '导出失败：' + (e as Error).message)
    } finally {
      setExporting(false)
    }
  }

  const inputStyle: React.CSSProperties = {
    padding: '6px 8px',
    border: '1px solid #d9d9d9',
    borderRadius: 4,
    fontSize: 14
  }

  const checkboxLabelStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    cursor: 'pointer',
    marginBottom: 8
  }

  const sectionStyle: React.CSSProperties = {
    marginBottom: 16,
    padding: 12,
    backgroundColor: '#fafafa',
    borderRadius: 6
  }

  const rangeGroupStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginLeft: 24,
    marginTop: 8
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
        width: 500,
        maxHeight: '80vh',
        overflow: 'auto',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
      }}>
        <h3 style={{ marginTop: 0, marginBottom: 16 }}>导出数据集 - {projectName}</h3>

        {/* 格式选择 */}
        <div style={sectionStyle}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 14 }}>导出格式</h4>
          <div style={{ display: 'flex', gap: 16 }}>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              border: format === 'yolo' ? '2px solid #1890ff' : '1px solid #d9d9d9',
              borderRadius: 6,
              cursor: 'pointer',
              backgroundColor: format === 'yolo' ? '#e6f7ff' : 'white'
            }}>
              <input
                type="radio"
                name="format"
                checked={format === 'yolo'}
                onChange={() => setFormat('yolo')}
              />
              <div>
                <div style={{ fontWeight: 500 }}>YOLO</div>
                <div style={{ fontSize: 11, color: '#666' }}>适合 Ultralytics 训练</div>
              </div>
            </label>
            <label style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              border: format === 'coco' ? '2px solid #1890ff' : '1px solid #d9d9d9',
              borderRadius: 6,
              cursor: 'pointer',
              backgroundColor: format === 'coco' ? '#e6f7ff' : 'white'
            }}>
              <input
                type="radio"
                name="format"
                checked={format === 'coco'}
                onChange={() => setFormat('coco')}
              />
              <div>
                <div style={{ fontWeight: 500 }}>COCO</div>
                <div style={{ fontSize: 11, color: '#666' }}>JSON 标注格式</div>
              </div>
            </label>
          </div>
        </div>

        {/* 基础设置 */}
        <div style={sectionStyle}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: 14 }}>基础设置</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <label>随机种子：</label>
            <input
              type="number"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              style={{ ...inputStyle, width: 100 }}
            />
          </div>
        </div>

        {/* 数据增强开关 */}
        <div style={sectionStyle}>
          <label style={checkboxLabelStyle}>
            <input
              type="checkbox"
              checked={augConfig.enabled}
              onChange={e => updateAugConfig('enabled', e.target.checked)}
            />
            <strong>启用数据增强</strong>
            <span style={{ color: '#888', fontSize: 12 }}>（仅应用于训练集）</span>
          </label>

          {augConfig.enabled && (
            <div style={{ marginTop: 12 }}>
              {/* 增强倍数 */}
              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  每张图增强数量：
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={augConfig.count}
                    onChange={e => updateAugConfig('count', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                  <span style={{ color: '#888', fontSize: 12 }}>（1-10张）</span>
                </label>
              </div>

              {/* 亮度 */}
              <label style={checkboxLabelStyle}>
                <input
                  type="checkbox"
                  checked={augConfig.brightness.enabled}
                  onChange={e => updateAugConfig('brightness.enabled', e.target.checked)}
                />
                亮度变化
              </label>
              {augConfig.brightness.enabled && (
                <div style={rangeGroupStyle}>
                  <span>范围：</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.brightness.min}
                    onChange={e => updateAugConfig('brightness.min', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                  <span>~</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.brightness.max}
                    onChange={e => updateAugConfig('brightness.max', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                  <span style={{ color: '#888', fontSize: 12 }}>（1.0为原始亮度）</span>
                </div>
              )}

              {/* 对比度 */}
              <label style={checkboxLabelStyle}>
                <input
                  type="checkbox"
                  checked={augConfig.contrast.enabled}
                  onChange={e => updateAugConfig('contrast.enabled', e.target.checked)}
                />
                对比度变化
              </label>
              {augConfig.contrast.enabled && (
                <div style={rangeGroupStyle}>
                  <span>范围：</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.contrast.min}
                    onChange={e => updateAugConfig('contrast.min', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                  <span>~</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.contrast.max}
                    onChange={e => updateAugConfig('contrast.max', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                </div>
              )}

              {/* 饱和度 */}
              <label style={checkboxLabelStyle}>
                <input
                  type="checkbox"
                  checked={augConfig.saturation.enabled}
                  onChange={e => updateAugConfig('saturation.enabled', e.target.checked)}
                />
                饱和度变化
              </label>
              {augConfig.saturation.enabled && (
                <div style={rangeGroupStyle}>
                  <span>范围：</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.saturation.min}
                    onChange={e => updateAugConfig('saturation.min', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                  <span>~</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="2"
                    value={augConfig.saturation.max}
                    onChange={e => updateAugConfig('saturation.max', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                </div>
              )}

              {/* 噪音 */}
              <label style={checkboxLabelStyle}>
                <input
                  type="checkbox"
                  checked={augConfig.noise.enabled}
                  onChange={e => updateAugConfig('noise.enabled', e.target.checked)}
                />
                添加噪音
              </label>
              {augConfig.noise.enabled && (
                <div style={{ ...rangeGroupStyle, flexWrap: 'wrap' }}>
                  <span>类型：</span>
                  <select
                    value={augConfig.noise.type}
                    onChange={e => updateAugConfig('noise.type', e.target.value)}
                    style={inputStyle}
                  >
                    <option value="gaussian">高斯噪音</option>
                    <option value="salt_pepper">椒盐噪音</option>
                  </select>
                  <span style={{ marginLeft: 8 }}>强度：</span>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="0.5"
                    value={augConfig.noise.intensity}
                    onChange={e => updateAugConfig('noise.intensity', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                </div>
              )}

              {/* 模糊 */}
              <label style={checkboxLabelStyle}>
                <input
                  type="checkbox"
                  checked={augConfig.blur.enabled}
                  onChange={e => updateAugConfig('blur.enabled', e.target.checked)}
                />
                高斯模糊
              </label>
              {augConfig.blur.enabled && (
                <div style={rangeGroupStyle}>
                  <span>模糊半径：</span>
                  <input
                    type="number"
                    step="0.5"
                    min="0.5"
                    max="5"
                    value={augConfig.blur.radius}
                    onChange={e => updateAugConfig('blur.radius', Number(e.target.value))}
                    style={{ ...inputStyle, width: 60 }}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* 图片Resize */}
        <div style={sectionStyle}>
          <label style={checkboxLabelStyle}>
            <input
              type="checkbox"
              checked={augConfig.resize.enabled}
              onChange={e => updateAugConfig('resize.enabled', e.target.checked)}
            />
            <strong>调整图片尺寸</strong>
            <span style={{ color: '#888', fontSize: 12 }}>（应用于所有图片）</span>
          </label>

          {augConfig.resize.enabled && (
            <div style={{ marginLeft: 24, marginTop: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span>宽度：</span>
                <input
                  type="number"
                  min="32"
                  max="4096"
                  placeholder="留空保持原比例"
                  value={augConfig.resize.width ?? ''}
                  onChange={e => updateAugConfig('resize.width', e.target.value ? Number(e.target.value) : null)}
                  style={{ ...inputStyle, width: 120 }}
                />
                <span>px</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span>高度：</span>
                <input
                  type="number"
                  min="32"
                  max="4096"
                  placeholder="留空保持原比例"
                  value={augConfig.resize.height ?? ''}
                  onChange={e => updateAugConfig('resize.height', e.target.value ? Number(e.target.value) : null)}
                  style={{ ...inputStyle, width: 120 }}
                />
                <span>px</span>
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={augConfig.resize.keepAspect}
                  onChange={e => updateAugConfig('resize.keepAspect', e.target.checked)}
                />
                保持宽高比
              </label>
            </div>
          )}
        </div>

        {/* 本地导出结果显示 */}
        {localResult && (
          <div style={{
            padding: 12,
            backgroundColor: '#f6ffed',
            border: '1px solid #b7eb8f',
            borderRadius: 6,
            marginBottom: 16
          }}>
            <div style={{ color: '#52c41a', fontWeight: 500, marginBottom: 8 }}>✓ 压缩包已生成</div>
            <div style={{ fontSize: 13, color: '#333' }}>
              <div style={{ marginBottom: 4 }}><strong>文件名：</strong>{localResult.filename}</div>
              <div style={{ marginBottom: 4 }}><strong>大小：</strong>{localResult.size_human}</div>
              <div style={{ marginBottom: 4 }}><strong>图片数：</strong>{localResult.total_images}</div>
              <div style={{ wordBreak: 'break-all' }}><strong>路径：</strong><code style={{ backgroundColor: '#f5f5f5', padding: '2px 6px', borderRadius: 3 }}>{localResult.path}</code></div>
            </div>
          </div>
        )}

        {/* 按钮 */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button
            onClick={onClose}
            disabled={exporting || localExporting}
            style={{
              padding: '8px 16px',
              border: '1px solid #d9d9d9',
              borderRadius: 4,
              backgroundColor: 'white',
              cursor: (exporting || localExporting) ? 'not-allowed' : 'pointer'
            }}
          >
            取消
          </button>
          <button
            onClick={handleLocalExport}
            disabled={exporting || localExporting}
            style={{
              padding: '8px 16px',
              border: '1px solid #1890ff',
              borderRadius: 4,
              backgroundColor: 'white',
              color: '#1890ff',
              cursor: (exporting || localExporting) ? 'not-allowed' : 'pointer'
            }}
          >
            {localExporting ? '生成中...' : '生成服务器压缩包'}
          </button>
          <button
            onClick={handleExport}
            disabled={exporting || localExporting}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: 4,
              backgroundColor: '#52c41a',
              color: 'white',
              cursor: (exporting || localExporting) ? 'not-allowed' : 'pointer'
            }}
          >
            {exporting ? '下载中...' : '下载到本地'}
          </button>
        </div>
      </div>
    </div>
  )
}
