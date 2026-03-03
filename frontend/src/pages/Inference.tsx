import React, { useEffect, useState, useRef, useCallback } from 'react'

type AvailableModel = {
  artifact_id: number
  job_id: number
  project_id: number
  project_name: string
  framework: string
  model_variant: string
  format: string
  path: string
  map50: number | null
  map50_95: number | null
  created_at: string
}

type Detection = {
  class_id: number
  class_name: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
}

type InferenceResult = {
  detections: Detection[]
  image_width: number
  image_height: number
  inference_time_ms: number
  model_info: {
    framework: string
    model_variant: string
    artifact_id: number
    job_id: number
  }
}

type VisualizeResult = InferenceResult & {
  image_data: string
}

// Color palette matching backend
const COLORS = [
  '#FF0000', // Red
  '#00FF00', // Green
  '#0000FF', // Blue
  '#FFFF00', // Yellow
  '#FF00FF', // Magenta
  '#00FFFF', // Cyan
  '#FF8000', // Orange
  '#8000FF', // Purple
  '#0080FF', // Light Blue
  '#FF0080', // Pink
  '#80FF00', // Lime
  '#00FF80', // Mint
]

export const Inference: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [models, setModels] = useState<AvailableModel[]>([])
  const [selectedModel, setSelectedModel] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [inferring, setInferring] = useState(false)

  // Image state
  const [uploadedImages, setUploadedImages] = useState<{ file: File; dataUrl: string }[]>([])
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [result, setResult] = useState<VisualizeResult | null>(null)

  // Parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [iouThreshold, setIouThreshold] = useState(0.45)
  const [imgsz, setImgsz] = useState(640)
  const [showOriginal, setShowOriginal] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  // Load available models
  useEffect(() => {
    setLoading(true)
    fetch('/api/inference/models')
      .then(r => r.json())
      .then(data => {
        setModels(data)
        if (data.length > 0) {
          setSelectedModel(data[0].artifact_id)
        }
      })
      .catch(() => setModels([]))
      .finally(() => setLoading(false))
  }, [])

  // Handle file selection
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const imageFiles = Array.from(files).filter(file => {
      const ext = file.name.toLowerCase().split('.').pop()
      return ['jpg', 'jpeg', 'png', 'webp'].includes(ext || '')
    })

    if (imageFiles.length === 0) {
      alert('No valid image files found')
      return
    }

    // Convert files to data URLs
    const promises = imageFiles.map(file => {
      return new Promise<{ file: File; dataUrl: string }>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve({ file, dataUrl: reader.result as string })
        reader.onerror = reject
        reader.readAsDataURL(file)
      })
    })

    Promise.all(promises).then(results => {
      setUploadedImages(results)
      setCurrentImageIndex(0)
      setResult(null)
    })

    // Reset input
    e.target.value = ''
  }, [])

  // Run inference
  const runInference = useCallback(async () => {
    if (!selectedModel || uploadedImages.length === 0) return

    const currentImage = uploadedImages[currentImageIndex]
    if (!currentImage) return

    setInferring(true)
    setResult(null)

    try {
      const response = await fetch('/api/inference/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          artifact_id: selectedModel,
          image_data: currentImage.dataUrl,
          confidence_threshold: confidenceThreshold,
          iou_threshold: iouThreshold,
          imgsz: imgsz,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Inference failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (e: any) {
      alert(e.message || 'Inference failed')
    } finally {
      setInferring(false)
    }
  }, [selectedModel, uploadedImages, currentImageIndex, confidenceThreshold, iouThreshold, imgsz])

  // Navigate images
  const goToPrevImage = useCallback(() => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1)
      setResult(null)
    }
  }, [currentImageIndex])

  const goToNextImage = useCallback(() => {
    if (currentImageIndex < uploadedImages.length - 1) {
      setCurrentImageIndex(currentImageIndex + 1)
      setResult(null)
    }
  }, [currentImageIndex, uploadedImages.length])

  // Clear all images
  const clearImages = useCallback(() => {
    setUploadedImages([])
    setCurrentImageIndex(0)
    setResult(null)
  }, [])

  // Get selected model info
  const selectedModelInfo = models.find(m => m.artifact_id === selectedModel)

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 16, maxWidth: 1600, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <button onClick={onBack} style={{ padding: '8px 16px' }}>
          ← Back
        </button>
        <h2 style={{ margin: 0 }}>Model Inference Verification</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: 24 }}>
        {/* Left Panel: Model Selection & Parameters */}
        <div>
          {/* Model Selection */}
          <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginBottom: 16 }}>
            <h3 style={{ marginTop: 0 }}>Select Model</h3>
            {loading ? (
              <p style={{ color: '#999' }}>Loading models...</p>
            ) : models.length === 0 ? (
              <p style={{ color: '#999' }}>No trained models available. Please train a model first.</p>
            ) : (
              <select
                value={selectedModel || ''}
                onChange={e => setSelectedModel(Number(e.target.value))}
                style={{
                  width: '100%',
                  padding: 10,
                  borderRadius: 4,
                  border: '1px solid #d9d9d9',
                  fontSize: 14,
                }}
              >
                {models.map(m => (
                  <option key={m.artifact_id} value={m.artifact_id}>
                    [{m.framework.toUpperCase()}] {m.project_name} - {m.model_variant}
                    {m.map50_95 ? ` (mAP: ${(m.map50_95 * 100).toFixed(1)}%)` : ''}
                  </option>
                ))}
              </select>
            )}

            {selectedModelInfo && (
              <div style={{ marginTop: 12, padding: 12, backgroundColor: '#f5f5f5', borderRadius: 6, fontSize: 13 }}>
                <div><strong>Framework:</strong> {selectedModelInfo.framework.toUpperCase()}</div>
                <div><strong>Model:</strong> {selectedModelInfo.model_variant}</div>
                <div><strong>Project:</strong> {selectedModelInfo.project_name}</div>
                {selectedModelInfo.map50 && (
                  <div><strong>mAP50:</strong> {(selectedModelInfo.map50 * 100).toFixed(2)}%</div>
                )}
                {selectedModelInfo.map50_95 && (
                  <div><strong>mAP50-95:</strong> {(selectedModelInfo.map50_95 * 100).toFixed(2)}%</div>
                )}
              </div>
            )}
          </section>

          {/* Parameters */}
          <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8, marginBottom: 16 }}>
            <h3 style={{ marginTop: 0 }}>Inference Parameters</h3>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
                Confidence Threshold: {confidenceThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={confidenceThreshold}
                onChange={e => setConfidenceThreshold(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#999' }}>
                <span>0.01</span>
                <span>0.99</span>
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
                IoU Threshold: {iouThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={iouThreshold}
                onChange={e => setIouThreshold(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#999' }}>
                <span>0.1</span>
                <span>0.9</span>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
                Image Size
              </label>
              <select
                value={imgsz}
                onChange={e => setImgsz(Number(e.target.value))}
                style={{
                  width: '100%',
                  padding: 8,
                  borderRadius: 4,
                  border: '1px solid #d9d9d9',
                }}
              >
                <option value={320}>320</option>
                <option value={416}>416</option>
                <option value={512}>512</option>
                <option value={640}>640</option>
                <option value={800}>800</option>
                <option value={1024}>1024</option>
                <option value={1280}>1280</option>
              </select>
            </div>
          </section>

          {/* Upload Section */}
          <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Upload Images</h3>

            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{
                  flex: 1,
                  padding: '10px 16px',
                  backgroundColor: '#1890ff',
                  color: 'white',
                  border: 'none',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Select Files
              </button>
              <button
                onClick={() => folderInputRef.current?.click()}
                style={{
                  flex: 1,
                  padding: '10px 16px',
                  backgroundColor: '#722ed1',
                  color: 'white',
                  border: 'none',
                  borderRadius: 6,
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                Select Folder
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              multiple
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />
            <input
              ref={folderInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              multiple
              {...{ webkitdirectory: '', directory: '' } as any}
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />

            {uploadedImages.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 14, color: '#666' }}>
                    {uploadedImages.length} image(s) loaded
                  </span>
                  <button
                    onClick={clearImages}
                    style={{
                      padding: '4px 12px',
                      fontSize: 12,
                      backgroundColor: '#ff4d4f',
                      color: 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: 'pointer',
                    }}
                  >
                    Clear All
                  </button>
                </div>

                {/* Image thumbnails */}
                <div style={{
                  display: 'flex',
                  gap: 8,
                  overflowX: 'auto',
                  padding: '8px 0',
                }}>
                  {uploadedImages.map((img, idx) => (
                    <div
                      key={idx}
                      onClick={() => {
                        setCurrentImageIndex(idx)
                        setResult(null)
                      }}
                      style={{
                        flexShrink: 0,
                        width: 60,
                        height: 60,
                        borderRadius: 4,
                        overflow: 'hidden',
                        border: idx === currentImageIndex ? '3px solid #1890ff' : '2px solid #eee',
                        cursor: 'pointer',
                      }}
                    >
                      <img
                        src={img.dataUrl}
                        alt={img.file.name}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>

        {/* Right Panel: Image Preview & Results */}
        <div>
          {/* Image Display */}
          <section style={{
            padding: 16,
            border: '1px solid #eee',
            borderRadius: 8,
            marginBottom: 16,
            minHeight: 500,
          }}>
            {uploadedImages.length === 0 ? (
              <div style={{
                height: 400,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#999',
                fontSize: 16,
                backgroundColor: '#fafafa',
                borderRadius: 8,
              }}>
                Upload images to start inference
              </div>
            ) : (
              <>
                {/* Navigation & Controls */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 12,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <button
                      onClick={goToPrevImage}
                      disabled={currentImageIndex === 0}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: currentImageIndex === 0 ? '#d9d9d9' : '#f0f0f0',
                        border: '1px solid #d9d9d9',
                        borderRadius: 4,
                        cursor: currentImageIndex === 0 ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Previous
                    </button>
                    <span style={{ fontSize: 14, color: '#666' }}>
                      {currentImageIndex + 1} / {uploadedImages.length}
                    </span>
                    <button
                      onClick={goToNextImage}
                      disabled={currentImageIndex >= uploadedImages.length - 1}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: currentImageIndex >= uploadedImages.length - 1 ? '#d9d9d9' : '#f0f0f0',
                        border: '1px solid #d9d9d9',
                        borderRadius: 4,
                        cursor: currentImageIndex >= uploadedImages.length - 1 ? 'not-allowed' : 'pointer',
                      }}
                    >
                      Next
                    </button>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    {result && (
                      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
                        <input
                          type="checkbox"
                          checked={showOriginal}
                          onChange={e => setShowOriginal(e.target.checked)}
                        />
                        Show Original
                      </label>
                    )}
                    <button
                      onClick={runInference}
                      disabled={inferring || !selectedModel}
                      style={{
                        padding: '8px 24px',
                        backgroundColor: inferring || !selectedModel ? '#d9d9d9' : '#52c41a',
                        color: 'white',
                        border: 'none',
                        borderRadius: 6,
                        cursor: inferring || !selectedModel ? 'not-allowed' : 'pointer',
                        fontWeight: 500,
                        fontSize: 15,
                      }}
                    >
                      {inferring ? 'Running...' : 'Run Inference'}
                    </button>
                  </div>
                </div>

                {/* Image */}
                <div style={{
                  backgroundColor: '#1e1e1e',
                  borderRadius: 8,
                  padding: 16,
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  minHeight: 400,
                }}>
                  <img
                    src={showOriginal || !result ? uploadedImages[currentImageIndex]?.dataUrl : result.image_data}
                    alt="Inference result"
                    style={{
                      maxWidth: '100%',
                      maxHeight: 600,
                      objectFit: 'contain',
                      borderRadius: 4,
                    }}
                  />
                </div>

                {/* File info */}
                <div style={{ marginTop: 8, fontSize: 13, color: '#666' }}>
                  <strong>File:</strong> {uploadedImages[currentImageIndex]?.file.name}
                  {result && (
                    <span style={{ marginLeft: 16 }}>
                      <strong>Size:</strong> {result.image_width} x {result.image_height}
                    </span>
                  )}
                </div>
              </>
            )}
          </section>

          {/* Detection Results */}
          {result && (
            <section style={{ padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <h3 style={{ margin: 0 }}>Detection Results</h3>
                <div style={{ fontSize: 14, color: '#666' }}>
                  Inference Time: <strong>{result.inference_time_ms.toFixed(1)}ms</strong>
                </div>
              </div>

              {result.detections.length === 0 ? (
                <div style={{
                  padding: 24,
                  textAlign: 'center',
                  color: '#999',
                  backgroundColor: '#fafafa',
                  borderRadius: 6,
                }}>
                  No detections found. Try lowering the confidence threshold.
                </div>
              ) : (
                <>
                  {/* Summary */}
                  <div style={{
                    display: 'flex',
                    gap: 16,
                    marginBottom: 16,
                    padding: 12,
                    backgroundColor: '#f6ffed',
                    borderRadius: 6,
                    border: '1px solid #b7eb8f',
                  }}>
                    <div>
                      <strong>Total Detections:</strong> {result.detections.length}
                    </div>
                    <div>
                      <strong>Classes Found:</strong>{' '}
                      {[...new Set(result.detections.map(d => d.class_name))].join(', ')}
                    </div>
                  </div>

                  {/* Detection list */}
                  <div style={{
                    maxHeight: 300,
                    overflowY: 'auto',
                    border: '1px solid #eee',
                    borderRadius: 6,
                  }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                      <thead>
                        <tr style={{ backgroundColor: '#fafafa' }}>
                          <th style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #eee' }}>#</th>
                          <th style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #eee' }}>Class</th>
                          <th style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #eee' }}>Confidence</th>
                          <th style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #eee' }}>Position</th>
                          <th style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #eee' }}>Size</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((det, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #f0f0f0' }}>
                            <td style={{ padding: '8px 12px' }}>{idx + 1}</td>
                            <td style={{ padding: '8px 12px' }}>
                              <span style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 6,
                              }}>
                                <span style={{
                                  width: 12,
                                  height: 12,
                                  borderRadius: 2,
                                  backgroundColor: COLORS[det.class_id % COLORS.length],
                                }} />
                                {det.class_name}
                              </span>
                            </td>
                            <td style={{ padding: '8px 12px' }}>
                              <span style={{
                                padding: '2px 8px',
                                borderRadius: 4,
                                backgroundColor: det.confidence > 0.7 ? '#f6ffed' : det.confidence > 0.4 ? '#fffbe6' : '#fff1f0',
                                color: det.confidence > 0.7 ? '#52c41a' : det.confidence > 0.4 ? '#faad14' : '#ff4d4f',
                              }}>
                                {(det.confidence * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td style={{ padding: '8px 12px', color: '#666', fontFamily: 'monospace' }}>
                              ({det.x.toFixed(0)}, {det.y.toFixed(0)})
                            </td>
                            <td style={{ padding: '8px 12px', color: '#666', fontFamily: 'monospace' }}>
                              {det.width.toFixed(0)} x {det.height.toFixed(0)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Class statistics */}
                  <div style={{ marginTop: 16 }}>
                    <h4 style={{ margin: '0 0 8px 0', fontSize: 14 }}>Class Statistics</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {Object.entries(
                        result.detections.reduce((acc, det) => {
                          acc[det.class_name] = (acc[det.class_name] || 0) + 1
                          return acc
                        }, {} as Record<string, number>)
                      ).map(([className, count]) => {
                        const det = result.detections.find(d => d.class_name === className)
                        return (
                          <div
                            key={className}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6,
                              padding: '6px 12px',
                              backgroundColor: '#f5f5f5',
                              borderRadius: 4,
                              fontSize: 13,
                            }}
                          >
                            <span style={{
                              width: 10,
                              height: 10,
                              borderRadius: 2,
                              backgroundColor: COLORS[(det?.class_id || 0) % COLORS.length],
                            }} />
                            <span>{className}:</span>
                            <strong>{count}</strong>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  )
}
