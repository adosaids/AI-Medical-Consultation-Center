import { useState } from 'react'
import { Card, Upload, Button, List, Tag, message, Spin } from 'antd'
import { UploadOutlined, FilePdfOutlined, CheckCircleOutlined } from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import { uploadPDF } from '../services/api'

interface UploadedFile {
  filename: string
  totalChunks: number
  storedChunks: number
  textLength: number
  uploadTime: Date
}

export default function KnowledgePage() {
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])

  const handleUpload = async (file: File) => {
    setUploading(true)
    try {
      const result = await uploadPDF(file)

      if (result.success && result.data) {
        message.success(`${file.name} 上传成功`)
        setUploadedFiles(prev => [...prev, {
          filename: result.data!.filename,
          totalChunks: result.data!.total_chunks,
          storedChunks: result.data!.stored_chunks,
          textLength: result.data!.text_length,
          uploadTime: new Date()
        }])
        setFileList([])
      } else {
        message.error(result.error || '上传失败')
      }
    } catch (error) {
      message.error('上传失败，请检查网络连接')
    } finally {
      setUploading(false)
    }
  }

  const handleManualUpload = () => {
    const file = fileList[0]
    if (!file || !file.originFileObj) {
      message.warning('请先选择文件')
      return
    }
    handleUpload(file.originFileObj as File)
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <Card title="知识库管理" style={{ marginBottom: 24 }}>
        <p style={{ marginBottom: 16, color: '#666' }}>
          上传医学文献或教材 PDF，系统将自动提取文本并向量化存储，供问诊时检索使用。
        </p>

        <Upload.Dragger
          fileList={fileList}
          onChange={({ fileList }) => setFileList(fileList.slice(-1))}
          beforeUpload={(file) => {
            const isPDF = file.type === 'application/pdf'
            if (!isPDF) {
              message.error('只能上传 PDF 文件!')
              return Upload.LIST_IGNORE
            }
            return false
          }}
          accept=".pdf"
          maxCount={1}
          disabled={uploading}
        >
          <p className="ant-upload-drag-icon">
            {uploading ? <Spin /> : <UploadOutlined />}
          </p>
          <p className="ant-upload-text">点击或拖拽 PDF 文件到此处上传</p>
          <p className="ant-upload-hint">
            支持单个 PDF 文件上传，文件将被分块处理并存储到向量数据库
          </p>
        </Upload.Dragger>

        <Button
          type="primary"
          onClick={handleManualUpload}
          disabled={fileList.length === 0 || uploading}
          loading={uploading}
          style={{ marginTop: 16 }}
          block
        >
          {uploading ? '上传中...' : '开始上传'}
        </Button>
      </Card>

      <Card title="已上传文档">
        <List
          dataSource={uploadedFiles}
          locale={{ emptyText: '暂无已上传的文档' }}
          renderItem={(file) => (
            <List.Item
              actions={[
                <Tag color="success">
                  <CheckCircleOutlined /> 已处理
                </Tag>
              ]}
            >
              <List.Item.Meta
                avatar={<FilePdfOutlined style={{ fontSize: 24, color: '#ff4d4f' }} />}
                title={file.filename}
                description={
                  <div>
                    <Tag>文本长度: {file.textLength.toLocaleString()} 字符</Tag>
                    <Tag>分块数: {file.totalChunks}</Tag>
                    <Tag>成功存储: {file.storedChunks}</Tag>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  )
}
