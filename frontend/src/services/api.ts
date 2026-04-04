import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    console.log('请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API 错误:', error)
    return Promise.reject(error)
  }
)

// 发送消息到护士接口
export async function sendMessage(message: string): Promise<{ 护士: string }> {
  const response = await api.post(`/huanzhejiekou/${encodeURIComponent(message)}`)
  return response as unknown as { 护士: string }
}

// 获取诊断结果
export async function getDiagnosis(request: string): Promise<{
  诊断推理: string[]
  治疗规划: string[]
}> {
  const response = await api.post(`/zhenduantuili/${encodeURIComponent(request)}`)
  return response as unknown as {
    诊断推理: string[]
    治疗规划: string[]
  }
}

// 上传 PDF 文件
export async function uploadPDF(file: File): Promise<{
  success: boolean
  message: string
  data?: {
    filename: string
    total_chunks: number
    stored_chunks: number
    text_length: number
  }
  error?: string
}> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await axios.post('/api/upload_pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: 60000
  })

  return response.data
}

export default api
