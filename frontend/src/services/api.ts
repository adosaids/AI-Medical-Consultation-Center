import axios from 'axios'
import { getWebSocketService, WebSocketService } from './websocket'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
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

// ========== 原有 HTTP API（保留兼容性） ==========

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

  const response = await axios.post('http://127.0.0.1:8000/upload_pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: 60000
  })

  return response.data
}

// ========== WebSocket 流式 API ==========

// 获取 WebSocket 服务实例
export function getWS(): WebSocketService {
  return getWebSocketService()
}

// 连接 WebSocket
export async function connectWebSocket(): Promise<WebSocketService> {
  const ws = getWebSocketService()
  await ws.connect()
  return ws
}

// 断开 WebSocket
export function disconnectWebSocket() {
  const ws = getWebSocketService()
  ws.disconnect()
}

// 发送流式聊天消息
export function sendStreamingMessage(
  message: string,
  onChunk: (chunk: string) => void,
  onComplete: (fullResponse: string) => void,
  onError: (error: string) => void
): void {
  const ws = getWebSocketService()

  // 清理之前的监听器
  const unsubscribers: (() => void)[] = []

  // 监听开始
  unsubscribers.push(
    ws.on('chat_start', () => {
      console.log('开始接收流式回复')
    })
  )

  // 监听数据块
  let fullContent = ''
  let chunkIndex = 0
  unsubscribers.push(
    ws.on('chat_chunk', (msg) => {
      const chunk = msg.content || ''
      chunkIndex++
      console.log(`[Frontend] Chunk #${chunkIndex}: ${JSON.stringify(chunk)}`)
      fullContent += chunk
      onChunk(chunk)
    })
  )

  // 监听完成
  unsubscribers.push(
    ws.on('chat_complete', () => {
      onComplete(fullContent)
      // 清理监听器
      unsubscribers.forEach(unsub => unsub())
    })
  )

  // 监听错误
  unsubscribers.push(
    ws.on('chat_error', (msg) => {
      onError(msg.error || '未知错误')
      unsubscribers.forEach(unsub => unsub())
    })
  )

  // 发送消息
  ws.sendChatMessage(message)
}

// 开始流式诊断
export function startStreamingDiagnosis(
  request: string,
  handlers: {
    onPhaseStart?: (phase: string) => void
    onChunk?: (data: {
      phase: string
      role: string
      turn: number
      content: string
    }) => void
    onTurnComplete?: (data: {
      phase: string
      turn: number
      userRes: string
      assRes: string
    }) => void
    onPhaseComplete?: (phase: string, turns: any[]) => void
    onComplete?: () => void
    onError?: (error: string) => void
    onPatientCaseUpdate?: (patientCase: any) => void
  }
): void {
  const ws = getWebSocketService()
  const unsubscribers: (() => void)[] = []

  // 监听诊断开始
  unsubscribers.push(
    ws.on('diagnosis_start', () => {
      console.log('诊断流程开始')
    })
  )

  // 监听阶段开始
  unsubscribers.push(
    ws.on('diagnosis_phase_start', (msg) => {
      handlers.onPhaseStart?.(msg.phase)
    })
  )

  // 监听数据块
  unsubscribers.push(
    ws.on('diagnosis_chunk', (msg) => {
      handlers.onChunk?.({
        phase: msg.phase,
        role: msg.role,
        turn: msg.turn,
        content: msg.content
      })
    })
  )

  // 监听轮次完成
  unsubscribers.push(
    ws.on('diagnosis_turn_complete', (msg) => {
      handlers.onTurnComplete?.({
        phase: msg.phase,
        turn: msg.turn,
        userRes: msg.user_res,
        assRes: msg.ass_res
      })
    })
  )

  // 监听阶段完成
  unsubscribers.push(
    ws.on('diagnosis_phase_complete', (msg) => {
      handlers.onPhaseComplete?.(msg.phase, msg.turns)
    })
  )

  // 监听诊断完成
  unsubscribers.push(
    ws.on('diagnosis_complete', (msg) => {
      // 如果有 patient_case 数据，回调给前端
      if (msg.patient_case) {
        handlers.onPatientCaseUpdate?.(msg.patient_case)
      }
      handlers.onComplete?.()
      unsubscribers.forEach(unsub => unsub())
    })
  )

  // 监听错误
  unsubscribers.push(
    ws.on('diagnosis_error', (msg) => {
      handlers.onError?.(msg.error || '诊断流程错误')
      unsubscribers.forEach(unsub => unsub())
    })
  )

  // 发送开始诊断命令
  ws.startDiagnosis(request)
}

export default api
