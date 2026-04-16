// WebSocket 服务 - 实现前后端实时双向通信

export type WebSocketMessage = {
  type: string
  [key: string]: any
}

export type MessageHandler = (message: WebSocketMessage) => void

export class WebSocketService {
  private ws: WebSocket | null = null
  private clientId: string
  private messageHandlers: Map<string, MessageHandler[]> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 3000
  private isConnecting = false

  constructor(clientId?: string) {
    this.clientId = clientId || this.generateClientId()
  }

  private generateClientId(): string {
    return 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
  }

  getClientId(): string {
    return this.clientId
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      if (this.isConnecting) {
        // 等待连接完成
        const checkInterval = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            clearInterval(checkInterval)
            resolve()
          }
        }, 100)
        return
      }

      this.isConnecting = true

      const wsUrl = `ws://127.0.0.1:8000/ws/${this.clientId}`
      this.ws = new WebSocket(wsUrl)

      this.ws.onopen = () => {
        console.log('WebSocket 连接成功')
        this.reconnectAttempts = 0
        this.isConnecting = false
        resolve()
      }

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('[WebSocket] 收到消息:', message.type, message)
          this.handleMessage(message)
        } catch (error) {
          console.error('解析 WebSocket 消息失败:', error)
        }
      }

      this.ws.onclose = () => {
        console.log('WebSocket 连接关闭')
        this.isConnecting = false
        this.attemptReconnect()
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket 错误:', error)
        this.isConnecting = false
        reject(error)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`尝试重新连接... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
      setTimeout(() => {
        this.connect().catch(console.error)
      }, this.reconnectDelay)
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  send(message: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.error('WebSocket 未连接')
    }
  }

  // 发送聊天消息（流式）
  sendChatMessage(message: string): void {
    this.send({
      action: 'chat',
      message: message
    })
  }

  // 开始诊断流程（异步流式）
  startDiagnosis(request: string): void {
    this.send({
      action: 'start_diagnosis',
      request: request
    })
  }

  // 发送心跳
  ping(): void {
    this.send({ action: 'ping' })
  }

  // 提交补充信息（在诊断推理过程中）
  submitSupplementaryInfo(answer: string): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log('[WebSocket] 准备提交补充信息:', answer)

      // 先监听确认消息
      const unsub = this.on('supplementary_info_accepted', (msg) => {
        console.log('[WebSocket] 收到 supplementary_info_accepted:', msg)
        unsub()
        resolve()
      })

      // 发送补充信息
      const payload = {
        action: 'submit_supplementary_info',
        answer: answer,
        request_id: Date.now().toString()
      }
      console.log('[WebSocket] 发送 submit_supplementary_info:', payload)
      this.send(payload)

      // 超时处理（30秒，给后端足够时间处理）
      setTimeout(() => {
        console.error('[WebSocket] 提交补充信息超时')
        unsub()
        reject(new Error('提交补充信息超时'))
      }, 30000)
    })
  }

  // 注册消息处理器
  on(type: string, handler: MessageHandler): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, [])
    }
    this.messageHandlers.get(type)!.push(handler)

    // 返回取消订阅函数
    return () => {
      const handlers = this.messageHandlers.get(type)
      if (handlers) {
        const index = handlers.indexOf(handler)
        if (index > -1) {
          handlers.splice(index, 1)
        }
      }
    }
  }

  private handleMessage(message: WebSocketMessage) {
    const handlers = this.messageHandlers.get(message.type) || []
    handlers.forEach(handler => {
      try {
        handler(message)
      } catch (error) {
        console.error('消息处理器错误:', error)
      }
    })

    // 也触发通配符处理器
    const wildcardHandlers = this.messageHandlers.get('*') || []
    wildcardHandlers.forEach(handler => {
      try {
        handler(message)
      } catch (error) {
        console.error('通配符消息处理器错误:', error)
      }
    })
  }

  // 设置全局补充信息处理器
  setGlobalSupplementHandler(handler: (question: string) => void) {
    globalSupplementHandler = handler
  }

  // 检查连接状态
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// 单例实例
let wsService: WebSocketService | null = null

export function getWebSocketService(): WebSocketService {
  if (!wsService) {
    wsService = new WebSocketService()
  }
  return wsService
}

export function initWebSocket(): WebSocketService {
  return getWebSocketService()
}

export function closeWebSocket() {
  if (wsService) {
    wsService.disconnect()
    wsService = null
  }
}

// 全局补充信息处理器（用于页面刷新后仍能处理补充信息）
let globalSupplementHandler: ((question: string) => void) | null = null

export function setGlobalSupplementHandler(handler: (question: string) => void) {
  globalSupplementHandler = handler
}
