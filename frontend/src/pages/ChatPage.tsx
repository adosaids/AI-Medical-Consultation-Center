import { useState, useRef, useEffect } from 'react'
import { Card, Input, Button, List, Avatar, Spin, Tag, Steps } from 'antd'
import { UserOutlined, RobotOutlined, MedicineBoxOutlined, SolutionOutlined, FileTextOutlined } from '@ant-design/icons'
import { sendStreamingMessage, startStreamingDiagnosis, connectWebSocket } from '../services/api'

const { TextArea } = Input

interface Message {
  id: string
  role: 'user' | 'nurse' | 'doctor' | 'system' | 'expert' | 'assistant'
  content: string
  timestamp: Date
}

interface DiagnosisStep {
  step_number: number
  hypothesis: string
  reasoning: string
  rejected_reason: string
  is_accepted: boolean
  is_rejected: boolean
  evidence: string[]
}

interface DiagnosisProcess {
  steps: DiagnosisStep[]
  final_diagnosis: string
  confidence: number
  summary: string
  total_steps: number
}

interface PatientCase {
  request: string
  Vital_Signs: Record<string, string>
  Diagnosis_Process: DiagnosisProcess | null
  Diagnosis: string
  Treatment_Plan: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'system',
      content: '您好！我是医宝智能问诊助手。请告诉我您哪里不舒服，我会帮您收集症状信息。',
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [patientCase, setPatientCase] = useState<PatientCase | null>(null)

  // 使用 ref 来跟踪流式消息
  const streamingIdRef = useRef<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsInitialized = useRef(false)

  // 初始化 WebSocket 连接
  useEffect(() => {
    if (!wsInitialized.current) {
      wsInitialized.current = true
      connectWebSocket().catch(console.error)
    }
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    // 创建流式消息占位
    const streamingId = (Date.now() + 1).toString()
    streamingIdRef.current = streamingId

    // 先添加一个空的护士消息
    const nurseMessage: Message = {
      id: streamingId,
      role: 'nurse',
      content: '',
      timestamp: new Date()
    }
    setMessages(prev => [...prev, nurseMessage])

    let receivedContent = ''

    try {
      sendStreamingMessage(
        input,
        // onChunk - 收到每个数据块
        (chunk) => {
          receivedContent += chunk
          // 直接更新消息数组中的内容
          setMessages(prevMessages => {
            const newMessages = [...prevMessages]
            const nurseIndex = newMessages.findIndex(m => m.id === streamingId)
            if (nurseIndex >= 0) {
              newMessages[nurseIndex] = {
                ...newMessages[nurseIndex],
                content: receivedContent
              }
            }
            return newMessages
          })
        },
        // onComplete - 流式传输完成
        () => {
          streamingIdRef.current = ''
          setLoading(false)
        },
        // onError - 发生错误
        (error) => {
          setMessages(prev => [...prev, {
            id: (Date.now() + 1).toString(),
            role: 'system',
            content: '抱歉，服务出现了一些问题：' + error,
            timestamp: new Date()
          }])
          streamingIdRef.current = ''
          setLoading(false)
        }
      )
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: '抱歉，服务出现了一些问题，请稍后再试。',
        timestamp: new Date()
      }])
      streamingIdRef.current = ''
      setLoading(false)
    }
  }

  const handleDiagnosis = async () => {
    setLoading(true)
    setCurrentStep(1)

    setMessages(prev => [...prev, {
      id: (Date.now() + 1).toString(),
      role: 'doctor',
      content: '开始诊断推理...',
      timestamp: new Date()
    }])

    try {
      startStreamingDiagnosis(
        '开始诊断',
        {
          onPhaseStart: (phase) => {
            setMessages(prev => [...prev, {
              id: (Date.now() + Math.random()).toString(),
              role: 'system',
              content: `=== ${phase} ===`,
              timestamp: new Date()
            }])
            // 更新当前步骤
            if (phase === '诊断推理') {
              setCurrentStep(1)
            } else if (phase === '治疗规划') {
              setCurrentStep(2)
            }
          },
          onChunk: (data) => {
            // 同时添加到聊天框显示
            const role = data.phase === '诊断推理' ? 'expert' : 'assistant'
            const roleName = data.phase === '诊断推理' ? `推理专家-${data.turn}` : `治疗规划-${data.turn}`

            setMessages(prev => {
              // 查找是否已存在该角色的消息
              const msgId = `${data.phase}_${data.role}_${data.turn}`
              const existingIndex = prev.findIndex(m => m.id === msgId)

              if (existingIndex >= 0) {
                // 更新已有消息
                const newMessages = [...prev]
                newMessages[existingIndex] = {
                  ...newMessages[existingIndex],
                  content: newMessages[existingIndex].content + data.content
                }
                return newMessages
              } else {
                // 添加新消息
                return [...prev, {
                  id: msgId,
                  role: role as any,
                  content: data.content,
                  timestamp: new Date()
                }]
              }
            })
          },
          onComplete: () => {
            setCurrentStep(2)
            setLoading(false)
          },
          onError: (error) => {
            console.error('诊断失败:', error)
            setLoading(false)
          },
          onPatientCaseUpdate: (caseData) => {
            setPatientCase(caseData)
          }
        }
      )
    } catch (error) {
      console.error('诊断失败:', error)
      setLoading(false)
    }
  }

  const getAvatar = (role: string) => {
    switch (role) {
      case 'user':
        return <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#87d068' }} />
      case 'nurse':
        return <Avatar icon={<MedicineBoxOutlined />} style={{ backgroundColor: '#1890ff' }} />
      case 'doctor':
        return <Avatar icon={<SolutionOutlined />} style={{ backgroundColor: '#722ed1' }} />
      case 'expert':
        return <Avatar icon={<FileTextOutlined />} style={{ backgroundColor: '#fa8c16' }} />
      case 'assistant':
        return <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#eb2f96' }} />
      default:
        return <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#faad14' }} />
    }
  }

  const getRoleName = (role: string) => {
    switch (role) {
      case 'user': return '患者'
      case 'nurse': return '护士'
      case 'doctor': return '医生'
      case 'expert': return '推理专家'
      case 'assistant': return '治疗规划师'
      default: return '系统'
    }
  }

  const stepItems = [
    { title: '症状收集', icon: <MedicineBoxOutlined /> },
    { title: '诊断推理', icon: <SolutionOutlined /> },
    { title: '治疗规划', icon: <FileTextOutlined /> },
  ]

  return (
    <div style={{ display: 'flex', gap: 24 }}>
      <Card style={{ flex: 1, height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
        <Steps current={currentStep} style={{ marginBottom: 24 }} items={stepItems} />

        <div style={{ flex: 1, overflow: 'auto', marginBottom: 16, padding: '0 8px' }}>
          <List
            dataSource={messages}
            renderItem={(msg) => (
              <List.Item
                style={{
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  padding: '8px 0'
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    alignItems: 'flex-start',
                    maxWidth: '80%'
                  }}
                >
                  {getAvatar(msg.role)}
                  <div
                    style={{
                      marginLeft: msg.role === 'user' ? 0 : 12,
                      marginRight: msg.role === 'user' ? 12 : 0,
                      padding: 12,
                      backgroundColor: msg.role === 'user' ? '#1890ff' : '#f0f2f5',
                      color: msg.role === 'user' ? '#fff' : '#333',
                      borderRadius: 12,
                      wordBreak: 'break-word',
                      minWidth: 100
                    }}
                  >
                    <div style={{ fontSize: 12, marginBottom: 4, opacity: 0.7 }}>
                      {getRoleName(msg.role)}
                      {msg.id === streamingIdRef.current && loading && (
                        <span style={{ marginLeft: 8 }}><Spin size="small" /></span>
                      )}
                    </div>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  </div>
                </div>
              </List.Item>
            )}
          />
          <div ref={messagesEndRef} />
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="请描述您的症状..."
            autoSize={{ minRows: 2, maxRows: 4 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={loading}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Button type="primary" onClick={handleSend} loading={loading && !!streamingIdRef.current}>
              发送
            </Button>
            <Button onClick={handleDiagnosis} loading={loading} type="dashed" danger disabled={loading}>
              开始诊断
            </Button>
          </div>
        </div>
      </Card>

      {patientCase && (
        <Card title="诊断结果" style={{ width: 450, overflow: 'auto', maxHeight: 'calc(100vh - 200px)' }}>
          {/* 症状信息 */}
          {Object.keys(patientCase.Vital_Signs).length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <Tag color="blue">症状信息</Tag>
              <div style={{ marginTop: 8, padding: 8, background: '#e6f7ff', borderRadius: 4 }}>
                {Object.entries(patientCase.Vital_Signs).map(([key, value], idx) => (
                  <div key={idx} style={{ marginBottom: 4, fontSize: 13 }}>
                    <span style={{ fontWeight: 'bold', color: '#1890ff' }}>{key}:</span>
                    <span style={{ marginLeft: 8 }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 诊断推理过程 */}
          {patientCase.Diagnosis_Process && patientCase.Diagnosis_Process.steps.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <Tag color="orange">诊断推理过程</Tag>
              <div style={{ marginTop: 8 }}>
                {patientCase.Diagnosis_Process.steps.map((step, idx) => (
                  <div key={idx} style={{
                    marginBottom: 12,
                    padding: 8,
                    background: step.is_accepted ? '#f6ffed' : step.is_rejected ? '#fff1f0' : '#fff7e6',
                    borderRadius: 4,
                    border: `1px solid ${step.is_accepted ? '#b7eb8f' : step.is_rejected ? '#ffa39e' : '#ffd591'}`
                  }}>
                    <div style={{ fontWeight: 'bold', fontSize: 12, color: '#666', marginBottom: 4 }}>
                      步骤 {step.step_number}
                      {step.is_accepted && <span style={{ color: '#52c41a', marginLeft: 8 }}>✓ 接受</span>}
                      {step.is_rejected && <span style={{ color: '#ff4d4f', marginLeft: 8 }}>✗ 排除</span>}
                    </div>
                    <div style={{ fontSize: 13, marginBottom: 4 }}>
                      <span style={{ fontWeight: 'bold' }}>假设:</span> {step.hypothesis.slice(0, 100)}{step.hypothesis.length > 100 ? '...' : ''}
                    </div>
                    <div style={{ fontSize: 12, color: '#666' }}>
                      <span style={{ fontWeight: 'bold' }}>理由:</span> {step.reasoning.slice(0, 80)}{step.reasoning.length > 80 ? '...' : ''}
                    </div>
                    {step.rejected_reason && (
                      <div style={{ fontSize: 12, color: '#ff4d4f', marginTop: 4 }}>
                        <span style={{ fontWeight: 'bold' }}>否定理由:</span> {step.rejected_reason.slice(0, 80)}{step.rejected_reason.length > 80 ? '...' : ''}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 最终诊断 */}
          {patientCase.Diagnosis && (
            <div style={{ marginBottom: 16 }}>
              <Tag color="purple">最终诊断</Tag>
              <div style={{ marginTop: 8, padding: 8, background: '#f9f0ff', borderRadius: 4, fontSize: 14 }}>
                {patientCase.Diagnosis}
              </div>
              {patientCase.Diagnosis_Process && patientCase.Diagnosis_Process.confidence > 0 && (
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                  置信度: {(patientCase.Diagnosis_Process.confidence * 100).toFixed(0)}%
                </div>
              )}
            </div>
          )}

          {/* 治疗规划 */}
          {patientCase.Treatment_Plan && (
            <div style={{ marginBottom: 16 }}>
              <Tag color="green">治疗规划</Tag>
              <div style={{ marginTop: 8, padding: 8, background: '#f6ffed', borderRadius: 4, fontSize: 14, whiteSpace: 'pre-wrap' }}>
                {patientCase.Treatment_Plan}
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
