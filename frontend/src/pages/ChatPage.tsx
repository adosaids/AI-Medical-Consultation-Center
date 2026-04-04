import { useState, useRef, useEffect } from 'react'
import { Card, Input, Button, List, Avatar, Spin, Tag, Steps } from 'antd'
import { UserOutlined, RobotOutlined, MedicineBoxOutlined, SolutionOutlined, FileTextOutlined } from '@ant-design/icons'
import { sendMessage, getDiagnosis } from '../services/api'

const { TextArea } = Input

interface Message {
  id: string
  role: 'user' | 'nurse' | 'doctor' | 'system'
  content: string
  timestamp: Date
}

interface DiagnosisResult {
  诊断推理: string[]
  治疗规划: string[]
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
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResult | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

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

    try {
      const response = await sendMessage(input)
      const nurseMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'nurse',
        content: response.护士,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, nurseMessage])
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: '抱歉，服务出现了一些问题，请稍后再试。',
        timestamp: new Date()
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleDiagnosis = async () => {
    setLoading(true)
    try {
      const result = await getDiagnosis('开始诊断')
      setDiagnosisResult(result)
      setCurrentStep(1)

      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'doctor',
        content: '诊断推理完成！已为您生成治疗规划建议。',
        timestamp: new Date()
      }])
    } catch (error) {
      console.error('诊断失败:', error)
    } finally {
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
      default:
        return <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#faad14' }} />
    }
  }

  const getRoleName = (role: string) => {
    switch (role) {
      case 'user': return '患者'
      case 'nurse': return '护士'
      case 'doctor': return '医生'
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
                      wordBreak: 'break-word'
                    }}
                  >
                    <div style={{ fontSize: 12, marginBottom: 4, opacity: 0.7 }}>
                      {getRoleName(msg.role)}
                    </div>
                    <div>{msg.content}</div>
                  </div>
                </div>
              </List.Item>
            )}
          />
          {loading && (
            <div style={{ textAlign: 'center', padding: 16 }}>
              <Spin tip="正在思考..." />
            </div>
          )}
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
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Button type="primary" onClick={handleSend} loading={loading}>
              发送
            </Button>
            <Button onClick={handleDiagnosis} loading={loading} type="dashed" danger>
              开始诊断
            </Button>
          </div>
        </div>
      </Card>

      {diagnosisResult && (
        <Card title="诊断结果" style={{ width: 400 }}>
          <div style={{ marginBottom: 16 }}>
            <Tag color="purple">诊断推理</Tag>
            <div style={{ marginTop: 8, padding: 8, background: '#f9f0ff', borderRadius: 4 }}>
              {diagnosisResult.诊断推理.map((item, idx) => (
                <div key={idx} style={{ marginBottom: 4 }}>• {item}</div>
              ))}
            </div>
          </div>
          <div>
            <Tag color="blue">治疗规划</Tag>
            <div style={{ marginTop: 8, padding: 8, background: '#e6f7ff', borderRadius: 4 }}>
              {diagnosisResult.治疗规划.map((item, idx) => (
                <div key={idx} style={{ marginBottom: 4 }}>• {item}</div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
