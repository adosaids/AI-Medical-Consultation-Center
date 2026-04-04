import { Layout, Menu, Typography } from 'antd'
import { MedicineBoxOutlined, MessageOutlined, FileTextOutlined } from '@ant-design/icons'
import { useState } from 'react'
import ChatPage from './pages/ChatPage'
import KnowledgePage from './pages/KnowledgePage'
import './App.css'

const { Header, Content, Sider } = Layout
const { Title } = Typography

type MenuKey = 'chat' | 'knowledge'

function App() {
  const [activeTab, setActiveTab] = useState<MenuKey>('chat')

  const menuItems = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '医疗问诊',
    },
    {
      key: 'knowledge',
      icon: <FileTextOutlined />,
      label: '知识库',
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', borderBottom: '1px solid #e8e8e8' }}>
        <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
          <MedicineBoxOutlined style={{ fontSize: 28, color: '#1890ff', marginRight: 12 }} />
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>医宝智能问诊系统</Title>
        </div>
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[activeTab]}
            style={{ height: '100%', borderRight: 0 }}
            items={menuItems}
            onClick={({ key }) => setActiveTab(key as MenuKey)}
          />
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, borderRadius: 8 }}>
            {activeTab === 'chat' && <ChatPage />}
            {activeTab === 'knowledge' && <KnowledgePage />}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  )
}

export default App
