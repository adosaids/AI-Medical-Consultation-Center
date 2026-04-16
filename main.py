import fix_qdrant  # 必须在最前面导入，修补 QdrantStorage

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from camel.messages import BaseMessage
from camel.types import RoleType
from ernie_types import ModelType
from agent import ErnieChineseAgent
from patient_case import PatientCase
import uvicorn
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import json
from typing import Dict, Optional

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: dict):
        msg_type = message.get("type", "unknown")
        print(f"[WebSocket] 发送消息: client_id={client_id}, type={msg_type}")
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
                print(f"[WebSocket] ✓ 消息已发送")
            except Exception as e:
                print(f"[WebSocket] ❌ 发送消息失败: {e}")
        else:
            print(f"[WebSocket] ❌ client_id={client_id} 不在活跃连接中")

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"广播消息失败: {e}")

manager = ConnectionManager()

from work import work, StreamingWork
from RAG import process_and_store_pdf
from concurrent.futures import ThreadPoolExecutor
from utils import prompts

# 创建线程池用于执行同步代码
executor = ThreadPoolExecutor(max_workers=10)

# 存储每个客户端的 StreamingWork 实例
client_work_instances: Dict[str, StreamingWork] = {}

# 存储每个客户端的 PatientCase
client_patient_cases: Dict[str, PatientCase] = {}

# 存储每个客户端的用户需求
client_requests: Dict[str, str] = {}

def create_patient_case_from_memory(client_id: str, request: str = "") -> PatientCase:
    """从护士对话记录创建 PatientCase

    使用护士agent的extract_vital_signs方法提取结构化体征信息，
    避免传递完整的对话记录导致上下文超限

    Args:
        client_id: 客户端ID
        request: 用户原始需求/请求

    Returns:
        PatientCase: 患者病例对象
    """
    case = PatientCase()
    case.set_request(request)

    # 使用护士agent提取结构化的体征信息
    vital_signs = agent.extract_vital_signs()
    case.set_vital_signs(vital_signs)

    print(f"[PatientCase] 提取的体征信息: {vital_signs}")

    return case


# 加载提示词
prompts_role, prompts_phase, chatchain = prompts()
nurse_prompt = prompts_role.get('huanzhejiekou', [])

# 处理 nurse_prompt：如果是字典列表，转换为 JSON 字符串
def format_prompt(prompt_item):
    """将提示词项格式化为字符串"""
    if isinstance(prompt_item, dict):
        import json
        return json.dumps(prompt_item, ensure_ascii=False, indent=2)
    return str(prompt_item)

# 统一转换为字符串列表
nurse_prompt_str = "".join([format_prompt(p) for p in nurse_prompt])

system_message = BaseMessage(
    role_name="system",
    role_type=RoleType.ASSISTANT,
    content="你是一个智能助手，负责回答用户的问题。",
    meta_dict=[{"a":"b"}]
)
prompt = nurse_prompt_str

# 从环境变量获取 ERNIE API Key 和 Secret Key
ernie_api_key = os.environ.get("ERNIE_API_KEY", "")
ernie_secret_key = os.environ.get("ERNIE_SECRET_KEY", "")

# 初始化 ErnieChineseAgent
agent = ErnieChineseAgent(
    system_message=system_message,
    ernie_api_key=ernie_api_key,
    ernie_secret_key=ernie_secret_key,
    model_type=ModelType.DEEPSEEK_V3_1_250821,
)

zhenduantuili_type = [ModelType.DEEPSEEK_V3_1_250821, ModelType.DEEPSEEK_V3_1_250821]
zhiliaoguihua_type = [ModelType.DEEPSEEK_V3_1_250821, ModelType.DEEPSEEK_V3_1_250821]
work_liu = work(agent_jiekou=agent, zhenduantype=zhenduantuili_type, zhiliaotype=zhiliaoguihua_type)


# ========== WebSocket 端点 ==========

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            action = message_data.get("action")
            print(f"\n[WebSocket] 收到消息: client_id={client_id}, action={action}")

            if action == "chat":
                # 处理聊天消息（流式）
                await handle_streaming_chat(client_id, message_data.get("message", ""))
            elif action == "start_diagnosis":
                # 开始诊断流程（异步流式）
                await handle_streaming_diagnosis(client_id, message_data.get("request", ""))
            elif action == "submit_supplementary_info":
                # 提交补充信息（在诊断推理过程中）
                await handle_supplementary_info(
                    client_id,
                    message_data.get("answer", ""),
                    message_data.get("request_id", "")
                )
            elif action == "ping":
                await manager.send_message(client_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        if client_id in client_work_instances:
            del client_work_instances[client_id]
        if client_id in client_patient_cases:
            del client_patient_cases[client_id]


async def handle_streaming_chat(client_id: str, message: str):
    """处理流式聊天消息"""
    mes = BaseMessage(
        role_name="患者",
        role_type=RoleType.USER,
        content=message,
        meta_dict={"患者接口": "输入"}
    )

    # 发送开始标记
    await manager.send_message(client_id, {
        "type": "chat_start",
        "role": "nurse"
    })

    # 使用流式模式获取回复
    response_chunks = []
    try:
        # 直接调用异步生成器（不使用 run_in_executor）
        async for chunk in agent.step_stream(mes, rolename='患者接口', prompt=prompt, agent_role='护士Agent'):
            response_chunks.append(chunk)
            await manager.send_message(client_id, {
                "type": "chat_chunk",
                "content": chunk,
                "role": "nurse"
            })

    except Exception as e:
        print(f"流式聊天错误: {e}")
        import traceback
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "chat_error",
            "error": str(e)
        })

    # 发送完成标记
    full_response = "".join(response_chunks)
    await manager.send_message(client_id, {
        "type": "chat_complete",
        "content": full_response,
        "role": "nurse"
    })


async def handle_supplementary_info(client_id: str, answer: str, request_id: str):
    """处理用户提交的补充信息

    在诊断推理过程中，当专科医生需要补充信息时，用户通过此接口提交回答
    """
    print(f"\n{'='*60}")
    print(f"[handle_supplementary_info] 收到补充信息")
    print(f"  client_id={client_id}")
    print(f"  request_id={request_id}")
    print(f"  answer={answer[:50]}...")
    print(f"{'='*60}\n")

    streaming_work = client_work_instances.get(client_id)
    if not streaming_work:
        print(f"[handle_supplementary_info] ❌ 错误: 没有找到 client_id={client_id} 的 StreamingWork 实例")
        await manager.send_message(client_id, {
            "type": "error",
            "error": "没有进行中的诊断流程"
        })
        return

    is_waiting = streaming_work.is_waiting_for_supplement()
    print(f"[handle_supplementary_info] 检查是否在等待补充: is_waiting={is_waiting}")

    if not is_waiting:
        print(f"[handle_supplementary_info] ⚠️ 警告: 当前不需要补充信息，忽略该请求")
        await manager.send_message(client_id, {
            "type": "error",
            "error": "当前不需要补充信息"
        })
        return

    try:
        # 提交补充信息，恢复诊断流程
        print(f"[handle_supplementary_info] ✓ 正在提交补充信息...")
        await streaming_work.submit_supplementary_info(answer)
        print(f"[handle_supplementary_info] ✓ 补充信息已提交，准备发送确认消息")

        await manager.send_message(client_id, {
            "type": "supplementary_info_accepted",
            "request_id": request_id
        })
        print(f"[handle_supplementary_info] ✓ 确认消息已发送")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"[handle_supplementary_info] ❌ 处理补充信息错误: {e}")
        import traceback
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "error",
            "error": f"处理补充信息失败: {str(e)}"
        })


async def handle_streaming_diagnosis(client_id: str, request: str):
    """处理流式诊断流程 - 使用 PatientCase 共享数据"""

    # 创建 PatientCase（从护士对话记录）
    patient_case = create_patient_case_from_memory(client_id, request)
    client_patient_cases[client_id] = patient_case

    # 创建流式工作实例
    streaming_work = StreamingWork(
        agent_jiekou=agent,
        zhenduantype=zhenduantuili_type,
        zhiliaotype=zhiliaoguihua_type,
        client_id=client_id,
        message_callback=manager.send_message
    )
    client_work_instances[client_id] = streaming_work

    # 发送诊断开始标记
    await manager.send_message(client_id, {
        "type": "diagnosis_start",
        "patient_case": patient_case.to_dict()
    })

    try:
        # 串行执行诊断推理和治疗规划
        await streaming_work.work_parallel(patient_case)
    except Exception as e:
        print(f"诊断流程错误: {e}")
        import traceback
        traceback.print_exc()
        await manager.send_message(client_id, {
            "type": "diagnosis_error",
            "error": str(e)
        })
    finally:
        # 发送诊断完成标记，包含完整的 PatientCase
        final_case = client_patient_cases.get(client_id)
        await manager.send_message(client_id, {
            "type": "diagnosis_complete",
            "patient_case": final_case.to_dict() if final_case else None
        })
        if client_id in client_work_instances:
            del client_work_instances[client_id]
        if client_id in client_patient_cases:
            del client_patient_cases[client_id]


# ========== 原有 HTTP API（保留兼容性） ==========

@app.post("/huanzhejiekou/{wenti}")
async def huanzhejiekou(wenti: str):
    mes = BaseMessage(
        role_name="患者",
        role_type=RoleType.USER,
        content=wenti,
        meta_dict={"患者接口": "输入"}
    )
    res = agent.step(mes, rolename='患者接口', prompt=prompt)
    return {
        "护士": res
    }


@app.post("/zhenduantuili/{request}")
async def zhenduantuili(request: str):
    li_tuili = work_liu.work()
    li_guihua = work_liu.work_guihua(request)
    return {
        "诊断推理": li_tuili,
        "治疗规划": li_guihua
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 文件接口"""
    try:
        content = await file.read()
        result = process_and_store_pdf(content, file.filename)
        return JSONResponse(content={
            "success": True,
            "message": "PDF 处理成功",
            "data": result
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "message": str(e),
            "error": str(e)
        }, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
