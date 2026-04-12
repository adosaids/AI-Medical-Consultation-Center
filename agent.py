from utils import wenjian
from camel.agents import ChatAgent
from factor import ModelFactory
from camel.types import RoleType, OpenAIBackendRole
from ernie_types import ModelType
from typing import Optional, Dict, List, Any, AsyncGenerator
from camel.messages import BaseMessage, OpenAIMessage
from ernie_adapters import ERNIETokenCounter
from camel.agents.chat_agent import FunctionCallingRecord
from camel.memories import MemoryRecord, ChatHistoryMemory, ScoreBasedContextCreator
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 线程池用于执行同步代码
_executor = ThreadPoolExecutor(max_workers=10)


# 继承ChatAgent创建专用子类
class ErnieChineseAgent(ChatAgent):
    def __init__(
            self,
            system_message,
            ernie_api_key: str,
            ernie_secret_key: str,
            model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
            model_config: Optional[Dict] = None,
            liu: bool = False,
            **kwargs
    ):
        # 初始化ERNIE后端
        ernie_model = ModelFactory.create(
            model_type=model_type,
            model_config_dict=model_config or {"temperature": 0.5},
            api_key=ernie_api_key,
            sk=ernie_secret_key,
            liu=liu
        )

        # 中文默认设置
        chinese_system_message = self._create_chinese_system_message(system_message)
        super().__init__(
            system_message=chinese_system_message,
            model=ernie_model,
            output_language="zh",
            **kwargs
        )
        self.memory = ChatHistoryMemory(context_creator=ScoreBasedContextCreator(
            token_counter=ERNIETokenCounter(model_type),
            token_limit=8000  # ERNIE-4.0-turbo-8k 的上下文长度
        )
        )
        self.ernie_api_key = ernie_api_key
        self.ernie_secret_key = ernie_secret_key
        self.model_type = model_type

    def _create_chinese_system_message(self, original_message: BaseMessage):
        """增强中文系统提示"""
        enhanced_content = f"{original_message.content}\n请使用简体中文进行交流，保持专业且自然的表达风格。"
        return BaseMessage(
            role_name="智能助手",
            role_type=RoleType.ASSISTANT,
            meta_dict=original_message.meta_dict,
            content=enhanced_content
        )

    def convert_to_ernie_messages(self, openai_messages: List[OpenAIMessage]):
        """将OpenAI格式消息转换为ERNIE所需格式"""
        return [{
            "role": "user" if msg["role"] == "assistant" else "assistant",
            "content": msg["content"]
        } for msg in openai_messages]

    def step(self, input_message: BaseMessage, rolename, prompt=None, need_memory: bool = True, liu: bool = False):
        """同步方式执行步骤（保留原有功能）"""
        ures = MemoryRecord(
            message=BaseMessage(
                role_name="患者",
                role_type=RoleType.USER,
                content=input_message.content,
                meta_dict={"department": "心血管科"}
            ),
            role_at_backend=OpenAIBackendRole.USER
        )
        self.memory.write_record(ures)
        openai_messages, num_tokens = self.memory.get_context()

        # 限制上下文只保留最近6轮对话（12条消息 + 系统提示）
        MAX_HISTORY_ROUNDS = 6
        if len(openai_messages) > MAX_HISTORY_ROUNDS * 2 + 1:
            # 保留系统消息（第一条）和最近的消息
            openai_messages = [openai_messages[0]] + openai_messages[-(MAX_HISTORY_ROUNDS * 2):]
            print(f"[{rolename}] 上下文已截断，保留最近 {MAX_HISTORY_ROUNDS} 轮对话")

        if need_memory:
            res = self.model_backend.run(openai_messages, prompt=prompt)
        else:
            res = self.model_backend.run([{"user": input_message}], prompt=prompt)
        ares = MemoryRecord(
            message=BaseMessage(
                role_name=rolename,
                role_type=RoleType.ASSISTANT,
                content=res,
                meta_dict={"department": "心血管科"}
            ),
            role_at_backend=OpenAIBackendRole.ASSISTANT
        )
        self.memory.write_record(ares)
        return res

    def extract_vital_signs(self) -> Dict[str, str]:
        """从对话记录中提取结构化的患者体征信息

        使用LLM分析对话内容，提取关键体征信息，避免上下文超限

        Returns:
            Dict[str, str]: 结构化的体征信息，如 {"症状": "头痛", "持续时间": "3天", ...}
        """
        # 获取对话记录
        memory_context, _ = self.memory.get_context()

        if not memory_context:
            return {"症状描述": "暂无信息"}

        # 构建对话文本（只取最近的几轮，避免过长）
        dialogue_text = ""
        for record in memory_context[-6:]:  # 只取最近6轮
            role = record.get('role', 'unknown')
            content = record.get('content', '')
            if role == 'user':
                dialogue_text += f"患者: {content}\n"
            else:
                dialogue_text += f"护士: {content}\n"

        # 构建提取提示词
        extraction_prompt = f"""请从以下医患对话中提取关键的患者体征信息。

对话记录：
{dialogue_text}

请提取以下信息（以JSON格式返回，如果没有相关信息则填"未知"或"无"）：
{{
    "主要症状": "患者的主要症状描述",
    "症状部位": "症状发生的身体部位",
    "持续时间": "症状持续了多长时间",
    "严重程度": "症状的严重程度（轻度/中度/重度）",
    "伴随症状": "伴随的其他症状",
    "既往病史": "患者提到的既往病史",
    "用药史": "患者提到的用药情况",
    "过敏史": "患者的过敏史",
    "年龄性别": "患者的年龄和性别（如果有提到）",
    "其他信息": "其他有助于诊断的信息"
}}

只返回JSON格式，不要其他解释。"""

        try:
            # 调用模型提取信息
            response = self.model_backend.run(
                messages=[{"role": "user", "content": extraction_prompt}],
                prompt=""
            )

            # 解析JSON响应
            import json
            # 尝试从响应中提取JSON部分
            response_text = response.strip()

            # 如果响应被markdown代码块包裹，提取内部内容
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            vital_signs = json.loads(response_text)

            # 确保所有必要字段都存在
            default_fields = [
                "主要症状", "症状部位", "持续时间", "严重程度",
                "伴随症状", "既往病史", "用药史", "过敏史", "年龄性别", "其他信息"
            ]
            for field in default_fields:
                if field not in vital_signs:
                    vital_signs[field] = "未知"

            return vital_signs

        except Exception as e:
            print(f"提取体征信息失败: {e}")
            # 如果提取失败，返回简化的信息
            return {
                "症状描述": memory_context[-1].get('content', '未知') if memory_context else "未知",
                "提取失败": "使用原始对话记录"
            }

    async def step_stream(self, input_message: BaseMessage, rolename, prompt=None, need_memory: bool = True, agent_role: str = "Agent") -> AsyncGenerator[str, None]:
        """流式方式执行步骤，返回异步生成器

        每生成一个 token 就 yield 一次，实现真正的流式输出

        Args:
            agent_role: 用于日志标识的角色名称，如"护士Agent"、"推理专家"等
        """
        # 写入用户消息到记忆
        ures = MemoryRecord(
            message=BaseMessage(
                role_name="患者",
                role_type=RoleType.USER,
                content=input_message.content,
                meta_dict={"department": "心血管科"}
            ),
            role_at_backend=OpenAIBackendRole.USER
        )
        self.memory.write_record(ures)

        # 获取上下文
        openai_messages, num_tokens = self.memory.get_context()

        # 限制上下文只保留最近6轮对话（12条消息 + 系统提示）
        MAX_HISTORY_ROUNDS = 6
        if len(openai_messages) > MAX_HISTORY_ROUNDS * 2 + 1:
            # 保留系统消息（第一条）和最近的消息
            openai_messages = [openai_messages[0]] + openai_messages[-(MAX_HISTORY_ROUNDS * 2):]
            print(f"[{agent_role}] 上下文已截断，保留最近 {MAX_HISTORY_ROUNDS} 轮对话")

        # 使用流式模式调用模型
        from factor import ModelFactory

        # 创建流式模型实例
        stream_model = ModelFactory.create(
            model_type=self.model_type,
            model_config_dict={"temperature": 0.5},
            api_key=self.ernie_api_key,
            sk=self.ernie_secret_key,
            liu=True  # 启用流式模式
        )

        full_response = []
        chunk_count = 0

        try:
            # 直接获取流式响应（同步生成器）
            stream_response = stream_model.run(openai_messages, prompt=prompt)

            # 迭代流式响应 - 将同步生成器转为异步
            for chunk in stream_response:
                chunk_count += 1
                # 确保 chunk 是字符串
                if isinstance(chunk, bytes):
                    text = chunk.decode('utf-8')
                else:
                    text = str(chunk)

                # ernie_model 已经返回完整片段，直接 yield
                if text:
                    full_response.append(text)
                    yield text

                # 让出控制权，避免阻塞事件循环
                await asyncio.sleep(0)

        except Exception as e:
            print(f"[{agent_role}] 流式响应错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 流式响应完成后，打印完整回复
            final_response = ''.join(full_response)
            if final_response:
                # 截断过长的日志
                log_content = final_response[:500] + "..." if len(final_response) > 500 else final_response
                print(f"\n{'='*60}")
                print(f"[{agent_role}] 完整回复 ({chunk_count} chunks, {len(final_response)} chars):")
                print(f"{'-'*60}")
                print(log_content)
                print(f"{'='*60}\n")

            # 将完整回复写入记忆
            if final_response:
                ares = MemoryRecord(
                    message=BaseMessage(
                        role_name=rolename,
                        role_type=RoleType.ASSISTANT,
                        content=final_response,
                        meta_dict={"department": "心血管科"}
                    ),
                    role_at_backend=OpenAIBackendRole.ASSISTANT
                )
                self.memory.write_record(ares)
