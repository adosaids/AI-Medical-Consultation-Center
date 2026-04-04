from utils import wenjian
from camel.agents import ChatAgent
from factor import ModelFactory
from camel.types import ModelPlatformType,RoleType,OpenAIBackendRole
from ernie_types import ModelType
from typing import Optional,Dict,List,Any
from camel.messages import BaseMessage,OpenAIMessage
from ernie_adapters import ERNIETokenCounter
from camel.agents.chat_agent import FunctionCallingRecord
from camel.memories import MemoryRecord,ChatHistoryMemory,ScoreBasedContextCreator
# 继承ChatAgent创建专用子类
class ErnieChineseAgent(ChatAgent):
    def __init__(
        self,
        system_message,
        ernie_api_key: str,
        ernie_secret_key: str,
        model_type: ModelType = ModelType.ERNIE_8K,
        model_config: Optional[Dict] = None,
        liu: bool = False,
        **kwargs
    ):
        # 初始化ERNIE后端
        ernie_model = ModelFactory.create(
            model_platform=ModelPlatformType.QIANFAN,
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
    def _create_chinese_system_message(self, original_message: BaseMessage):
        """增强中文系统提示"""
        enhanced_content = f"{original_message.content}\n请使用简体中文进行交流，保持专业且自然的表达风格。"
        return BaseMessage(
            role_name="智能助手",
            role_type=RoleType.ASSISTANT,
            meta_dict=original_message.meta_dict,
            content=enhanced_content
        )
    def convert_to_ernie_messages(self,openai_messages: List[OpenAIMessage]):
        """将OpenAI格式消息转换为ERNIE所需格式"""
        return [{
            "role": "user" if msg["role"] == "assistant" else "assistant",
            "content": msg["content"]
        } for msg in openai_messages]
    
    def step(self, input_message:BaseMessage,rolename,prompt = None,need_memory:bool = True,liu:bool = False):
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
        openai_messages,num_tokens = self.memory.get_context()
        if need_memory:
            res = self.model_backend.run(openai_messages,prompt = prompt)
        else:
            res = self.model_backend.run([{"user":input_message}],prompt = prompt)
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



