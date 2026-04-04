from camel.memories import (
    ChatHistoryBlock,
    LongtermAgentMemory,
    ScoreBasedContextCreator,
)
from camel.storages import BaseKeyValueStorage, BaseVectorStorage
from camel.memories.base import BaseContextCreator
from typing import TYPE_CHECKING, List, Optional,Union,Dict, Any
from qianfan import Embedding  # 导入百度 SDK
from camel.types import ChatCompletionMessageParam
from ernie_types import ModelType
from camel.utils import BaseTokenCounter
from camel.memories import MemoryBlock, MemoryRecord, ContextRecord,ChatHistoryMemory
from camel.storages.vectordb_storages import (
    BaseVectorStorage,
    QdrantStorage,
    VectorDBQuery,
    VectorRecord,
)
from ernie_types import ERNIEBackendRole
from collections import deque

OpenAIMessage = ChatCompletionMessageParam

class ERNIETokenCounter(BaseTokenCounter):
    def __init__(self, model: ModelType):
        self.model = model.value_for_tiktoken  # 假设已定义 ERNIE 模型枚举
        # ERNIE-4.0-turbo-8k 参数 (需根据百度文档确认)
        if "ERNIE" in self.model:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            raise NotImplementedError(f"Unsupported ERNIE model: {self.model}")

        # 加载百度分词器
        self.embed_client = Embedding()

    def count_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                if key == "content":
                    # 调用百度 Embedding API 计算 Token
                    resp = self.embed_client.do(texts=[str(value)], model="Embedding-V1")
                    num_tokens += len(resp.body["data"][0]["embedding"])
                elif key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += 3  # 假设与 OpenAI 类似，添加系统 Token
        return num_tokens


class ERNIEEmbedding():
    """ERNIE 文本向量化模块"""
    def __init__(self):
        self.client = Embedding()

    def embed(self, text) -> List[float]:
        """调用百度 API 获取向量"""
        if type(text) is str:
            resp = self.client.do(texts=[text], model="Embedding-V1")
        else:
            resp = self.client.do(texts=[text], model="Embedding-V1")

        return resp.body["data"][0]["embedding"]  # 384 维向量

    def embed_list(
        self,
        objs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        r"""为给定文本生成嵌入。

参数：
objs （list[str]）：要为其生成嵌入的文本。
**kwargs （Any）：传递给嵌入 API 的额外 kwargs。

返回：
list[list[float]]：表示生成的嵌入的列表
作为浮点数列表。
        """
        # TODO: count tokens
        response = self.client.do(
            model="tao-8k",
            texts=objs
        )
        return [data['embedding'] for data in response['data']]

    def get_output_dim(self) -> int:
        return 384  # ERNIE-Embedding-V1 的向量维度

class ERNIEVectorDBBlock(MemoryBlock):
    """适配 ERNIE 的向量数据库模块"""
    def __init__(
        self,
        storage: Optional[BaseVectorStorage] = None,
        embedding: Optional[ERNIEEmbedding] = None,
    ) -> None:
        self.embedding = embedding or ERNIEEmbedding()
        self.vector_dim = 384  # 固定维度
        self.storage = storage or QdrantStorage(vector_dim=self.vector_dim)

    def retrieve(self, keyword: str, limit: int = 3) -> List[ContextRecord]:
        query_vector = self.embedding.embed(keyword)
        results = self.storage.query(
            VectorDBQuery(query_vector=query_vector, top_k=limit))
        return [
            ContextRecord(
                memory_record=MemoryRecord.from_dict(result.record.payload),
                score=result.similarity,
            ) for result in results if result.record.payload
        ]

    def write_records(self, records: List[MemoryRecord]) -> None:
        v_records = [
            VectorRecord(
                vector=self.embedding.embed(record.message.content),
                payload=record.to_dict(),
                id=str(record.uuid),
            ) for record in records
        ]
        self.storage.add(v_records)

    def clear(self) -> None:
        self.storage.clear()
# 初始化 ERNIE 向量数据库块
# 在 camel/types.py 中定义 ERNIE 角色类型

class ERNIELongtermAgentMemory(LongtermAgentMemory):
    """ERNIE 专用的长期记忆模块"""
    def __init__(
        self,
        model_type: ModelType = ModelType.ERNIE_8K,
        retrieve_limit: int = 5  # 增大检索数量以提升召回率
    ):
        # 使用 ERNIE 的上下文生成策略
        context_creator = ScoreBasedContextCreator(
            token_counter=ERNIETokenCounter(model_type),
            token_limit=8000  # ERNIE-4.0-turbo-8k 的上下文长度
        )

        super().__init__(
            context_creator=context_creator,
            chat_history_block=ChatHistoryBlock(),
            vector_db_block=ERNIEVectorDBBlock(),  # 使用 ERNIE 向量模块
            retrieve_limit=retrieve_limit
        )
    def write_records(self, records: List[MemoryRecord]) -> None:
        super().write_records(records)

    # 新增：针对中文的当前话题提取优化
        for record in records:
            if record.role_at_backend == ERNIEBackendRole.USER:
                content = record.message.content

            # 提取关键词（示例：使用 jieba 分词）
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(content, topK=3)
                self._current_topic = " ".join(keywords)  # 示例结果："胸痛 持续 2小时"
    def current_topic(self):
        return self._current_topic


class ERNIEHistoryMemory(ChatHistoryMemory):
    r"""ERNIE 专用聊天历史记忆模块，保证严格时间顺序

    Args:
        context_creator (BaseContextCreator): 上下文生成器
        storage (BaseKeyValueStorage, optional): 存储后端（默认使用内存存储）
        window_size (int, optional): 上下文窗口大小（保留最近N条消息）
    """

    def __init__(
        self,
        context_creator: BaseContextCreator = None,
        storage: Optional[BaseKeyValueStorage] = None,
        model_type: ModelType = ModelType.ERNIE_8K,
        window_size: Optional[int] = 10,
    ) -> None:
        context_creator = ScoreBasedContextCreator(
            token_counter=ERNIETokenCounter(model_type),
            token_limit=8000  # ERNIE-4.0-turbo-8k 的上下文长度
        )
        super().__init__(context_creator, storage, window_size)

        # 添加时间顺序保障层
        self._ordered_storage = deque(maxlen=window_size)

    def retrieve(self) -> List[ContextRecord]:
        # 按时间顺序获取原始记录（旧→新）
        raw_records = list(self._ordered_storage)
        return self._context_creator.create_context(raw_records)

    def write_records(self, records: List[MemoryRecord]) -> None:
        # 按时间顺序写入
        for record in records:
            self._ordered_storage.append(record)
            self._chat_history_block.write_records([record])

        # 窗口截断
        if self._window_size:
            while len(self._ordered_storage) > self._window_size:
                self._ordered_storage.popleft()

    def clear(self) -> None:
        super().clear()
        self._ordered_storage.clear()
