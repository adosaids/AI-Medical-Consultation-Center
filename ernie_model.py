from camel.models import BaseModelBackend
from qianfan import ChatCompletion
from typing import List, Optional, Union, Dict, Any, Iterator
from camel.utils import BaseTokenCounter
from ernie_adapters import ERNIETokenCounter
from ernie_types import ModelType


class ErnieModel(BaseModelBackend):
    def __init__(
            self,
            model_type: ModelType,
            model_config_dict: Dict,
            api_key: Optional[str] = None,
            sk: Optional[str] = None,
            url: Optional[str] = None,
            token_counter: Optional[BaseTokenCounter] = None,
            liu: bool = False
    ) -> None:
        super().__init__(model_type, model_config_dict, api_key, url, token_counter)
        self.api_key = api_key
        self.url = url
        self._sk = sk
        self.model_type = model_type
        self.client = ChatCompletion(access_key=api_key, secret_key=sk, temperature=0.05, top_p=0.2)
        self.liu = liu

    def token_counter(self):
        self._token_counter = ERNIETokenCounter(self.model_type)
        return self._token_counter

    def run(
            self,
            messages: List,
            prompt: str
    ):
        """执行模型调用

        如果 liu=True，返回流式响应生成器
        否则返回完整响应字符串
        """
        resp = self.client.do(
            system=prompt,
            stop=["<stop>"],
            model=self.model_type.value,
            messages=messages,
            stream=self.liu
        )

        if self.liu:
            # 流式模式：返回生成器
            return self._stream_response(resp)
        else:
            # 非流式模式：返回结果字符串
            result = None
            if hasattr(resp, 'result'):
                result = resp.result
            elif isinstance(resp, dict) and 'result' in resp:
                result = resp['result']
            elif hasattr(resp, 'body'):
                body = resp.body
                if isinstance(body, dict) and 'result' in body:
                    result = body['result']

            if result is None:
                result = str(resp)

            # 确保是字符串
            if isinstance(result, bytes):
                result = result.decode('utf-8')
            else:
                result = str(result)

            return result

    def _stream_response(self, resp) -> Iterator[str]:
        """处理流式响应

        百度千帆返回的是独立片段，不是累积文本，直接 yield 每个片段
        """
        try:
            previous_text = ""

            for chunk in resp:
                # 处理百度千帆流式响应的不同格式
                text = None

                # 千帆 API 返回的是对象，需要提取 result 字段
                if hasattr(chunk, 'result'):
                    text = chunk.result
                elif isinstance(chunk, dict):
                    if 'result' in chunk:
                        text = chunk['result']
                    elif 'body' in chunk:
                        body = chunk['body']
                        if isinstance(body, dict) and 'result' in body:
                            text = body['result']
                        else:
                            text = str(body)
                elif hasattr(chunk, 'body'):
                    body = chunk.body
                    if isinstance(body, dict) and 'result' in body:
                        text = body['result']
                    else:
                        text = str(body)

                # 确保 text 是字符串
                if text is not None:
                    if isinstance(text, bytes):
                        text = text.decode('utf-8')
                    else:
                        text = str(text)

                    # 百度千帆返回的是独立片段，不是累积文本
                    # 直接 yield 每个非空 text
                    if text:
                        yield text

        except Exception as e:
            print(f"流式响应处理错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def check_model_config(self):
        pass

    def count_tokens_from_messages(self, messages: List) -> int:
        return self._token_counter.count_tokens_from_messages(messages=messages)

    @property
    def stream(self) -> bool:
        return self.liu

    def token_limit(self) -> int:
        return (
                self.model_config_dict.get("max_tokens")
                or self.model_type.token_limit
        )
