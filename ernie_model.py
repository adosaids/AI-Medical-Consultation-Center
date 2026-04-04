from camel.models import BaseModelBackend
from qianfan import ChatCompletion
from typing import List, Optional,Union,Dict, Any
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
        self.client = ChatCompletion(access_key=api_key, secret_key=sk,temperature=0.05,top_p=0.2)
        self.liu = liu

    def token_counter(self):
        self._token_counter = ERNIETokenCounter(self.model_type)
        return self._token_counter

    def run(
        self,
        messages: List,# [{"role":"user" or "assistant","content":"any"}]
        prompt: str
    ):
        resp = self.client.do(system=prompt,stop=["<stop>"],model=self.model_type.value,messages=messages,stream=self.liu)# 没有提供两个api
        if self.liu:
            return resp
        else:
            return resp['result']

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
