from camel.models import BaseModelBackend
from typing import Optional,Union,Dict, Any
from camel.utils import BaseTokenCounter
from camel.types import ModelPlatformType
from ernie_types import ModelType
from ernie_model import ErnieModel
class ModelFactory:
    @staticmethod
    def create(
        model_platform: ModelPlatformType,
        model_type: Union[ModelType, str],
        model_config_dict: Dict,
        token_counter: Optional[BaseTokenCounter] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        sk: Optional[str] = None,
        liu: bool = False
    ) -> BaseModelBackend:
        model_class: Any
        if isinstance(model_type, ModelType):
            if model_platform.is_qianfan and model_type.is_qianfan:
                model_class = ErnieModel
        else:
            raise ValueError(f"Invalid model type `{model_type}` provided.")
        if model_type.is_qianfan:
            return model_class(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            sk=sk,
            liu = liu
    )