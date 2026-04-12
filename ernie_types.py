from enum import Enum
class ERNIEBackendRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class TaskType(Enum):
    SOCIETY = "ai协作"
    CHAT = "与用户的交流"
    SHUCHU = "标准化输出"

class ModelType(Enum):
    ERNIE_8K_P= "ERNIE-4.0-8K-Preview"
    ERNIE_8K="ERNIE-3.5-8K"
    ERNIE_Character_8k = "ERNIE-Character-8K"
    ERNIE_Speed_P = "ERNIE-Speed-Pro-128K"
    ERNIE_Speed_8K = "ERNIE-Speed-8K"
    ERNIE_Speed_128K = "ERNIE-Speed-128K"
    DEEPSEEK_V3_1_250821 = "deepseek-v3.1-250821"

    @property
    def is_qianfan(self) -> bool:
        r"""Returns whether this type of models is a Qianfan (Baidu) model."""
        return self in {
            ModelType.ERNIE_8K,
            ModelType.ERNIE_8K_P,
            ModelType.ERNIE_Character_8k,
            ModelType.ERNIE_Speed_128K,
            ModelType.ERNIE_Speed_8K,
            ModelType.ERNIE_8K_P,
            ModelType.DEEPSEEK_V3_1_250821
        }

    @property
    def value_for_tiktoken(self) -> str:
        if self.is_openai or self.is_qianfan:
            return self.value
        return "gpt-3.5-turbo"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            ModelType.ERNIE_8K
        }
