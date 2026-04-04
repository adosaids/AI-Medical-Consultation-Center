from camel.societies.role_playing import RolePlaying
from typing import Optional,Dict,List
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.types import RoleType
from ernie_types import ModelType
from agent import ErnieChineseAgent
from ernie_types import TaskType
class ERNIERolePlaying():
    r"""两个代理之间的角色扮演。

参数：
assistant_role_name （str）：所扮演的角色名称
助理。
user_role_name （str）：用户扮演的角色名称。
critic_role_name （str，可选）：由
评论家。带有 ：obj：'"human"' 的角色名称会将 critic 设置为
：obj：'Human' 代理，否则将创建一个 ：obj：'CriticAgent'。
（默认 ：obj：'"critic"'）
task_prompt （str，可选）：要执行的任务的提示。
（默认 ：obj：'""'）
with_task_specify （bool， optional）：是否使用任务指定
代理。（默认 ：obj：'True'）
with_task_planner （bool， optional）：是否使用任务规划器
代理。（默认 ：obj：'False'）
with_critic_in_the_loop （bool， optional）：是否包含评论家
在循环中。（默认 ：obj：'False'）
critic_criteria （str， optional）：Critic 代理的 Critic 标准。
如果未指定，请设置条件以提高任务性能。
model （BaseModelBackend，可选）：用于的模型后端
生成响应。如果指定，它将覆盖
所有代理。（默认 ：obj：'None'）
task_type （TaskType，可选）：要执行的任务类型。
（默认 ：obj：'TaskType.AI_SOCIETY'）
assistant_agent_kwargs （Dict，可选）：要传递的其他参数
到助理代理。（默认 ：obj：'None'）
user_agent_kwargs （Dict，可选）：要传递给的其他参数
用户代理。（默认 ：obj：'None'）
task_specify_agent_kwargs （Dict，可选）：其他参数
传递给任务 指定代理。（默认 ：obj：'None'）
task_planner_agent_kwargs （Dict， optional）：其他参数
传递给 Task Planner 代理。（默认 ：obj：'None'）
critic_kwargs （Dict，可选）：要传递给
评论家。（默认 ：obj：'None'）
sys_msg_generator_kwargs （Dict， optional）：其他参数
传递给系统消息生成器。（默认 ：obj：'None'）
extend_sys_msg_meta_dicts （List[Dict]， optional）：要
扩展系统消息 Meta dicts with。（默认 ：obj：'None'）
extend_task_specify_meta_dict （Dict，可选）：用于扩展
task 指定元 dict with。（默认 ：obj：'None'）
output_language （str， optional）：由
代理。（默认 ：obj：'None'）
    """
    def __init__(
        self,
        assistant_role_name: str,
        user_role_name: str,
        assistant_api: str,
        assistant_sk: str,
        user_api: str,
        user_sk: str,
        user_prompt:str,
        assistant_prompt:str,
        user_model_type: ModelType = ModelType.ERNIE_8K,
        assistant_model_type: ModelType = ModelType.ERNIE_8K,
        task_type: TaskType = TaskType.SOCIETY,
    ) -> None:
        self.assistant_role_name,self.user_role_name = assistant_role_name,user_role_name
        self.user_model_type = user_model_type
        self.assistant_model_type = assistant_model_type
        self.user_model_api,self.user_model_sk = user_api,user_sk
        self.assistant_model_api,self.assistant_model_sk = assistant_api,assistant_sk
        self.task_type = task_type
        self.assistant_prompt = assistant_prompt
        self.user_prompt = user_prompt
        self._init_agents(assistant_prompt,user_prompt)
    def _init_agents(
            self,
            init_assistant_sys_msg: BaseMessage,
            init_user_sys_msg: BaseMessage,
            assistant_agent_kwargs: Optional[Dict] = None,
            user_agent_kwargs: Optional[Dict] = None,
            output_language: Optional[str] = None
    ):
        '''
        创建了两个agent代理
        '''
        system_message = BaseMessage(
        role_name="system",
        role_type=RoleType.ASSISTANT,
        content="你是一个智能助手，负责回答用户的问题。",
        meta_dict=[{"a":"b"}]
        )
        self.assistant_agent = ErnieChineseAgent(
            system_message=system_message,
            ernie_api_key=self.assistant_model_api,
            ernie_secret_key=self.assistant_model_sk,
            model_type=self.assistant_model_type,
            model_config=assistant_agent_kwargs
        )
        self.user_agent = ErnieChineseAgent(
            system_message=system_message,
            ernie_api_key=self.user_model_api,
            ernie_secret_key=self.user_model_sk,
            model_type=self.user_model_type,
            model_config=user_agent_kwargs
        )
        self.user_sys_message = init_user_sys_msg
        self.assistant_sys_message = init_assistant_sys_msg
    def step(self,mesg:str,user_name: str,need_memory:bool=True):
        system_message = BaseMessage(
            role_name="user",
            role_type=RoleType.ASSISTANT,
            content=mesg,
            meta_dict=[{"a":"b"}]
        )
        user_msg = self.user_agent.step(input_message=system_message,rolename=self.user_role_name,prompt=self.user_prompt,need_memory=need_memory)
        user = BaseMessage(
        role_name="system",
        role_type=RoleType.ASSISTANT,
        content=user_name+user_msg,
        meta_dict=[{"a":"b"}]
        )
        assistant_msg = self.assistant_agent.step(input_message=user,rolename=self.assistant_role_name,prompt=self.assistant_prompt,need_memory=need_memory)
        return (user_msg,assistant_msg)
