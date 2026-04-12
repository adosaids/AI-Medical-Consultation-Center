from camel.societies.role_playing import RolePlaying
from typing import Optional, Dict, List, AsyncGenerator
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.types import RoleType
from ernie_types import ModelType
from agent import ErnieChineseAgent
from ernie_types import TaskType


class ERNIERolePlaying:
    r"""两个代理之间的角色扮演。

    参数：
    assistant_role_name （str）：所扮演的角色名称
    助理。
    user_role_name （str）：用户扮演的角色名称。
    critic_role_name （str，可选）：由
    评论家。带有 ：obj：'"human"' 的角色名称会将 critic 设置为
    ：obj：'Human' 代理，否则将创建一个 ：obj：'CriticAgent'。
    （默认 ：obj：'"critic"'）
    task_prompt （str， optional）：要执行的任务的提示。
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
            user_prompt: str,
            assistant_prompt: str,
            user_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
            assistant_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
            task_type: TaskType = TaskType.SOCIETY,
    ) -> None:
        self.assistant_role_name, self.user_role_name = assistant_role_name, user_role_name
        self.user_model_type = user_model_type
        self.assistant_model_type = assistant_model_type
        self.user_model_api, self.user_model_sk = user_api, user_sk
        self.assistant_model_api, self.assistant_model_sk = assistant_api, assistant_sk
        self.task_type = task_type
        self.assistant_prompt = assistant_prompt
        self.user_prompt = user_prompt
        self._init_agents(assistant_prompt, user_prompt)

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
            meta_dict=[{"a": "b"}]
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

    def step(self, mesg: str, user_name: str, need_memory: bool = True):
        system_message = BaseMessage(
            role_name="user",
            role_type=RoleType.ASSISTANT,
            content=mesg,
            meta_dict=[{"a": "b"}]
        )
        user_msg = self.user_agent.step(input_message=system_message, rolename=self.user_role_name,
                                        prompt=self.user_prompt, need_memory=need_memory)
        user = BaseMessage(
            role_name="system",
            role_type=RoleType.ASSISTANT,
            content=user_name + user_msg,
            meta_dict=[{"a": "b"}]
        )
        assistant_msg = self.assistant_agent.step(input_message=user, rolename=self.assistant_role_name,
                                                  prompt=self.assistant_prompt, need_memory=need_memory)
        return (user_msg, assistant_msg)


class StreamingRolePlaying:
    r"""支持流式输出的角色扮演类

    与 ERNIERolePlaying 类似，但支持异步流式输出
    """

    def __init__(
            self,
            assistant_role_name: str,
            user_role_name: str,
            assistant_name: str,
            user_name: str,
            task_name: str,
            assistant_api: str = None,
            assistant_sk: str = None,
            user_api: str = None,
            user_sk: str = None,
            user_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
            assistant_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
            ass_model_type: ModelType = None,  # 别名兼容性
            task_type: TaskType = TaskType.SOCIETY,
    ) -> None:
        from utils import prompts
        prompts_role, prompts_phase, _ = prompts()

        self.assistant_role_name = assistant_role_name
        self.user_role_name = user_role_name
        self.task_name = task_name

        # 处理参数别名
        if ass_model_type is not None:
            assistant_model_type = ass_model_type

        # 从环境变量获取API密钥
        import os
        if assistant_api is None:
            assistant_api = os.environ.get("ERNIE_API_KEY", "")
        if assistant_sk is None:
            assistant_sk = os.environ.get("ERNIE_SECRET_KEY", "")
        if user_api is None:
            user_api = os.environ.get("ERNIE_API_KEY", "")
        if user_sk is None:
            user_sk = os.environ.get("ERNIE_SECRET_KEY", "")

        self.user_model_type = user_model_type
        self.assistant_model_type = assistant_model_type
        self.user_model_api = user_api
        self.user_model_sk = user_sk
        self.assistant_model_api = assistant_api
        self.assistant_model_sk = assistant_sk

        # 构建提示词
        import json

        def format_prompt(prompt_list, task_value):
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                if isinstance(prompt_list[0], dict):
                    # 将 task 插入到字典中，然后转 JSON
                    prompt_dict = prompt_list[0].copy()
                    prompt_dict['task'] = task_value
                    return json.dumps(prompt_dict, ensure_ascii=False, indent=2)
                else:
                    # 字符串列表，使用 format 替换 {task}
                    return "".join(prompt_list).format(task=task_value)
            return str(prompt_list)

        task_prompt = prompts_phase[task_name]['phase_prompt']
        self.assistant_prompt = format_prompt(prompts_role[assistant_name], task_prompt)
        self.user_prompt = format_prompt(prompts_role[user_name], task_prompt)

        self._init_agents()

    def _init_agents(self):
        '''初始化两个流式agent代理'''
        system_message = BaseMessage(
            role_name="system",
            role_type=RoleType.ASSISTANT,
            content="你是一个智能助手，负责回答用户的问题。",
            meta_dict=[{"a": "b"}]
        )

        self.assistant_agent = ErnieChineseAgent(
            system_message=system_message,
            ernie_api_key=self.assistant_model_api,
            ernie_secret_key=self.assistant_model_sk,
            model_type=self.assistant_model_type,
        )
        self.user_agent = ErnieChineseAgent(
            system_message=system_message,
            ernie_api_key=self.user_model_api,
            ernie_secret_key=self.user_model_sk,
            model_type=self.user_model_type,
        )

    async def step_stream(self, mesg: str, user_name: str, need_memory: bool = True) -> AsyncGenerator[
        tuple[str, str], None]:
        """流式执行一步角色扮演

        首先执行用户智能体，然后执行助手智能体，每个都流式返回结果

        Yields:
            ("user_chunk", content) 或 ("assistant_chunk", content)
        """
        # 用户智能体输入
        user_input = BaseMessage(
            role_name="user",
            role_type=RoleType.ASSISTANT,
            content=mesg,
            meta_dict=[{"a": "b"}]
        )

        # 流式执行用户智能体
        user_full_response = []
        async for chunk in self.user_agent.step_stream(
                input_message=user_input,
                rolename=self.user_role_name,
                prompt=self.user_prompt,
                need_memory=need_memory,
                agent_role=self.user_role_name  # 传入角色名用于日志
        ):
            user_full_response.append(chunk)
            yield ("user_chunk", chunk)

        user_msg = "".join(user_full_response)

        # 助手智能体输入（使用用户智能体的输出）
        assistant_input = BaseMessage(
            role_name="system",
            role_type=RoleType.ASSISTANT,
            content=user_name + user_msg,
            meta_dict=[{"a": "b"}]
        )

        # 流式执行助手智能体
        assistant_full_response = []
        async for chunk in self.assistant_agent.step_stream(
                input_message=assistant_input,
                rolename=self.assistant_role_name,
                prompt=self.assistant_prompt,
                need_memory=need_memory,
                agent_role=self.assistant_role_name  # 传入角色名用于日志
        ):
            assistant_full_response.append(chunk)
            yield ("assistant_chunk", chunk)
