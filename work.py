from utils import prompts
from agent import ErnieChineseAgent
from roleplay import ERNIERolePlaying
from camel.messages import BaseMessage
from camel.types import RoleType
from ernie_types import ModelType
from typing import List,Dict
import os
prompts_role,prompts_phase,chatchain = prompts()
class work():
    def __init__(self,
     agent_jiekou: ErnieChineseAgent,
     zhenduantype:List,
     zhiliaotype:List):
        self.agent_jiekou = agent_jiekou
        self.zhenduantuili_list = []
        self.zhiliaoguihua_list = []
        self.zhenduantype = zhenduantype
        self.zhiliaotype = zhiliaotype
        self.zhenduantuiili,self.zhiliaoguihua = self.init_agents()

    def init_agents(self):
        a = base_two_agent(
            assistant_name="zhuankeyisheng",
            user_name="zhenduantuili",
            task_name="zhenduantuili",
            ass_model_type=self.zhenduantype[0],
            user_model_type=self.zhenduantype[1]
        )
        b = base_two_agent(
            assistant_name="lunlihegui",
            user_name="zhiliaoguihua",
            task_name="zhiliaoguihua",
            user_model_type=self.zhiliaotype[1],
            ass_model_type=self.zhiliaotype[0]
        )
        return a,b
        
    def work(self):
        memory_jiekou,_ = self.agent_jiekou.memory.get_context()
        n = 1
        while n < chatchain['zhenduantuili']['max_turn_step']:
            if n == 1:
                user_res,ass_res = self.zhenduantuiili.step("护士给出的患者信息如下：{xinxi}".format(xinxi=memory_jiekou[-1]['content']),us_name="推理专家：")
            else:
                user_res,ass_res = self.zhenduantuiili.step("回复专科医生发送的消息，消息如下：专科医生：\n"+ass_res,us_name="回复推理专家发送的消息，消息如下：推理专家：")
            print(f"信息：{memory_jiekou[-1]['content']}\n")
            print("推理专家：\n")
            print(user_res+"\n")
            print("专科医生：\n")
            print(ass_res+"\n")
            self.zhenduantuili_list.append(user_res)
            self.zhenduantuili_list.append(ass_res)
            if "<stop>" in user_res or "<stop>" in ass_res:
                break
            n+=1
        return self.zhenduantuili_list
    
    def work_guihua(self,request: str,inpu=None):
        n = 1
        while n < chatchain['zhiliaoguihua']['max_turn_step']:
            if n == 1:
#                user_res,ass_res = self.zhiliaoguihua.step("专科专家的专科建议：{jieguo}，用户需求如下{xuqiu}".format(jieguo = self.zhenduantuili_list[-1],xuqiu = request),us_name="治疗规划师：")
                user_res,ass_res = self.zhiliaoguihua.step("专科专家的专科建议：{jieguo}，用户需求如下{xuqiu}".format(jieguo = inpu,xuqiu = request),us_name="治疗规划师：")
            else:
                user_res,ass_res = self.zhiliaoguihua.step(ass_res,us_name="治疗规划师：")
 #           print(f"结果{self.zhenduantuili_list[-1]}，要求：{request}\n")
            print("治疗规划师：\n")
            print(user_res+"\n")
            print("伦理合规检测：\n")
            print(ass_res+"\n")
            self.zhiliaoguihua_list.append(user_res)
            self.zhiliaoguihua_list.append(ass_res)
            if "<stop>" in user_res or "<stop>" in ass_res:
                break
            n+=1
        return self.zhiliaoguihua_list
        

class base_one_agent():
    def __init__(self,
        system_message:BaseMessage = BaseMessage(
            role_name="system",
            role_type=RoleType.ASSISTANT,
            content="你是一个智能助手，负责回答用户的问题。",
            meta_dict=[{"a":"b"}]),
        prompt: List = prompts_role['huanzhejiekou'],
        api: str = None,
        sk: str = None,
        model_type: ModelType = ModelType.ERNIE_8K,
        role_name: str = "护士"
        ):
        # 从环境变量获取API密钥（如果未传入）
        if api is None:
            api = os.environ.get("ERNIE_API_KEY", "")
        if sk is None:
            sk = os.environ.get("ERNIE_SECRET_KEY", "")
        self.role_name = role_name
        self.system_message = system_message
        self.prompt = "".join(prompt)
        self.api = api
        self.sk = sk
        self.model_type = model_type
        self.agent = ErnieChineseAgent(
        system_message=self.system_message,
        ernie_api_key=self.api,
        ernie_secret_key=self.sk,
        model_type = self.model_type,
        )
    def step(self,message:str,prompt_task:str=None):
        msg = BaseMessage(
            role_name=self.role_name,
            role_type=RoleType.ASSISTANT,
            content=message,
            meta_dict={"huanzhejiekou":"msg"}
        )
        res = self.agent.step(input_message=msg,rolename=self.role_name,prompt=self.prompt.format(task=prompt_task))
        return res
    
class base_two_agent():
    def __init__(self,
        assistant_name: str,
        user_name: str,
        task_name: str,
        ass_api: str = None,
        ass_sk: str = None,
        user_api: str = None,
        user_sk: str = None,
        ass_model_type: ModelType = ModelType.ERNIE_8K,
        user_model_type: ModelType = ModelType.ERNIE_8K,
        ass_role_name: str = "护士",
        user_role_name: str = "医生"
        ):
        # 从环境变量获取API密钥（如果未传入）
        if ass_api is None:
            ass_api = os.environ.get("ERNIE_API_KEY", "")
        if ass_sk is None:
            ass_sk = os.environ.get("ERNIE_SECRET_KEY", "")
        if user_api is None:
            user_api = os.environ.get("ERNIE_API_KEY", "")
        if user_sk is None:
            user_sk = os.environ.get("ERNIE_SECRET_KEY", "")
        self.task_name = task_name
        self.ass_api = ass_api
        self.ass_sk = ass_sk
        self.user_api = user_api
        self.user_sk = user_sk
        self.prompt_ass = "".join(prompts_role[assistant_name]).format(task = prompts_phase[task_name]['phase_prompt'])
        self.prompt_user = "".join(prompts_role[user_name]).format(task = prompts_phase[task_name]['phase_prompt'])
        self.ass_role_name = ass_role_name
        self.user_role_name = user_role_name
        self.socity = ERNIERolePlaying(
            assistant_role_name=self.ass_role_name,
            user_role_name=self.user_role_name,
            assistant_api=ass_api,
            assistant_sk=ass_sk,
            user_api=user_api,
            user_sk=user_sk,
            user_prompt=self.prompt_user,
            assistant_prompt=self.prompt_ass,
            assistant_model_type=ass_model_type,
            user_model_type=user_model_type
        )
    def step(self,message: str,us_name:str,need_memory:bool = True):
        user_res,ass_res = self.socity.step(mesg=message,user_name=us_name,need_memory=need_memory)
        return (user_res,ass_res)