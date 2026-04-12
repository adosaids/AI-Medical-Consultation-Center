from utils import prompts
from agent import ErnieChineseAgent
from roleplay import ERNIERolePlaying, StreamingRolePlaying
from camel.messages import BaseMessage
from camel.types import RoleType
from ernie_types import ModelType
from patient_case import PatientCase
from diagnosis_step import DiagnosisStep, DiagnosisProcess
from typing import List, Dict, Optional, Callable, Awaitable
import os
import asyncio

prompts_role, prompts_phase, chatchain = prompts()


class work():
    def __init__(self,
                 agent_jiekou: ErnieChineseAgent,
                 zhenduantype: List,
                 zhiliaotype: List):
        self.agent_jiekou = agent_jiekou
        self.zhenduantuili_list = []
        self.zhiliaoguihua_list = []
        self.zhenduantype = zhenduantype
        self.zhiliaotype = zhiliaotype
        self.zhenduantuiili, self.zhiliaoguihua = self.init_agents()

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
        return a, b

    def work(self):
        memory_jiekou, _ = self.agent_jiekou.memory.get_context()
        n = 1
        while n < chatchain['zhenduantuili']['max_turn_step']:
            if n == 1:
                user_res, ass_res = self.zhenduantuiili.step(
                    "护士给出的患者信息如下：{xinxi}".format(xinxi=memory_jiekou[-1]['content']), us_name="推理专家：")
            else:
                user_res, ass_res = self.zhenduantuiili.step(
                    "回复专科医生发送的消息，消息如下：专科医生：\n" + ass_res, us_name="回复推理专家发送的消息，消息如下：推理专家：")
            print(f"信息：{memory_jiekou[-1]['content']}\n")
            print("推理专家：\n")
            print(user_res + "\n")
            print("专科医生：\n")
            print(ass_res + "\n")
            self.zhenduantuili_list.append(user_res)
            self.zhenduantuili_list.append(ass_res)
            if "<stop>" in user_res or "<stop>" in ass_res:
                break
            n += 1
        return self.zhenduantuili_list

    def work_guihua(self, request: str, inpu=None):
        n = 1
        while n < chatchain['zhiliaoguihua']['max_turn_step']:
            if n == 1:
                user_res, ass_res = self.zhiliaoguihua.step(
                    "专科专家的专科建议：{jieguo}，用户需求如下{xuqiu}".format(jieguo=inpu, xuqiu=request), us_name="治疗规划师：")
            else:
                user_res, ass_res = self.zhiliaoguihua.step(ass_res, us_name="治疗规划师：")
            print("治疗规划师：\n")
            print(user_res + "\n")
            print("伦理合规检测：\n")
            print(ass_res + "\n")
            self.zhiliaoguihua_list.append(user_res)
            self.zhiliaoguihua_list.append(ass_res)
            if "<stop>" in user_res or "<stop>" in ass_res:
                break
            n += 1
        return self.zhiliaoguihua_list


class StreamingWork:
    """支持流式输出的工作类

    串行执行诊断推理和治疗规划，使用 PatientCase 共享数据
    """

    def __init__(self,
                 agent_jiekou: ErnieChineseAgent,
                 zhenduantype: List,
                 zhiliaotype: List,
                 client_id: str,
                 message_callback: Callable[[str, dict], Awaitable[None]]):
        self.agent_jiekou = agent_jiekou
        self.zhenduantype = zhenduantype
        self.zhiliaotype = zhiliaotype
        self.client_id = client_id
        self.message_callback = message_callback
        self.zhenduantuiili, self.zhiliaoguihua = self.init_agents()
        self.patient_case: Optional[PatientCase] = None  # 当前处理的病例

    def init_agents(self):
        """初始化流式角色扮演智能体"""
        zhenduantuili = StreamingRolePlaying(
            assistant_role_name="专科医生",
            user_role_name="推理专家",
            assistant_name="zhuankeyisheng",
            user_name="zhenduantuili",
            task_name="zhenduantuili",
            ass_model_type=self.zhenduantype[0],
            user_model_type=self.zhenduantype[1]
        )
        zhiliaoguihua = StreamingRolePlaying(
            assistant_role_name="伦理合规",
            user_role_name="治疗规划师",
            assistant_name="lunlihegui",
            user_name="zhiliaoguihua",
            task_name="zhiliaoguihua",
            ass_model_type=self.zhiliaotype[0],
            user_model_type=self.zhiliaotype[1]
        )
        return zhenduantuili, zhiliaoguihua

    async def work_parallel(self, patient_case: PatientCase):
        """串行执行诊断推理和治疗规划

        Args:
            patient_case: 患者病例对象，包含症状信息和用户需求
        """
        self.patient_case = patient_case

        # 打印初始 PatientCase 状态
        print(f"\n{'='*60}")
        print(f"[PatientCase 初始状态]")
        print(f"{'-'*60}")
        print(f"个人需求: {self.patient_case.request}")
        print(f"症状信息:")
        for key, value in self.patient_case.Vital_Signs.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        # 1. 先执行诊断推理，将结果写入 patient_case
        await self._run_diagnosis_stream()

        # 打印诊断完成后的 PatientCase 状态
        print(f"\n{'='*60}")
        print(f"[PatientCase 诊断完成后]")
        print(f"{'-'*60}")
        print(f"诊断推理步骤数: {len(self.patient_case.Diagnosis_Process.steps) if self.patient_case.Diagnosis_Process else 0}")
        print(f"最终诊断: {self.patient_case.Diagnosis[:100]}..." if len(self.patient_case.Diagnosis) > 100 else f"最终诊断: {self.patient_case.Diagnosis}")
        print(f"{'='*60}\n")

        # 2. 诊断推理完成后，执行治疗规划
        await self._run_treatment_stream()

        # 打印最终 PatientCase 状态
        print(f"\n{'='*60}")
        print(f"[PatientCase 最终状态]")
        print(f"{'-'*60}")
        print(self.patient_case)
        print(f"{'='*60}\n")

    async def _run_diagnosis_stream(self):
        """流式执行诊断推理，记录每一步推理过程到 PatientCase"""
        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_start",
            "phase": "诊断推理"
        })

        # 创建诊断推理过程
        diagnosis_process = self.patient_case.create_diagnosis_process()

        # 获取症状信息文本
        vital_signs_text = self.patient_case.get_vital_signs_text()

        n = 1
        turn_messages = []

        # 限制症状信息长度，避免上下文超限
        MAX_SYMPTOM_LENGTH = 800
        if len(vital_signs_text) > MAX_SYMPTOM_LENGTH:
            vital_signs_text = vital_signs_text[:MAX_SYMPTOM_LENGTH] + "\n...[症状信息过长，已截断]"

        while n < chatchain['zhenduantuili']['max_turn_step']:
            # 创建当前步骤记录
            step = diagnosis_process.create_step(n)

            if n == 1:
                message = "护士给出的患者信息如下：{xinxi}".format(xinxi=vital_signs_text)
                us_name = "推理专家："
            else:
                # 使用上一轮的专家回复，限制长度
                last_ass_res = turn_messages[-1]['ass_res'] if turn_messages else ""
                if len(last_ass_res) > 1000:
                    last_ass_res = last_ass_res[:1000] + "\n...[内容过长，已截断]"
                message = "回复专科医生发送的消息，消息如下：专科医生：\n" + last_ass_res
                us_name = "回复推理专家发送的消息，消息如下：推理专家："

            # 流式执行一步
            user_chunks = []
            ass_chunks = []

            async for chunk_type, chunk_content in self.zhenduantuiili.step_stream(message, us_name):
                if chunk_type == "user_chunk":
                    user_chunks.append(chunk_content)
                    await self.message_callback(self.client_id, {
                        "type": "diagnosis_chunk",
                        "phase": "诊断推理",
                        "role": "推理专家",
                        "turn": n,
                        "content": chunk_content
                    })
                elif chunk_type == "assistant_chunk":
                    ass_chunks.append(chunk_content)
                    await self.message_callback(self.client_id, {
                        "type": "diagnosis_chunk",
                        "phase": "诊断推理",
                        "role": "专科医生",
                        "turn": n,
                        "content": chunk_content
                    })

            user_res = "".join(user_chunks)
            ass_res = "".join(ass_chunks)

            # 记录到步骤中
            step.hypothesis = user_res
            step.reasoning = ass_res

            turn_messages.append({"user_res": user_res, "ass_res": ass_res})

            # 打印步骤信息
            print(f"\n{'-'*50}")
            print(f"[诊断推理步骤 {n}] 更新:")
            print(f"  假设: {user_res[:100]}..." if len(user_res) > 100 else f"  假设: {user_res}")
            print(f"  理由: {ass_res[:100]}..." if len(ass_res) > 100 else f"  理由: {ass_res}")
            print(f"{'-'*50}\n")

            # 检查是否包含否定关键词（排除某个假设）
            if "排除" in user_res or "排除" in ass_res or "不是" in user_res or "不是" in ass_res:
                step.is_rejected = True
                step.rejected_reason = user_res if "排除" in user_res or "不是" in user_res else ass_res
                print(f"[诊断推理步骤 {n}] 状态: ❌ 已排除")

            # 发送本轮完成标记
            await self.message_callback(self.client_id, {
                "type": "diagnosis_turn_complete",
                "phase": "诊断推理",
                "turn": n,
                "user_res": user_res,
                "ass_res": ass_res
            })

            # 检查是否结束
            should_stop = "<stop>" in user_res or "<stop>" in ass_res

            if should_stop:
                # 最后一步设为接受
                step.is_accepted = True
                print(f"[诊断推理步骤 {n}] 状态: ✅ 已接受为最终诊断")
                print(f"[诊断推理] 诊断完成，共 {n} 个步骤\n")
                break

            n += 1

        # 设置最终诊断结果
        if turn_messages:
            final_ass_res = turn_messages[-1].get('ass_res', '')
            self.patient_case.set_diagnosis(final_ass_res)

        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_complete",
            "phase": "诊断推理",
            "turns": turn_messages,
            "diagnosis_process": diagnosis_process.to_dict()
        })

    def _build_diagnosis_context(self) -> str:
        """从 PatientCase 构建精简的诊断信息上下文

        只包含关键信息，避免上下文过长
        """
        lines = []
        lines.append("【患者症状信息】")
        lines.append(self.patient_case.get_vital_signs_text())

        # 只添加最终诊断结果（最后一条被接受的步骤）
        if self.patient_case.Diagnosis_Process:
            diagnosis_process = self.patient_case.Diagnosis_Process
            accepted_steps = diagnosis_process.get_accepted_steps()

            if accepted_steps:
                final_step = accepted_steps[-1]
                lines.append("\n【诊断结果】")
                # 只保留最终诊断的核心内容，限制长度
                diagnosis_text = final_step.reasoning[:500] if len(final_step.reasoning) > 500 else final_step.reasoning
                lines.append(diagnosis_text)

            if diagnosis_process.confidence > 0:
                lines.append(f"\n诊断置信度: {diagnosis_process.confidence:.0%}")
        else:
            # 如果没有详细过程，使用简化的诊断字段
            lines.append("\n【诊断结果】")
            diagnosis_text = self.patient_case.Diagnosis[:500] if len(self.patient_case.Diagnosis) > 500 else self.patient_case.Diagnosis
            lines.append(diagnosis_text)

        return "\n".join(lines)

    async def _run_treatment_stream(self):
        """流式执行治疗规划

        从 PatientCase 读取完整的诊断信息和用户需求，生成治疗规划
        """
        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_start",
            "phase": "治疗规划"
        })

        n = 1
        turn_messages = []

        # 从 PatientCase 获取完整的诊断上下文和用户需求
        diagnosis_context = self._build_diagnosis_context()
        request = self.patient_case.request

        # 打印诊断上下文信息
        print(f"\n{'='*60}")
        print(f"[治疗规划] 从 PatientCase 获取的诊断信息:")
        print(f"{'-'*60}")
        print(f"患者需求: {request}")
        print(f"原始诊断信息长度: {len(self.patient_case.Diagnosis)} chars")
        print(f"处理后诊断信息长度: {len(diagnosis_context)} chars")
        if self.patient_case.Diagnosis_Process:
            print(f"诊断步骤数: {len(self.patient_case.Diagnosis_Process.steps)}")
        print(f"{'='*60}\n")

        # 限制诊断上下文总长度
        MAX_DIAGNOSIS_LENGTH = 1500
        if len(diagnosis_context) > MAX_DIAGNOSIS_LENGTH:
            diagnosis_context = diagnosis_context[:MAX_DIAGNOSIS_LENGTH] + "\n...[诊断信息过长，已截断]"

        while n < chatchain['zhiliaoguihua']['max_turn_step']:
            if n == 1:
                message = "基于以下诊断信息制定治疗规划：\n\n{diagnosis}\n\n患者的个人需求：{xuqiu}".format(
                    diagnosis=diagnosis_context, xuqiu=request)
                us_name = "治疗规划师："
            else:
                last_ass_res = turn_messages[-1]['ass_res'] if turn_messages else ""
                # 限制上一轮回复长度
                if len(last_ass_res) > 1000:
                    last_ass_res = last_ass_res[:1000] + "\n...[内容过长，已截断]"
                message = last_ass_res
                us_name = "治疗规划师："

            # 流式执行一步
            user_chunks = []
            ass_chunks = []

            async for chunk_type, chunk_content in self.zhiliaoguihua.step_stream(message, us_name):
                if chunk_type == "user_chunk":
                    user_chunks.append(chunk_content)
                    await self.message_callback(self.client_id, {
                        "type": "diagnosis_chunk",
                        "phase": "治疗规划",
                        "role": "治疗规划师",
                        "turn": n,
                        "content": chunk_content
                    })
                elif chunk_type == "assistant_chunk":
                    ass_chunks.append(chunk_content)
                    await self.message_callback(self.client_id, {
                        "type": "diagnosis_chunk",
                        "phase": "治疗规划",
                        "role": "伦理合规",
                        "turn": n,
                        "content": chunk_content
                    })

            user_res = "".join(user_chunks)
            ass_res = "".join(ass_chunks)

            turn_messages.append({"user_res": user_res, "ass_res": ass_res})

            # 发送本轮完成标记
            await self.message_callback(self.client_id, {
                "type": "diagnosis_turn_complete",
                "phase": "治疗规划",
                "turn": n,
                "user_res": user_res,
                "ass_res": ass_res
            })

            if "<stop>" in user_res or "<stop>" in ass_res:
                break

            n += 1

        # 保存治疗规划结果到 PatientCase
        if turn_messages:
            treatment_plan = turn_messages[-1].get('ass_res', '')
            self.patient_case.set_treatment_plan(treatment_plan)
            print(f"\n{'='*60}")
            print(f"[治疗规划完成]")
            print(f"{'-'*60}")
            print(f"治疗规划 ({len(treatment_plan)} chars):")
            print(treatment_plan[:300] + "..." if len(treatment_plan) > 300 else treatment_plan)
            print(f"{'='*60}\n")

        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_complete",
            "phase": "治疗规划",
            "turns": turn_messages,
            "patient_case": self.patient_case.to_dict()
        })


class base_one_agent():
    def __init__(self,
                 system_message: BaseMessage = BaseMessage(
                     role_name="system",
                     role_type=RoleType.ASSISTANT,
                     content="你是一个智能助手，负责回答用户的问题。",
                     meta_dict=[{"a": "b"}]),
                 prompt: List = prompts_role['huanzhejiekou'],
                 api: str = None,
                 sk: str = None,
                 model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
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
            model_type=self.model_type,
        )

    def step(self, message: str, prompt_task: str = None):
        msg = BaseMessage(
            role_name=self.role_name,
            role_type=RoleType.ASSISTANT,
            content=message,
            meta_dict={"huanzhejiekou": "msg"}
        )
        res = self.agent.step(input_message=msg, rolename=self.role_name,
                              prompt=self.prompt.format(task=prompt_task))
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
                 ass_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
                 user_model_type: ModelType = ModelType.DEEPSEEK_V3_1_250821,
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
        import json
        # 处理 prompts_role 中的字典列表，转换为 JSON 字符串
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

        self.prompt_ass = format_prompt(prompts_role[assistant_name], prompts_phase[task_name]['phase_prompt'])
        self.prompt_user = format_prompt(prompts_role[user_name], prompts_phase[task_name]['phase_prompt'])
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

    def step(self, message: str, us_name: str, need_memory: bool = True):
        user_res, ass_res = self.socity.step(mesg=message, user_name=us_name, need_memory=need_memory)
        return (user_res, ass_res)
