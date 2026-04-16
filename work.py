from utils import prompts
from agent import ErnieChineseAgent
from roleplay import ERNIERolePlaying, StreamingRolePlaying
from camel.messages import BaseMessage
from camel.types import RoleType
from ernie_types import ModelType
from patient_case import PatientCase
from diagnosis_step import DiagnosisStep, DiagnosisProcess, HypothesisManager, HypothesisStatus, ReviewRound
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
    支持在诊断过程中暂停以获取用户补充信息
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

        # 补充信息流程控制
        self._supplement_queue = asyncio.Queue()  # 用于暂停/恢复诊断流程的队列
        self._is_waiting_for_supplement = False   # 是否正在等待补充信息
        self._current_turn_messages: List[Dict] = []   # 当前轮次的对话记录
        self._diagnosis_paused = False  # 诊断是否已暂停
        print(f"[StreamingWork] 初始化完成，队列 ID: {id(self._supplement_queue)}")

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
        """流式执行诊断推理，使用多假设并行追踪方法"""
        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_start",
            "phase": "诊断推理"
        })

        # 创建假设管理器
        hypothesis_manager = self.patient_case.create_hypothesis_manager()

        # 获取症状信息文本
        vital_signs_text = self.patient_case.get_vital_signs_text()

        # 限制症状信息长度
        MAX_SYMPTOM_LENGTH = 800
        if len(vital_signs_text) > MAX_SYMPTOM_LENGTH:
            vital_signs_text = vital_signs_text[:MAX_SYMPTOM_LENGTH] + "\n...[症状信息过长，已截断]"

        # 阶段1: 生成假设列表
        print(f"\n{'='*60}")
        print(f"[阶段1] 生成诊断假设")
        print(f"{'='*60}\n")

        await self._generate_hypothesis_list(hypothesis_manager, vital_signs_text)

        # 检查是否有假设生成
        if not hypothesis_manager.hypotheses:
            print("[诊断错误] 未能生成任何诊断假设")
            await self.message_callback(self.client_id, {
                "type": "diagnosis_error",
                "error": "未能生成诊断假设，请提供更详细的症状信息"
            })
            return

        # 打印初始假设列表
        print(hypothesis_manager.to_string())

        # 阶段2: 逐一验证假设
        print(f"\n{'='*60}")
        print(f"[阶段2] 假设验证循环")
        print(f"{'='*60}\n")

        total_rounds = 0
        MAX_TOTAL_ROUNDS = 20  # 防止无限循环

        while hypothesis_manager.has_active_hypotheses() and total_rounds < MAX_TOTAL_ROUNDS:
            # 获取下一个待验证的假设
            current_hypothesis = hypothesis_manager.get_next_pending_hypothesis()

            # 如果没有待验证的，取审查中且置信度最高的
            if current_hypothesis is None:
                current_hypothesis = hypothesis_manager.get_best_alternative()

            # 如果还是没有，说明所有都被排除了或确认了
            if current_hypothesis is None:
                break

            # 开始审查该假设
            hypothesis_manager.start_review_hypothesis(current_hypothesis)

            # 执行一轮验证
            should_stop = await self._verify_single_hypothesis(
                hypothesis_manager,
                current_hypothesis,
                vital_signs_text,
                total_rounds
            )

            total_rounds += 1

            # 检查是否已确认某个假设
            if hypothesis_manager.has_confirmed_hypothesis():
                print(f"\n{'='*60}")
                print(f"[诊断完成] 已确认诊断: {hypothesis_manager.final_diagnosis}")
                print(f"{'='*60}\n")
                break

            # 检查是否所有假设都被排除
            if hypothesis_manager.is_all_rejected():
                print(f"\n{'='*60}")
                print(f"[诊断异常] 所有假设都被排除，需要重新生成假设")
                print(f"{'='*60}\n")

                # 请求补充信息并重新生成假设
                need_success = await self._request_more_info_for_new_hypotheses(
                    hypothesis_manager,
                    vital_signs_text
                )

                if not need_success:
                    break

                # 重新开始验证循环
                continue

        # 检查最终状态
        if not hypothesis_manager.has_confirmed_hypothesis():
            # 如果没有确认的，取置信度最高的或使用默认
            if hypothesis_manager.get_best_alternative():
                best_hypothesis = hypothesis_manager.get_best_alternative()
                hypothesis_manager.confirm_hypothesis(
                    best_hypothesis,
                    "基于现有证据的最佳选择"
                )
            else:
                print("[诊断异常] 未能确认任何诊断，使用默认诊断")
                hypothesis_manager.final_diagnosis = "待进一步检查"

        # 设置最终诊断到 PatientCase
        self.patient_case.set_diagnosis(hypothesis_manager.final_diagnosis)

        # 打印最终状态
        print(hypothesis_manager.to_string())

        # 发送完成消息
        await self.message_callback(self.client_id, {
            "type": "diagnosis_phase_complete",
            "phase": "诊断推理",
            "hypothesis_manager": hypothesis_manager.to_dict(),
            "final_diagnosis": hypothesis_manager.final_diagnosis
        })

    def _extract_need_info(self, text: str) -> Optional[str]:
        """从文本中提取 <need_info> 标签内容

        Args:
            text: 专科医生的回复文本

        Returns:
            需要补充的问题，如果没有则返回 None
        """
        import re
        match = re.search(r'<need_info>(.*?)</need_info>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    async def submit_supplementary_info(self, answer: str):
        """提交用户补充信息（由外部调用，如 WebSocket 处理器）

        Args:
            answer: 用户的补充回答
        """
        print(f"\n[StreamingWork] ===== 提交补充信息 =====")
        print(f"[StreamingWork] 当前状态:")
        print(f"  _is_waiting_for_supplement: {self._is_waiting_for_supplement}")
        print(f"[StreamingWork] 收到答案: {answer[:100]}...")

        # 将答案放入队列
        await self._supplement_queue.put(answer)
        print(f"[StreamingWork] ✓ 已将答案放入队列")
        print(f"[StreamingWork] ===== 提交补充信息完成 =====\n")

    def is_waiting_for_supplement(self) -> bool:
        """检查是否正在等待补充信息"""
        print(f"[StreamingWork] 检查是否等待补充: {self._is_waiting_for_supplement}")
        return self._is_waiting_for_supplement

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

    async def _generate_hypothesis_list(self, hypothesis_manager: HypothesisManager, vital_signs_text: str):
        """生成诊断假设列表（阶段1）

        Args:
            hypothesis_manager: 假设管理器
            vital_signs_text: 症状信息文本
        """
        message = f"""
护士给出的患者信息如下：
{vital_signs_text}

请根据上述患者信息，列出所有可能的诊断假设（3-5个），按可能性从高到低排序。

必须使用以下格式输出：
【假设列表】
1. [疾病名称1]
2. [疾病名称2]
3. [疾病名称3]
...

只输出假设列表，不要其他内容。
"""

        # 流式获取推理专家回复
        chunks = []
        async for chunk_type, chunk_content in self.zhenduantuiili.step_stream(message, "推理专家："):
            if chunk_type == "user_chunk":
                chunks.append(chunk_content)
                await self.message_callback(self.client_id, {
                    "type": "diagnosis_chunk",
                    "phase": "假设生成",
                    "role": "推理专家",
                    "content": chunk_content
                })

        full_response = "".join(chunks)
        print(f"[假设生成] 推理专家回复:\n{full_response}\n")

        # 解析假设列表
        disease_names = self._extract_hypothesis_list(full_response)

        # 创建假设
        if disease_names:
            hypothesis_manager.create_hypotheses(disease_names)
            await self.message_callback(self.client_id, {
                "type": "hypothesis_list_generated",
                "hypotheses": [{"id": f"H{i+1:03d}", "name": name, "status": "pending"} for i, name in enumerate(disease_names)]
            })
        else:
            print("[假设生成] 解析失败，使用默认假设")

    async def _verify_single_hypothesis(
        self,
        hypothesis_manager: HypothesisManager,
        hypothesis,
        vital_signs_text: str,
        round_number: int
    ) -> bool:
        """验证单个假设（阶段2中的一轮）

        Args:
            hypothesis_manager: 假设管理器
            hypothesis: 当前待验证的假设
            vital_signs_text: 症状信息
            round_number: 轮次序号

        Returns:
            bool: 是否应该停止验证流程（已确认）
        """
        print(f"\n{'='*60}")
        print(f"[假设验证] 当前: {hypothesis.disease_name} ({hypothesis.hypothesis_id})")
        print(f"{'='*60}\n")

        # 通知前端开始验证当前假设
        await self.message_callback(self.client_id, {
            "type": "hypothesis_review_start",
            "hypothesis_id": hypothesis.hypothesis_id,
            "disease_name": hypothesis.disease_name
        })

        # 构建验证消息
        message = f"""
当前正在验证的诊断假设：【{hypothesis.disease_name}】

患者症状信息：
{vital_signs_text}

请针对上述假设进行分析，输出支持点和矛盾点，并给出明确结论。

必须使用以下格式：
【当前假设】：{hypothesis.disease_name}
【支持点】：
- [具体支持证据]
【矛盾点】：
- [具体矛盾证据]
【疑问】：
- [需要补充的信息]
【我的结论】：[确认/排除/存疑]
<status>CONFIRMED/REJECTED/PENDING</status>
【理由】：[简要说明]
"""

        # 获取假设列表文本（用于专科医生参考）
        hypothesis_list_text = self._build_hypothesis_list_text(hypothesis_manager)

        # 流式执行验证
        user_chunks = []
        ass_chunks = []

        # 推理专家分析
        async for chunk_type, chunk_content in self.zhenduantuiili.step_stream(message, "推理专家："):
            if chunk_type == "user_chunk":
                user_chunks.append(chunk_content)
                await self.message_callback(self.client_id, {
                    "type": "diagnosis_chunk",
                    "phase": "假设验证",
                    "role": "推理专家",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "content": chunk_content
                })

        user_res = "".join(user_chunks)

        # 提取推理专家的结论
        expert_status = self._extract_status(user_res)
        expert_reason = self._extract_reason(user_res)

        # 专科医生审查
        specialist_message = f"""
【专科医生审查】{hypothesis.disease_name}

患者症状信息：
{vital_signs_text}

{hypothesis_list_text}

推理专家对该假设的分析：
{user_res}

请作为专科医生独立审查该假设，做出明确决策。

必须使用以下格式：
【决策】：[确认(CONFIRMED)/排除(REJECTED)/存疑(PENDING)]
【理由】：[具体说明]
【下一步】：[确认：输出治疗方案 / 排除：建议验证下一个假设 / 存疑：需要补充什么信息]
<decision>CONFIRMED/REJECTED/PENDING</decision>
"""

        async for chunk_type, chunk_content in self.zhenduantuiili.step_stream(specialist_message, "专科医生："):
            if chunk_type == "assistant_chunk":
                ass_chunks.append(chunk_content)
                await self.message_callback(self.client_id, {
                    "type": "diagnosis_chunk",
                    "phase": "假设验证",
                    "role": "专科医生",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "content": chunk_content
                })

        ass_res = "".join(ass_chunks)

        # 提取专科医生的决策
        specialist_decision = self._extract_decision(ass_res)
        specialist_reason = self._extract_reason(ass_res)

        # 创建审查记录
        review_round = ReviewRound(
            round_number=round_number,
            reviewer="推理专家",
            analysis=user_res[:500],
            conclusion=expert_status,
            status_change=f"PENDING→{expert_status}"
        )
        hypothesis_manager.add_review_round(hypothesis, review_round)

        review_round2 = ReviewRound(
            round_number=round_number,
            reviewer="专科医生",
            analysis=ass_res[:500],
            conclusion=specialist_decision,
            status_change=f"{expert_status}→{specialist_decision}"
        )
        hypothesis_manager.add_review_round(hypothesis, review_round2)

        # 根据专科医生决策更新假设状态
        should_stop = False

        if specialist_decision == "CONFIRMED":
            hypothesis_manager.confirm_hypothesis(hypothesis, specialist_reason)
            should_stop = True

        elif specialist_decision == "REJECTED":
            hypothesis_manager.reject_hypothesis(hypothesis, specialist_reason)

        elif specialist_decision == "PENDING":
            # 检查是否需要补充信息
            need_info = self._extract_need_info(ass_res)

            if need_info:
                # 请求补充信息
                supplement_success = await self._request_supplementary_info(need_info)

                if not supplement_success:
                    # 补充失败，标记为排除
                    hypothesis_manager.reject_hypothesis(hypothesis, "无法获取补充信息")
                else:
                    # 补充成功，存疑重试
                    hypothesis_manager.pend_hypothesis(hypothesis)
            else:
                # 没有需要补充的信息，存疑但重新排队
                hypothesis_manager.pend_hypothesis(hypothesis)

        # 发送本轮完成
        await self.message_callback(self.client_id, {
            "type": "hypothesis_review_complete",
            "hypothesis_id": hypothesis.hypothesis_id,
            "status": hypothesis.status.value,
            "expert_status": expert_status,
            "specialist_decision": specialist_decision
        })

        # 打印本轮结果
        print(f"[验证结果] {hypothesis.disease_name} → {hypothesis.status.value}")

        return should_stop

    async def _request_supplementary_info(self, question: str) -> bool:
        """请求用户补充信息

        Args:
            question: 需要补充的问题

        Returns:
            bool: 是否成功获取到补充信息
        """
        print(f"\n[补充信息] 请求补充")
        print(f"  问题: {question[:100]}...")

        await self.message_callback(self.client_id, {
            "type": "request_supplementary_info",
            "phase": "假设验证",
            "question": question
        })
        print(f"  ✓ 已发送 request_supplementary_info 消息")

        self._is_waiting_for_supplement = True
        self.patient_case.set_pending_question(question)

        # 等待补充信息（使用队列机制）
        print(f"  开始等待补充信息...")
        try:
            # 设置超时（60秒）
            answer = await asyncio.wait_for(self._supplement_queue.get(), timeout=60.0)
            print(f"  ✓ 收到答案: {answer[:100]}...")

            self.patient_case.add_supplementary_info(
                question=question,
                answer=answer,
                source="假设验证"
            )

            await self.message_callback(self.client_id, {
                "type": "supplementary_info_received",
                "phase": "假设验证",
                "question": question,
                "answer": answer
            })
            print(f"  ✓ 已发送 supplementary_info_received 消息")

            self._is_waiting_for_supplement = False
            self.patient_case.clear_pending_question()

            print(f"  ✓ 补充信息处理完成\n")
            return True
        except asyncio.TimeoutError:
            print(f"  ⏱️ 等待超时（60秒）")
            self._is_waiting_for_supplement = False
            return False
        except Exception as e:
            print(f"  ❌ 等待异常: {e}")
            self._is_waiting_for_supplement = False
            return False

    async def _request_more_info_for_new_hypotheses(
        self,
        hypothesis_manager: HypothesisManager,
        vital_signs_text: str
    ) -> bool:
        """当所有假设都被排除时，请求更多信息并重新生成假设

        Args:
            hypothesis_manager: 假设管理器
            vital_signs_text: 症状信息

        Returns:
            bool: 是否成功生成新假设
        """
        question = "基于当前症状，所有可能的诊断假设都被排除了。为了给出更准确的诊断，请补充以下信息：\n" + \
                   "1. 症状的详细描述（部位、性质、发作频率）\n" + \
                   "2. 既往病史（是否有类似疾病、慢性病史）\n" + \
                   "3. 用药情况（是否服用过药物）\n" + \
                   "4. 其他相关信息"

        print(f"[重新生成假设] 所有假设被排除，请求更多信息")

        supplement_success = await self._request_supplementary_info(question)

        if not supplement_success:
            return False

        # 基于新信息重新生成假设
        updated_vital_signs = self.patient_case.get_vital_signs_text()

        print(f"[重新生成假设] 基于新信息重新生成")
        await self._generate_hypothesis_list(hypothesis_manager, updated_vital_signs)

        return bool(hypothesis_manager.hypotheses)

    def _extract_hypothesis_list(self, text: str) -> List[str]:
        """从文本中提取假设列表

        Args:
            text: 包含假设列表的文本

        Returns:
            List[str]: 疾病名称列表
        """
        import re

        # 尝试提取【假设列表】后的内容
        if "【假设列表】" in text:
            list_text = text.split("【假设列表】")[1]

            # 提取所有带编号的项
            matches = re.findall(r'\d+\.\s*([^\n]+)', list_text)
            if matches:
                return [m.strip() for m in matches[:5]]  # 最多5个

        # 备用：提取所有看起来像疾病名的行
        lines = text.split('\n')
        diseases = []
        for line in lines:
            line = line.strip()
            # 排除格式标记
            if any(skip in line for skip in ['【', '】', '支持点', '矛盾点', '疑问', '理由', '结论']):
                continue
            if len(line) > 2 and len(line) < 30:
                diseases.append(line)
            if len(diseases) >= 5:
                break

        return diseases

    def _extract_status(self, text: str) -> str:
        """从文本中提取 <status> 标签

        Args:
            text: 分析文本

        Returns:
            str: CONFIRMED/REJECTED/PENDING/UNKNOWN
        """
        import re
        match = re.search(r'<status>(.*?)</status>', text, re.DOTALL | re.IGNORECASE)
        if match:
            status = match.group(1).strip().upper()
            if status in ['CONFIRMED', 'REJECTED', 'PENDING']:
                return status
        return 'UNKNOWN'

    def _extract_decision(self, text: str) -> str:
        """从文本中提取 <decision> 标签

        Args:
            text: 专科医生回复文本

        Returns:
            str: CONFIRMED/REJECTED/PENDING/UNKNOWN
        """
        import re
        match = re.search(r'<decision>(.*?)</decision>', text, re.DOTALL | re.IGNORECASE)
        if match:
            decision = match.group(1).strip().upper()
            if decision in ['CONFIRMED', 'REJECTED', 'PENDING']:
                return decision
        return 'UNKNOWN'

    def _extract_reason(self, text: str) -> str:
        """从文本中提取理由

        Args:
            text: 分析文本

        Returns:
            str: 理由文本
        """
        import re

        # 提取【理由】后的内容
        if "【理由】" in text:
            reason = text.split("【理由】")[1].split('\n')[0].strip()
            return reason[:200]

        # 备用：提取最后一句
        sentences = text.split('。')
        if sentences:
            return sentences[-1].strip()[:200]

        return text[:200]

    def _build_hypothesis_list_text(self, hypothesis_manager: HypothesisManager) -> str:
        """构建假设列表文本（用于专科医生参考）

        Args:
            hypothesis_manager: 假设管理器

        Returns:
            str: 假设列表文本
        """
        lines = ["【当前假设列表】"]

        status_icons = {
            HypothesisStatus.PENDING: "⏳",
            HypothesisStatus.UNDER_REVIEW: "🔍",
            HypothesisStatus.CONFIRMED: "✅",
            HypothesisStatus.REJECTED: "❌"
        }

        for h in hypothesis_manager.hypotheses:
            icon = status_icons.get(h.status, "?")
            lines.append(f"{icon} {h.disease_name} ({h.hypothesis_id}) - {h.status.value}")

        return "\n".join(lines)


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
