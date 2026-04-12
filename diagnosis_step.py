"""诊断推理记录模块

用于保存病症推理过程中的每一步推理、理由及否定理由
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DiagnosisStep:
    """单次诊断推理步骤

    记录诊断推理过程中的每一个假设及其论证过程
    """
    step_number: int                          # 步骤序号
    hypothesis: str = ""                      # 当前假设/推理内容（推理专家提出的）
    reasoning: str = ""                       # 支持这个推理的理由（专科医生的分析）
    rejected_reason: str = ""                 # 否定/排除这个假设的理由
    is_accepted: bool = False                 # 是否被接受
    is_rejected: bool = False                 # 是否被明确排除
    evidence: List[str] = field(default_factory=list)  # 支持证据（来自Vital_Signs的key）

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "step_number": self.step_number,
            "hypothesis": self.hypothesis,
            "reasoning": self.reasoning,
            "rejected_reason": self.rejected_reason,
            "is_accepted": self.is_accepted,
            "is_rejected": self.is_rejected,
            "evidence": self.evidence
        }


class DiagnosisProcess:
    """完整的诊断推理过程

    管理整个诊断推理的所有步骤，包含完整的思考链条
    """

    def __init__(self):
        self.steps: List[DiagnosisStep] = []      # 所有推理步骤
        self.final_diagnosis: str = ""            # 最终诊断结论
        self.confidence: float = 0.0              # 置信度 (0-1)
        self.summary: str = ""                    # 诊断总结

    def add_step(self, step: DiagnosisStep):
        """添加一个推理步骤"""
        self.steps.append(step)

    def create_step(self, step_number: int) -> DiagnosisStep:
        """创建并添加一个新步骤，返回该步骤"""
        step = DiagnosisStep(step_number=step_number)
        self.add_step(step)
        return step

    def get_step(self, step_number: int) -> Optional[DiagnosisStep]:
        """获取指定序号的步骤"""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_last_step(self) -> Optional[DiagnosisStep]:
        """获取最后一个步骤"""
        if self.steps:
            return self.steps[-1]
        return None

    def set_final_diagnosis(self, diagnosis: str, confidence: float = 0.0):
        """设置最终诊断结果"""
        self.final_diagnosis = diagnosis
        self.confidence = confidence

    def set_summary(self, summary: str):
        """设置诊断总结"""
        self.summary = summary

    def get_accepted_steps(self) -> List[DiagnosisStep]:
        """获取所有被接受的步骤"""
        return [s for s in self.steps if s.is_accepted]

    def get_rejected_steps(self) -> List[DiagnosisStep]:
        """获取所有被拒绝的步骤"""
        return [s for s in self.steps if s.is_rejected]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "final_diagnosis": self.final_diagnosis,
            "confidence": self.confidence,
            "summary": self.summary,
            "total_steps": len(self.steps)
        }

    def to_string(self) -> str:
        """转换为可读字符串"""
        lines = []
        lines.append("=" * 50)
        lines.append("诊断推理过程")
        lines.append("=" * 50)

        for step in self.steps:
            lines.append(f"\n【步骤 {step.step_number}】")
            lines.append(f"假设: {step.hypothesis}")
            lines.append(f"理由: {step.reasoning}")
            if step.rejected_reason:
                lines.append(f"否定理由: {step.rejected_reason}")
            if step.is_accepted:
                lines.append("结果: ✓ 接受")
            elif step.is_rejected:
                lines.append("结果: ✗ 排除")
            else:
                lines.append("结果: 待定")

        lines.append("\n" + "=" * 50)
        lines.append(f"最终诊断: {self.final_diagnosis}")
        if self.confidence > 0:
            lines.append(f"置信度: {self.confidence:.0%}")
        if self.summary:
            lines.append(f"总结: {self.summary}")
        lines.append("=" * 50)

        return "\n".join(lines)
