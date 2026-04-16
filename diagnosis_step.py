"""诊断推理记录模块

用于保存病症推理过程中的每一步推理、理由及否定理由
支持多假设并行追踪
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class HypothesisStatus(Enum):
    """假设状态枚举"""
    PENDING = "pending"           # 待验证
    UNDER_REVIEW = "under_review" # 验证中
    CONFIRMED = "confirmed"       # 已确认
    REJECTED = "rejected"         # 已排除


@dataclass
class ReviewRound:
    """单次审查轮次记录

    记录针对某个假设的一轮专家审查
    """
    round_number: int                    # 轮次序号
    reviewer: str                        # 审查者："推理专家" / "专科医生"
    analysis: str = ""                   # 分析内容
    conclusion: str = ""                 # 本轮结论
    status_change: Optional[str] = None  # 状态变化，如 "PENDING->UNDER_REVIEW"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "round_number": self.round_number,
            "reviewer": self.reviewer,
            "analysis": self.analysis,
            "conclusion": self.conclusion,
            "status_change": self.status_change,
            "timestamp": self.timestamp
        }


@dataclass
class Hypothesis:
    """诊断假设

    记录一个可能的诊断假设及其验证过程
    """
    hypothesis_id: str                      # 唯一标识
    disease_name: str                       # 疾病名称
    status: HypothesisStatus = HypothesisStatus.PENDING  # 当前状态
    confidence: int = 0                     # 置信度 0-100
    supporting_evidence: List[str] = field(default_factory=list)   # 支持证据
    conflicting_evidence: List[str] = field(default_factory=list)  # 矛盾证据
    review_rounds: List[ReviewRound] = field(default_factory=list) # 审查记录
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def start_review(self) -> None:
        """开始审查，状态从 PENDING 变为 UNDER_REVIEW"""
        if self.status == HypothesisStatus.PENDING:
            self.status = HypothesisStatus.UNDER_REVIEW

    def confirm(self, reason: str = "") -> None:
        """确认该假设为最终诊断"""
        self.status = HypothesisStatus.CONFIRMED
        self.confidence = 100
        if reason:
            self.supporting_evidence.append(f"【最终确认】{reason}")

    def reject(self, reason: str = "") -> None:
        """排除该假设"""
        self.status = HypothesisStatus.REJECTED
        self.confidence = 0
        if reason:
            self.conflicting_evidence.append(f"【排除原因】{reason}")

    def add_review_round(self, review_round: ReviewRound) -> None:
        """添加一轮审查记录"""
        self.review_rounds.append(review_round)

    def get_latest_review(self) -> Optional[ReviewRound]:
        """获取最新的审查记录"""
        return self.review_rounds[-1] if self.review_rounds else None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "hypothesis_id": self.hypothesis_id,
            "disease_name": self.disease_name,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "review_rounds": [r.to_dict() for r in self.review_rounds],
            "created_at": self.created_at
        }


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


class HypothesisManager:
    """假设管理器

    管理诊断过程中的多个假设，实现并行追踪
    """

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []           # 所有假设
        self.current_hypothesis: Optional[Hypothesis] = None  # 当前正在审查的假设
        self.final_diagnosis: str = ""                   # 最终确认的诊断
        self.confirmed_hypothesis: Optional[Hypothesis] = None  # 已确认的假设
        self.total_rounds: int = 0                       # 总审查轮数

    def create_hypotheses(self, disease_names: List[str]) -> None:
        """根据疾病名称列表创建假设

        Args:
            disease_names: 疾病名称列表
        """
        for i, name in enumerate(disease_names, 1):
            hypothesis = Hypothesis(
                hypothesis_id=f"H{i:03d}",
                disease_name=name,
                status=HypothesisStatus.PENDING
            )
            self.hypotheses.append(hypothesis)
        print(f"[HypothesisManager] 创建了 {len(self.hypotheses)} 个假设: {[h.disease_name for h in self.hypotheses]}")

    def get_pending_hypotheses(self) -> List[Hypothesis]:
        """获取所有待验证的假设"""
        return [h for h in self.hypotheses if h.status == HypothesisStatus.PENDING]

    def get_under_review_hypotheses(self) -> List[Hypothesis]:
        """获取所有审查中的假设"""
        return [h for h in self.hypotheses if h.status == HypothesisStatus.UNDER_REVIEW]

    def get_confirmed_hypothesis(self) -> Optional[Hypothesis]:
        """获取已确认的假设"""
        for h in self.hypotheses:
            if h.status == HypothesisStatus.CONFIRMED:
                return h
        return None

    def get_rejected_hypotheses(self) -> List[Hypothesis]:
        """获取所有已排除的假设"""
        return [h for h in self.hypotheses if h.status == HypothesisStatus.REJECTED]

    def get_next_pending_hypothesis(self) -> Optional[Hypothesis]:
        """获取下一个待验证的假设"""
        pending = self.get_pending_hypotheses()
        return pending[0] if pending else None

    def start_review_hypothesis(self, hypothesis: Hypothesis) -> None:
        """开始审查指定假设"""
        hypothesis.start_review()
        self.current_hypothesis = hypothesis
        print(f"[HypothesisManager] 开始审查假设: {hypothesis.disease_name}")

    def confirm_hypothesis(self, hypothesis: Hypothesis, reason: str = "") -> None:
        """确认指定假设为最终诊断"""
        hypothesis.confirm(reason)
        self.confirmed_hypothesis = hypothesis
        self.final_diagnosis = hypothesis.disease_name
        print(f"[HypothesisManager] ✅ 确认诊断: {hypothesis.disease_name}")

    def reject_hypothesis(self, hypothesis: Hypothesis, reason: str = "") -> None:
        """排除指定假设"""
        hypothesis.reject(reason)
        print(f"[HypothesisManager] ❌ 排除假设: {hypothesis.disease_name}, 原因: {reason[:50]}...")

    def pend_hypothesis(self, hypothesis: Hypothesis) -> None:
        """将假设重新置为待验证状态（存疑后重试）"""
        hypothesis.status = HypothesisStatus.PENDING
        print(f"[HypothesisManager] ⏸️ 假设存疑，重新排队: {hypothesis.disease_name}")

    def add_review_round(self, hypothesis: Hypothesis, review_round: ReviewRound) -> None:
        """为指定假设添加审查记录"""
        hypothesis.add_review_round(review_round)
        self.total_rounds += 1

    def has_confirmed_hypothesis(self) -> bool:
        """检查是否已有确认的假设"""
        return any(h.status == HypothesisStatus.CONFIRMED for h in self.hypotheses)

    def has_pending_hypotheses(self) -> bool:
        """检查是否还有待验证的假设"""
        return any(h.status == HypothesisStatus.PENDING for h in self.hypotheses)

    def has_active_hypotheses(self) -> bool:
        """检查是否还有活跃的假设（待验证或审查中）"""
        return any(h.status in [HypothesisStatus.PENDING, HypothesisStatus.UNDER_REVIEW] for h in self.hypotheses)

    def get_best_alternative(self) -> Optional[Hypothesis]:
        """获取最佳的替代假设（置信度最高的审查中或待验证假设）"""
        candidates = [h for h in self.hypotheses if h.status in [HypothesisStatus.UNDER_REVIEW, HypothesisStatus.PENDING]]
        if not candidates:
            return None
        return max(candidates, key=lambda h: h.confidence)

    def is_all_rejected(self) -> bool:
        """检查是否所有假设都被排除"""
        return len(self.hypotheses) > 0 and all(h.status == HypothesisStatus.REJECTED for h in self.hypotheses)

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "total": len(self.hypotheses),
            "pending": len(self.get_pending_hypotheses()),
            "under_review": len(self.get_under_review_hypotheses()),
            "confirmed": 1 if self.confirmed_hypothesis else 0,
            "rejected": len(self.get_rejected_hypotheses()),
            "total_rounds": self.total_rounds,
            "current": self.current_hypothesis.disease_name if self.current_hypothesis else None,
            "final_diagnosis": self.final_diagnosis
        }

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "current_hypothesis": self.current_hypothesis.to_dict() if self.current_hypothesis else None,
            "final_diagnosis": self.final_diagnosis,
            "confirmed_hypothesis": self.confirmed_hypothesis.to_dict() if self.confirmed_hypothesis else None,
            "total_rounds": self.total_rounds,
            "status_summary": self.get_status_summary()
        }

    def to_string(self) -> str:
        """转换为可读字符串"""
        lines = []
        lines.append("=" * 60)
        lines.append("诊断假设追踪")
        lines.append("=" * 60)

        status_icons = {
            HypothesisStatus.PENDING: "⏳",
            HypothesisStatus.UNDER_REVIEW: "🔍",
            HypothesisStatus.CONFIRMED: "✅",
            HypothesisStatus.REJECTED: "❌"
        }

        for h in self.hypotheses:
            icon = status_icons.get(h.status, "?")
            lines.append(f"\n{icon} 【{h.disease_name}】({h.hypothesis_id})")
            lines.append(f"   状态: {h.status.value}")
            lines.append(f"   置信度: {h.confidence}%")
            if h.supporting_evidence:
                lines.append(f"   支持: {', '.join(h.supporting_evidence[:3])}")
            if h.conflicting_evidence:
                lines.append(f"   矛盾: {', '.join(h.conflicting_evidence[:3])}")
            lines.append(f"   审查轮数: {len(h.review_rounds)}")

        lines.append("\n" + "=" * 60)
        if self.final_diagnosis:
            lines.append(f"🎯 最终诊断: {self.final_diagnosis}")
        else:
            lines.append("🎯 最终诊断: 待定")
        lines.append(f"总审查轮数: {self.total_rounds}")
        lines.append("=" * 60)

        return "\n".join(lines)