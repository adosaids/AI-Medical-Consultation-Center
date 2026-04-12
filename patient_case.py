"""患者病例模块

用于在多个agent之间共享患者信息
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from diagnosis_step import DiagnosisProcess


@dataclass
class PatientCase:
    """患者病例类

    用于在症状收集、诊断推理、治疗规划等多个模块之间共享患者信息
    """
    request: str = ""                                    # 用户的个人需求
    Vital_Signs: dict = field(default_factory=dict)      # 患者的症状信息（字典）
    Diagnosis_Process: Optional[DiagnosisProcess] = None # 诊断推理过程（详细步骤）
    Diagnosis: str = ""                                  # 最终诊断结果
    Treatment_Plan: str = ""                             # 治疗规划

    def set_request(self, request: str):
        """设置用户个人需求"""
        self.request = request

    def add_vital_sign(self, key: str, value: Any):
        """添加一个症状信息"""
        self.Vital_Signs[key] = value

    def set_vital_signs(self, signs: dict):
        """批量设置症状信息"""
        self.Vital_Signs = signs

    def get_vital_sign(self, key: str) -> Any:
        """获取指定症状信息"""
        return self.Vital_Signs.get(key)

    def create_diagnosis_process(self) -> DiagnosisProcess:
        """创建诊断推理过程"""
        self.Diagnosis_Process = DiagnosisProcess()
        return self.Diagnosis_Process

    def set_diagnosis(self, diagnosis: str):
        """设置诊断结果（同时更新Diagnosis_Process中的结果）"""
        self.Diagnosis = diagnosis
        if self.Diagnosis_Process:
            self.Diagnosis_Process.set_final_diagnosis(diagnosis)

    def set_treatment_plan(self, plan: str):
        """设置治疗规划"""
        self.Treatment_Plan = plan

    def get_vital_signs_text(self, max_length: int = 500) -> str:
        """将症状信息转换为文本格式，用于prompt

        Args:
            max_length: 每个症状值的最大长度，超过则截断
        """
        if not self.Vital_Signs:
            return "暂无症状信息"

        lines = []
        for key, value in self.Vital_Signs.items():
            # 限制每个值的长度
            value_str = str(value)
            if len(value_str) > max_length:
                value_str = value_str[:max_length] + "..."
            lines.append(f"{key}: {value_str}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转换为字典（用于前端展示或序列化）"""
        return {
            "request": self.request,
            "Vital_Signs": self.Vital_Signs,
            "Diagnosis": self.Diagnosis,
            "Treatment_Plan": self.Treatment_Plan,
            "Diagnosis_Process": self.Diagnosis_Process.to_dict() if self.Diagnosis_Process else None
        }

    def is_ready_for_diagnosis(self) -> bool:
        """检查是否准备好进行诊断推理"""
        return bool(self.Vital_Signs)

    def is_ready_for_treatment(self) -> bool:
        """检查是否准备好进行治疗规划"""
        return bool(self.Diagnosis)

    def __str__(self) -> str:
        """字符串表示"""
        lines = []
        lines.append("=" * 50)
        lines.append("患者病例")
        lines.append("=" * 50)
        lines.append(f"\n【个人需求】\n{self.request}")
        lines.append(f"\n【症状信息】\n{self.get_vital_signs_text()}")

        if self.Diagnosis:
            lines.append(f"\n【诊断结果】\n{self.Diagnosis}")
        if self.Treatment_Plan:
            lines.append(f"\n【治疗规划】\n{self.Treatment_Plan}")

        lines.append("=" * 50)
        return "\n".join(lines)
