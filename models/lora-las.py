import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel, LoraLayer, PeftModel, PeftType, TaskType, get_peft_model
from peft.utils import _get_submodules
from peft.tuners.lora import is_bnb_available, is_bnb_4bit_available
from typing import Optional, Union, List
import re


# ====================== 核心非线性模块：LAS (Label-Aware Scaling) ======================
class LASModule(nn.Module):
    """
    实现论文公式(4):
        f(x) = x + LeakyReLU(x) ⊙ A · sin(ω · (x ⊙ s))
    其中 s = sigmoid(W_s c) 是由情感上下文 c 动态生成的缩放向量
    """

    def __init__(
        self,
        output_dim: int,
        label_dim: int = 768,  # 情感嵌入维度（如 Sentence-BERT 输出）
        A: float = 2e-4,
        omega: float = 1e4,
    ):
        super().__init__()
        self.A = A
        self.omega = omega
        self.proj = nn.Linear(label_dim, output_dim)  # 轻量投影：c → s
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, label_context: torch.Tensor):
        """
        x: [B, L, out_features] 或 [B, out_features]
        label_context: [B, label_dim]
        """
        B = x.size(0)
        s = torch.sigmoid(self.proj(label_context))  # [B, out_features]

        # 扩展 s 到与 x 同形
        if x.dim() == 3:
            s = s.unsqueeze(1)  # [B, 1, out_features]
        # x: [B, L, D], s: [B, 1, D] → 广播

        # LeakyReLU 激活
        leaky_x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # 正弦调制项
        sin_input = self.omega * (x * s)
        sin_term = torch.sin(sin_input)
        interference = self.A * leaky_x * sin_term
        return x + interference


# ====================== LoRA-LAS Layer ======================
class LoRA_LAS_Layer(LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_A: Optional[nn.Parameter] = None,
        lora_B: Optional[nn.Parameter] = None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        bias: str = "none",
        label_dim: int = 768,
        A: float = 2e-4,
        omega: float = 1e4,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            init_lora_weights=True,  # 保持默认初始化
            use_rslora=False,
            merge_weights=merge_weights,
        )
        if bias != "none":
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.las = LASModule(out_features, label_dim, A, omega)

    def forward(self, x: torch.Tensor, label_context: Optional[torch.Tensor] = None):
        # 原始主干输出
        base_out = torch.matmul(x, self.weight.T if self.fan_in_fan_out else self.weight)

        if self.disable_adapters or label_context is None:
            return base_out + (self.bias if self.bias is not None else 0.0)

        # LoRA 分支计算
        dropout = self.lora_dropout(x)
        lora_out = torch.matmul(dropout, self.lora_A)
        lora_out = torch.matmul(lora_out, self.lora_B)
        if self.fan_in_fan_out:
            lora_out = lora_out.T

        # 应用 LAS 非线性（关键！）
        lora_out = self.las(lora_out, label_context)  # [B, L, out_features]

        # 缩放 + 合并
        result = base_out + self.scaling * lora_out
        if self.bias is not None:
            result += self.bias

        return result


# ====================== LoRA-LAS Config ======================
class LoRA_LAS_Config(LoraConfig):
    def __init__(
        self,
        label_dim: int = 768,
        A: float = 2e-4,
        omega: float = 1e4,
        **kwargs,
    ):
        if 'bias' not in kwargs:
            kwargs['bias'] = 'none'
        super().__init__(**kwargs)
        self.label_dim = label_dim
        self.A = A
        self.omega = omega
        self.peft_type = PeftType.LORA  # 仍注册为 LORA 类型（兼容性）


# ====================== LoRA-LAS Model ======================
class LoRA_LAS_Model(LoraModel):
    def __init__(self, model, config, adapter_name):
        super(LoraModel, self).__init__()
        self.model = model
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_config[adapter_name] = config
        self._label_context = None  # 全局暂存 label_context
        self._replace_modules(adapter_name)

    def set_label_context(self, label_context: torch.Tensor):
        """在 forward 前调用，注入情感上下文"""
        self._label_context = label_context

    def _replace_modules(self, adapter_name):
        config = self.peft_config[adapter_name]
        modules_to_save = config.modules_to_save
        is_target_modules_in_base_model = False

        for name, module in self.model.named_modules():
            if name not in self.model._modules:
                continue
            if any([module_name == name for module_name in modules_to_save]):
                continue
            if any(module_match in name for module_match in config.target_modules):
                is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, name)
                if hasattr(target, "bias"):
                    bias = target.bias is not None
                else:
                    bias = False

                new_module = LoRA_LAS_Layer(
                    in_features=target.in_features,
                    out_features=target.out_features,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    fan_in_fan_out=config.fan_in_fan_out,
                    merge_weights=config.merge_weights,
                    bias="none" if not bias else ("bias" if target.bias.requires_grad else "none"),
                    label_dim=config.label_dim,
                    A=config.A,
                    omega=config.omega,
                )
                setattr(parent, target_name, new_module)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def forward(self, *args, **kwargs):
        # 注入 label_context 到所有 LoRA_LAS_Layer
        for module in self.model.modules():
            if isinstance(module, LoRA_LAS_Layer):
                # 临时 monkey patch forward
                original_forward = module.forward
                def new_forward(x):
                    return original_forward(x, label_context=self._label_context)
                module.forward = new_forward

        try:
            output = self.model(*args, **kwargs)
        finally:
            # 恢复原始 forward
            for module in self.model.modules():
                if isinstance(module, LoRA_LAS_Layer):
                    module.forward = module.__class__.forward

        return output


# ====================== 注册（可选，这里直接使用自定义模型类） ======================
def get_peft_model_lora_las(
    model,
    peft_config: LoRA_LAS_Config,
    adapter_name: str = "default",
):
    """
    替代 get_peft_model，返回 LoRA_LAS_Model
    """
    peft_model = LoRA_LAS_Model(model, peft_config, adapter_name)
    return peft_model


# ====================== 应用接口 ======================
def apply_lora_las(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Union[List[str], str]] = None,
    label_dim: int = 768,
    A: float = 2e-4,
    omega: float = 1e4,
    task_type: TaskType = TaskType.CAUSAL_LM,
    adapter_name: str = "lora_las",
):
    if target_modules is None:
        # 默认 Qwen/LLaMA 目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    config = LoRA_LAS_Config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        label_dim=label_dim,
        A=A,
        omega=omega,
        bias="none",
    )

    peft_model = get_peft_model_lora_las(model, config, adapter_name)
    peft_model.print_trainable_parameters()
    return peft_model
