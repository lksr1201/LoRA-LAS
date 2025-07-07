import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer, LoraModel
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils import PeftType


# ====================== 核心非线性模块：Sinter ======================
class Sinter(nn.Module):
    """
    实现公式：f(x) = x + α·ReLU(x) ⊙ sin(Ωx + φ)
    - α：缩放因子（sinter_alpha）
    - Ω：频率（sinter_omega，逐元素作用，简化为标量）
    - φ：相位偏移（sinter_phi，默认0）
    """

    def __init__(self, alpha=5e-5, omega=1e4, phi=0.0):
        super().__init__()
        self.alpha = alpha  # 对应公式中的α
        self.omega = omega  # 对应公式中的Ω（标量，逐元素相乘）
        self.phi = phi  # 对应公式中的φ（相位偏移）

    def forward(self, x: torch.Tensor):
        relu_x = torch.nn.functional.elu(x)  # ReLU预处理：提取正激活部分\
        torch
        sin_term = torch.sin(self.omega * x + self.phi)  # 正弦调制：周期非线性
        interference = self.alpha * relu_x * sin_term  # 逐元素相乘 + 缩放
        return x + interference  # 残差连接


# ====================== LoRAN Layer（替换标准LoRA Layer） ======================
class LoRANLayer(LoraLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sinter_alpha=5e-5,  # 替换原sinter_amplitude，对应公式α
            sinter_omega=1e4,  # 频率Ω
            sinter_phi=0.0,  # 相位φ
            **kwargs,
    ):
        # 强制默认无bias（如需bias可显式传入）
        if 'bias' not in kwargs:
            kwargs['bias'] = 'none'
        super().__init__(in_features, out_features, **kwargs)
        self.sinter = Sinter(sinter_alpha, sinter_omega, sinter_phi)  # 初始化非线性模块

    def forward(self, x: torch.Tensor):
        # 原始LoRA前向（无非线性）
        base_output = super().forward(x)

        # 仅对LoRA分支应用非线性变换（若启用适配器）
        if self.disable_adapters:
            return base_output

        # 提取LoRA分支输出并应用Sinter变换
        lora_output = torch.matmul(x, self.lora_B @ self.lora_A)  # LoRA核心计算
        lora_output = self.sinter(lora_output)  # 应用ReLU+正弦调制的非线性

        # 合并结果（原始权重 + 缩放后的LoRA非线性输出 + bias）
        bias = self.bias if self.bias is not None else 0.0
        return torch.matmul(x, self.weight) + self.scaling * lora_output + bias


# ====================== LoRAN Model（替换标准LoRA Model） ======================
class LoRANModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
        self.forward = self._forward  # 保持PEFT前向逻辑

    def _create_and_replace(
            self,
            lora_config,
            adapter_name,
            target,
            target_name,
            parent,
            **optional_kwargs,
    ):
        # 替换为自定义LoRANLayer
        new_module = LoRANLayer(
            in_features=target.in_features,
            out_features=target.out_features,
            sinter_alpha=lora_config.sinter_alpha,  # 传递Sinter参数
            sinter_omega=lora_config.sinter_omega,
            sinter_phi=lora_config.sinter_phi,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            fan_in_fan_out=lora_config.fan_in_fan_out,
            merge_weights=lora_config.merge_weights,
            bias=lora_config.bias,
        )
        self._replace_module(parent, target_name, new_module, target)
        return new_module


# ====================== LoRAN Config（扩展LoRA配置） ======================
class LoRANConfig(LoraConfig):
    def __init__(
            self,
            sinter_alpha=5e-5,  # 替换原sinter_amplitude，对应公式α
            sinter_omega=1e4,  # 频率Ω
            sinter_phi=0.0,  # 相位φ
            **kwargs,
    ):
        # 强制默认无bias（如需bias可显式传入）
        if 'bias' not in kwargs:
            kwargs['bias'] = 'none'
        super().__init__(**kwargs)
        self.sinter_alpha = sinter_alpha  # 暴露给用户的参数
        self.sinter_omega = sinter_omega
        self.sinter_phi = sinter_phi
        self.peft_type = PeftType.LORA  # 注册PEFT类型


# ====================== 注册LoRAN到PEFT生态 ======================
# 确保类型映射正确
PEFT_TYPE_TO_CONFIG_MAPPING["LORAN"] = LoRANConfig
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["LORAN"] = LoRANModel


def register_loran():
    """兜底注册，避免多进程等场景下的注册丢失"""
    if "LORAN" not in PEFT_TYPE_TO_CONFIG_MAPPING:
        PEFT_TYPE_TO_CONFIG_MAPPING["LORAN"] = LoRANConfig
    if "LORAN" not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING:
        MODEL_TYPE_TO_PEFT_MODEL_MAPPING["LORAN"] = LoRANModel


# ====================== 模型应用接口 ======================
def apply_loran(
        model,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        sinter_alpha=5e-5,  # Sinter的α
        sinter_omega=1e4,  # Sinter的Ω
        sinter_phi=0.0,  # Sinter的φ
        frozen_llm=False  # 是否冻结大模型主体
):
    register_loran()  # 确保注册

    # 定义目标模块（以LLAMA类模型为例，需根据实际模型调整）
    layer_num = len(model.model.layers)
    target_modules = [
        f"model.layers.{i}.{k}"
        for i in range(layer_num)
        for k in ["self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"]
    ]

    # 构建LoRAN配置
    peft_config = LoRANConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        sinter_alpha=sinter_alpha,
        sinter_omega=sinter_omega,
        sinter_phi=sinter_phi,
    )

    # 注入LoRAN
    model = get_peft_model(model, peft_config)
    print(f"LoRAF配置参数: {peft_config}")
    print("\n===== LoRAF核心参数 =====")
    print(f"低秩维度(r): {peft_config.r}")
    print(f"LoRA缩放(alpha): {peft_config.lora_alpha}")
    print(f"Dropout: {peft_config.lora_dropout}")
    print(f"Sinter-α: {peft_config.sinter_alpha}")
    print(f"Sinter-Ω: {peft_config.sinter_omega}")
    print(f"Sinter-φ: {peft_config.sinter_phi}")
    print("========================\n")

    # 冻结大模型（可选）
    if frozen_llm:
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("✅ 大模型主体已冻结，仅LoRAN参数可训练")

    model.print_trainable_parameters()  # 打印可训练参数统计
    return model