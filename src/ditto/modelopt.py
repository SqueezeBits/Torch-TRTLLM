import torch
from transformers import PreTrainedModel
from peft import PeftModel

from tensorrt_llm.quantization import QuantAlgo

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.export.quantization_utils import get_quantization_format, to_quantized_weight

from modelopt.torch.export.model_config import QUANTIZATION_INT4_AWQ

from .literals import ModelOptQuantFormat

SUPPORTED_MODELOPT_TARGETS = [
    "nn.Linear",
]

SUPPORTED_MODELOPT_QUANTIZATION = [
    QUANTIZATION_INT4_AWQ, 
]

def get_modelopt_modules(model: PreTrainedModel | PeftModel) -> list[DynamicModule]:
    """Get modelopt defined modules from given model.

    Args:
        model (PreTrainedModel | PeftModel): The pretrained model

    Returns:
        list[DynamicModule]: List of modelopt defined modules
    """
    return [module for module in model.modules() if isinstance(module, DynamicModule)]



def infer_modelopt_quanitzation(modelopt_modules: list[DynamicModule]) -> tuple[ModelOptQuantFormat, QuantAlgo, int]:
    """Verify modelopt modules and infer quantization algorithm and block size.

    Args:
        modelopt_modules (list[DynamicModule]): Modelopt modules to infer information from

    Returns:
        tuple[ModelOptQuantFormat, QuantAlgo, int]: Inferred informations
    """

    unsupported_targets = [
        target for m in modelopt_modules 
        if (target := QuantModuleRegistry.get_key_from_dm(m)) not in SUPPORTED_MODELOPT_TARGETS
    ]
    assert len(unsupported_targets) == 0, f"Found unsupported quantization targets:\n\t {unsupported_targets}"
    
    assert (
        all(
            (not (m.input_quantizer.is_enabled or m.output_quantizer.is_enabled)) # type: ignore
            for m in modelopt_modules 
        ), "Activation quantization is not supported yet"
    ) 
                
    active_modules = [m for m in modelopt_modules if m.weight_quantizer.is_enabled] # type: ignore

    # currently, we only support INT4 WoQ without pre-quant scale
    # however, modelopt does not distinguishs it from INT4_AWQ
    quant_format: ModelOptQuantFormat = get_quantization_format(active_modules[0]) # type: ignore
    assert quant_format in SUPPORTED_MODELOPT_QUANTIZATION, f"Unsupported modelopt quantization scheme: {quant_format}"
    assert all(
        m.input_quantizer.pre_quant_scale is None and m.output_quantizer.pre_quant_scale is None # type: ignore
        for m in modelopt_modules
    ), "Pre-quantization scales are not supported"

    assert all(
        m.weight_quantizer.block_sizes == active_modules[0].weight_quantizer.block_sizes for m in active_modules # type: ignore
    ), "Quantization granurality must be identical across the model"
    block_size: int = active_modules[0].weight_quantizer.block_sizes[-1] # type: ignore
    
    return quant_format, QuantAlgo.W4A16_AWQ, block_size


def export_modelopt_weight_and_scale(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Export weight and modelopt scale tensors to proper dtype and shape.

    Args:
        weight (torch.Tensor): Unquantized weight
        scale (torch.Tensor): Quantization scale
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Exported weight and scale
    """
    original_shape = weight.shape
    reshaped_scale = scale.reshape(original_shape[0], -1)
    int4x2_qweight = to_quantized_weight(weight, reshaped_scale, "int4_awq")
    int4x2_qweight_interleaved = torch.repeat_interleave(int4x2_qweight, 2, dim=0)
    bitmask = torch.tensor([[0x0F], [0xF0]], dtype=torch.uint8).repeat(int4x2_qweight.size(0), 1)
    int4x2_qweight_interleaved = int4x2_qweight_interleaved.view(torch.uint8) & bitmask
    int4x2_qweight_interleaved[1::2] = int4x2_qweight_interleaved[1::2] >> 4
    
    return int4x2_qweight_interleaved.reshape(original_shape), reshaped_scale
