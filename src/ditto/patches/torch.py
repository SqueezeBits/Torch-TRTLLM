from loguru import logger
from torch.nn.modules import Module, ModuleList

original_modulelist_getitem = ModuleList.__getitem__


def patched_modulelist_getitem(self, idx: int | slice) -> ModuleList | Module:
    # This patch is required for `torch.export` failure for transformers>=4.47.0
    # at the lines like `for decoder_layer in self.layers[': self.config.num_hidden_layers]:`.
    # For example, see https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/llama/modeling_llama.py#L896
    if (
        isinstance(idx, slice)
        and self.__class__.__name__ == "ModuleList"
        and self.__class__.__module__ == "torch.nn.modules.container"
    ):
        # The direct access of `self.__class__` at this line is causing the failure.
        # return self.__class__(list(self._modules.values())[idx])
        return ModuleList(list(self._modules.values())[idx])
    return original_modulelist_getitem(self, idx)


ModuleList.__getitem__ = patched_modulelist_getitem

logger.info("ditto patches for torch are applied!")
