from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Final, Generic

import torch
from torch import nn

from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import ModelBuilderProtocol, StateDict, StateDictLoader
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


def _check_uninitialized(model: nn.Module) -> list[str]:
    """Return the names of parameters or buffers left on the meta device."""

    names = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    names.extend(name for name, value in model.named_buffers() if value.device.type == "meta")
    return names


class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType]):
    """Build a model from one complete checkpoint or a set of checkpoint shards."""

    def __init__(
        self,
        model_class_configurator: type[ModelConfigurator[ModelType]],
        model_path: str | tuple[str, ...],
        model_sd_ops: SDOps | None = None,
        module_ops: tuple[ModuleOps, ...] = (),
        model_loader: StateDictLoader | None = None,
        registry: Registry | None = None,
    ) -> None:
        self._model_class_configurator: Final = model_class_configurator
        self._model_path = model_path
        self._model_sd_ops = model_sd_ops
        self._module_ops = module_ops
        self._model_loader = model_loader or SafetensorsModelStateDictLoader()
        self._registry = registry or DummyRegistry()

    @property
    def model_sd_ops(self) -> SDOps | None:
        return self._model_sd_ops

    @property
    def module_ops(self) -> tuple[ModuleOps, ...]:
        return self._module_ops

    @property
    def registry(self) -> Registry:
        return self._registry

    @property
    def model_path(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def checkpoint(self) -> str | tuple[str, ...]:
        return self._model_path

    @property
    def model_loader(self) -> StateDictLoader:
        return self._model_loader

    def with_sd_ops(self, sd_ops: SDOps | None) -> Self:
        clone = copy.copy(self)
        clone._model_sd_ops = sd_ops
        return clone

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._module_ops = module_ops
        return clone

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._registry = registry
        return clone

    def model_config(self) -> dict:
        return read_model_config(self._model_path, self._model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        return create_meta_model(self._model_class_configurator, config, module_ops)

    def load_sd(
        self,
        paths: list[str],
        registry: Registry,
        device: torch.device | None,
        sd_ops: SDOps | None = None,
    ) -> StateDict:
        return load_state_dict(paths, self._model_loader, registry, device, sd_ops)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        model = self.meta_model(self.model_config(), self._module_ops)
        state = load_state_dict(
            self._model_path,
            self._model_loader,
            self._registry,
            device,
            self._model_sd_ops,
        )
        values = state.sd
        if dtype is not None:
            values = {key: value.to(dtype=dtype) for key, value in values.items()}
        model.load_state_dict(values, strict=False, assign=True)

        uninitialized = _check_uninitialized(model)
        if uninitialized:
            logger.warning("Uninitialized parameters or buffers: %s", uninitialized)
            return model
        return model.to(device)
