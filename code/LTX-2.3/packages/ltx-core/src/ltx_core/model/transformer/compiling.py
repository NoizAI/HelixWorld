from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

# Defaults applied inside the patched forward. Overriding via CompilationConfig
# replaces these wholesale; it does not merge.
_DEFAULT_INDUCTOR_CONFIG: dict[str, Any] = {}
_DEFAULT_DYNAMO_CONFIG: dict[str, Any] = {"inline_inbuilt_nn_modules": True, "cache_size_limit": 256}


@dataclass(frozen=True)
class CompilationConfig:
    """``torch.compile`` configuration for transformer blocks. ``None`` keeps eager."""

    mode: str | None = None
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None
    inductor_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_INDUCTOR_CONFIG))
    dynamo_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_DYNAMO_CONFIG))


class CompiledBlockPerturbationsProcessor(BlockPerturbationsProcessor):
    """Per-block input prep for compiled blocks: mark the seq dim dynamic, then attach perturbation
    as config-independent runtime masks so the block traces ONCE.
    The ``mark_dynamic`` calls keep the per-block compile artifact shape-polymorphic; they run in
    eager mode (this processor lives outside the compiled region) on the tensors about to cross into
    the trace. Both keep-masks are then attached UNCONDITIONALLY and the skip flags pinned False, so
    the trace is identical for every pass (cond / uncond / STG): the block never sees a flipped
    Python bool (``self_attn_all_perturbed``) or a None-vs-tensor mask that Dynamo would specialise
    on, so the STG pass no longer triggers a recompile. An all-keep mask blends to a no-op
    (``out*1 + v*0``); an all-zero mask reproduces the skip. Reads only ``mask`` (runtime tensor
    indexing), never the host-side ``all_in_batch`` / ``any_in_batch``.
    """

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> TransformerArgs:
        # Positional embeddings are second-from-last regardless of rope type:
        # split rope is (B, H, T, D//2) -- dim -2 == 2; interleaved rope is (B, T, D)
        # -- dim -2 == 1. Both work via the negative index.
        torch._dynamo.mark_dynamic(args.x, 1)
        cos, sin = args.positional_embeddings
        torch._dynamo.mark_dynamic(cos, cos.ndim - 2)
        torch._dynamo.mark_dynamic(sin, sin.ndim - 2)
        if args.cross_positional_embeddings is not None:
            cross_cos, cross_sin = args.cross_positional_embeddings
            torch._dynamo.mark_dynamic(cross_cos, cross_cos.ndim - 2)
            torch._dynamo.mark_dynamic(cross_sin, cross_sin.ndim - 2)
        if args.self_attention_mask is not None:
            # Dense form is (B, 1, T, T); key-padding form (from the SP wrapper)
            # is (B, 1, 1, T) -- leave the size-1 query dim static so Dynamo
            # keeps the broadcast.
            if args.self_attention_mask.shape[2] > 1:
                torch._dynamo.mark_dynamic(args.self_attention_mask, 2)
            torch._dynamo.mark_dynamic(args.self_attention_mask, 3)
        if args.context_mask is not None:
            torch._dynamo.mark_dynamic(args.context_mask, 2)
        # `timesteps` / `embedded_timestep` are per-token when conditioning sets a
        # per-position denoise mask, in which case their dim 1 equals the seq length
        # and must vary with it. When they're a single timestep broadcast across the
        # sequence (dim 1 == 1), leaving them static lets Dynamo keep the size-1
        # broadcast.
        if args.timesteps.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.timesteps, 1)
        if args.embedded_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.embedded_timestep, 1)
        # `cross_scale_shift_timestep` is the cross-attn AdaLN scale/shift input
        # derived from the own-modality per-token timesteps (denoise_mask * sigma),
        # so its dim 1 equals the seq length when conditioning is per-token.
        # `cross_gate_timestep` is the cross-modality sigma scalar -- dim 1 is 1
        # and broadcasts, leave it static. Same guard pattern as `timesteps`.
        if args.cross_scale_shift_timestep is not None and args.cross_scale_shift_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.cross_scale_shift_timestep, 1)
        if args.video_control is not None:
            control = args.video_control
            local_control_tensors = (
                control.camera_intrinsics,
                control.camera_w2c,
                control.camera_valid_mask,
                control.action_ids,
                control.action_valid_mask,
            )
            for tensor in local_control_tensors:
                if tensor is not None and tensor.shape[1] > 1:
                    torch._dynamo.mark_dynamic(tensor, 1)
            if (
                control.camera_key_mask is not None
                and control.camera_key_mask is not control.camera_valid_mask
                and control.camera_key_mask.shape[1] > 1
            ):
                # Under sequence parallelism this is the padded global key
                # mask, whereas the matrices/query mask above are local.
                torch._dynamo.mark_dynamic(control.camera_key_mask, 1)
        # Perturbation as config-independent runtime masks (skip flags pinned False -> no recompile).
        return replace(
            args,
            self_attn_perturbation_mask=perturbations.mask(self_attn_type, block_idx),
            self_attn_all_perturbed=False,
            cross_attn_perturbation_mask=perturbations.mask(cross_attn_type, block_idx),
            cross_attn_skip_all=False,
        )


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Compile each transformer block via ``torch.compile`` with the given settings.
    The patched forward emits ``torch.compiler.cudagraph_mark_step_begin()`` once
    per step. Under CUDA-graph-enabling modes (``"reduce-overhead"`` /
    ``"max-autotune"``) this overrides Dynamo's per-invocation auto-mark
    heuristic, which would otherwise fire once per compiled block call (48 per
    forward) and treat each block call as a fresh iteration. Under other modes
    the mark is a no-op (decrements an unread counter).
    """
    model.transformer_blocks = torch.nn.ModuleList(
        torch.compile(m, mode=config.mode, backend=config.backend, fullgraph=config.fullgraph, dynamic=config.dynamic)
        for m in model.transformer_blocks
    )
    model.block_input_processor = CompiledBlockPerturbationsProcessor()

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        with (
            torch._inductor.config.patch(**config.inductor_config),
            torch._dynamo.config.patch(**config.dynamo_config),  # type: ignore[attr-defined]
        ):
            return model.forward_without_compilation(*args, **kwargs)

    model.forward_without_compilation = model.forward
    model.forward = patched_dynamo_forward
    return model


def compile_transformer_blocks_in_place(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Compile each block's bound forward without replacing registered modules.

    FSDP auto-wrapping and checkpoint serialization both key off the original
    :class:`BasicAVTransformerBlock` objects.  Assigning a compiled callable to
    ``forward`` keeps those objects and every state-dict key intact, while FSDP
    unshard/reshard hooks stay outside the compiled Tensor region.
    """

    state_keys_before = tuple(model.state_dict().keys())
    for block_index, block in enumerate(model.transformer_blocks):
        if type(block) is not BasicAVTransformerBlock:
            raise TypeError(
                "in-place block compilation requires exact BasicAVTransformerBlock modules; "
                f"block {block_index} is {type(block).__name__}"
            )
        eager_forward = block.forward
        compiled_forward = torch.compile(
            eager_forward,
            mode=config.mode,
            backend=config.backend,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
        )

        @wraps(eager_forward)
        def configured_forward(*args: object, _compiled=compiled_forward, **kwargs: object):
            with (
                torch._inductor.config.patch(**config.inductor_config),
                torch._dynamo.config.patch(**config.dynamo_config),  # type: ignore[attr-defined]
            ):
                return _compiled(*args, **kwargs)

        block.forward = configured_forward

    model.set_precompute_camera_projective_matrices(True)
    state_keys_after = tuple(model.state_dict().keys())
    if state_keys_after != state_keys_before or any("_orig_mod" in key for key in state_keys_after):
        raise RuntimeError("in-place block compilation changed transformer state-dict keys")
    return model


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    """Build a ``ModuleOps`` that compiles transformer blocks with the given settings."""
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
