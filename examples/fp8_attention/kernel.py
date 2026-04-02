import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

FP8_DTYPE = jnp.float8_e4m3fn


def _fp8_flash_attention_kernel(
    sm_scale_ref,
    kv_len_ref,
    q_ref,
    k_hbm,
    v_hbm,
    o_ref,
    m_scr,
    l_scr,
    acc_scr,
    *,
    blk_q,
    blk_kv,
    kv_tiles,
    q_seq_len,
    head_dim,
):
  batch_idx = pl.program_id(0)
  head_idx = pl.program_id(1)
  q_tile_idx = pl.program_id(2)

  sm_scale = sm_scale_ref[0]
  kv_len = kv_len_ref[0]

  m_scr[...] = jnp.full((blk_q, 1), -jnp.inf, jnp.float32)
  l_scr[...] = jnp.zeros((blk_q, 1), jnp.float32)
  acc_scr[...] = jnp.zeros((blk_q, head_dim), jnp.float32)

  q_fp8 = q_ref[0, 0, :, :]

  def kv_body(kv_tile_indices, k_tile_ref, v_tile_ref):
    kv_tile_idx = kv_tile_indices[0]
    k_fp8 = k_tile_ref[0, 0, :, :]
    v_fp8 = v_tile_ref[0, 0, :, :]

    m_p = m_scr[...]
    l_p = l_scr[...]
    acc_p = acc_scr[:, :]

    logits = jax.lax.dot_general(
        q_fp8,
        k_fp8,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    logits = (logits * sm_scale * 1.44269504).astype(jnp.float32)

    q_span = (q_tile_idx * blk_q) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 0
    )
    k_span = (kv_tile_idx * blk_kv) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 1
    )
    mask = (q_span < q_seq_len) & (k_span < kv_len)
    logits = jnp.where(mask, logits, -jnp.inf)

    m_curr = jnp.max(logits, axis=1)[:, None]
    m_next = jnp.maximum(m_p, m_curr)

    p_f32 = jnp.exp2(logits - m_next)
    alpha_f32 = jnp.exp2(m_p - m_next)
    l_next = l_p * alpha_f32 + jnp.sum(p_f32, axis=1)[:, None]

    p_fp8 = p_f32.astype(FP8_DTYPE)

    acc_next = acc_p * alpha_f32 + jax.lax.dot_general(
        p_fp8,
        v_fp8,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    m_scr[...] = m_next
    l_scr[...] = l_next
    acc_scr[:, :] = acc_next.astype(jnp.float32)

  pltpu.emit_pipeline(
      kv_body,
      grid=(kv_tiles,),
      in_specs=[
          pl.BlockSpec(
              (1, 1, blk_kv, head_dim),
              lambda i: (batch_idx, head_idx, i, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
          pl.BlockSpec(
              (1, 1, blk_kv, head_dim),
              lambda i: (batch_idx, head_idx, i, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
      _explicit_indices=True,
  )(k_hbm, v_hbm)

  res_f32 = acc_scr[:, :] / l_scr[...]
  o_ref[0, 0, :, :] = res_f32.astype(FP8_DTYPE)


def _fp8_flash_attention_kernel_hd64(
    sm_scale_ref,
    kv_len_ref,
    q_ref,
    kv_hbm,
    o_ref,
    m_scr,
    l_scr,
    acc_scr,
    *,
    blk_q,
    blk_kv,
    kv_tiles,
    q_seq_len,
    head_dim_x2,
):
  batch_idx = pl.program_id(0)
  head_idx = pl.program_id(1)
  q_tile_idx = pl.program_id(2)

  sm_scale = sm_scale_ref[0]
  kv_len = kv_len_ref[0]

  m_scr[...] = jnp.full((blk_q, 1), -jnp.inf, jnp.float32)
  l_scr[...] = jnp.zeros((blk_q, 1), jnp.float32)
  acc_scr[...] = jnp.zeros((blk_q, head_dim_x2), jnp.float32)

  q_fp8 = q_ref[0, 0, :, :]

  def kv_body(kv_tile_indices, kv_tile_ref):
    kv_tile_idx = kv_tile_indices[0]

    # Load KV merged in FP8
    kv_fp8 = kv_tile_ref[0, 0, :, :]

    m_p = m_scr[...]
    l_p = l_scr[...]
    acc_p = acc_scr[:, :]

    # 1st Matmul: Q (fp8) @ KV^T (fp8)
    # Because Q's last 64 elements are padded to 0, Q @ KV^T == Q @ K^T
    logits = jax.lax.dot_general(
        q_fp8,
        kv_fp8,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    logits = (logits * sm_scale * 1.44269504).astype(jnp.float32)

    q_span = (q_tile_idx * blk_q) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 0
    )
    k_span = (kv_tile_idx * blk_kv) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 1
    )
    mask = (q_span < q_seq_len) & (k_span < kv_len)
    logits = jnp.where(mask, logits, -jnp.inf)

    m_curr = jnp.max(logits, axis=1)[:, None]
    m_next = jnp.maximum(m_p, m_curr)

    p_f32 = jnp.exp2(logits - m_next)
    alpha_f32 = jnp.exp2(m_p - m_next)
    l_next = l_p * alpha_f32 + jnp.sum(p_f32, axis=1)[:, None]

    # 2nd Matmul: P (bf16/f32) @ V (fp8)
    acc_next = acc_p * alpha_f32 + jax.lax.dot_general(
        p_f32.astype(jnp.bfloat16),
        kv_fp8,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    m_scr[...] = m_next
    l_scr[...] = l_next
    acc_scr[:, :] = acc_next.astype(jnp.float32)

  pltpu.emit_pipeline(
      kv_body,
      grid=(kv_tiles,),
      in_specs=[
          pl.BlockSpec(
              (1, 1, blk_kv, head_dim_x2),
              lambda i: (batch_idx, head_idx, i, 0),
              pipeline_mode=pl.Buffered(buffer_count=2),
          ),
      ],
      _explicit_indices=True,
  )(kv_hbm)

  res_f32 = acc_scr[:, :] / l_scr[...]
  o_ref[0, 0, :, :] = res_f32.astype(FP8_DTYPE)


def flash_attention_fp8(
    q_fp8,  # [batch, heads, q_len, head_dim]
    k_fp8,  # [batch, heads, kv_len, head_dim]
    v_fp8,  # [batch, heads, kv_len, head_dim]
    blk_q=1024,
    blk_kv=1024,
):
  batch, heads, q_len, head_dim = q_fp8.shape
  _, _, kv_len, _ = k_fp8.shape

  sm_scale = jnp.array([1.0 / (head_dim**0.5)], dtype=jnp.float32)
  kv_seq_len_arr = jnp.array([kv_len], dtype=jnp.int32)
  grid = (batch, heads, (q_len + blk_q - 1) // blk_q)
  kv_tiles = (kv_len + blk_kv - 1) // blk_kv

  flops = 4 * batch * heads * q_len * kv_len * head_dim
  input_bytes = (
      math.prod(q_fp8.shape) + math.prod(k_fp8.shape) + math.prod(v_fp8.shape)
  )
  output_bytes = math.prod(q_fp8.shape)
  cost_estimate = pl.CostEstimate(
      flops=flops, bytes_accessed=input_bytes + output_bytes, transcendentals=0
  )

  if head_dim == 64:
    # Concatenate K and V to form [..., 128]
    kv_merged = jnp.concatenate([k_fp8, v_fp8], axis=-1)
    # Pad Q with 64 zeros so Q @ KV^T works mathematically (zeros cancel V)
    q_padded = jnp.pad(
        q_fp8,
        ((0, 0), (0, 0), (0, 0), (0, 64)),
        constant_values=0.0,
    )
    head_dim_x2 = 128

    out_padded = pl.pallas_call(
        kernel=functools.partial(
            _fp8_flash_attention_kernel_hd64,
            blk_q=blk_q,
            blk_kv=blk_kv,
            kv_tiles=kv_tiles,
            q_seq_len=q_len,
            head_dim_x2=head_dim_x2,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=[
                pl.BlockSpec(
                    (1, 1, blk_q, head_dim_x2),
                    lambda b, h, qt, _sm, _kv: (b, h, qt, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (1, 1, blk_q, head_dim_x2),
                lambda b, h, qt, _sm, _kv: (b, h, qt, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, head_dim_x2), jnp.float32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(
            shape=q_padded.shape,
            dtype=FP8_DTYPE,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
        ),
        cost_estimate=cost_estimate,
    )(sm_scale, kv_seq_len_arr, q_padded, kv_merged)

    # Slice out P @ V (the last 64 elements)
    return out_padded[..., 64:]

  else:
    return pl.pallas_call(
        kernel=functools.partial(
            _fp8_flash_attention_kernel,
            blk_q=blk_q,
            blk_kv=blk_kv,
            kv_tiles=kv_tiles,
            q_seq_len=q_len,
            head_dim=head_dim,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=[
                pl.BlockSpec(
                    (1, 1, blk_q, head_dim),
                    lambda b, h, qt, _sm, _kv: (b, h, qt, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (1, 1, blk_q, head_dim),
                lambda b, h, qt, _sm, _kv: (b, h, qt, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, head_dim), jnp.float32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(shape=q_fp8.shape, dtype=FP8_DTYPE),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
        ),
        cost_estimate=cost_estimate,
    )(sm_scale, kv_seq_len_arr, q_fp8, k_fp8, v_fp8)
