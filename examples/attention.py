import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools


def _flash_attention_kernel(
    sm_scale_ref,
    kv_len_ref,
    q_hbm,
    k_hbm,
    v_hbm,
    o_hbm,
    kv_bufs,
    sems,
    m_scr,
    l_scr,
    acc_scr,
    q_vmem,
    o_vmem,
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

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _fetch_q(qi, wait=False):
    start_q = qi * blk_q
    sz = jnp.minimum(blk_q, q_seq_len - start_q)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    _async_copy(
        q_hbm.at[batch_idx, head_idx, pl.ds(start_q, sz), :],
        q_vmem.at[pl.ds(0, sz), :],
        sems.at[2, 0],
        wait,
    )

  def _fetch_bkv(kvi, buf_idx, wait=False):
    start_kv = kvi * blk_kv
    sz = jnp.minimum(blk_kv, kv_len - start_kv)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    _async_copy(
        k_hbm.at[batch_idx, head_idx, pl.ds(start_kv, sz), :],
        kv_bufs.at[buf_idx, pl.ds(0, sz), :],
        sems.at[0, buf_idx],
        wait,
    )
    _async_copy(
        v_hbm.at[batch_idx, head_idx, pl.ds(start_kv, sz), :],
        kv_bufs.at[buf_idx + 2, pl.ds(0, sz), :],
        sems.at[1, buf_idx],
        wait,
    )

  def _send_bo(qi, wait=False):
    start_q = qi * blk_q
    sz = jnp.minimum(blk_q, q_seq_len - start_q)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    if not wait:
      res = (acc_scr[:, :] / l_scr[...]).astype(o_hbm.dtype)
      o_vmem[...] = res
    _async_copy(
        o_vmem.at[pl.ds(0, sz), :],
        o_hbm.at[batch_idx, head_idx, pl.ds(start_q, sz), :],
        sems.at[3, 0],
        wait,
    )

  _fetch_q(q_tile_idx, wait=False)
  _fetch_bkv(0, 0, wait=False)
  _fetch_q(q_tile_idx, wait=True)
  q = q_vmem[:blk_q, :]

  def body_func(kvi, _):
    buf_idx = kvi % 2
    @pl.when(kvi + 1 < kv_tiles)
    def _():
      _fetch_bkv(kvi + 1, 1 - buf_idx, wait=False)

    _fetch_bkv(kvi, buf_idx, wait=True)

    k = kv_bufs[buf_idx, :, :]
    v = kv_bufs[buf_idx + 2, :, :]
    v_span = (kvi * blk_kv) + jax.lax.broadcasted_iota(jnp.int32, v.shape, 0)
    v = jnp.where(v_span < kv_len, v, 0.0)

    m_p = m_scr[...]
    l_p = l_scr[...]
    acc_p = acc_scr[:, :]

    logits = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    logits = (logits * sm_scale).astype(jnp.float32)

    q_span_logits = (q_tile_idx * blk_q) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 0
    )
    k_span_logits = (kvi * blk_kv) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 1
    )
    mask = (q_span_logits < q_seq_len) & (k_span_logits < kv_len)
    logits = jnp.where(mask, logits, -jnp.inf)

    m_curr = jnp.max(logits, axis=1)[:, None]
    m_next = jnp.maximum(m_p, m_curr)
    p = jnp.exp(logits - m_next)
    alpha = jnp.exp(m_p - m_next)
    l_next = l_p * alpha + jnp.sum(p, axis=1)[:, None]
    acc_next = acc_p * alpha + jax.lax.dot_general(
        p.astype(v.dtype),
        v,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    m_scr[...] = m_next
    l_scr[...] = l_next
    acc_scr[:, :] = acc_next.astype(jnp.float32)
    return None

  jax.lax.fori_loop(0, kv_tiles, body_func, None)
  _send_bo(q_tile_idx, wait=False)
  _send_bo(q_tile_idx, wait=True)


def _flash_attention_kernel_hd64(
    sm_scale_ref,
    kv_len_ref,
    q_hbm,
    kv_hbm,
    o_hbm,
    kv_bufs,
    sems,
    m_scr,
    l_scr,
    acc_scr,
    q_vmem,
    o_vmem,
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

  def _async_copy(src, dst, sem, wait):
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
      cp.wait()
    else:
      cp.start()

  def _fetch_q(qi, wait=False):
    start_q = qi * blk_q
    sz = jnp.minimum(blk_q, q_seq_len - start_q)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    _async_copy(
        q_hbm.at[batch_idx, head_idx, pl.ds(start_q, sz), :],
        q_vmem.at[pl.ds(0, sz), :],
        sems.at[1, 0],
        wait,
    )

  def _fetch_bkv(kvi, buf_idx, wait=False):
    start_kv = kvi * blk_kv
    sz = jnp.minimum(blk_kv, kv_len - start_kv)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    _async_copy(
        kv_hbm.at[batch_idx, head_idx, pl.ds(start_kv, sz), :],
        kv_bufs.at[buf_idx, pl.ds(0, sz), :],
        sems.at[0, buf_idx],
        wait,
    )

  def _send_bo(qi, wait=False):
    start_q = qi * blk_q
    sz = jnp.minimum(blk_q, q_seq_len - start_q)
    sz = pl.multiple_of((sz + 7) // 8 * 8, 8)
    if not wait:
      res = (acc_scr[:, :] / l_scr[...]).astype(o_hbm.dtype)
      o_vmem[...] = res
    _async_copy(
        o_vmem.at[pl.ds(0, sz), :],
        o_hbm.at[batch_idx, head_idx, pl.ds(start_q, sz), :],
        sems.at[2, 0],
        wait,
    )

  _fetch_q(q_tile_idx, wait=False)
  _fetch_bkv(0, 0, wait=False)
  _fetch_q(q_tile_idx, wait=True)
  q = q_vmem[:blk_q, :]

  def body_func(kvi, _):
    buf_idx = kvi % 2

    @pl.when(kvi + 1 < kv_tiles)
    def _():
      _fetch_bkv(kvi + 1, 1 - buf_idx, wait=False)

    _fetch_bkv(kvi, buf_idx, wait=True)

    kv = kv_bufs[buf_idx, :, :]
    kv_span = (kvi * blk_kv) + jax.lax.broadcasted_iota(jnp.int32, kv.shape, 0)
    kv = jnp.where(kv_span < kv_len, kv, 0.0)

    m_p = m_scr[...]
    l_p = l_scr[...]
    acc_p = acc_scr[:, :]

    logits = jax.lax.dot_general(
        q, kv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    logits = (logits * sm_scale).astype(jnp.float32)

    q_span_logits = (q_tile_idx * blk_q) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 0
    )
    k_span_logits = (kvi * blk_kv) + jax.lax.broadcasted_iota(
        jnp.int32, logits.shape, 1
    )
    mask = (q_span_logits < q_seq_len) & (k_span_logits < kv_len)
    logits = jnp.where(mask, logits, -jnp.inf)

    m_curr = jnp.max(logits, axis=1)[:, None]
    m_next = jnp.maximum(m_p, m_curr)
    p = jnp.exp(logits - m_next)
    alpha = jnp.exp(m_p - m_next)
    l_next = l_p * alpha + jnp.sum(p, axis=1)[:, None]

    acc_next = acc_p * alpha + jax.lax.dot_general(
        p.astype(kv.dtype),
        kv,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    m_scr[...] = m_next
    l_scr[...] = l_next
    acc_scr[:, :] = acc_next.astype(jnp.float32)
    return None

  jax.lax.fori_loop(0, kv_tiles, body_func, None)
  _send_bo(q_tile_idx, wait=False)
  _send_bo(q_tile_idx, wait=True)


def flash_attention(
    q,  # [batch, heads, q_len, head_dim]
    k,  # [batch, heads, kv_len, head_dim]
    v,  # [batch, heads, kv_len, head_dim]
    blk_q=128,
    blk_kv=128,
):
  batch, heads, q_len, head_dim = q.shape
  _, _, kv_len, _ = k.shape

  sm_scale = jnp.array([1.0 / (head_dim**0.5)], dtype=jnp.float32)
  kv_seq_len_arr = jnp.array([kv_len], dtype=jnp.int32)
  grid = (batch, heads, (q_len + blk_q - 1) // blk_q)
  kv_tiles = (kv_len + blk_kv - 1) // blk_kv

  if head_dim == 64:
    kv_merged = jnp.concatenate([k, v], axis=-1)
    q_padded = jnp.pad(
        q, ((0, 0), (0, 0), (0, 0), (0, 64)), constant_values=0.0
    )
    head_dim_x2 = 128

    out_padded = pl.pallas_call(
        kernel=functools.partial(
            _flash_attention_kernel_hd64,
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
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM(
                    (2, blk_kv, head_dim_x2), q.dtype
                ),
                pltpu.SemaphoreType.DMA((3, 2)),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, head_dim_x2), jnp.float32),
                pltpu.VMEM((blk_q, head_dim_x2), q.dtype),
                pltpu.VMEM((blk_q, head_dim_x2), q.dtype),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(
            shape=q_padded.shape, dtype=q_padded.dtype
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
        ),
    )(sm_scale, kv_seq_len_arr, q_padded, kv_merged)

    return out_padded[..., 64:]

  else:
    return pl.pallas_call(
        kernel=functools.partial(
            _flash_attention_kernel,
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
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((4, blk_kv, head_dim), q.dtype),
                pltpu.SemaphoreType.DMA((4, 2)),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, 1), jnp.float32),
                pltpu.VMEM((blk_q, head_dim), jnp.float32),
                pltpu.VMEM((blk_q, head_dim), q.dtype),
                pltpu.VMEM((blk_q, head_dim), q.dtype),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
            disable_bounds_checks=True,
        ),
    )(sm_scale, kv_seq_len_arr, q, k, v)
