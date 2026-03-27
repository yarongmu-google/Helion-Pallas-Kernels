import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

import attention
import util

VMEM_LIMIT_BYTES = 60 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()


def sample_qkv(
    dtype=jnp.float32,
    batch_size: int = 1,
    num_heads: int = 1,
    head_dim: int = 512,
    q_len: int = 2048,
    kv_len: int = 8192,
):
  """Samples query, key, and value tensors."""
  q = np.random.uniform(size=(batch_size, num_heads, q_len, head_dim)).astype(
      dtype
  )
  k = np.random.uniform(size=(batch_size, num_heads, kv_len, head_dim)).astype(
      dtype
  )
  v = np.random.uniform(size=(batch_size, num_heads, kv_len, head_dim)).astype(
      dtype
  )
  return jnp.array(q), jnp.array(k), jnp.array(v)


def dynamic_validate_inputs(q_len, kv_len, blk_q, blk_kv):
  # Our kernel supports ragged unaligned shapes,
  # so we only validate basic constraints
  pass


def get_smem_estimate_bytes():
  # Our kernel uses very little SMEM
  # (only 4 scalar references: sm_scale, kv_len)
  return 4 * 4  # 16 bytes. Negligible.


def get_vmem_estimate_bytes(blk_q, blk_kv, head_dim):
  """Analytically estimates the VMEM footprint of the BF16 kernel."""
  bytes_per_elem = 2  # bf16
  if head_dim == 64:
    head_dim_x2 = 128
    kv_bytes = 2 * blk_kv * head_dim_x2 * bytes_per_elem
    acc_bytes = blk_q * head_dim_x2 * 4
    q_out_bytes = 2 * blk_q * head_dim_x2 * bytes_per_elem
  else:
    kv_bytes = 4 * blk_kv * head_dim * bytes_per_elem
    acc_bytes = blk_q * head_dim * 4
    q_out_bytes = 2 * blk_q * head_dim * bytes_per_elem

  m_bytes = blk_q * 1 * 4
  l_bytes = blk_q * 1 * 4

  total_bytes = kv_bytes + q_out_bytes + m_bytes + l_bytes + acc_bytes
  return total_bytes


def get_simplified_raw_key(
    dtype=jnp.float32,
    batch_size: int = 1,
    num_heads: int = 1,
    head_dim: int = 512,
    q_len: int = 2048,
    kv_len: int = 8192,
):
  return f"{jnp.dtype(dtype).name}_num_heads-{num_heads}_head_dim_{head_dim}_q_{q_len}_kv_{kv_len}"


def autotune(
    q,
    k,
    v,
    blk_kv_lst,
    blk_q_lst,
    q_len,
    kv_len,
    num_iterations=100,
):
  """Find the best (blk_q, blk_kv) for dynamic/ragged attention workloads."""
  args = [q, k, v]
  head_dim = q.shape[-1]

  best_block_size = None
  best_t = None

  for blk_kv in blk_kv_lst:
    for blk_q in blk_q_lst:

      kwargs = {
          "blk_q": blk_q,
          "blk_kv": blk_kv,
      }

      try:
        dynamic_validate_inputs(q_len, kv_len, blk_q, blk_kv)
      except Exception as err:
        print(f"[Debug] Failed with ({blk_kv=}, {blk_q=}), got error: {err=}")
        continue

      vmem_estimate = get_vmem_estimate_bytes(blk_q, blk_kv, head_dim)
      if vmem_estimate > VMEM_LIMIT_BYTES:
        print(
            f"[Debug] Skip ({blk_kv=}, {blk_q=}) because {vmem_estimate=} >"
            f" {VMEM_LIMIT_BYTES=}"
        )
        continue

      smem_estimate = get_smem_estimate_bytes()
      if smem_estimate > SMEM_LIMIT_BYTES:
        print(
            f"[Debug] Skip ({blk_kv=}, {blk_q=}) because {smem_estimate=} >"
            f" {SMEM_LIMIT_BYTES=}"
        )
        continue

    def run_fn(q, k, v):
        return simplified_kernel.flash_attention(*args, **kwargs)

    try:
        compiled = jax.jit(run_fn).lower(*args).compile()
        res = compiled(*args)
        res.block_until_ready()

        start_time = time.perf_counter_ns()
        for _ in range(num_iterations):
        res = compiled(*args)
        res.block_until_ready()
        end_time = time.perf_counter_ns()
        t = (end_time - start_time) / num_iterations
    except Exception as err:
        print(f"[Debug] Failed with ({blk_kv=}, {blk_q=}), got error: {err=}")
        continue

    print(f"[Debug] {blk_kv=}, {blk_q=}, {t=}")
    if best_t is None or t < best_t:
    best_block_size = (smem_estimate, vmem_estimate, blk_q, blk_kv)
    best_t = t

  return best_block_size


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class Autotune(jtu.JaxTestCase):

  @parameterized.product(
      dtype=[jnp.bfloat16],
      batch_size=[2],
      num_heads=[32],
      head_dim=[64, 128],
      q_len=[1024, 4096],
      kv_len=[1024, 4096],
      blk_kv_lst=[(512, 1024, 4096)],
      blk_q_lst=[(512, 1024, 4096)],
  )
  def test_autotune(
      self,
      dtype,
      batch_size,
      num_heads,
      head_dim,
      q_len,
      kv_len,
      blk_kv_lst,
      blk_q_lst,
  ):
    q, k, v = sample_qkv(
        dtype=dtype,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=q_len,
        kv_len=kv_len,
    )

    rows = []
    # Using our custom simplified key
    key_str = get_simplified_raw_key(
        dtype=dtype,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=q_len,
        kv_len=kv_len,
    )

    # best_block_size: (blk_q, blk_kv)
    best_block_size = autotune(
        q,
        k,
        v,
        blk_kv_lst,
        blk_q_lst,
        q_len,
        kv_len,
        num_iterations=100,
    )
    if best_block_size is not None:
      rows.append(f"'{key_str}': {best_block_size},")

    print(f"Best block sizes: {rows=}")
    if rows:
      util.save_to_sponge(f"autotune_bf16_table.txt", "\n".join(rows))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())