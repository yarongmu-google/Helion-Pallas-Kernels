from collections.abc import Sequence
import time

from absl import app
import jax
from jax import numpy as jnp
import numpy as np

import attention
import example_utils
import jax_kernel


# Best tuned block sizes for Simplified Kernel: (blk_q, blk_kv)
SIMPLIFIED_TUNED_CONFIGS = {
    (1024, 1024, 64): (1024, 1024),
    (1024, 4096, 64): (1024, 4096),
    (4096, 1024, 64): (4096, 1024),
    (4096, 4096, 64): (1024, 4096),
    (1024, 1024, 128): (1024, 1024),
    (1024, 4096, 128): (1024, 4096),
    (4096, 1024, 128): (4096, 1024),
    (4096, 4096, 128): (1024, 4096),
}

# Best tuned block sizes for JAX Kernel: (blk_q, blk_k_major, blk_k, blk_b)
JAX_TUNED_CONFIGS = {
    (1024, 1024, 64): (512, 512, 128, 1),
    (1024, 4096, 64): (512, 512, 128, 1),
    (4096, 1024, 64): (512, 512, 128, 1),
    (4096, 4096, 64): (512, 512, 128, 1),
    (1024, 1024, 128): (512, 512, 128, 1),
    (1024, 4096, 128): (512, 512, 128, 1),
    # Note: Using a safe fallback for 4K/128 as (4096, 512, 512, 2) OOMs VMEM
    (4096, 1024, 128): (512, 512, 128, 1),
    (4096, 4096, 128): (512, 512, 128, 1),
}


def run_benchmark(name, batch, heads, q_len, kv_len, dim):
  print(f"\n--- Benchmarking: {name} ---")
  print(f"Shape: B={batch}, H={heads}, Q={q_len}, KV={kv_len}, D={dim}")

  # Lookup configs
  key = (q_len, kv_len, dim)
  simp_config = SIMPLIFIED_TUNED_CONFIGS.get(key, (1024, 1024))
  jax_config = JAX_TUNED_CONFIGS.get(key, (1024, 1024, 128, 1))

  q, k, v = example_utils.sample_qkv(
      batch, heads, q_len, kv_len, dim, dtype=jnp.bfloat16
  )

  sm_scale = 1.0 / (dim**0.5)

  # Total theoretical FLOPs for Forward Flash Attention
  total_flops = 4.0 * batch * heads * q_len * kv_len * dim

  # 1. Simplified Kernel
  def run_simplified():
    return simplified_kernel.flash_attention(
        q, k, v, blk_q=simp_config[0], blk_kv=simp_config[1]
    )

  lowered_s = jax.jit(run_simplified).lower()
  compiled_s = lowered_s.compile()

  # Warmup
  res_s = compiled_s()
  res_s.block_until_ready()

  num_iters = 100
  start = time.time()
  for _ in range(num_iters):
    res_s = compiled_s()
    res_s.block_until_ready()
  time_s = (time.time() - start) / num_iters

  flops_s = (
      compiled_s.cost_analysis()[0].get("flops", 0)
      if isinstance(compiled_s.cost_analysis(), list)
      else compiled_s.cost_analysis().get("flops", 0)
  )

  # 2. OSS JAX Kernel
  block_sizes = jax_kernel.BlockSizes(
      block_q=jax_config[0],
      block_k_major=jax_config[1],
      block_k=jax_config[2],
      block_b=jax_config[3],
  )

  def run_oss():
    return jax_kernel.flash_attention(
        q, k, v, sm_scale=sm_scale, block_sizes=block_sizes
    )

  lowered_o = jax.jit(run_oss).lower()
  compiled_o = lowered_o.compile()

  # Warmup
  res_o = compiled_o()
  res_o.block_until_ready()

  start = time.time()
  for _ in range(num_iters):
    res_o = compiled_o()
    res_o.block_until_ready()
  time_o = (time.time() - start) / num_iters

  flops_o = (
      compiled_o.cost_analysis()[0].get("flops", 0)
      if isinstance(compiled_o.cost_analysis(), list)
      else compiled_o.cost_analysis().get("flops", 0)
  )

  # 3. Verification
  np.testing.assert_allclose(
      res_s, res_o, atol=1e-2, err_msg=f"Output mismatch in {name}"
  )

  print(f"Simplified: {time_s:.6f}s, {flops_s/time_s/1e12:.2f} TFLOPs/s")
  print(f"OSS JAX:    {time_o:.6f}s, {flops_o/time_o/1e12:.2f} TFLOPs/s")
  print(f"Speedup:    {time_o/time_s:.2f}x")
  print("✅ Numerical Verification Passed")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # List of cases to test
  cases = [
      ("Small/Sanity", 1, 1, 128, 128, 64),
      ("Meta/Helion Use Case", 2, 32, 1024, 1024, 64),
      ("Meta/Helion Use Case (Long Seq)", 2, 32, 4096, 4096, 64),
      ("Large Seq", 1, 8, 2048, 8192, 128),
      ("Wide Head", 1, 16, 1024, 1024, 256),
      # Additional Tuned Cases
      ("Tuned 1K_4K_64", 2, 32, 1024, 4096, 64),
      ("Tuned 4K_1K_64", 2, 32, 4096, 1024, 64),
      ("Tuned 1K_1K_128", 2, 32, 1024, 1024, 128),
      ("Tuned 1K_4K_128", 2, 32, 1024, 4096, 128),
      ("Tuned 4K_1K_128", 2, 32, 4096, 1024, 128),
      ("Tuned 4K_4K_128", 2, 32, 4096, 4096, 128),
  ]

  for name, b, h, ql, kl, d in cases:
    try:
      run_benchmark(name, b, h, ql, kl, d)
    except Exception as e:
      print(f"❌ Failed case {name}: {e}")

if __name__ == "__main__":
  app.run(main)