from collections.abc import Sequence
import functools
import time

from absl import app
import jax
from jax import numpy as jnp
import numpy as np

import attention
import example_utils
import jax_kernel


def run_benchmark(
    name, batch, heads, q_len, kv_len, dim, blk_q=128, blk_kv=128
):
  print(f"\n--- Benchmarking: {name} ---")
  print(f"Shape: B={batch}, H={heads}, Q={q_len}, KV={kv_len}, D={dim}")

  q, k, v = example_utils.sample_qkv(
      batch, heads, q_len, kv_len, dim, dtype=jnp.bfloat16
  )

  sm_scale = 1.0 / (dim**0.5)

  # Total theoretical FLOPs for Forward Flash Attention
  # 2 * (Q @ K^T) + 2 * (P @ V)
  total_flops = 4.0 * batch * heads * q_len * kv_len * dim

  # 1. Simplified Kernel
  def run_simplified():
    return attention.flash_attention(
        q, k, v, blk_q=blk_q, blk_kv=blk_kv
    )

  lowered_s = jax.jit(run_simplified).lower()
  compiled_s = lowered_s.compile()

  start = time.time()
  res_s = compiled_s()
  res_s.block_until_ready()
  time_s = time.time() - start

  flops_s = (
      compiled_s.cost_analysis()[0].get("flops", 0)
      if isinstance(compiled_s.cost_analysis(), list)
      else compiled_s.cost_analysis().get("flops", 0)
  )

  # 2. OSS JAX Kernel
  block_sizes = jax_kernel.BlockSizes(
      block_q=blk_q, block_k_major=blk_kv, block_k=128, block_b=1
  )

  def run_oss():
    return jax_kernel.flash_attention(
        q, k, v, sm_scale=sm_scale, block_sizes=block_sizes
    )

  lowered_o = jax.jit(run_oss).lower()
  compiled_o = lowered_o.compile()

  start = time.time()
  res_o = compiled_o()
  res_o.block_until_ready()
  time_o = time.time() - start
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
  ]

  for name, b, h, ql, kl, d in cases:
    try:
      run_benchmark(name, b, h, ql, kl, d)
    except Exception as e:
      print(f"❌ Failed case {name}: {e}")

  # Finally, try Pacchetto on the simplified kernel for the Helion case
  print("\n--- Attempting Pacchetto for Simplified Kernel ---")
  q, k, v = example_utils.sample_qkv(2, 32, 1024, 1024, 128, dtype=jnp.bfloat16)

  from google3.platforms.xla.tools import pacchetto as pc

  # Use named_scope so pacchetto can find it
  func = jax.named_scope("simplified_kernel_pacchetto")(
      lambda q, k, v: attention.flash_attention(
          q, k, v, blk_q=128, blk_kv=128
      )
  )

  try:
    bundles = pc.get_bundles(
        func,
        hlo_pattern=".*",
        enable_trace=True,
    )(q, k, v)
    print("Got bundles. Parsing POST_RA...")
    parsed_bundle = bundles.get_parsed_bundles(pc.BundleType.POST_RA)
    print("Parsed. Dumping to sponge...")
    parsed_bundle.dump_to_sponge(name="pacchetto-simplified-flash-attn")
    print("✅ Successfully dumped Pacchetto trace to Sponge!")
  except Exception as e:
    print(f"❌ Pacchetto failed: {e}")


if __name__ == "__main__":
  app.run(main)
