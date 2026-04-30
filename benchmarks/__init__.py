# Parallel-chain MCMC — must be set before jax/numpyro import anywhere
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
