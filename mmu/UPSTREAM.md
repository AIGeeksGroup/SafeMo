# Upstream

The MMU code uses a runtime-only snapshot of the `closd` Python package from GuyTevet/CLoSD commit `d8950bdb78312fcd2844f336b3fc079e7368c87f`.

Only the DiP model, diffusion, HumanML3D evaluation, autoregressive evaluation sampler, and dependency resolver needed by the released MMU commands are included. The simulator, original training loop, standalone sampling entry point, and unrelated CLoSD components are excluded. The included upstream source files are unmodified. Keep the upstream MIT license as `CLoSD_LICENSE`.
