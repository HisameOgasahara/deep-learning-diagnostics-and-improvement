# Practice notebooks

These notebooks are executable architecture/algorithm references. The reduction rule is strict:

- **May be reduced:** tensor widths, channel sizes, state/rank widths, vocabulary size, batch size, sequence/action length, image resolution, point count, and short optimization budgets used only for sanity checks.
- **Must not be reduced or replaced:** model depth, stage/block counts, branch topology, attention schedules, architectural head/expert counts, expert top-k, routing rules, residual mechanisms, positional-encoding type, objective/loss definitions, solver equations, and training/inference algorithm paths.

Every architecture-specific notebook contains executable assertions for its important invariants. The notebooks are kept in readable, expanded Python rather than code-golfed implementations. When one notebook becomes too large to remain readable, the topic is split instead of compressing the implementation.

| Notebook | Structural checks kept |
|---|---|
| 01 | tensor layout / broadcast / reduction / gather-scatter semantics |
| 02 | core PyTorch forward-backward-optimizer path |
| 03 | GPT-2 Small 12 layers/12 heads, ViT-B 12/12, DiT-B/2 12/12 with fixed 2D sin-cos PE |
| 04 | GLU / GEGLU / SwiGLU complete two-input-projection + output-projection path |
| 05 | BatchNorm / InstanceNorm / GroupNorm / LayerNorm / RMSNorm axes and statistics |
| 06 | persistent mHC streams across 43 layers; K3 93-site Block AttnRes with block size 12 |
| 07 | RoPE Q/K rotations and 2D/3D axial extensions |
| 08 | DeepSeek-V4-Flash 43-layer attention schedule, 64 Q heads, KV=1, SWA128, CSA4, HCA128, index top-k512 |
| 09 | Kimi-K3 93 layers, 69 KDA + 24 MLA, 1 dense + 92 MoE, 896 experts/top-16/+2 shared, AttnRes12 |
| 10 | DeepSeek-V4 256 experts/top-6/+1 shared, checkpoint hash routing; K3 896/top-16/+2; V4 MTP path |
| 11 | Prodigy state chain, canonical Muon scaling, V4 hybrid Newton-Schulz, K3 96-head per-head Muon |
| 12 | ResNet bottleneck and ConvNeXt block including LayerScale and DropPath |
| 13 | Swin-T [2,2,6,2] / [3,6,12,24], resolution-aware window/shift, DropPath, FPN, CenterNet loss/decode |
| 14 | diffusion targets, Flow Matching, Rectified Flow reflow, MeanFlow JVP/stop-gradient objective |
| 15 | DPM-Solver++ analytical updates and UniPC UniP/UniC path with no extra corrector NFE |
| 16 | DiT-B/2 12/12 and SD3-Medium 24 joint blocks/24 heads; fixed 2D sin-cos PE and final context-pre-only block |
| 17 | Mamba-130M 24-block stack, conv kernel4, expansion2, low-rank timestep selective scan |
| 18 | NeRF stratified + hierarchical sampling; 3DGS degree-3 SH, anisotropic rendering, clone/split/prune/reset |
| 19 | PointNet++ hierarchy, DGCNN EdgeConv stack, Point Transformer vector attention, PointPillars PFN, CenterPoint loss/decode |
| 20 | REINFORCE/PPO/DPO/GRPO tensor objectives; GRPO completion-wise token reduction |
| 21 | SD-v1-style four-stage conditional U-Net, 1000-step base schedule, 50 stochastic DDIM transitions, CFG, DDPO PPO ratios |
| 22 | ACT ResNet18 + posterior padding mask + 4/4/7 Transformer depths; ConditionalUnet1D + 100-step DDPM |
| 23 | π0 27-layer SigLIP + 18 joint layers + GQA/RoPE/block mask/flow matching; FAST DCT/quantization/BPE round trip |
