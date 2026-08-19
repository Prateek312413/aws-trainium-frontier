# AWS Trainium Frontier Competition: Registration & Application Guide

## 1. Hackathon Overview & Deadlines
- **Devpost Challenge URL**: [https://trainium-frontier.devpost.com/](https://trainium-frontier.devpost.com/)
- **Registration Form**: [https://app.smartsheet.com/b/form/019f19eaab4b772bb689d86d346ba558](https://app.smartsheet.com/b/form/019f19eaab4b772bb689d86d346ba558)
- **Phase 1 Cap**: First **100 teams** selected based on competition proposal quality.
- **Phase 1 Deadline**: September 30, 2026 @ 11:00pm PDT.
- **Phase 2 Top 10 Scaled Training**: October 7 – November 11, 2026.
- **NeurIPS 2026 Presentation**: Sydney, Australia (December 6–12, 2026).

---

## 2. Recommended Smartsheet Application Answers

Use these tailored responses when submitting the Smartsheet registration form:

### Project Title
> **NeuronFrontier-LM: Hardware-Co-Designed 30-Minute Speedrun LLM & NKI Custom Kernel Engine for Trainium2**

### Executive Summary & Technical Proposal (Copy & Paste)
> Our team is co-designing a hardware-native transformer architecture and custom Neuron Kernel Interface (NKI) kernels strictly tailored for the architectural nuances of Trainium2 (Trn2) silicon.
>
> **Core Innovations**:
> 1. **TensorEngine Systolic Tiling (128x128)**: All model dimensions, projections, GQA heads, and SwiGLU hidden layers are strictly quantized to 128-byte multiples to eliminate systolic array idling.
> 2. **NKI Tiled FlashAttention with SBUF Scratchpad**: A custom NKI attention forward/backward kernel operating inside Trainium2's 24MB SBUF SRAM with online softmax, reducing HBM memory transactions from \(O(N^2)\) to \(O(N)\).
> 3. **Dual Muon + AdamW Optimization**: Integrating 5th-order Newton-Schulz matrix orthogonalization (Muon) for 2D weights with decoupled AdamW for embeddings/norms, combined with QK-Norm to prevent entropy collapse under aggressive learning rates.
> 4. **Chunked SBUF Cross-Entropy Loss**: Online LogSumExp computation that projects vocabulary in 512-token chunks, completely eliminating the materialization of the 1.6GB logits tensor in HBM.
> 5. **Hardware-Aligned Sparse MoE**: Top-2 fine-grained expert routing with balanced capacity factors for maximum parameter capacity under the fixed 30-minute compute envelope.
>
> Our pipeline targets leading validation bits-per-byte (`val_bpb`) on the official leaderboard and seamless scaling to multi-node clusters in Phase 2.

### What experience do you have with AWS Trainium / Inferentia or Kernel Programming?
> Extensive experience with PyTorch, CUDA, Triton, and high-performance ML systems. We have implemented custom NKI tiled attention kernels, fused activation layers, and distributed training recipes for AWS Neuron SDK (`torch_neuronx`, `neuronx-cc`).

### AWS Account ID & GitHub Usernames
> [Insert your AWS 12-digit Account ID and GitHub Handle(s)]

---

## 3. Post-Registration Checklist
1. [ ] Submit Smartsheet registration form.
2. [ ] Join the Devpost Hackathon: [https://trainium-frontier.devpost.com/register](https://trainium-frontier.devpost.com/register).
3. [ ] Accept invitation to private AWS Annapurna Labs GitHub repository.
4. [ ] Request AWS Promotional Credits for Trainium2 (`trn2.48xlarge`) instances.
5. [ ] Clone this repository to your EC2 instance and run `python train_speedrun.py`.
