1. Large Language Models & Reasoning (LLMs)

The "Reasoning Era" began in late 2024, focusing on how models can "think" before they speak.

1. **DeepSeek-R1** (Shao et al., 2024): Introduced **Group Relative Policy Optimization (GRPO)** , proving that LLMs can achieve OpenAI-level reasoning using pure Reinforcement Learning without needing a separate reward model.
2. **The Llama 3 Herd of Models** (Grattafiori et al., 2024): The definitive technical report on Meta’s Llama 3, detailing how to scale open-weights models to 405B parameters with multimodal capabilities.
3. **Kimi K1.5: Scaling Reinforcement Learning with LLMs** (Kimi Team, 2025): Explores scaling compute during training through RL to overcome the shortage of high-quality human text data.
4. **Gemma: Open Models Based on Gemini Research** (Mesnard et al., 2024): Google’s release of high-performance small models (2B and 7B), establishing new benchmarks for efficiency.
5. **Phi-3 Technical Report** (Abdin et al., 2024): Microsoft's "Small Language Model" breakthrough, showing that high-quality "textbook-style" data allows a 3.8B model to rival models 20x its size.
6. **Qwen2 Technical Report** (Yang et al., 2024): Details Alibaba’s world-leading open-source models, emphasizing Mixture-of-Experts (MoE) efficiency.
7. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** (Zelikman et al., 2024): Proposes a way for LLMs to generate "internal rationale" for every token they produce.
8. **Direct Language Model Alignment from Online AI Feedback** (Guo et al., 2024): A move beyond RLHF toward **RLAIF** (Reinforcement Learning from AI Feedback), automating the alignment process.
9. **BitNet b1.58: One-bit LLMs** (Ma et al., 2024): Proposes a 1-bit LLM where weights are ternary **$\{-1, 0, 1\}$**, drastically reducing energy and memory costs.
10. **Chain-of-Thought Empowers LLMs to Solve Mathematical Problems** (SBSC Team, 2025): A specialized look at improving Olympiad-level math through step-by-step coding.

---

## 2. Vision & Multimodal Learning

Models are now moving from static image generation to high-fidelity, consistent video and spatial reasoning.

1. **SAM 2: Segment Anything in Images and Videos** (Ravi et al., 2025): Meta’s update to the Segment Anything Model, adding real-time object tracking in video through a "memory bank" architecture.
2. **Visual Autoregressive Modeling (VAR)** (Tian et al., 2024): A NeurIPS award-winner that treats image generation like "next-token" prediction, making vision models scale as predictably as LLMs.
3. **Sora: Video Generation Models as World Simulators** (OpenAI, 2024): Though a technical report, it detailed the "Diffusion Transformer" (DiT) architecture that has since become the standard for video.
4. **DeepSeek-VL: Towards Real-World Vision-Language Understanding** (Lu et al., 2024): An architecture optimized for mobile and real-world visual tasks rather than just captions.
5. **Vision Transformers Need Registers** (Darcet et al., 2024): Identifies "artifacts" in ViT feature maps and fixes them with register tokens—essential for high-accuracy object detection.
6. **Pyramidal Flow Matching for Video** (ICLR, 2025): Introduces a more efficient way to generate video frames by matching "flows" at different resolutions.
7. **Inference Optimal VLMs Need Fewer Visual Tokens** (DeepMind, 2025): Proposes methods to reduce the computational cost of "looking" at images by 50% without losing accuracy.
8. **V-JEPA: Video Joint-Embedding Predictive Architecture** (Meta, 2024): LeCun’s vision for world models that learn by predicting missing parts of a video rather than pixels.
9. **Oryx MLLM: On-Demand Spatial-Temporal Understanding** (2025): A multimodal model that can process video at arbitrary resolutions for high-precision spatial tasks.
10. **4Real: Photorealistic 4D Scene Generation** (NeurIPS, 2024): Combines video diffusion with 3D Gaussian Splatting to create interactive 4D environments.

---

## 3. Novel Architectures & Science

2024-2025 has seen the first real challenges to the "Transformer" monopoly.

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2024): The paper that launched the **SSM** (State Space Model) revolution, offering Transformer-level performance with much faster inference.
2. **AlphaFold 3** (Abramson et al., 2024): Can predict structures for almost all life molecules (DNA, RNA, ligands), not just proteins. This helped secure the **2024 Nobel Prize** .
3. **KAN: Kolmogorov-Arnold Networks** (Liu et al., 2024): Proposes a total alternative to Multi-Layer Perceptrons (MLPs) where weights are functions on edges, offering better interpretability.
4. **The AI Scientist** (Lu et al., 2024): A framework for fully automated scientific discovery—from hypothesis to writing the final PDF.
5. **Vision Mamba (Vim)** (Zhu et al., 2024): Applies Mamba's linear scaling to vision, proving Transformers aren't the only way to do high-res image processing.
6. **Jamba: A Hybrid Transformer-Mamba Architecture** (AI21, 2024): Proves that mixing Mamba and Transformer layers combines the strengths of both (memory efficiency + high capacity).
7. **Sparse Attention by DeepSeek** (2025): A technical deep-dive into replacing standard attention to achieve near-infinite context windows.
8. **Small Language Models are the Future of Agentic AI** (Belcak et al., 2025): Argues that for specific tasks, specialized SLMs are more reliable than generalized LLMs.
9. **The AdEMAMix Optimizer** (2025): A new optimizer that outperforms Adam by remembering "past gradients" more effectively, speeding up training for large models.
10. **IgGM: A Generative Model for Antibody Design** (2025): A breakthrough in using deep learning for generative medicine and nanobody synthesis.

---
