# vLLM Asychronous Optimization 
This code leverages vLLM ![vLLM](https://img.shields.io/badge/vLLM-GPU_Inference-FF6F00?style=flat) and the OpenAI API ![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey) wrapper for faster, asynchronous LLM inference. It was developed over May 2025-May 2026 via extensive experiments over high-performance NVIDIA GPUs over university servers.

The implementation leverages available GPU resources for local server endpoint hosting via the OpenAI API completion wrapper and vLLM's fast inference capability.
The code offers a walkthrough of resource optimization methods, including optional parallel compute and multi-server hosting. 
