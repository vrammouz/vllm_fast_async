# vLLM Asychronous Optimization 
![vLLM](https://img.shields.io/badge/vLLM-GPU_Inference-FF6F00?style=flat)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey) 

This code leverages vLLM and the OpenAI API wrapper for faster, asynchronous LLM inference. It was developed over May 2025-May 2026 via extensive experiments over high-performance NVIDIA GPUs over university servers.

The implementation leverages available GPU resources for local server endpoint hosting via the OpenAI API completion wrapper and vLLM's fast inference capability.
The code offers a walkthrough of resource optimization methods, including optional parallel compute and multi-server hosting. 

The idea is inspired by coroutines and async background tasks from prior experience. Proudly developed.
