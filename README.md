# AMCP AI Engine

> Training, evaluation, model registry, and inference services for the AI layer of the AI Modular Cybersecurity Platform.

## What lives here

This repository contains the AI-specific implementation for model development and serving.

Includes:
- training pipelines
- dataset handling and preprocessing
- model definitions and shared bases
- evaluation utilities
- inference serving code
- ONNX export and model packaging

## Repository layout

```text
amcp-ai-engine/
├── ai/
└── README.md
```

## How to use it

1. Keep model code and training logic here.
2. Expose inference through stable APIs or events.
3. Depend on platform-core contracts rather than copying them.

## Related repos

- amcp-platform-core
- amcp-runtime-microservices
- amcp-clients
