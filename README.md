# RedFuser

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ASPLOS 2026](https://img.shields.io/badge/ASPLOS-2026-orange.svg)](https://asplos-conference.org/)

This repository contains the artifact for our ASPLOS 2026 paper: **"RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators"**.

## Overview

RedFuser is a novel framework for optimizing cascaded reductions in deep learning compilers. Built on top of [Apache TVM](https://github.com/apache/tvm), RedFuser introduces a series of compiler transformation passes that enable efficient fusion of reduction operations with other computations, particularly targeting modern GPU architectures.

## Update

- \[2026-01\]: RedFuser is now avaliable with flash-attention example.
- \[2025-11\]: 🎉RedFuser is accepted by ASPLOS 2026!

## Roadmap

- [x] flash-attention
- [ ] flash-decoding
- [ ] moe-routing
- [ ] fp8 quant+gemm

## Getting Started

Please follow https://tvm.apache.org/docs/install/index.html to install.

## Example

For flash-attention example, see [`python/tvm/redfuser/example/flash_attention.py`](python/tvm/redfuser/example/flash_attention.py).

## Structure

```
redfuser/
├── python/tvm/redfuser/       # RedFuser Python implementation
│   ├── transform/             # Core transformation passes
│   └── example/               # Example workloads
│ ...
```

## License

RedFuser is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

This project builds upon [Apache TVM](https://github.com/apache/tvm). We thank the TVM community for their excellent infrastructure and support.
