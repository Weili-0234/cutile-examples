# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU kernel example repository showcasing implementations of various machine learning operators using **Cutile**, a CUDA tile-based GPU programming framework. Each Python file is a standalone, self-contained kernel example that can be run directly.

## Running Examples

Each file is self-contained and executable:
```bash
python _01_sum.py
python _10_rmsnorm.py
# etc.
```

No build system, package manager config, or test framework is configured. Dependencies: `torch`, `cuda.tile` (Cutile library).

## Code Architecture

### Kernel Structure Pattern

All kernels follow this pattern:

```python
@ct.kernel
def kernel_name(arrays: ct.Array, constants: ct.Constant) -> None:
    block_x, block_y = ct.bid(0), ct.bid(1)
    tile = ct.load(array, indices, shape, padding_mode=...)
    tile = process(tile)
    ct.store(output, indices, tile)
```

Reusable tile operations use `@ct.function`:
```python
@ct.function
def helper(tile: ct.Tile) -> ct.Tile:
    return process(tile)
```

### Kernel Launch

```python
ct.launch(
    torch.cuda.current_stream(),
    (blocks_x, blocks_y, blocks_z),  # Grid dimensions
    kernel_func,
    (args, tile_size, constants)
)
```

### Key Cutile Abstractions

- **ct.Array**: Global GPU memory arrays (inputs/outputs)
- **ct.Tile**: Registered tile (block working memory)
- **ct.Constant**: Compile-time constants for unrolling
- **ct.bid(d)**: Block ID in dimension d
- **ct.load() / ct.store()**: Memory operations
- **ct.mma()**: Matrix-multiply-accumulate
- **ct.atomic_add()**: Thread-safe reduction

### Common Patterns

1. **Reduction**: Load tile → reduce locally → `ct.atomic_add()` to global
2. **Normalization**: Compute stats → normalize → apply affine transform
3. **Tile-based GEMM**: Loop over K tiles with `ct.mma()` accumulator
4. **Data type handling**: Upcast to float32 for computation, downcast for output

### Conventions

- **File naming**: `_NN_description.py` (sequential numbering)
- **Padding**: Use `ct.PaddingMode.ZERO` for out-of-bounds handling
- **Numerical stability**: Use `eps` constants (1e-6 to 1e-8)
- **Input validation**: Check CUDA device, contiguity, and shape
