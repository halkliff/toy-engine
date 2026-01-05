# Core

The `core` crate provides fundamental mathematical primitives and operations for game and graphics engine development. It includes high-performance implementations of vectors, matrices, and color spaces with conversions and transformations.

## Overview

This crate forms the mathematical foundation of the ToyEngine, offering essential data structures and operations needed for 3D graphics, physics simulations, and general game development. All types are thoroughly documented with docstrings, and comprehensive API documentation can be generated using `cargo doc`.

**Note:** This crate is under active development. Additional modules (such as `math`) and features will be added as the engine grows.

## Modules

### Primitives

The `primitives` module contains all fundamental data types used throughout the engine:

#### Vectors (`vec`)
- **Vec2** - 2D vector for positions, directions, and UV coordinates
- **Vec3** - 3D vector for positions, directions, normals, and RGB colors
- **Vec4** - 4D vector for homogeneous coordinates and RGBA colors

#### Matrices (`mat`)
- **Matrix2** - 2×2 matrix for 2D transformations
- **Matrix3** - 3×3 matrix for 2D transformations with translation and 3D rotations
- **Matrix4** - 4×4 matrix for 3D transformations (translation, rotation, scale, projection)

#### Colors (`color`)
- **RGB** - Standard RGB color (0-255 per channel)
- **ARGB** - RGB with alpha channel
- **LinearRGB** - RGB in linear color space (gamma-corrected)
- **LinearARGB** - Linear RGB with alpha channel
- **PremultipliedARGB** - Alpha-premultiplied ARGB for efficient blending
- **HSL** - Hue, Saturation, Lightness color model
- **HSLA** - HSL with alpha channel
- **HSV** - Hue, Saturation, Value color model
- **HSVA** - HSV with alpha channel
- **Percentage** - Type-safe wrapper for percentage values (0.0-1.0)
- **Hue** - Type-safe wrapper for hue values (0.0-360.0°)

## Quick Reference

| Category | Struct | Description |
|----------|--------|-------------|
| **Vectors** | `Vec2` | 2D vector (x, y) |
| | `Vec3` | 3D vector (x, y, z) |
| | `Vec4` | 4D vector (x, y, z, w) |
| **Matrices** | `Matrix2` | 2×2 transformation matrix |
| | `Matrix3` | 3×3 transformation matrix |
| | `Matrix4` | 4×4 transformation matrix |
| **Color Spaces** | `RGB` | 8-bit RGB color |
| | `ARGB` | 8-bit ARGB color with alpha |
| | `LinearRGB` | Linear RGB (gamma-corrected) |
| | `LinearARGB` | Linear ARGB with alpha |
| | `PremultipliedARGB` | Alpha-premultiplied ARGB |
| | `HSL` | Hue, Saturation, Lightness |
| | `HSLA` | HSL with alpha |
| | `HSV` | Hue, Saturation, Value |
| | `HSVA` | HSV with alpha |
| **Color Helpers** | `Percentage` | Type-safe percentage (0.0-1.0) |
| | `Hue` | Type-safe hue angle (0.0-360.0°) |

## Features

### Vector Operations
- Arithmetic operations (add, subtract, multiply, divide)
- Dot and cross products
- Length, normalization, and distance calculations
- Component-wise operations
- Linear interpolation (lerp)

### Matrix Operations
- Matrix arithmetic (add, subtract, multiply)
- Matrix-vector multiplication
- Determinant and inverse calculations
- Transpose operations
- Common transformations (translation, rotation, scale)

### Color Conversions
- Seamless conversion between RGB, HSL, and HSV color spaces
- Linear and gamma-corrected color space support
- Alpha-premultiplication for efficient blending
- Packing and unpacking to/from 32-bit integers
- Luminance calculations

## Usage

```rust
use core::primitives::{Vec3, Matrix4, RGB, HSV};

// Vector operations
let v1 = Vec3::new(1.0, 2.0, 3.0);
let v2 = Vec3::new(4.0, 5.0, 6.0);
let dot = v1.dot(v2);
let cross = v1.cross(v2);

// Matrix transformations
let transform = Matrix4::identity()
    .translate(Vec3::new(10.0, 0.0, 0.0))
    .rotate_y(90.0);

// Color space conversions
let rgb = RGB::new(255, 128, 64);
let hsv: HSV = rgb.into();
let back_to_rgb: RGB = hsv.into();
```

## Future Additions

The following modules and features are planned for future releases:
- **math** - Advanced mathematical functions (trigonometry, interpolation, etc.)
- **Quaternions** - For smooth 3D rotations
- **Bounding volumes** - AABB, OBB, spheres for collision detection
- **Curves** - Bezier curves and splines
- **Random** - Pseudo-random number generation utilities

## Documentation

For detailed API documentation, including method descriptions, examples, and implementation notes, run:

```bash
cargo doc --open
```

## License

This crate is part of the ToyEngine project.
