//! Matrix primitives for 2x2, 3x3, and 4x4 matrices.
//!
//! This module provides high-performance matrix types used throughout the graphics engine
//! for positions, directions, velocities, colors, and transformations.
//! The matrices are optimized for common operations such as addition, subtraction,
//! multiplication, inversion and scaling.
//! They are designed to work seamlessly with vector types defined in the `vec` module.

pub mod mat2;
pub mod mat3;
pub mod mat4;

pub use mat2::*;
pub use mat3::*;
pub use mat4::*;