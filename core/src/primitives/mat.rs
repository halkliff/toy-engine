//! Matrix primitives for 2x2, 3x3, and 4x4 matrices.
//!
//! This module provides high-performance matrix types used throughout the graphics engine
//! for positions, directions, velocities, colors, and transformations.
//! The matrices are optimized for common operations such as addition, subtraction,
//! multiplication, inversion and scaling.
//! They are designed to work seamlessly with vector types defined in the `vec` module.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::mat::{Matrix2, Matrix3, Matrix4};
//! use toyengine::primitives::vec::{Vec2, Vec3, Vec4};
//!
//! // 2x2 matrix for 2D transformations
//! let rotation_2d = Matrix2::from_angle(std::f32::consts::FRAC_PI_4);
//! let point_2d = Vec2::new(1.0, 0.0);
//! let rotated_2d = rotation_2d * point_2d;
//!
//! // 3x3 matrix operations
//! let mat3 = Matrix3::identity();
//! let scaled = mat3 * 2.0;
//!
//! // 4x4 matrix for 3D transformations
//! let transform = Matrix4::identity();
//! let point_3d = Vec4::new(1.0, 2.0, 3.0, 1.0);
//! let transformed = transform * point_3d;
//! ```

/** `Matrix2` Struct and implementation */
pub mod mat2;
pub use mat2::*;

/** `Matrix3` Struct and implementation */
pub mod mat3;
pub use mat3::*;

/** `Matrix4` Struct and implementation */
pub mod mat4;
pub use mat4::*;
