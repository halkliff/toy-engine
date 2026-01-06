//! Vector primitives for 2D, 3D, and 4D mathematics.
//!
//! This module provides high-performance vector types used throughout the graphics engine
//! for positions, directions, velocities, colors, and transformations.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::vec::{Vec2, Vec3, Vec4};
//!
//! // 2D vector operations
//! let position = Vec2::new(10.0, 20.0);
//! let velocity = Vec2::new(1.0, -2.0);
//! let new_position = position + velocity;
//!
//! // 3D vector operations
//! let point = Vec3::new(1.0, 2.0, 3.0);
//! let direction = Vec3::new(0.0, 1.0, 0.0).normalized();
//! let distance = point.length();
//! let dot = point.dot(direction);
//!
//! // 4D vector for homogeneous coordinates
//! let homogeneous = Vec4::new(1.0, 2.0, 3.0, 1.0);
//! ```
//!
//! # Structs
//!
//! - [`Vec2`] - 2D vector with x and y components
//! - [`Vec3`] - 3D vector with x, y, and z components
//! - [`Vec4`] - 4D vector with x, y, z, and w components

/** `Vec2` Struct and implementation */
pub mod vec2;
pub use vec2::*;

/** `Vec3` Struct and implementation */
pub mod vec3;
pub use vec3::*;

/** `Vec4` Struct and implementation */
pub mod vec4;
pub use vec4::*;

/// Precision threshold for determining if a vector is normalized.
pub const NORMALIZED_PRECISION_THRESHOLD: f32 = 2e-4;
