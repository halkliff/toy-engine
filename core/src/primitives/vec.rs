//! Vector primitives for 2D, 3D, and 4D mathematics.
//!
//! This module provides high-performance vector types used throughout the graphics engine
//! for positions, directions, velocities, colors, and transformations.

pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use vec2::*;
pub use vec3::*;
pub use vec4::*;

/// Precision threshold for determining if a vector is normalized.
pub const NORMALIZED_PRECISION_THRESHOLD: f32 = 2e-4;
