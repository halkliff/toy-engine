//! Primitive types for colors, vectors, and matrices.
//!
//! This module provides fundamental data structures and operations for handling colors,
//! 2D/3D/4D vectors, and matrices used throughout the graphics engine.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::{vec::Vec3, mat::Matrix4, color::RGB, quat::Quaternion};
//!
//! // Working with vectors
//! let position = Vec3::new(1.0, 2.0, 3.0);
//! let direction = Vec3::new(0.0, 1.0, 0.0);
//! let scaled = position * 2.0;
//!
//! // Working with matrices
//! let transform = Matrix4::identity();
//! let scaled_transform = transform * 2.0;
//!
//! // Working with colors
//! let red = RGB::new(255, 0, 0);
//! let hsl_color = red.to_hsl();
//!
//! // Working with quaternions
//! let rotation = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
//! let rotated = rotation * Vec3::X;
//! ```

pub mod color;
pub mod vec;
pub mod mat;
pub mod quat;