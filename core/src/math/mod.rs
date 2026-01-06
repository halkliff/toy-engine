//! Mathematical utilities for 2D and 3D transformations.
//!
//! This module provides transformation structures for handling position, rotation, and scale
//! in both 2D and 3D space. These are essential building blocks for game engines, graphics
//! applications, physics simulations, and scene graphs.
//!
//! # Modules
//!
//! - [`transform_2d`] - 2D transformations with position, rotation (angle), and scale
//! - [`transform_3d`] - 3D transformations with position, rotation (quaternion), and scale
//!
//! # Examples
//!
//! ```rust
//! use toyengine::math::transform_2d::Transform2D;
//! use toyengine::math::transform_3d::Transform3D;
//! use toyengine::primitives::vec::{Vec2, Vec3};
//! use toyengine::primitives::quat::Quaternion;
//!
//! // 2D transformation for sprites, UI elements, etc.
//! let transform_2d = Transform2D::new(
//!     Vec2::new(100.0, 50.0),           // position
//!     std::f32::consts::FRAC_PI_4,      // 45° rotation
//!     Vec2::new(2.0, 2.0)                // 2x scale
//! );
//!
//! // 3D transformation for game objects, camera, etc.
//! let transform_3d = Transform3D::new(
//!     Vec3::new(10.0, 5.0, -20.0),      // position
//!     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4),
//!     Vec3::one()                        // uniform scale
//! );
//! ```

pub mod transform_2d;
pub mod transform_3d;
