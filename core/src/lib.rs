//! Core library for the ToyEngine graphics engine.
//!
//! This library provides fundamental data structures and algorithms
//! for mathematical operations, color representations, and other low-level
//! functionality required by the graphics engine.
//!
//! It includes modules for handling colors in various color spaces,
//! such as RGB, HSL, and HSV, as well as vector and matrix mathematics
//! for 2D, 3D, and 4D transformations.
//!
//! # Examples
//! ```rust
//! use toyengine::primitives::color::RGB;
//! let color = RGB::new(255, 0, 0); // Pure red
//! let (r, g, b): (u8, u8, u8) = color.into();
//! assert_eq!((r, g, b), (255, 0, 0));
//! let linear_color: LinearRGB = color.into();
//! assert!(linear_color.r > 0.5); // Linear red channel
//! let vec: Vec3 = color.into();
//! assert_eq!(vec, Vec3::new(1.0, 0.0, 0.0)); // RGB as normalized vector
//! ```

pub mod primitives;
pub mod math;