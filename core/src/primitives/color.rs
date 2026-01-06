//! Color primitives and conversions between color spaces.
//!
//! Includes RGB, ARGB, HSL, HSLA, HSV, and HSVA color types with associated methods and traits.
//! Provides conversions, packing/unpacking, and basic color operations.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::color::{RGB, ARGB, HSL, HSV, LinearRGB};
//! use toyengine::primitives::vec::Vec3;
//!
//! // Create colors in different color spaces
//! let red = RGB::new(255, 0, 0);
//! let transparent_red = ARGB::new(128, 255, 0, 0);
//!
//! // Convert between color spaces
//! let hsl_color = red.to_hsl();
//! let hsv_color = red.to_hsv();
//! let linear_color: LinearRGB = red.into();
//!
//! // Manipulate colors in HSL space
//! let hue_shifted = HSL::new(180.0, 1.0, 0.5); // Cyan
//! let rgb_result: RGB = hue_shifted.into();
//!
//! // Convert colors to vectors for calculations
//! let color_vec: Vec3 = red.into();
//! ```

mod conversions;
mod traits;
mod spaces;

pub use spaces::*;
pub use traits::*;

// ===== Color Space Constants =====
// These constants define the fundamental ranges and conversion factors used throughout the color module.
// Extracting them as named constants improves code clarity and enables easy reference searches.

/// Minimum value for percentage-based color components (0.0)
pub(crate) const MIN_PERCENTAGE: f32 = 0.0;

/// Maximum value for percentage-based color components (1.0)
pub(crate) const MAX_PERCENTAGE: f32 = 1.0;

/// Maximum value for 8-bit color channels (0-255 range)
pub(crate) const U8_MAX_F32: f32 = 255.0;

/// Maximum value for 8-bit color channels as u32 for calculations
pub(crate) const U8_MAX_U32: u32 = 255;

/// Inverse of 255 for efficient normalization (1/255 ≈ 0.00392157)
pub(crate) const INV_U8_MAX: f32 = 1.0 / 255.0;

/// Bit shift amount for fast division by 256 (close approximation to 255)
/// Used in performance-critical blending operations: (x * y) >> 8 ≈ (x * y) / 255
pub(crate) const FAST_DIV_255_SHIFT: u32 = 8;

/// Maximum hue value in degrees (color wheel wraps at 360°)
pub(crate) const HUE_MAX_DEGREES: f32 = 360.0;

/// Size of each hue segment on the color wheel (360° / 6 primary/secondary colors)
pub(crate) const HUE_SEGMENT_DEGREES: f32 = 60.0;

/// Standard gamma value for sRGB color space (approximate)
/// Used for converting between linear and gamma-corrected RGB
pub(crate) const SRGB_GAMMA: f32 = 2.2;
/// Inverse gamma for sRGB (1/2.2 ≈ 0.4545)
pub(crate) const SRGB_INV_GAMMA: f32 = 1.0 / 2.2;

/// Red channel weight for luminance calculation (ITU-R BT.601 standard)
/// Reflects human eye sensitivity: green > red > blue
pub(crate) const LUMINANCE_RED_WEIGHT: f32 = 0.299;
/// Green channel weight for luminance calculation (ITU-R BT.601 standard)
pub(crate) const LUMINANCE_GREEN_WEIGHT: f32 = 0.587;

/// Blue channel weight for luminance calculation (ITU-R BT.601 standard)
pub(crate) const LUMINANCE_BLUE_WEIGHT: f32 = 0.114;
