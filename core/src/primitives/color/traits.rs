//! Traits and utilities for color types and components.

use crate::primitives::color::{
    ARGB, HSL, HSLA, HSV, HSVA, HUE_MAX_DEGREES, MAX_PERCENTAGE, MIN_PERCENTAGE, RGB, U8_MAX_U32,
};
use crate::primitives::color::spaces::{LinearRGB, LinearARGB, PremultipliedARGB};

/// Trait for numeric types that have defined bounds and can clamp values within range.
/// Used internally for color components like hue (0-360°) and percentages (0.0-1.0).
pub trait BoundedNum<T: PartialOrd + Copy + Sized + 'static> {
    /// Minimum valid value for this bounded type
    const MIN: T;
    /// Maximum valid value for this bounded type
    const MAX: T;

    /// Returns the value clamped or wrapped within valid bounds
    fn bounded(self) -> T;
}

/// Represents a percentage value in the range [0.0, 1.0].
/// Used for saturation, lightness, and alpha components in color spaces.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Percentage(pub f32);

/// Represents a hue value in degrees [0.0, 360.0).
/// Automatically wraps around (e.g., 361° becomes 1°).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hue(pub f32);

impl BoundedNum<f32> for Percentage {
    const MIN: f32 = MIN_PERCENTAGE;
    const MAX: f32 = MAX_PERCENTAGE;

    /// Clamps the percentage value to [0.0, 1.0] range
    #[inline]
    fn bounded(self) -> f32 {
        self.0.clamp(Self::MIN, Self::MAX)
    }
}

impl BoundedNum<f32> for Hue {
    const MIN: f32 = MIN_PERCENTAGE;
    const MAX: f32 = HUE_MAX_DEGREES;

    /// Wraps the hue value using modulo arithmetic (e.g., 370° → 10°, -10° → 350°)
    #[inline]
    fn bounded(self) -> f32 {
        self.0.rem_euclid(Self::MAX)
    }
}

impl From<f32> for Percentage {
    fn from(value: f32) -> Self {
        Percentage(value.clamp(Percentage::MIN, Percentage::MAX))
    }
}

impl From<f32> for Hue {
    fn from(value: f32) -> Self {
        Hue(value.rem_euclid(Hue::MAX))
    }
}

/// Marker trait for color types without an alpha channel (RGB, HSL).
/// Use this as a trait bound when you need to ensure a color is fully opaque.
pub trait Solid {}

/// Base trait for all color types providing conversion and raw packing methods.
/// All color types can convert to any other color type via the Into trait bounds.
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::{RGB, ARGB, HSV, Color};
/// let rgb = RGB::new(255, 128, 64);
/// let argb: ARGB = rgb.into();
/// let hsv: HSV = rgb.into();
/// let packed = rgb.to_argb_raw(); // u32 representation
/// ```
pub trait Color:
    Into<ARGB>
    + Into<HSLA>
    + Into<RGB>
    + Into<HSL>
    + Into<HSV>
    + Into<HSVA>
    + Into<LinearRGB>
    + Into<LinearARGB>
    + Into<PremultipliedARGB>
{
    /// Converts the color to a packed u32 in ARGB format (0xAARRGGBB).
    /// Alpha channel is set to 255 (opaque).
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::{RGB, Color};
    /// let rgb = RGB::new(255, 128, 64);
    /// let packed = rgb.to_argb_raw(); // 0xFFFF8040
    /// ```
    fn to_rgb_raw(self) -> u32 {
        let rgb: RGB = self.into();
        U8_MAX_U32 << 24 | ((rgb.red as u32) << 16) | ((rgb.green as u32) << 8) | (rgb.blue as u32)
    }

    /// Converts the color to a packed u32 in ARGB format (0xAARRGGBB).
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::{ARGB, Color};
    /// let argb = ARGB::new(128, 255, 0, 0); // 50% opaque red
    /// let packed = argb.to_argb_raw(); // 0x80FF0000
    /// ```
    fn to_argb_raw(self) -> u32 {
        let argb: ARGB = self.into();
        ((argb.alpha as u32) << 24)
            | ((argb.red as u32) << 16)
            | ((argb.green as u32) << 8)
            | (argb.blue as u32)
    }
}
