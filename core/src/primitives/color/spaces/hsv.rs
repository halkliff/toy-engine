//! HSV and HSVA color space definitions and utilities.
//!
//! HSV (Hue, Saturation, Value) and HSVA (with Alpha) color types.
//! HSV is often preferred by artists and designers because the Value component
//! directly controls brightness, making it more intuitive than HSL's Lightness.
//! Provides methods for hue rotation, saturation/value adjustment, and alpha fading.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::color::{HSV, HSVA, RGB};
//!
//! // Create colors in HSV space
//! let bright_red = HSV::new(0.0, 1.0, 1.0);
//! let dark_green = HSV::new(120.0, 1.0, 0.5);
//!
//! // Create with alpha
//! let semi_transparent_yellow = HSVA::new(60.0, 1.0, 1.0, 0.5);
//!
//! // Convert from RGB
//! let rgb = RGB::new(255, 0, 128);
//! let hsv = rgb.to_hsv();
//!
//! // Adjust value for brightness control
//! let darker = hsv.with_value(0.5);
//!
//! // Rotate hue for complementary color
//! let complementary = hsv.with_hue_offset(180.0);
//!
//! // Convert back to RGB
//! let result: RGB = complementary.into();
//! ```

use crate::primitives::color::{HUE_MAX_DEGREES, MAX_PERCENTAGE, MIN_PERCENTAGE, Hue, Percentage, Solid, Color};

/// HSV color with hue, saturation, and value channels (no alpha).
/// Often preferred by artists for its intuitive brightness control.
///
/// # Components
/// - Hue: 0-360° on the color wheel (0=red, 120=green, 240=blue)
/// - Saturation: 0.0-1.0 (0=grayscale, 1=vivid color)
/// - Value: 0.0-1.0 (0=black, 1=maximum brightness for that hue)
///
/// # HSV vs HSL
/// - **HSV Value**: Controls brightness directly (0=black, 1=bright color)
/// - **HSL Lightness**: 0=black, 0.5=pure color, 1=white
/// - HSV is often more intuitive for color picking and adjustment
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HSV {
    /// Hue in degrees [0, 360)
    pub hue: Hue,
    /// Saturation [0.0, 1.0]
    pub saturation: Percentage,
    /// Value (brightness) [0.0, 1.0]
    pub value: Percentage,
}

/// HSVA color with hue, saturation, value, and alpha channels.
/// Often preferred by artists for its intuitive brightness control.
///
/// # Components
/// - Hue: 0-360° on the color wheel (0=red, 120=green, 240=blue)
/// - Saturation: 0.0-1.0 (0=grayscale, 1=vivid color)
/// - Value: 0.0-1.0 (0=black, 1=maximum brightness for that hue)
/// - Alpha: 0.0-1.0 (0=transparent, 1=opaque)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HSVA {
    /// Hue in degrees [0, 360)
    pub hue: Hue,
    /// Saturation [0.0, 1.0]
    pub saturation: Percentage,
    /// Value (brightness) [0.0, 1.0]
    pub value: Percentage,
    /// Alpha/opacity [0.0, 1.0]
    pub alpha: Percentage,
}

impl Solid for HSV {}
impl Color for HSV {}
impl Color for HSVA {}

/// Manual implementation of `f32::rem_euclid` for use in const context.
///
/// Performs Euclidean modulo operation (always returns positive remainder).
/// Required because `f32::rem_euclid` is not const-stable yet.
///
/// # Examples
/// - `__const_rem_euclid_f32(370.0, 360.0)` returns `10.0`
/// - `__const_rem_euclid_f32(-10.0, 360.0)` returns `350.0`
const fn __const_rem_euclid_f32(value: f32, modulus: f32) -> f32 {
    let remainder = value % modulus;
    if remainder < 0.0 {
        remainder + modulus
    } else {
        remainder
    }
}

impl HSV {
    /// Creates a new HSV color with automatic bounds enforcement.
    ///
    /// # Arguments
    /// * `hue` - Hue in degrees (automatically wrapped to [0, 360), e.g., 370° becomes 10°)
    /// * `saturation` - Saturation [0.0, 1.0] (0.0 = grayscale, 1.0 = vivid)
    /// * `value` - Value/brightness [0.0, 1.0] (0.0 = black, 1.0 = maximum brightness)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSV;
    /// let red = HSV::new(0.0, 1.0, 1.0);        // Pure red at max brightness
    /// let dark_blue = HSV::new(240.0, 1.0, 0.5); // Dark blue (50% brightness)
    /// let gray = HSV::new(0.0, 0.0, 0.5);        // Mid gray (hue irrelevant)
    /// ```
    pub const fn new(hue: f32, saturation: f32, value: f32) -> Self {
        HSV {
            hue: Hue(__const_rem_euclid_f32(hue, HUE_MAX_DEGREES)),
            saturation: Percentage(saturation.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            value: Percentage(value.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
        }
    }

    /// Rotates the hue around the color wheel by the specified degrees.
    ///
    /// Positive values rotate clockwise, negative values counter-clockwise.
    /// Hue automatically wraps (e.g., 360° = 0°).
    ///
    /// # Color Wheel Reference
    /// - 0° = Red
    /// - 60° = Yellow
    /// - 120° = Green
    /// - 180° = Cyan
    /// - 240° = Blue
    /// - 300° = Magenta
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSV;
    /// let mut red = HSV::new(0.0, 1.0, 1.0);
    /// red.rotate_hue(60.0);   // Shift to yellow
    /// red.rotate_hue(120.0);  // Now cyan
    /// ```
    #[inline]
    pub fn rotate_hue(&mut self, degrees: f32) {
        self.hue = (self.hue.0 + degrees).rem_euclid(HUE_MAX_DEGREES).into();
    }

    /// Adjusts saturation (color intensity/vividness).
    ///
    /// Adds the amount to current saturation, clamped to [0.0, 1.0].
    /// - Positive amount: More vivid/intense
    /// - Negative amount: More washed out/grayscale
    ///
    /// # Saturation Scale
    /// - 0.0 = Complete grayscale (no color)
    /// - 0.5 = Muted/pastel colors
    /// - 1.0 = Maximum color intensity
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSV;
    /// let mut pastel = HSV::new(120.0, 0.3, 0.8);  // Muted green
    /// pastel.saturate(0.5);                         // Now 0.8 saturation
    /// pastel.saturate(-0.8);                        // Now 0.0 (grayscale)
    /// ```
    #[inline]
    pub fn saturate(&mut self, amount: f32) {
        self.saturation = (self.saturation.0 + amount).clamp(0.0, 1.0).into();
    }

    /// Adjusts value (brightness).
    ///
    /// Adds the amount to current value, clamped to [0.0, 1.0].
    /// - Positive amount: Brighter
    /// - Negative amount: Darker
    ///
    /// # Value Scale
    /// - 0.0 = Black (no brightness)
    /// - 0.5 = Medium brightness
    /// - 1.0 = Maximum brightness (pure color at full intensity)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSV;
    /// let mut color = HSV::new(240.0, 1.0, 0.6);  // Blue at 60% brightness
    /// color.adjust_value(0.2);                     // Now 0.8 brightness
    /// color.adjust_value(-0.5);                    // Now 0.3 brightness
    /// ```
    #[inline]
    pub fn adjust_value(&mut self, amount: f32) {
        self.value = (self.value.0 + amount).clamp(0.0, 1.0).into();
    }

    // Chaining methods (consume self)

    /// Chainable version of rotate_hue
    #[inline]
    pub fn with_rotate_hue(mut self, degrees: f32) -> Self {
        self.rotate_hue(degrees);
        self
    }

    /// Chainable version of saturate
    #[inline]
    pub fn with_saturate(mut self, amount: f32) -> Self {
        self.saturate(amount);
        self
    }

    /// Chainable version of adjust_value
    #[inline]
    pub fn with_adjust_value(mut self, amount: f32) -> Self {
        self.adjust_value(amount);
        self
    }
}

impl HSVA {
    /// Creates a new HSVA color with automatic bounds enforcement.
    ///
    /// # Arguments
    /// * `hue` - Hue in degrees (wrapped to [0, 360), e.g., 370° becomes 10°)
    /// * `saturation` - Saturation [0.0, 1.0] (0.0 = grayscale, 1.0 = vivid)
    /// * `value` - Value/brightness [0.0, 1.0] (0.0 = black, 1.0 = maximum brightness)
    /// * `alpha` - Opacity [0.0, 1.0] (0.0 = transparent, 1.0 = opaque)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let semi_red = HSVA::new(0.0, 1.0, 1.0, 0.5);        // 50% transparent red
    /// let opaque_blue = HSVA::new(240.0, 1.0, 1.0, 1.0);   // Solid blue
    /// ```
    pub const fn new(hue: f32, saturation: f32, value: f32, alpha: f32) -> Self {
        HSVA {
            hue: Hue(__const_rem_euclid_f32(hue, HUE_MAX_DEGREES)),
            saturation: Percentage(saturation.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            value: Percentage(value.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            alpha: Percentage(alpha.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
        }
    }

    /// Rotates the hue around the color wheel by the specified degrees.
    ///
    /// See [`HSV::rotate_hue`](HSV::rotate_hue) for color wheel reference.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let mut semi_red = HSVA::new(0.0, 1.0, 1.0, 0.5);
    /// semi_red.rotate_hue(180.0);  // Now cyan
    /// ```
    #[inline]
    pub fn rotate_hue(&mut self, degrees: f32) {
        self.hue = (self.hue.0 + degrees).rem_euclid(HUE_MAX_DEGREES).into();
    }

    /// Adjusts saturation (color intensity).
    ///
    /// See [`HSV::saturate`](HSV::saturate) for detailed explanation.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let mut color = HSVA::new(120.0, 0.5, 0.8, 0.8);
    /// color.saturate(0.3);   // Now 0.8 saturation
    /// color.saturate(-0.3);  // Now 0.5 saturation
    /// ```
    #[inline]
    pub fn saturate(&mut self, amount: f32) {
        self.saturation = (self.saturation.0 + amount).clamp(0.0, 1.0).into();
    }

    /// Adjusts value (brightness).
    ///
    /// See [`HSV::adjust_value`](HSV::adjust_value) for detailed explanation.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let mut color = HSVA::new(240.0, 1.0, 0.6, 1.0);
    /// color.adjust_value(0.2);  // Now 0.8 brightness
    /// color.adjust_value(-0.3); // Now 0.5 brightness
    /// ```
    #[inline]
    pub fn adjust_value(&mut self, amount: f32) {
        self.value = (self.value.0 + amount).clamp(0.0, 1.0).into();
    }

    /// Adjusts alpha (opacity).
    ///
    /// Adds the amount to current alpha, clamped to [0.0, 1.0].
    /// - Positive amount: More opaque
    /// - Negative amount: More transparent
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let mut color = HSVA::new(0.0, 1.0, 1.0, 0.8);  // 80% opaque
    /// color.fade(0.2);                                 // Now 100% opaque
    /// color.fade(-0.5);                                // Now 50% opaque
    /// ```
    #[inline]
    pub fn fade(&mut self, amount: f32) {
        self.alpha = (self.alpha.0 + amount).clamp(0.0, 1.0).into();
    }

    // Chaining methods (consume self for fluent API)

    /// Chainable version of [`rotate_hue`](Self::rotate_hue).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSVA;
    /// let result = HSVA::new(0.0, 1.0, 1.0, 1.0)
    ///     .with_rotate_hue(30.0)
    ///     .with_saturate(-0.2)
    ///     .with_adjust_value(0.1)
    ///     .with_fade(-0.3);
    /// ```
    #[inline]
    pub fn with_rotate_hue(mut self, degrees: f32) -> Self {
        self.rotate_hue(degrees);
        self
    }

    /// Chainable version of [`saturate`](Self::saturate).
    #[inline]
    pub fn with_saturate(mut self, amount: f32) -> Self {
        self.saturate(amount);
        self
    }

    /// Chainable version of [`adjust_value`](Self::adjust_value).
    #[inline]
    pub fn with_adjust_value(mut self, amount: f32) -> Self {
        self.adjust_value(amount);
        self
    }

    /// Chainable version of [`fade`](Self::fade).
    #[inline]
    pub fn with_fade(mut self, amount: f32) -> Self {
        self.fade(amount);
        self
    }
}
