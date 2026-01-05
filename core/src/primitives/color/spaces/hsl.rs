//! HSL and HSLA color space definitions and utilities.
//!
//! HSL (Hue, Saturation, Lightness) and HSLA (with Alpha) color types.
//! More intuitive for color adjustments compared to RGB.
//! Provides methods for hue rotation, saturation/lightness adjustment, and alpha fading.

use crate::primitives::color::{HUE_MAX_DEGREES, MAX_PERCENTAGE, MIN_PERCENTAGE, Hue, Percentage, Solid, Color};

/// HSLA color with hue, saturation, lightness, and alpha channels.
/// More intuitive for color adjustments than RGB.
///
/// # Components
/// - Hue: 0-360° on the color wheel (0=red, 120=green, 240=blue)
/// - Saturation: 0.0-1.0 (0=grayscale, 1=vivid color)
/// - Lightness: 0.0-1.0 (0=black, 0.5=pure color, 1=white)
/// - Alpha: 0.0-1.0 (0=transparent, 1=opaque)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HSLA {
    /// Hue in degrees [0, 360)
    pub hue: Hue,
    /// Saturation [0.0, 1.0]
    pub saturation: Percentage,
    /// Lightness [0.0, 1.0]
    pub lightness: Percentage,
    /// Alpha/opacity [0.0, 1.0]
    pub alpha: Percentage,
}

/// HSL color with hue, saturation, and lightness channels (no alpha).
/// More intuitive for color adjustments than RGB.
///
/// # Components
/// - Hue: 0-360° on the color wheel (0=red, 120=green, 240=blue)
/// - Saturation: 0.0-1.0 (0=grayscale, 1=vivid color)
/// - Lightness: 0.0-1.0 (0=black, 0.5=pure color, 1=white)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HSL {
    /// Hue in degrees [0, 360)
    pub hue: Hue,
    /// Saturation [0.0, 1.0]
    pub saturation: Percentage,
    /// Lightness [0.0, 1.0]
    pub lightness: Percentage,
}

impl Solid for HSL {}
impl Color for HSL {}
impl Color for HSLA {}

// Operator trait implementations for idiomatic Rust

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

impl HSL {
    /// Creates a new HSL color with automatic bounds enforcement.
    ///
    /// # Arguments
    /// * `hue` - Hue in degrees (automatically wrapped to [0, 360), e.g., 370° becomes 10°)
    /// * `saturation` - Saturation [0.0, 1.0] (0.0 = grayscale, 1.0 = vivid)
    /// * `lightness` - Lightness [0.0, 1.0] (0.0 = black, 0.5 = pure color, 1.0 = white)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSL;
    /// let red = HSL::new(0.0, 1.0, 0.5);        // Pure red
    /// let pastel_blue = HSL::new(240.0, 0.5, 0.75); // Light blue
    /// let gray = HSL::new(0.0, 0.0, 0.5);       // Mid gray (hue irrelevant)
    /// ```
    pub const fn new(hue: f32, saturation: f32, lightness: f32) -> Self {
        HSL {
            hue: Hue(__const_rem_euclid_f32(hue, HUE_MAX_DEGREES)),
            saturation: Percentage(saturation.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            lightness: Percentage(lightness.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
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
    /// # use toyengine::primitives::color::HSL;
    /// let mut red = HSL::new(0.0, 1.0, 0.5);
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
    /// # use toyengine::primitives::color::HSL;
    /// let mut pastel = HSL::new(120.0, 0.3, 0.5);  // Muted green
    /// pastel.saturate(0.5);                         // Now 0.8 saturation
    /// pastel.saturate(-0.8);                        // Now 0.0 (grayscale)
    /// ```
    #[inline]
    pub fn saturate(&mut self, amount: f32) {
        self.saturation = (self.saturation.0 + amount).clamp(0.0, 1.0).into();
    }

    /// Adjusts lightness (brightness).
    ///
    /// Adds the amount to current lightness, clamped to [0.0, 1.0].
    /// - Positive amount: Brighter
    /// - Negative amount: Darker
    ///
    /// # Lightness Scale
    /// - 0.0 = Black (no light)
    /// - 0.5 = Pure color at full intensity
    /// - 1.0 = White (full light, color washed out)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSL;
    /// let mut color = HSL::new(240.0, 1.0, 0.5);  // Pure blue
    /// color.lighten(0.2);                          // Now 0.7 lightness
    /// color.lighten(-0.5);                         // Now 0.2 lightness
    /// ```
    #[inline]
    pub fn lighten(&mut self, amount: f32) {
        self.lightness = (self.lightness.0 + amount).clamp(0.0, 1.0).into();
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

    /// Chainable version of lighten
    #[inline]
    pub fn with_lighten(mut self, amount: f32) -> Self {
        self.lighten(amount);
        self
    }
}

impl HSLA {
    /// Creates a new HSLA color with automatic bounds enforcement.
    ///
    /// # Arguments
    /// * `hue` - Hue in degrees (wrapped to [0, 360), e.g., 370° becomes 10°)
    /// * `saturation` - Saturation [0.0, 1.0] (0.0 = grayscale, 1.0 = vivid)
    /// * `lightness` - Lightness [0.0, 1.0] (0.0 = black, 0.5 = pure color, 1.0 = white)
    /// * `alpha` - Opacity [0.0, 1.0] (0.0 = transparent, 1.0 = opaque)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSLA;
    /// let semi_red = HSLA::new(0.0, 1.0, 0.5, 0.5);        // 50% transparent red
    /// let opaque_blue = HSLA::new(240.0, 1.0, 0.5, 1.0);   // Solid blue
    /// ```
    pub const fn new(hue: f32, saturation: f32, lightness: f32, alpha: f32) -> Self {
        HSLA {
            hue: Hue(__const_rem_euclid_f32(hue, HUE_MAX_DEGREES)),
            saturation: Percentage(saturation.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            lightness: Percentage(lightness.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
            alpha: Percentage(alpha.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE)),
        }
    }

    /// Rotates the hue around the color wheel by the specified degrees.
    ///
    /// See [`HSL::rotate_hue`](HSL::rotate_hue) for color wheel reference.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSLA;
    /// let mut semi_red = HSLA::new(0.0, 1.0, 0.5, 0.5);
    /// semi_red.rotate_hue(180.0);  // Now cyan
    /// ```
    #[inline]
    pub fn rotate_hue(&mut self, degrees: f32) {
        self.hue = (self.hue.0 + degrees).rem_euclid(HUE_MAX_DEGREES).into();
    }

    /// Adjusts saturation (color intensity).
    ///
    /// See [`HSL::saturate`](HSL::saturate) for detailed explanation.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSLA;
    /// let mut color = HSLA::new(120.0, 0.5, 0.5, 0.8);
    /// color.saturate(0.3);   // Now 0.8 saturation
    /// color.saturate(-0.3);  // Now 0.5 saturation
    /// ```
    #[inline]
    pub fn saturate(&mut self, amount: f32) {
        self.saturation = (self.saturation.0 + amount).clamp(0.0, 1.0).into();
    }

    /// Adjusts lightness (brightness).
    ///
    /// See [`HSL::lighten`](HSL::lighten) for detailed explanation.
    /// Alpha is preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::HSLA;
    /// let mut color = HSLA::new(240.0, 1.0, 0.5, 1.0);
    /// color.lighten(0.2);  // Now 0.7 lightness
    /// color.lighten(-0.3); // Now 0.4 lightness
    /// ```
    #[inline]
    pub fn lighten(&mut self, amount: f32) {
        self.lightness = (self.lightness.0 + amount).clamp(0.0, 1.0).into();
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
    /// # use toyengine::primitives::color::HSLA;
    /// let mut color = HSLA::new(0.0, 1.0, 0.5, 0.8);  // 80% opaque
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
    /// # use toyengine::primitives::color::HSLA;
    /// let result = HSLA::new(0.0, 1.0, 0.5, 1.0)
    ///     .with_rotate_hue(30.0)
    ///     .with_saturate(-0.2)
    ///     .with_lighten(0.1)
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

    /// Chainable version of [`lighten`](Self::lighten).
    #[inline]
    pub fn with_lighten(mut self, amount: f32) -> Self {
        self.lighten(amount);
        self
    }

    /// Chainable version of [`fade`](Self::fade).
    #[inline]
    pub fn with_fade(mut self, amount: f32) -> Self {
        self.fade(amount);
        self
    }
}
