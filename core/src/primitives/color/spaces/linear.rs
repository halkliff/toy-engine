//! Linear RGB color space for physically accurate rendering.
//!
//! Linear RGB is essential for correct lighting calculations, blending, and color interpolation.
//! Unlike sRGB (which has gamma correction), linear RGB values are proportional to light intensity.

use crate::primitives::color::{Color, MAX_PERCENTAGE, MIN_PERCENTAGE, Solid};
use crate::primitives::vec;

/// Linear RGB color with floating-point channels in [0.0, 1.0] range.
///
/// This color space is physically linear, meaning:
/// - Doubling a value doubles the light intensity
/// - Blending colors produces correct results
/// - Required for proper lighting calculations
///
/// # When to Use
/// - Lighting calculations (Phong, PBR)
/// - Physically accurate color interpolation
/// - Texture filtering
/// - Post-processing effects
///
/// # Conversion
/// sRGB → Linear: Apply inverse gamma (~2.2)
/// Linear → sRGB: Apply gamma (~1/2.2)
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::{RGB, LinearRGB};
/// // Convert from sRGB
/// let srgb = RGB::new(128, 128, 128);
/// let linear: LinearRGB = srgb.into();
///
/// // Do lighting calculations in linear space
/// let lit = LinearRGB::new(linear.r * 2.0, linear.g * 2.0, linear.b * 2.0);
///
/// // Convert back to sRGB for display
/// let display: RGB = lit.into();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRGB {
    /// Red component in linear space [0.0, 1.0]
    pub red: f32,
    /// Green component in linear space [0.0, 1.0]
    pub green: f32,
    /// Blue component in linear space [0.0, 1.0]
    pub blue: f32,
}
/// Linear ARGB color with alpha channel for physically accurate rendering with transparency.
///
/// Extends LinearRGB with an alpha channel, maintaining physically linear RGB values
/// while supporting transparency. Unlike PremultipliedARGB, the RGB values are NOT
/// multiplied by alpha (straight alpha).
///
/// # When to Use
/// - Lighting calculations with transparency
/// - Blending operations that need separate RGB and alpha
/// - Post-processing effects with alpha
///
/// # Conversion
/// - To PremultipliedARGB: Multiply RGB by alpha for GPU-friendly format
/// - To ARGB: Apply gamma correction and convert to u8
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::LinearARGB;
/// let semi_transparent = LinearARGB::new(0.8, 0.6, 0.4, 0.5);
/// let doubled = semi_transparent * 2.0; // Doubles RGB, preserves alpha
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearARGB {
    /// Red component in linear space [0.0, 1.0]
    pub red: f32,
    /// Green component in linear space [0.0, 1.0]
    pub green: f32,
    /// Blue component in linear space [0.0, 1.0]
    pub blue: f32,
    /// Alpha component [0.0, 1.0]
    pub alpha: f32,
}

impl Solid for LinearRGB {}
impl Color for LinearRGB {}
impl Color for LinearARGB {}

/// Macro to define LinearRGB color constants
/// Usage: linear_rgb_colors! { NAME = (r, g, b); ... }
#[macro_export]
macro_rules! linear_rgb_colors {
    ($($name:ident = ($r:expr, $g:expr, $b:expr));* $(;)?) => {
        $(
            pub const $name: Self = LinearRGB {
                red: $r,
                green: $g,
                blue: $b,
            };
        )*
    };
}

impl LinearRGB {
    /// Creates a new linear RGB color.
    ///
    /// Values are clamped to [0.0, 1.0] range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(0.5, 0.3, 0.1);
    /// let bright = LinearRGB::new(2.0, 2.0, 2.0); // Clamped to 1.0
    /// ```
    #[inline]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        LinearRGB {
            red: r.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            green: g.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            blue: b.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }

    /// Creates a linear RGB color without clamping.
    ///
    /// Use this when you know values are already in valid range
    /// or when you want to allow HDR values > 1.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let hdr = LinearRGB::new_unchecked(2.5, 1.8, 3.0); // HDR values
    /// ```
    #[inline]
    pub const fn new_unchecked(r: f32, g: f32, b: f32) -> Self {
        LinearRGB {
            red: r,
            green: g,
            blue: b,
        }
    }

    /// Linear interpolation between two colors.
    ///
    /// This produces physically correct results since we're in linear space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let dark = LinearRGB::new(0.1, 0.1, 0.1);
    /// let bright = LinearRGB::new(0.9, 0.9, 0.9);
    /// let mid = dark.lerp(&bright, 0.5);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        LinearRGB {
            red: self.red + (other.red - self.red) * t,
            green: self.green + (other.green - self.green) * t,
            blue: self.blue + (other.blue - self.blue) * t,
        }
    }

    /// Multiply by a scalar (brightness adjustment).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut color = LinearRGB::new(0.5, 0.3, 0.2);
    /// color.scale(2.0); // Double brightness
    /// ```
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        self.red = (self.red * factor).clamp(0.0, 1.0);
        self.green = (self.green * factor).clamp(0.0, 1.0);
        self.blue = (self.blue * factor).clamp(0.0, 1.0);
    }

    linear_rgb_colors!(
        BLACK = (0.0, 0.0, 0.0);
        WHITE = (1.0, 1.0, 1.0);
        RED = (1.0, 0.0, 0.0);
        GREEN = (0.0, 1.0, 0.0);
        BLUE = (0.0, 0.0, 1.0);
        YELLOW = (1.0, 1.0, 0.0);
        CYAN = (0.0, 1.0, 1.0);
        MAGENTA = (1.0, 0.0, 1.0);
    );
}

/// Macro to define LinearARGB color constants
/// Usage: linear_argb_colors! { NAME = (a, r, g, b); ... }
#[macro_export]
macro_rules! linear_argb_colors {
    ($($name:ident = ($a:expr, $r:expr, $g:expr, $b:expr));* $(;)?) => {
        $(
            pub const $name: Self = LinearARGB {
                alpha: $a,
                red: $r,
                green: $g,
                blue: $b,
            };
        )*
    };
}

impl LinearARGB {
    /// Creates a new linear ARGB color.
    ///
    /// Values are clamped to [0.0, 1.0] range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(0.5, 0.3, 0.1, 0.8);
    /// ```
    #[inline]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        LinearARGB {
            red: r.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            green: g.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            blue: b.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            alpha: a.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }

    /// Creates a linear ARGB color without clamping.
    ///
    /// Use this when you know values are already in valid range
    /// or when you want to allow HDR values > 1.0 for RGB channels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let hdr = LinearARGB::new_unchecked(2.5, 1.8, 3.0, 1.0); // HDR RGB values
    /// ```
    #[inline]
    pub const fn new_unchecked(r: f32, g: f32, b: f32, a: f32) -> Self {
        LinearARGB {
            red: r,
            green: g,
            blue: b,
            alpha: a,
        }
    }

    /// Linear interpolation between two colors.
    ///
    /// Interpolates both RGB and alpha linearly. For correct alpha blending,
    /// consider converting to PremultipliedARGB first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let opaque = LinearARGB::new(1.0, 0.0, 0.0, 1.0);
    /// let transparent = LinearARGB::new(0.0, 0.0, 1.0, 0.0);
    /// let mid = opaque.lerp(&transparent, 0.5);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        LinearARGB {
            red: self.red + (other.red - self.red) * t,
            green: self.green + (other.green - self.green) * t,
            blue: self.blue + (other.blue - self.blue) * t,
            alpha: self.alpha + (other.alpha - self.alpha) * t,
        }
    }

    /// Multiply RGB by a scalar, preserving alpha.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut color = LinearARGB::new(0.5, 0.3, 0.2, 0.8);
    /// color.scale(2.0); // Doubles RGB brightness, alpha unchanged
    /// ```
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        self.red = (self.red * factor).clamp(0.0, 1.0);
        self.green = (self.green * factor).clamp(0.0, 1.0);
        self.blue = (self.blue * factor).clamp(0.0, 1.0);
    }

    linear_argb_colors!(
        BLACK = (1.0, 0.0, 0.0, 0.0);
        WHITE = (1.0, 1.0, 1.0, 1.0);
        RED = (1.0, 1.0, 0.0, 0.0);
        GREEN = (1.0, 0.0, 1.0, 0.0);
        BLUE = (1.0, 0.0, 0.0, 1.0);
        YELLOW = (1.0, 1.0, 1.0, 0.0);
        CYAN = (1.0, 0.0, 1.0, 1.0);
        MAGENTA = (1.0, 1.0, 0.0, 1.0);
    );
}

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// ===== LinearRGB Operators =====

impl Add for LinearRGB {
    type Output = Self;

    /// Adds two LinearRGB colors component-wise (clamping at 1.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let light1 = LinearRGB::new(0.5, 0.3, 0.1);
    /// let light2 = LinearRGB::new(0.3, 0.4, 0.2);
    /// let combined = light1 + light2;  // Physically correct light addition
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        LinearRGB {
            red: (self.red + other.red).clamp(0.0, 1.0),
            green: (self.green + other.green).clamp(0.0, 1.0),
            blue: (self.blue + other.blue).clamp(0.0, 1.0),
        }
    }
}

impl Add<&Self> for LinearRGB {
    type Output = Self;

    /// Adds two LinearRGB colors component-wise (clamping at 1.0).
    #[inline]
    fn add(self, other: &Self) -> Self {
        self + *other
    }
}

impl Add<&Self> for &LinearRGB {
    type Output = LinearRGB;

    /// Adds two LinearRGB colors component-wise (clamping at 1.0).
    #[inline]
    fn add(self, other: &Self) -> LinearRGB {
        *self + *other
    }
}

impl Add<LinearRGB> for &LinearRGB {
    type Output = LinearRGB;

    /// Adds two LinearRGB colors component-wise (clamping at 1.0).
    #[inline]
    fn add(self, other: LinearRGB) -> LinearRGB {
        *self + other
    }
}

impl AddAssign for LinearRGB {
    /// Adds another LinearRGB color component-wise in-place (clamping at 1.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut light = LinearRGB::new(0.5, 0.3, 0.1);
    /// let additional = LinearRGB::new(0.3, 0.4, 0.2);
    /// light += additional;  // Adds light in-place
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.red = (self.red + other.red).clamp(0.0, 1.0);
        self.green = (self.green + other.green).clamp(0.0, 1.0);
        self.blue = (self.blue + other.blue).clamp(0.0, 1.0);
    }
}

impl AddAssign<&Self> for LinearRGB {
    /// Adds another LinearRGB color component-wise in-place (clamping at 1.0).
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl Sub for LinearRGB {
    type Output = Self;

    /// Subtracts two LinearRGB colors component-wise (saturating at 0.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let bright = LinearRGB::new(0.8, 0.6, 0.4);
    /// let dim = LinearRGB::new(0.3, 0.2, 0.1);
    /// let difference = bright - dim;
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        LinearRGB {
            red: (self.red - other.red).max(0.0),
            green: (self.green - other.green).max(0.0),
            blue: (self.blue - other.blue).max(0.0),
        }
    }
}

impl Sub<&Self> for LinearRGB {
    type Output = Self;

    /// Subtracts two LinearRGB colors component-wise (saturating at 0.0).
    #[inline]
    fn sub(self, other: &Self) -> Self {
        self - *other
    }
}

impl Sub<&Self> for &LinearRGB {
    type Output = LinearRGB;

    /// Subtracts two LinearRGB colors component-wise (saturating at 0.0).
    #[inline]
    fn sub(self, other: &Self) -> LinearRGB {
        *self - *other
    }
}

impl Sub<LinearRGB> for &LinearRGB {
    type Output = LinearRGB;

    /// Subtracts two LinearRGB colors component-wise (saturating at 0.0).
    #[inline]
    fn sub(self, other: LinearRGB) -> LinearRGB {
        *self - other
    }
}

impl SubAssign for LinearRGB {
    /// Subtracts another LinearRGB color component-wise in-place (saturating at 0.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut color = LinearRGB::new(0.8, 0.6, 0.4);
    /// let subtract = LinearRGB::new(0.3, 0.2, 0.1);
    /// color -= subtract;
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.red = (self.red - other.red).max(0.0);
        self.green = (self.green - other.green).max(0.0);
        self.blue = (self.blue - other.blue).max(0.0);
    }
}

impl SubAssign<&Self> for LinearRGB {
    /// Subtracts another LinearRGB color component-wise in-place (saturating at 0.0).
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        *self -= *other;
    }
}

impl Mul for LinearRGB {
    type Output = Self;

    /// Multiplies two LinearRGB colors component-wise (color modulation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let light = LinearRGB::new(1.0, 0.8, 0.6);
    /// let surface = LinearRGB::new(0.5, 0.5, 0.5);
    /// let reflected = light * surface;  // Physically correct modulation
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        LinearRGB {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
        }
    }
}

impl Mul<&Self> for LinearRGB {
    type Output = Self;

    /// Multiplies two LinearRGB colors component-wise (color modulation).
    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Self> for &LinearRGB {
    type Output = LinearRGB;

    /// Multiplies two LinearRGB colors component-wise (color modulation).
    #[inline]
    fn mul(self, other: &Self) -> LinearRGB {
        *self * *other
    }
}

impl Mul<LinearRGB> for &LinearRGB {
    type Output = LinearRGB;

    /// Multiplies two LinearRGB colors component-wise (color modulation).
    #[inline]
    fn mul(self, other: LinearRGB) -> LinearRGB {
        *self * other
    }
}

impl MulAssign for LinearRGB {
    /// Multiplies this LinearRGB color by another component-wise in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut light = LinearRGB::new(1.0, 0.8, 0.6);
    /// let filter = LinearRGB::new(0.5, 0.5, 0.5);
    /// light *= filter;  // Modulate light in-place
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.red *= other.red;
        self.green *= other.green;
        self.blue *= other.blue;
    }
}

impl MulAssign<&Self> for LinearRGB {
    /// Multiplies this LinearRGB color by another component-wise in-place.
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

impl Mul<f32> for LinearRGB {
    type Output = Self;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(0.5, 0.3, 0.2);
    /// let brighter = color * 2.0;  // Double brightness
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        LinearRGB {
            red: (self.red * scalar).clamp(0.0, 1.0),
            green: (self.green * scalar).clamp(0.0, 1.0),
            blue: (self.blue * scalar).clamp(0.0, 1.0),
        }
    }
}

impl Mul<&f32> for LinearRGB {
    type Output = Self;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        self * *scalar
    }
}

impl Mul<&f32> for &LinearRGB {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, scalar: &f32) -> LinearRGB {
        *self * *scalar
    }
}

impl Mul<f32> for &LinearRGB {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, scalar: f32) -> LinearRGB {
        *self * scalar
    }
}

impl MulAssign<f32> for LinearRGB {
    /// Scales all LinearRGB channels by a scalar factor in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut color = LinearRGB::new(0.5, 0.3, 0.2);
    /// color *= 2.0;  // Doubles brightness in-place
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.red = (self.red * scalar).clamp(0.0, 1.0);
        self.green = (self.green * scalar).clamp(0.0, 1.0);
        self.blue = (self.blue * scalar).clamp(0.0, 1.0);
    }
}

impl MulAssign<&f32> for LinearRGB {
    /// Scales all LinearRGB channels by a scalar factor in-place.
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        *self *= *scalar;
    }
}

impl Mul<LinearRGB> for f32 {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(0.5, 0.3, 0.2);
    /// let brighter = 2.0 * color;  // Double brightness
    /// ```
    #[inline]
    fn mul(self, color: LinearRGB) -> LinearRGB {
        color * self
    }
}

impl Mul<&LinearRGB> for f32 {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, color: &LinearRGB) -> LinearRGB {
        *color * self
    }
}

impl Mul<&LinearRGB> for &f32 {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, color: &LinearRGB) -> LinearRGB {
        *color * *self
    }
}

impl Mul<LinearRGB> for &f32 {
    type Output = LinearRGB;

    /// Multiplies all LinearRGB channels by a scalar factor (brightness adjustment).
    #[inline]
    fn mul(self, color: LinearRGB) -> LinearRGB {
        color * *self
    }
}

impl Div<f32> for LinearRGB {
    type Output = Self;

    /// Divides all LinearRGB channels by a scalar factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(0.8, 0.6, 0.4);
    /// let dimmer = color / 2.0;  // Half brightness
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv = 1.0 / scalar;
        LinearRGB {
            red: (self.red * inv).clamp(0.0, 1.0),
            green: (self.green * inv).clamp(0.0, 1.0),
            blue: (self.blue * inv).clamp(0.0, 1.0),
        }
    }
}

impl Div<&f32> for LinearRGB {
    type Output = Self;

    /// Divides all LinearRGB channels by a scalar factor.
    #[inline]
    fn div(self, scalar: &f32) -> Self {
        self / *scalar
    }
}

impl Div<&f32> for &LinearRGB {
    type Output = LinearRGB;

    /// Divides all LinearRGB channels by a scalar factor.
    #[inline]
    fn div(self, scalar: &f32) -> LinearRGB {
        *self / *scalar
    }
}

impl Div<f32> for &LinearRGB {
    type Output = LinearRGB;

    /// Divides all LinearRGB channels by a scalar factor.
    #[inline]
    fn div(self, scalar: f32) -> LinearRGB {
        *self / scalar
    }
}

impl DivAssign<f32> for LinearRGB {
    /// Divides all LinearRGB channels by a scalar factor in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let mut color = LinearRGB::new(0.8, 0.6, 0.4);
    /// color /= 2.0;  // Halves brightness in-place
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        let inv = 1.0 / scalar;
        self.red = (self.red * inv).clamp(0.0, 1.0);
        self.green = (self.green * inv).clamp(0.0, 1.0);
        self.blue = (self.blue * inv).clamp(0.0, 1.0);
    }
}

impl DivAssign<&f32> for LinearRGB {
    /// Divides all LinearRGB channels by a scalar factor in-place.
    #[inline]
    fn div_assign(&mut self, scalar: &f32) {
        *self /= *scalar;
    }
}

// ===== LinearARGB Operators =====

impl Add for LinearARGB {
    type Output = Self;

    /// Adds two LinearARGB colors component-wise (clamping at 1.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let light1 = LinearARGB::new(0.5, 0.3, 0.1, 1.0);
    /// let light2 = LinearARGB::new(0.3, 0.4, 0.2, 1.0);
    /// let combined = light1 + light2;  // Physically correct light addition
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        LinearARGB {
            red: (self.red + other.red).clamp(0.0, 1.0),
            green: (self.green + other.green).clamp(0.0, 1.0),
            blue: (self.blue + other.blue).clamp(0.0, 1.0),
            alpha: (self.alpha + other.alpha).clamp(0.0, 1.0),
        }
    }
}

impl AddAssign for LinearARGB {
    /// Adds another LinearARGB color component-wise in-place (clamping at 1.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut light = LinearARGB::new(0.5, 0.3, 0.1, 1.0);
    /// let additional = LinearARGB::new(0.3, 0.4, 0.2, 0.0);
    /// light += additional;  // Adds light in-place
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.red = (self.red + other.red).clamp(0.0, 1.0);
        self.green = (self.green + other.green).clamp(0.0, 1.0);
        self.blue = (self.blue + other.blue).clamp(0.0, 1.0);
        self.alpha = (self.alpha + other.alpha).clamp(0.0, 1.0);
    }
}

impl Sub for LinearARGB {
    type Output = Self;

    /// Subtracts two LinearARGB colors component-wise (saturating at 0.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let bright = LinearARGB::new(0.8, 0.6, 0.4, 1.0);
    /// let dim = LinearARGB::new(0.3, 0.2, 0.1, 0.0);
    /// let difference = bright - dim;
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        LinearARGB {
            red: (self.red - other.red).max(0.0),
            green: (self.green - other.green).max(0.0),
            blue: (self.blue - other.blue).max(0.0),
            alpha: (self.alpha - other.alpha).max(0.0),
        }
    }
}

impl SubAssign for LinearARGB {
    /// Subtracts another LinearARGB color component-wise in-place (saturating at 0.0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut color = LinearARGB::new(0.8, 0.6, 0.4, 1.0);
    /// let subtract = LinearARGB::new(0.3, 0.2, 0.1, 0.0);
    /// color -= subtract;
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.red = (self.red - other.red).max(0.0);
        self.green = (self.green - other.green).max(0.0);
        self.blue = (self.blue - other.blue).max(0.0);
        self.alpha = (self.alpha - other.alpha).max(0.0);
    }
}

impl Mul<f32> for LinearARGB {
    type Output = Self;

    /// Multiplies RGB channels by a scalar factor (brightness adjustment).
    ///
    /// **Alpha is preserved unchanged** - only RGB channels are scaled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(0.5, 0.3, 0.2, 0.8);
    /// let brighter = color * 2.0;  // Double brightness, alpha stays 0.8
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        LinearARGB {
            red: (self.red * scalar).clamp(0.0, 1.0),
            green: (self.green * scalar).clamp(0.0, 1.0),
            blue: (self.blue * scalar).clamp(0.0, 1.0),
            alpha: self.alpha, // Alpha preserved
        }
    }
}

impl Mul<LinearARGB> for f32 {
    type Output = LinearARGB;

    /// Multiplies RGB channels by a scalar factor (brightness adjustment).
    ///
    /// **Alpha is preserved unchanged** - only RGB channels are scaled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(0.5, 0.3, 0.2, 0.8);
    /// let brighter = 2.0 * color;  // Double brightness, alpha stays 0.8
    /// ```
    #[inline]
    fn mul(self, color: LinearARGB) -> LinearARGB {
        color * self
    }
}

impl MulAssign<f32> for LinearARGB {
    /// Scales RGB channels by a scalar factor in-place (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut color = LinearARGB::new(0.5, 0.3, 0.2, 0.8);
    /// color *= 2.0;  // Doubles brightness in-place, alpha unchanged
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.red = (self.red * scalar).clamp(0.0, 1.0);
        self.green = (self.green * scalar).clamp(0.0, 1.0);
        self.blue = (self.blue * scalar).clamp(0.0, 1.0);
        // Alpha preserved
    }
}

impl Mul for LinearARGB {
    type Output = Self;
    /// Component-wise RGB multiplication, preserving alpha.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let light = LinearARGB::new(1.0, 0.8, 0.6, 1.0);
    /// let surface = LinearARGB::new(0.5, 0.5, 0.5, 0.8);
    /// let result = light * surface; // RGB modulated, min alpha
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        LinearARGB {
            red: self.red * other.red,
            green: self.green * other.green,
            blue: self.blue * other.blue,
            alpha: self.alpha.min(other.alpha), // Take minimum alpha
        }
    }
}

impl MulAssign for LinearARGB {
    /// Multiplies this LinearARGB color by another component-wise in-place.
    ///
    /// RGB channels are modulated, alpha is set to minimum of the two.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut light = LinearARGB::new(1.0, 0.8, 0.6, 1.0);
    /// let filter = LinearARGB::new(0.5, 0.5, 0.5, 0.8);
    /// light *= filter;  // Modulate light in-place
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.red *= other.red;
        self.green *= other.green;
        self.blue *= other.blue;
        self.alpha = self.alpha.min(other.alpha);
    }
}

impl Div<f32> for LinearARGB {
    type Output = Self;

    /// Divides RGB channels by a scalar factor (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(0.8, 0.6, 0.4, 1.0);
    /// let dimmer = color / 2.0;  // Half brightness, alpha stays 1.0
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv = 1.0 / scalar;
        LinearARGB {
            red: (self.red * inv).clamp(0.0, 1.0),
            green: (self.green * inv).clamp(0.0, 1.0),
            blue: (self.blue * inv).clamp(0.0, 1.0),
            alpha: self.alpha, // Alpha preserved
        }
    }
}

impl DivAssign<f32> for LinearARGB {
    /// Divides RGB channels by a scalar factor in-place (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let mut color = LinearARGB::new(0.8, 0.6, 0.4, 1.0);
    /// color /= 2.0;  // Halves brightness in-place
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        let inv = 1.0 / scalar;
        self.red = (self.red * inv).clamp(0.0, 1.0);
        self.green = (self.green * inv).clamp(0.0, 1.0);
        self.blue = (self.blue * inv).clamp(0.0, 1.0);
        // Alpha preserved
    }
}

// ===== LinearRGB Array/Tuple Conversions =====

impl From<[f32; 3]> for LinearRGB {
    /// Converts from a 3-element array `[f32; 3]` into [`LinearRGB`].
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let arr = [1.0, 0.5, 0.0];
    /// let color: LinearRGB = arr.into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(arr: [f32; 3]) -> Self {
        LinearRGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
        }
    }
}

impl From<&[f32; 3]> for LinearRGB {
    /// Converts from a reference to a 3-element array `&[f32; 3]` into [`LinearRGB`].
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let arr = [1.0, 0.5, 0.0];
    /// let color: LinearRGB = (&arr).into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(arr: &[f32; 3]) -> Self {
        LinearRGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
        }
    }
}

impl From<LinearRGB> for [f32; 3] {
    /// Converts between [`LinearRGB`] and a 3-element array `[f32; 3]`.
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let arr: [f32; 3] = color.into();
    /// assert_eq!(arr, [1.0, 0.5, 0.0]);
    /// ```
    fn from(color: LinearRGB) -> Self {
        [color.red, color.green, color.blue]
    }
}

impl From<&LinearRGB> for [f32; 3] {
    /// Converts between a reference to [`LinearRGB`] and a 3-element array `[f32; 3]`.
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let arr: [f32; 3] = (&color).into();
    /// assert_eq!(arr, [1.0, 0.5, 0.0]);
    /// ```
    fn from(color: &LinearRGB) -> Self {
        [color.red, color.green, color.blue]
    }
}

impl From<(f32, f32, f32)> for LinearRGB {
    /// Converts from a tuple `(f32, f32, f32)` into [`LinearRGB`].
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let tup = (1.0, 0.5, 0.0);
    /// let color: LinearRGB = tup.into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(tup: (f32, f32, f32)) -> Self {
        LinearRGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
        }
    }
}

impl From<&(f32, f32, f32)> for LinearRGB {
    /// Converts from a reference to a tuple `&(f32, f32, f32)` into [`LinearRGB`].
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let tup = (1.0, 0.5, 0.0);
    /// let color: LinearRGB = (&tup).into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(tup: &(f32, f32, f32)) -> Self {
        LinearRGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
        }
    }
}

impl From<LinearRGB> for (f32, f32, f32) {
    /// Converts between [`LinearRGB`] and a tuple `(f32, f32, f32)`.
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let tup: (f32, f32, f32) = color.into();
    /// assert_eq!(tup, (1.0, 0.5, 0.0));
    /// ```
    fn from(color: LinearRGB) -> Self {
        (color.red, color.green, color.blue)
    }
}

impl From<&LinearRGB> for (f32, f32, f32) {
    /// Converts between a reference to [`LinearRGB`] and a tuple `(f32, f32, f32)`.
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let tup: (f32, f32, f32) = (&color).into();
    /// assert_eq!(tup, (1.0, 0.5, 0.0));
    /// ```
    fn from(color: &LinearRGB) -> Self {
        (color.red, color.green, color.blue)
    }
}

// ===== LinearRGB Vector Conversions =====

impl From<vec::Vec3> for LinearRGB {
    /// Converts from a 3D vector representation ([`Vec3`](crate::primitives::vec::Vec3)) into [`LinearRGB`].
    /// The conversion maps RGB channels to vector components:
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// # use toyengine::primitives::vec;
    /// let vec = vec::Vec3 { x: 1.0, y: 0.5, z: 0.0 };
    /// let color: LinearRGB = vec.into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(v: vec::Vec3) -> Self {
        LinearRGB {
            red: v.x,
            green: v.y,
            blue: v.z,
        }
    }
}

impl From<&vec::Vec3> for LinearRGB {
    fn from(v: &vec::Vec3) -> Self {
        let rgb: LinearRGB = (*v).into();
        rgb
    }
}

impl From<vec::Vec4> for LinearRGB {
    /// Converts from a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)) into [`LinearRGB`].
    /// The conversion maps RGB channels to vector components:
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    ///
    /// The w component is ignored.
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// # use toyengine::primitives::vec;
    /// let vec = vec::Vec4 { x: 1.0, y: 0.5, z: 0.0, w: 1.0 };
    /// let color: LinearRGB = vec.into();
    /// assert_eq!(color, LinearRGB::new(1.0, 0.5, 0.0));
    /// ```
    fn from(v: vec::Vec4) -> Self {
        LinearRGB {
            red: v.x,
            green: v.y,
            blue: v.z,
        }
    }
}

impl From<&vec::Vec4> for LinearRGB {
    fn from(v: &vec::Vec4) -> Self {
        let rgb: LinearRGB = (*v).into();
        rgb
    }
}

impl From<LinearRGB> for vec::Vec3 {
    /// Converts between [`LinearRGB`] and a 3D vector representation ([`Vec3`](crate::primitives::vec::Vec3)).
    /// The conversion maps RGB channels to vector components:
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// # use toyengine::primitives::vec;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let vec: vec::Vec3 = color.into();
    /// assert_eq!(vec, vec::Vec3 { x: 1.0, y: 0.5, z: 0.0 });
    /// ```
    fn from(color: LinearRGB) -> Self {
        vec::Vec3 {
            x: color.red.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: color.green.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: color.blue.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }
}

impl From<&LinearRGB> for vec::Vec3 {
    fn from(color: &LinearRGB) -> Self {
        let vec: vec::Vec3 = (*color).into();
        vec
    }
}

impl From<LinearRGB> for vec::Vec4 {
    /// Converts between [`LinearRGB`] and a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)).
    /// The conversion maps RGB channels to vector components:
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    /// - w -> Alpha (set to 1.0)
    ///
    /// The w component is set to 1.0 (fully opaque).
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearRGB;
    /// # use toyengine::primitives::vec;
    /// let color = LinearRGB::new(1.0, 0.5, 0.0);
    /// let vec: vec::Vec4 = color.into();
    /// assert_eq!(vec, vec::Vec4 { x: 1.0, y: 0.5, z: 0.0, w: 1.0 });
    /// ```
    fn from(color: LinearRGB) -> Self {
        vec::Vec4 {
            x: color.red.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: color.green.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: color.blue.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            w: MAX_PERCENTAGE,
        }
    }
}

impl From<&LinearRGB> for vec::Vec4 {
    fn from(color: &LinearRGB) -> Self {
        let vec: vec::Vec4 = (*color).into();
        vec
    }
}

// ===== LinearARGB Array/Tuple Conversions =====

impl From<[f32; 4]> for LinearARGB {
    /// Converts from a 4-element array `[f32; 4]` into [`LinearARGB`].
    /// The array elements correspond to the ARGB channels in order: [Red, Green, Blue, Alpha].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let arr = [1.0, 0.5, 0.0, 0.8];
    /// let color: LinearARGB = arr.into();
    /// assert_eq!(color, LinearARGB::new(1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(arr: [f32; 4]) -> Self {
        LinearARGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
            alpha: arr[3],
        }
    }
}

impl From<&[f32; 4]> for LinearARGB {
    /// Converts from a reference to a 4-element array `&[f32; 4]` into [`LinearARGB`].
    /// The array elements correspond to the ARGB channels in order: [Red, Green, Blue, Alpha].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let arr = [1.0, 0.5, 0.0, 0.8];
    /// let color: LinearARGB = (&arr).into();
    /// assert_eq!(color, LinearARGB::new(1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(arr: &[f32; 4]) -> Self {
        LinearARGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
            alpha: arr[3],
        }
    }
}

impl From<LinearARGB> for [f32; 4] {
    /// Converts between [`LinearARGB`] and a 4-element array `[f32; 4]`.
    /// The array elements correspond to the ARGB channels in order: [Red, Green, Blue, Alpha].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(1.0, 0.5, 0.0, 0.8);
    /// let arr: [f32; 4] = color.into();
    /// assert_eq!(arr, [1.0, 0.5, 0.0, 0.8]);
    /// ```
    fn from(color: LinearARGB) -> Self {
        [color.red, color.green, color.blue, color.alpha]
    }
}

impl From<&LinearARGB> for [f32; 4] {
    /// Converts between a reference to [`LinearARGB`] and a 4-element array `[f32; 4]`.
    /// The array elements correspond to the ARGB channels in order: [Red, Green, Blue, Alpha].
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(1.0, 0.5, 0.0, 0.8);
    /// let arr: [f32; 4] = (&color).into();
    /// assert_eq!(arr, [1.0, 0.5, 0.0, 0.8]);
    /// ```
    fn from(color: &LinearARGB) -> Self {
        [color.red, color.green, color.blue, color.alpha]
    }
}

impl From<(f32, f32, f32, f32)> for LinearARGB {
    /// Converts from a tuple `(f32, f32, f32, f32)` into [`LinearARGB`].
    /// The tuple elements correspond to the ARGB channels in order: (Red, Green, Blue, Alpha).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let tup = (1.0, 0.5, 0.0, 0.8);
    /// let color: LinearARGB = tup.into();
    /// assert_eq!(color, LinearARGB::new(1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(tup: (f32, f32, f32, f32)) -> Self {
        LinearARGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
            alpha: tup.3,
        }
    }
}

impl From<&(f32, f32, f32, f32)> for LinearARGB {
    /// Converts from a reference to a tuple `&(f32, f32, f32, f32)` into [`LinearARGB`].
    /// The tuple elements correspond to the ARGB channels in order: (Red, Green, Blue, Alpha).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let tup = (1.0, 0.5, 0.0, 0.8);
    /// let color: LinearARGB = (&tup).into();
    /// assert_eq!(color, LinearARGB::new(1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(tup: &(f32, f32, f32, f32)) -> Self {
        LinearARGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
            alpha: tup.3,
        }
    }
}

impl From<LinearARGB> for (f32, f32, f32, f32) {
    /// Converts between [`LinearARGB`] and a tuple `(f32, f32, f32, f32)`.
    /// The tuple elements correspond to the ARGB channels in order: (Red, Green, Blue, Alpha).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(1.0, 0.5, 0.0, 0.8);
    /// let tup: (f32, f32, f32, f32) = color.into();
    /// assert_eq!(tup, (1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(color: LinearARGB) -> Self {
        (color.red, color.green, color.blue, color.alpha)
    }
}

impl From<&LinearARGB> for (f32, f32, f32, f32) {
    /// Converts between a reference to [`LinearARGB`] and a tuple `(f32, f32, f32, f32)`.
    /// The tuple elements correspond to the ARGB channels in order: (Red, Green, Blue, Alpha).
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// let color = LinearARGB::new(1.0, 0.5, 0.0, 0.8);
    /// let tup: (f32, f32, f32, f32) = (&color).into();
    /// assert_eq!(tup, (1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(color: &LinearARGB) -> Self {
        (color.red, color.green, color.blue, color.alpha)
    }
}

// ===== LinearARGB Vector Conversions =====

impl From<vec::Vec4> for LinearARGB {
    /// Converts from a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)) into [`LinearARGB`].
    /// The conversion maps ARGB channels to vector components:
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    /// - Alpha -> w
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// # use toyengine::primitives::vec;
    /// let vec = vec::Vec4 { x: 1.0, y: 0.5, z: 0.0, w: 0.8 };
    /// let color: LinearARGB = vec.into();
    /// assert_eq!(color, LinearARGB::new(1.0, 0.5, 0.0, 0.8));
    /// ```
    fn from(v: vec::Vec4) -> Self {
        LinearARGB {
            red: v.x,
            green: v.y,
            blue: v.z,
            alpha: v.w,
        }
    }
}

impl From<&vec::Vec4> for LinearARGB {
    fn from(v: &vec::Vec4) -> Self {
        let argb: LinearARGB = (*v).into();
        argb
    }
}

impl From<LinearARGB> for vec::Vec4 {
    /// Converts between [`LinearARGB`] and a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)).
    /// The conversion maps ARGB channels to vector components:
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    /// - w -> Alpha
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::color::LinearARGB;
    /// # use toyengine::primitives::vec;
    /// let color = LinearARGB::new(1.0, 0.5, 0.0, 0.8);
    /// let vec: vec::Vec4 = color.into();
    /// assert_eq!(vec, vec::Vec4 { x: 1.0, y: 0.5, z: 0.0, w: 0.8 });
    /// ```
    fn from(color: LinearARGB) -> Self {
        vec::Vec4 {
            x: color.red.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: color.green.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: color.blue.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            w: color.alpha.clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }
}

impl From<&LinearARGB> for vec::Vec4 {
    fn from(color: &LinearARGB) -> Self {
        let vec: vec::Vec4 = (*color).into();
        vec
    }
}
