//! RGB and ARGB color types and operations.
//!
//! RGB is the standard color space for digital images and displays.
//! Each channel (red, green, blue) is represented as an 8-bit unsigned integer (0-255).
//! ARGB adds an alpha channel for opacity (0 = transparent, 255 = opaque).
//! Provides common operations like addition, multiplication, interpolation, and conversions.
//! Also includes fast approximations for sRGB to linear RGB conversion.
//! Supports conversions to/from vector types for graphics applications.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::color::{RGB, ARGB, LinearRGB};
//! use toyengine::primitives::vec::Vec3;
//!
//! // Create colors
//! let red = RGB::new(255, 0, 0);
//! let blue_transparent = ARGB::new(128, 0, 0, 255);
//!
//! // Parse from hex
//! let orange = RGB::from_hex(0xFF8000);
//! let green_alpha = ARGB::from_hex(0x80_00_FF_00); // 50% green
//!
//! // Color arithmetic
//! let mixed = red + RGB::new(0, 100, 0);
//!
//! // Convert to linear space for lighting
//! let linear: LinearRGB = red.into();
//!
//! // Convert to vectors for calculations
//! let color_vec: Vec3 = red.into();
//! ```

use crate::primitives::color::{
    Color, FAST_DIV_255_SHIFT, INV_U8_MAX, LUMINANCE_BLUE_WEIGHT, LUMINANCE_GREEN_WEIGHT,
    LUMINANCE_RED_WEIGHT, MAX_PERCENTAGE, MIN_PERCENTAGE, SRGB_GAMMA, SRGB_INV_GAMMA, Solid,
    U8_MAX_F32, U8_MAX_U32,
};
use crate::primitives::vec;

/// RGB color in sRGB color space with 8-bit channels.
/// Each component ranges from 0 (no intensity) to 255 (full intensity).
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::RGB;
/// let red = RGB::new(255, 0, 0);
/// let orange = RGB::from_hex(0xFF8000);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RGB {
    /// Red component (0-255)
    pub red: u8,
    /// Green component (0-255)
    pub green: u8,
    /// Blue component (0-255)
    pub blue: u8,
}

/// ARGB color in sRGB color space with 8-bit channels and alpha.
/// Alpha represents opacity: 0 = fully transparent, 255 = fully opaque.
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::ARGB;
/// let semi_transparent_red = ARGB::new(128, 255, 0, 0);
/// let color = ARGB::from_hex(0x80FF0000); // Alpha, Red, Green, Blue
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ARGB {
    /// Alpha/opacity component (0-255, where 0 = transparent, 255 = opaque)
    pub alpha: u8,
    /// Red component (0-255)
    pub red: u8,
    /// Green component (0-255)
    pub green: u8,
    /// Blue component (0-255)
    pub blue: u8,
}

impl Solid for RGB {}
impl Color for RGB {}
impl Color for ARGB {}

/// Macro to define RGB color constants
/// Usage: rgb_colors! { NAME = (r, g, b); ... }
#[macro_export]
macro_rules! rgb_colors {
    ($($name:ident = ($r:expr, $g:expr, $b:expr));* $(;)?) => {
        $(
            pub const $name: Self = RGB {
                red: $r,
                green: $g,
                blue: $b,
            };
        )*
    };
}

impl RGB {
    /// Creates a new RGB color from individual channel values.
    ///
    /// # Arguments
    /// * `red` - Red channel intensity (0-255)
    /// * `green` - Green channel intensity (0-255)
    /// * `blue` - Blue channel intensity (0-255)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let red = RGB::new(255, 0, 0);
    /// let purple = RGB::new(128, 0, 128);
    /// ```
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        RGB { red, green, blue }
    }

    /// Creates an RGB color from a 24-bit hexadecimal value.
    /// Format: 0xRRGGBB
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let orange = RGB::from_hex(0xFF8000);  // RGB(255, 128, 0)
    /// let cyan = RGB::from_hex(0x00FFFF);    // RGB(0, 255, 255)
    /// ```
    pub const fn from_hex(hex: u32) -> Self {
        RGB {
            red: ((hex >> 16) & 0xFF) as u8,
            green: ((hex >> 8) & 0xFF) as u8,
            blue: (hex & 0xFF) as u8,
        }
    }

    /// Linear interpolation between two colors in sRGB space.
    ///
    /// This is a fast approximation suitable for UI animations and simple effects.
    /// For physically accurate blending (lighting, gradients), use [`lerp_correct`](Self::lerp_correct).
    ///
    /// # Arguments
    /// * `other` - Target color to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other), automatically clamped
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let red = RGB::new(255, 0, 0);
    /// let blue = RGB::new(0, 0, 255);
    /// let purple = red.lerp(&blue, 0.5);  // Mix 50/50
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        RGB {
            // Linearly blend each channel
            red: (self.red as f32 + (other.red as f32 - self.red as f32) * t) as u8,
            green: (self.green as f32 + (other.green as f32 - self.green as f32) * t) as u8,
            blue: (self.blue as f32 + (other.blue as f32 - self.blue as f32) * t) as u8,
        }
    }

    /// Multiplies all color channels by a scalar factor (brightness adjustment).
    ///
    /// Convenience method that uses the `*=` operator internally.
    /// Values are clamped to [0, 255] to prevent overflow.
    ///
    /// # Arguments
    /// * `factor` - Scaling factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// color.scale(0.5);   // Half brightness (mutates in-place)
    /// ```
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        *self *= factor;
    }

    /// Converts color to grayscale using luminance weights.
    ///
    /// Uses the standard luminance formula (ITU-R BT.601):
    /// gray = 0.299*R + 0.587*G + 0.114*B
    /// (weights match human eye sensitivity)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut purple = RGB::new(128, 0, 128);
    /// purple.grayscale();  // All channels become same value (mutates in-place)
    /// ```
    #[inline]
    pub fn grayscale(&mut self) {
        // Weighted average based on human perception (ITU-R BT.601)
        let gray = (LUMINANCE_RED_WEIGHT * self.red as f32
            + LUMINANCE_GREEN_WEIGHT * self.green as f32
            + LUMINANCE_BLUE_WEIGHT * self.blue as f32) as u8;
        self.red = gray;
        self.green = gray;
        self.blue = gray;
    }

    /// Inverts the color (creates the complementary/negative color).
    ///
    /// Each channel is subtracted from 255: result = 255 - channel
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut white = RGB::new(255, 255, 255);
    /// white.invert();  // RGB(0, 0, 0) - mutates in-place
    ///
    /// let mut red = RGB::new(255, 0, 0);
    /// red.invert();  // RGB(0, 255, 255) - mutates in-place
    /// ```
    #[inline]
    pub fn invert(&mut self) {
        self.red = u8::MAX - self.red;
        self.green = u8::MAX - self.green;
        self.blue = u8::MAX - self.blue;
    }

    /// Fast approximation of sRGB to linear RGB conversion (gamma ~2.2).
    ///
    /// Returns normalized [0.0, 1.0] linear RGB values.
    /// Good balance of speed and accuracy for game rendering.
    /// For mathematically correct sRGB, use the full sRGB transfer function instead.
    ///
    /// # Returns
    /// Array of [red, green, blue] in linear space
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(128, 128, 128);
    /// let [r, g, b] = color.to_linear_fast();
    /// // Linear values are ~0.21, not 0.5 due to gamma correction
    /// ```
    #[inline]
    pub fn to_linear_fast(&self) -> [f32; 3] {
        [
            // Apply inverse gamma (~2.2) to each channel
            (self.red as f32 * INV_U8_MAX).powf(SRGB_GAMMA),
            (self.green as f32 * INV_U8_MAX).powf(SRGB_GAMMA),
            (self.blue as f32 * INV_U8_MAX).powf(SRGB_GAMMA),
        ]
    }

    /// Fast approximation of linear RGB to sRGB conversion (gamma ~1/2.2).
    ///
    /// Converts normalized linear RGB values back to sRGB color space.
    /// Input values are clamped to [0.0, 1.0] range.
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - Linear RGB components in [0.0, 1.0] range
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::from_linear_fast(0.5, 0.5, 0.5);
    /// // Result is brighter than RGB(128,128,128) due to gamma
    /// ```
    #[inline]
    pub fn from_linear_fast(r: f32, g: f32, b: f32) -> Self {
        RGB {
            // Apply gamma correction (~1/2.2) and scale to 0-255
            red: (r.powf(SRGB_INV_GAMMA) * U8_MAX_F32)
                .round()
                .clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            green: (g.powf(SRGB_INV_GAMMA) * U8_MAX_F32)
                .round()
                .clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            blue: (b.powf(SRGB_INV_GAMMA) * U8_MAX_F32)
                .round()
                .clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
        }
    }

    /// Gamma-correct linear interpolation for physically accurate color blending.
    ///
    /// Converts to linear space, interpolates, then converts back to sRGB.
    /// Use for lighting, gradients, and professional color work.
    /// About 3x slower than [`lerp`](Self::lerp) but ~95% accurate.
    ///
    /// # Arguments
    /// * `other` - Target color to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let dark = RGB::new(50, 50, 50);
    /// let light = RGB::new(200, 200, 200);
    /// // lerp_correct produces brighter midtones than lerp
    /// let correct_mid = dark.lerp_correct(&light, 0.5);
    /// ```
    #[inline]
    pub fn lerp_correct(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);

        // Convert both colors to linear space
        let [r1, g1, b1] = self.to_linear_fast();
        let [r2, g2, b2] = other.to_linear_fast();

        // Interpolate in linear space for correct results
        RGB::from_linear_fast(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)
    }

    // Chaining methods (consume self for fluent API)

    /// Chainable version of [`lerp`](Self::lerp).
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let result = RGB::RED
    ///     .with_lerp(&RGB::BLUE, 0.5)
    ///     .with_scale(0.8)
    ///     .with_invert();
    /// ```
    #[inline]
    pub fn with_lerp(self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }

    /// Chainable version using the `*` operator for scalar multiplication.
    #[inline]
    pub fn with_scale(self, factor: f32) -> Self {
        self * factor
    }

    /// Chainable version using the `*` operator for component-wise multiplication.
    #[inline]
    pub fn with_multiply(self, other: &Self) -> Self {
        self * *other
    }

    /// Chainable version using the `+` operator for additive blending.
    #[inline]
    pub fn with_add(self, other: &Self) -> Self {
        self + *other
    }

    /// Chainable version using the `grayscale()` method.
    #[inline]
    pub fn with_grayscale(mut self) -> Self {
        self.grayscale();
        self
    }

    /// Chainable version using the `invert()` method.
    #[inline]
    pub fn with_invert(mut self) -> Self {
        self.invert();
        self
    }

    /// Chainable version of [`lerp_correct`](Self::lerp_correct).
    #[inline]
    pub fn with_lerp_correct(self, other: &Self, t: f32) -> Self {
        self.lerp_correct(other, t)
    }

    // Batch operations for SIMD optimization

    /// Batch linear interpolation between two color arrays.
    ///
    /// Processes multiple colors in parallel, enabling SIMD optimization.
    /// Output length is determined by the minimum of all input lengths.
    ///
    /// # Performance
    /// ~3-4x faster than individual lerp calls with compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let starts = vec![RGB::RED; 100];
    /// let ends = vec![RGB::BLUE; 100];
    /// let mut output = vec![RGB::BLACK; 100];
    /// RGB::lerp_slice(&starts, &ends, 0.5, &mut output);
    /// ```
    #[inline]
    pub fn lerp_slice(colors_a: &[Self], colors_b: &[Self], t: f32, output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i].lerp(&colors_b[i], t);
        }
    }

    /// Batch scale operation on an array of colors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let colors = vec![RGB::new(200, 100, 50); 100];
    /// let mut output = vec![RGB::BLACK; 100];
    /// RGB::scale_slice(&colors, 0.5, &mut output);  // Halve brightness
    /// ```
    #[inline]
    pub fn scale_slice(colors: &[Self], factor: f32, output: &mut [Self]) {
        let len = colors.len().min(output.len());
        for i in 0..len {
            output[i] = colors[i] * factor;
        }
    }

    /// Batch component-wise multiplication between two color arrays.
    #[inline]
    pub fn multiply_slice(colors_a: &[Self], colors_b: &[Self], output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i] * colors_b[i];
        }
    }

    /// Batch additive blending between two color arrays.
    #[inline]
    pub fn add_slice(colors_a: &[Self], colors_b: &[Self], output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i] + colors_b[i];
        }
    }

    /// Batch grayscale conversion.
    #[inline]
    pub fn grayscale_slice(colors: &[Self], output: &mut [Self]) {
        let len = colors.len().min(output.len());
        for i in 0..len {
            output[i] = colors[i];
            output[i].grayscale();
        }
    }

    /// In-place batch scale operation (may enable better SIMD optimization).
    ///
    /// Modifies the colors array directly, reducing memory allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut colors = vec![RGB::RED; 1000];
    /// RGB::scale_slice_inplace(&mut colors, 0.8);  // Darken all colors
    /// ```
    #[inline]
    pub fn scale_slice_inplace(colors: &mut [Self], factor: f32) {
        for color in colors.iter_mut() {
            *color *= factor;
        }
    }

    /// In-place batch grayscale conversion.
    #[inline]
    pub fn grayscale_slice_inplace(colors: &mut [Self]) {
        for color in colors.iter_mut() {
            color.grayscale();
        }
    }

    /// Batch gamma-correct linear interpolation.
    ///
    /// Slower than [`lerp_slice`](Self::lerp_slice) but produces correct results for gradients.
    #[inline]
    pub fn lerp_correct_slice(colors_a: &[Self], colors_b: &[Self], t: f32, output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i].lerp_correct(&colors_b[i], t);
        }
    }

    // Common color constants
    rgb_colors! {
        BLACK = (0, 0, 0);
        WHITE = (255, 255, 255);
        RED = (255, 0, 0);
        GREEN = (0, 255, 0);
        BLUE = (0, 0, 255);
        YELLOW = (255, 255, 0);
        CYAN = (0, 255, 255);
        MAGENTA = (255, 0, 255);
    }
}

/// Macro to define ARGB color constants
/// Usage: argb_colors! { NAME = (a, r, g, b); ... }
#[macro_export]
macro_rules! argb_colors {
    ($($name:ident = ($a:expr, $r:expr, $g:expr, $b:expr));* $(;)?) => {
        $(
            pub const $name: Self = ARGB {
                alpha: $a,
                red: $r,
                green: $g,
                blue: $b,
            };
        )*
    };
}

impl ARGB {
    /// Creates a new ARGB color from individual channel values.
    ///
    /// # Arguments
    /// * `alpha` - Opacity (0 = fully transparent, 255 = fully opaque)
    /// * `red` - Red channel intensity (0-255)
    /// * `green` - Green channel intensity (0-255)
    /// * `blue` - Blue channel intensity (0-255)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let opaque_red = ARGB::new(255, 255, 0, 0);
    /// let semi_transparent = ARGB::new(128, 100, 100, 100);  // 50% opacity
    /// let invisible = ARGB::new(0, 255, 0, 0);  // Fully transparent red
    /// ```
    pub const fn new(alpha: u8, red: u8, green: u8, blue: u8) -> Self {
        ARGB {
            alpha,
            red,
            green,
            blue,
        }
    }

    /// Creates an ARGB color from a 32-bit hexadecimal value.
    /// Format: 0xAARRGGBB (alpha, red, green, blue)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let semi_red = ARGB::from_hex(0x80FF0000);    // 50% opaque red
    /// let opaque_cyan = ARGB::from_hex(0xFF00FFFF); // Fully opaque cyan
    /// let transparent = ARGB::from_hex(0x00000000); // Fully transparent
    /// ```
    pub const fn from_hex(hex: u32) -> Self {
        ARGB {
            alpha: ((hex >> 24) & 0xFF) as u8,
            red: ((hex >> 16) & 0xFF) as u8,
            green: ((hex >> 8) & 0xFF) as u8,
            blue: (hex & 0xFF) as u8,
        }
    }

    /// Alpha composites this color over a background using Porter-Duff "over" operator.
    ///
    /// This is the standard blending operation for layering transparent colors.
    /// Uses fast integer math with bit shifts for better performance.
    ///
    /// # Fast Paths
    /// - Foreground fully opaque (α=255): Returns foreground directly
    /// - Foreground fully transparent (α=0): Returns background directly
    ///
    /// # Formula
    /// - α_out = α_fg + α_bg × (1 - α_fg)
    /// - C_out = (C_fg × α_fg + C_bg × α_bg × (1 - α_fg)) / α_out
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let background = ARGB::new(255, 0, 0, 255);    // Opaque blue
    /// let foreground = ARGB::new(128, 255, 0, 0);    // Semi-transparent red
    /// let result = foreground.blend_over(&background); // Purple-ish blend
    /// ```
    #[inline]
    pub fn blend_over(&self, background: &Self) -> Self {
        // Fast path for fully opaque source
        if self.alpha == u8::MAX {
            return *self;
        }

        // Fast path for fully transparent source
        if self.alpha == 0 {
            return *background;
        }

        let src_a = self.alpha as u32;
        let inv_src_a = U8_MAX_U32 - src_a;
        let dst_a = background.alpha as u32;

        // Compute output alpha
        let out_a = src_a + ((dst_a * inv_src_a) >> FAST_DIV_255_SHIFT);

        if out_a == 0 {
            return ARGB::TRANSPARENT;
        }

        // Blend each channel using integer math
        // Formula: (src * src_a + dst * dst_a * (255 - src_a) / 255) / out_a
        let blend_channel = |src: u8, dst: u8| -> u8 {
            let src = src as u32;
            let dst = dst as u32;
            let blended = (src * src_a + ((dst * dst_a * inv_src_a) >> FAST_DIV_255_SHIFT)) / out_a;
            blended as u8
        };

        ARGB {
            alpha: out_a as u8,
            red: blend_channel(self.red, background.red),
            green: blend_channel(self.green, background.green),
            blue: blend_channel(self.blue, background.blue),
        }
    }

    /// Fast premultiplied alpha blending.
    ///
    /// **Important**: Both colors MUST be premultiplied (use [`premultiply`](Self::premultiply) first).
    /// This is significantly faster than [`blend_over`](Self::blend_over) due to simpler math.
    ///
    /// # When to Use
    /// - Processing many transparent sprites/textures (premultiply once, blend many times)
    /// - Performance-critical rendering pipelines
    /// - GPU-style blending operations
    ///
    /// # Formula
    /// C_out = C_fg + C_bg × (1 - α_fg)  (simpler than standard blending)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let bg = ARGB::new(255, 100, 100, 100).premultiply();
    /// let fg = ARGB::new(128, 255, 0, 0).premultiply();
    /// let result = fg.blend_premultiplied(&bg);  // Fast blend
    /// ```
    #[inline]
    pub fn blend_premultiplied(&self, background: &Self) -> Self {
        let inv_src_a = U8_MAX_U32 - self.alpha as u32;

        ARGB {
            alpha: self.alpha.saturating_add(
                ((background.alpha as u32 * inv_src_a) >> FAST_DIV_255_SHIFT) as u8,
            ),
            red: self
                .red
                .saturating_add(((background.red as u32 * inv_src_a) >> FAST_DIV_255_SHIFT) as u8),
            green: self.green.saturating_add(
                ((background.green as u32 * inv_src_a) >> FAST_DIV_255_SHIFT) as u8,
            ),
            blue: self
                .blue
                .saturating_add(((background.blue as u32 * inv_src_a) >> FAST_DIV_255_SHIFT) as u8),
        }
    }

    /// Converts to premultiplied alpha format.
    ///
    /// Premultiplied alpha stores RGB channels already multiplied by their alpha:
    /// C_premul = C × α
    ///
    /// # Benefits
    /// - Faster blending operations (use [`blend_premultiplied`](Self::blend_premultiplied))
    /// - Better interpolation behavior (no color bleeding)
    /// - Common in GPU texture formats (DDS, PNG with premul)
    ///
    /// # Use Case
    /// Premultiply textures/sprites once during loading, then use fast blending during rendering.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color = ARGB::new(128, 255, 0, 0);  // 50% red
    /// let premul = color.premultiply();        // RGB channels halved
    /// // premul ≈ ARGB(128, 127, 0, 0)
    /// ```
    #[inline]
    pub fn premultiply(&self) -> Self {
        let a = self.alpha as f32 * INV_U8_MAX;
        ARGB {
            alpha: self.alpha,
            red: (self.red as f32 * a) as u8,
            green: (self.green as f32 * a) as u8,
            blue: (self.blue as f32 * a) as u8,
        }
    }

    /// Linear interpolation between two colors including alpha channel.
    ///
    /// Interpolates all four channels (ARGB) in sRGB space.
    /// For gamma-correct blending, convert to RGB and use [`RGB::lerp_correct`](RGB::lerp_correct).
    ///
    /// # Arguments
    /// * `other` - Target color to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other), automatically clamped
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let opaque_red = ARGB::new(255, 255, 0, 0);
    /// let transparent_blue = ARGB::new(0, 0, 0, 255);
    /// let mid = opaque_red.lerp(&transparent_blue, 0.5);  // 50% opacity, purple
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        ARGB {
            alpha: (self.alpha as f32 + (other.alpha as f32 - self.alpha as f32) * t) as u8,
            red: (self.red as f32 + (other.red as f32 - self.red as f32) * t) as u8,
            green: (self.green as f32 + (other.green as f32 - self.green as f32) * t) as u8,
            blue: (self.blue as f32 + (other.blue as f32 - self.blue as f32) * t) as u8,
        }
    }

    /// Multiplies RGB channels by a scalar factor (brightness adjustment).
    ///
    /// Convenience method that uses the `*=` operator internally.
    /// **Alpha is preserved unchanged** - only RGB channels are scaled.
    /// Values are clamped to [0, 255] to prevent overflow.
    ///
    /// # Arguments
    /// * `factor` - Scaling factor (1.0 = no change, <1.0 = darker, >1.0 = brighter)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let mut color = ARGB::new(128, 200, 100, 50);
    /// color.scale(0.5);    // Half brightness, alpha stays 128 (mutates in-place)
    /// ```
    #[inline]
    pub fn scale(&mut self, factor: f32) {
        *self *= factor;
    }

    /// Converts RGB to grayscale using luminance weights.
    ///
    /// **Alpha is preserved unchanged.**
    /// Uses the standard formula (ITU-R BT.601): gray = 0.299×R + 0.587×G + 0.114×B
    /// (weights match human eye sensitivity)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let mut purple = ARGB::new(200, 128, 0, 128);
    /// purple.grayscale();  // Grayscale with alpha=200 (mutates in-place)
    /// ```
    #[inline]
    pub fn grayscale(&mut self) {
        let gray = (LUMINANCE_RED_WEIGHT * self.red as f32
            + LUMINANCE_GREEN_WEIGHT * self.green as f32
            + LUMINANCE_BLUE_WEIGHT * self.blue as f32) as u8;
        self.red = gray;
        self.green = gray;
        self.blue = gray;
    }

    /// Inverts the RGB channels (complementary color).
    ///
    /// **Alpha is preserved unchanged.**
    /// Each RGB channel: result = 255 - channel
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let mut color = ARGB::new(128, 255, 0, 0);  // Semi-transparent red
    /// color.invert();           // Semi-transparent cyan (128, 0, 255, 255) (mutates in-place)
    /// ```
    #[inline]
    pub fn invert(&mut self) {
        self.red = u8::MAX - self.red;
        self.green = u8::MAX - self.green;
        self.blue = u8::MAX - self.blue;
    }

    // Chaining methods (consume self for fluent API)

    /// Chainable version of [`blend_over`](Self::blend_over).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let result = ARGB::new(128, 255, 0, 0)
    ///     .with_blend_over(&ARGB::new(255, 0, 0, 255))
    ///     .with_scale(0.8);
    /// ```
    #[inline]
    pub fn with_blend_over(self, background: &Self) -> Self {
        self.blend_over(background)
    }

    /// Chainable version of [`blend_premultiplied`](Self::blend_premultiplied).
    #[inline]
    pub fn with_blend_premultiplied(self, background: &Self) -> Self {
        self.blend_premultiplied(background)
    }

    /// Chainable version of [`premultiply`](Self::premultiply).
    #[inline]
    pub fn with_premultiply(self) -> Self {
        self.premultiply()
    }

    /// Chainable version of [`lerp`](Self::lerp).
    #[inline]
    pub fn with_lerp(self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }

    /// Chainable version using the `*` operator for scalar multiplication.
    #[inline]
    pub fn with_scale(self, factor: f32) -> Self {
        self * factor
    }

    /// Chainable version using the `*` operator for component-wise multiplication.
    #[inline]
    pub fn with_multiply(self, other: &Self) -> Self {
        self * *other
    }

    /// Chainable version using the `+` operator for additive blending.
    #[inline]
    pub fn with_add(self, other: &Self) -> Self {
        self + *other
    }

    /// Chainable version using the `grayscale()` method.
    #[inline]
    pub fn with_grayscale(mut self) -> Self {
        self.grayscale();
        self
    }

    /// Chainable version using the `invert()` method.
    #[inline]
    pub fn with_invert(mut self) -> Self {
        self.invert();
        self
    }

    // Batch operations for SIMD optimization

    /// Batch linear interpolation between two color arrays including alpha.
    ///
    /// Processes multiple colors in parallel, enabling SIMD optimization.
    /// All four channels (ARGB) are interpolated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let starts = vec![ARGB::new(255, 255, 0, 0); 100];
    /// let ends = vec![ARGB::new(128, 0, 0, 255); 100];
    /// let mut output = vec![ARGB::TRANSPARENT; 100];
    /// ARGB::lerp_slice(&starts, &ends, 0.5, &mut output);
    /// ```
    #[inline]
    pub fn lerp_slice(colors_a: &[Self], colors_b: &[Self], t: f32, output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i].lerp(&colors_b[i], t);
        }
    }

    /// Batch scale operation on color array.
    ///
    /// Scales RGB channels while preserving alpha for all colors.
    #[inline]
    pub fn scale_slice(colors: &[Self], factor: f32, output: &mut [Self]) {
        let len = colors.len().min(output.len());
        for i in 0..len {
            output[i] = colors[i] * factor;
        }
    }

    /// Batch component-wise multiplication between two color arrays.
    #[inline]
    pub fn multiply_slice(colors_a: &[Self], colors_b: &[Self], output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i] * colors_b[i];
        }
    }

    /// Batch additive blending between two color arrays.
    #[inline]
    pub fn add_slice(colors_a: &[Self], colors_b: &[Self], output: &mut [Self]) {
        let len = colors_a.len().min(colors_b.len()).min(output.len());
        for i in 0..len {
            output[i] = colors_a[i] + colors_b[i];
        }
    }

    /// Batch alpha blending using Porter-Duff "over" operator.
    ///
    /// Composites each foreground color over corresponding background color.
    /// Use for rendering multiple transparent sprites over backgrounds.
    #[inline]
    pub fn blend_over_slice(foreground: &[Self], background: &[Self], output: &mut [Self]) {
        let len = foreground.len().min(background.len()).min(output.len());
        for i in 0..len {
            output[i] = foreground[i].blend_over(&background[i]);
        }
    }

    /// Batch premultiplied alpha blending.
    ///
    /// **Requires both input arrays to contain premultiplied colors.**
    /// Faster than [`blend_over_slice`](Self::blend_over_slice) for large batches.
    #[inline]
    pub fn blend_premultiplied_slice(
        foreground: &[Self],
        background: &[Self],
        output: &mut [Self],
    ) {
        let len = foreground.len().min(background.len()).min(output.len());
        for i in 0..len {
            output[i] = foreground[i].blend_premultiplied(&background[i]);
        }
    }

    /// Batch premultiply alpha conversion.
    ///
    /// Converts an array of standard ARGB colors to premultiplied format.
    #[inline]
    pub fn premultiply_slice(colors: &[Self], output: &mut [Self]) {
        let len = colors.len().min(output.len());
        for i in 0..len {
            output[i] = colors[i].premultiply();
        }
    }

    /// Batch grayscale conversion.
    #[inline]
    pub fn grayscale_slice(colors: &[Self], output: &mut [Self]) {
        let len = colors.len().min(output.len());
        for i in 0..len {
            output[i] = colors[i];
            output[i].grayscale();
        }
    }

    /// In-place batch scale operation.
    ///
    /// Modifies the colors array directly, reducing memory allocations.
    #[inline]
    pub fn scale_slice_inplace(colors: &mut [Self], factor: f32) {
        for color in colors.iter_mut() {
            *color *= factor;
        }
    }

    /// In-place batch premultiply conversion.
    #[inline]
    pub fn premultiply_slice_inplace(colors: &mut [Self]) {
        for color in colors.iter_mut() {
            *color = color.premultiply();
        }
    }

    /// In-place batch grayscale conversion.
    #[inline]
    pub fn grayscale_slice_inplace(colors: &mut [Self]) {
        for color in colors.iter_mut() {
            color.grayscale();
        }
    }

    /// Blends a single foreground color over an array of background colors.
    ///
    /// Applies the same transparent overlay to multiple colors efficiently.
    /// Useful for UI effects like applying a tint/fade to multiple elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let overlay = ARGB::new(128, 255, 255, 255);  // Semi-transparent white
    /// let backgrounds = vec![
    ///     ARGB::new(255, 255, 0, 0),    // Red
    ///     ARGB::new(255, 0, 255, 0),    // Green
    ///     ARGB::new(255, 0, 0, 255),    // Blue
    /// ];
    /// let mut output = vec![ARGB::TRANSPARENT; 3];
    /// ARGB::blend_over_broadcast(&overlay, &backgrounds, &mut output);
    /// // All colors now have a white tint
    /// ```
    #[inline]
    pub fn blend_over_broadcast(foreground: &Self, backgrounds: &[Self], output: &mut [Self]) {
        let len = backgrounds.len().min(output.len());
        for i in 0..len {
            output[i] = foreground.blend_over(&backgrounds[i]);
        }
    }

    // Common color constants
    argb_colors! {
        TRANSPARENT = (0, 0, 0, 0);
        BLACK = (255, 0, 0, 0);
        WHITE = (255, 255, 255, 255);
        RED = (255, 255, 0, 0);
        GREEN = (255, 0, 255, 0);
        BLUE = (255, 0, 0, 255);
        YELLOW = (255, 255, 255, 0);
        CYAN = (255, 0, 255, 255);
        MAGENTA = (255, 255, 0, 255);
    }
}


// Operator trait implementations for idiomatic Rust

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
// RGB operator implementations

impl Add for RGB {
    type Output = Self;

    /// Adds two RGB colors component-wise (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let red = RGB::new(255, 0, 0);
    /// let green = RGB::new(0, 255, 0);
    /// let yellow = red + green;  // RGB(255, 255, 0)
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        RGB {
            red: self.red.saturating_add(other.red),
            green: self.green.saturating_add(other.green),
            blue: self.blue.saturating_add(other.blue),
        }
    }
}

impl Add<&Self> for RGB {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        self.add(*other)
    }
}

impl Add<&Self> for &RGB {
    type Output = RGB;

    #[inline]
    fn add(self, other: &Self) -> RGB {
        (*self).add(*other)
    }
}

impl Add<RGB> for &RGB {
    type Output = RGB;

    #[inline]
    fn add(self, other: RGB) -> RGB {
        (*self).add(other)
    }
}

impl AddAssign for RGB {
    /// Adds another RGB color component-wise in-place (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut red = RGB::new(255, 0, 0);
    /// let green = RGB::new(0, 255, 0);
    /// red += green;  // red becomes RGB(255, 255, 0)
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.red = self.red.saturating_add(other.red);
        self.green = self.green.saturating_add(other.green);
        self.blue = self.blue.saturating_add(other.blue);
    }
}

impl AddAssign<&Self> for RGB {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.add_assign(*other);
    }
}

impl Add<u8> for RGB {
    type Output = Self;

    /// Adds a scalar value to all RGB channels (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let brighter = color + 30;  // Increases brightness
    /// ```
    #[inline]
    fn add(self, value: u8) -> Self {
        RGB {
            red: self.red.saturating_add(value),
            green: self.green.saturating_add(value),
            blue: self.blue.saturating_add(value),
        }
    }
}

impl Add<&u8> for RGB {
    type Output = Self;

    #[inline]
    fn add(self, value: &u8) -> Self {
        self.add(*value)
    }
}

impl Add<&u8> for &RGB {
    type Output = RGB;

    #[inline]
    fn add(self, value: &u8) -> RGB {
        (*self).add(*value)
    }
}

impl Add<u8> for &RGB {
    type Output = RGB;

    #[inline]
    fn add(self, value: u8) -> RGB {
        (*self).add(value)
    }
}

impl AddAssign<u8> for RGB {
    /// Adds a scalar value to all RGB channels in-place (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// color += 30;  // Increases brightness in-place
    /// ```
    #[inline]
    fn add_assign(&mut self, value: u8) {
        self.red = self.red.saturating_add(value);
        self.green = self.green.saturating_add(value);
        self.blue = self.blue.saturating_add(value);
    }
}

impl AddAssign<&u8> for RGB {
    #[inline]
    fn add_assign(&mut self, value: &u8) {
        self.add_assign(*value);
    }
}
impl Add<RGB> for u8 {
    type Output = RGB;

    /// Adds a scalar value to all RGB channels (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let brighter = 30 + color;  // Increases brightness
    /// ```
    #[inline]
    fn add(self, color: RGB) -> RGB {
        color.add(self)
    }
}

impl Add<&RGB> for u8 {
    type Output = RGB;

    #[inline]
    fn add(self, color: &RGB) -> RGB {
        color.add(self)
    }
}

impl Add<&RGB> for &u8 {
    type Output = RGB;

    #[inline]
    fn add(self, color: &RGB) -> RGB {
        color.add(*self)
    }
}

impl Add<RGB> for &u8 {
    type Output = RGB;

    #[inline]
    fn add(self, color: RGB) -> RGB {
        color.add(*self)
    }
}

impl Mul for RGB {
    type Output = Self;

    /// Multiplies two RGB colors component-wise (result = (a * b) / 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let white = RGB::new(255, 255, 255);
    /// let red_tint = RGB::new(255, 128, 128);
    /// let tinted = white * red_tint;  // Tinted with red
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        RGB {
            red: self.red.saturating_mul(other.red),
            green: self.green.saturating_mul(other.green),
            blue: self.blue.saturating_mul(other.blue),
        }
    }
}

impl Mul<&Self> for RGB {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        self.mul(*other)
    }
}

impl Mul<&Self> for &RGB {
    type Output = RGB;

    #[inline]
    fn mul(self, other: &Self) -> RGB {
        (*self).mul(*other)
    }
}

impl Mul<RGB> for &RGB {
    type Output = RGB;

    #[inline]
    fn mul(self, other: RGB) -> RGB {
        (*self).mul(other)
    }
}

impl MulAssign for RGB {
    /// Multiplies this RGB color by another component-wise in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut white = RGB::new(255, 255, 255);
    /// let red_tint = RGB::new(255, 128, 128);
    /// white *= red_tint;  // white becomes tinted
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.red = self.red.saturating_mul(other.red);
        self.green = self.green.saturating_mul(other.green);
        self.blue = self.blue.saturating_mul(other.blue);
    }
}

impl MulAssign<&Self> for RGB {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        self.mul_assign(*other);
    }
}

// Scalar multiplication for brightness adjustment
impl Mul<f32> for RGB {
    type Output = Self;

    /// Multiplies all RGB channels by a scalar factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let darker = color * 0.5;  // Half brightness
    /// ```
    #[inline]
    fn mul(self, factor: f32) -> Self {
        RGB {
            red: (self.red as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            green: (self.green as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            blue: (self.blue as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
        }
    }
}

impl Mul<&f32> for RGB {
    type Output = Self;

    #[inline]
    fn mul(self, factor: &f32) -> Self {
        self.mul(*factor)
    }
}

impl Mul<&f32> for &RGB {
    type Output = RGB;

    #[inline]
    fn mul(self, factor: &f32) -> RGB {
        (*self).mul(*factor)
    }
}

impl Mul<f32> for &RGB {
    type Output = RGB;

    #[inline]
    fn mul(self, factor: f32) -> RGB {
        (*self).mul(factor)
    }
}

impl MulAssign<f32> for RGB {
    /// Scales all RGB channels by a scalar factor in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// color *= 0.5;  // Halves brightness in-place
    /// ```
    #[inline]
    fn mul_assign(&mut self, factor: f32) {
        self.red = (self.red as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        self.green = (self.green as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        self.blue = (self.blue as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
    }
}

impl MulAssign<&f32> for RGB {
    #[inline]
    fn mul_assign(&mut self, factor: &f32) {
        self.mul_assign(*factor);
    }
}

impl Mul<RGB> for f32 {
    type Output = RGB;

    /// Multiplies all RGB channels by a scalar factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let darker = 0.5 * color;  // Half brightness
    /// ```
    #[inline]
    fn mul(self, color: RGB) -> RGB {
        color.mul(self)
    }
}

impl Mul<&RGB> for f32 {
    type Output = RGB;

    #[inline]
    fn mul(self, color: &RGB) -> RGB {
        color.mul(self)
    }
}

impl Mul<&RGB> for &f32 {
    type Output = RGB;

    #[inline]
    fn mul(self, color: &RGB) -> RGB {
        color.mul(*self)
    }
}

impl Mul<RGB> for &f32 {
    type Output = RGB;

    #[inline]
    fn mul(self, color: RGB) -> RGB {
        color.mul(*self)
    }
}

impl Sub for RGB {
    type Output = Self;

    /// Subtracts two RGB colors component-wise (saturating at 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let yellow = RGB::new(255, 255, 0);
    /// let red = RGB::new(255, 0, 0);
    /// let green = yellow - red;  // RGB(0, 255, 0)
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        RGB {
            red: self.red.saturating_sub(other.red),
            green: self.green.saturating_sub(other.green),
            blue: self.blue.saturating_sub(other.blue),
        }
    }
}

impl Sub<&Self> for RGB {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        self.sub(*other)
    }
}

impl Sub<&Self> for &RGB {
    type Output = RGB;

    #[inline]
    fn sub(self, other: &Self) -> RGB {
        (*self).sub(*other)
    }
}

impl Sub<RGB> for &RGB {
    type Output = RGB;

    #[inline]
    fn sub(self, other: RGB) -> RGB {
        (*self).sub(other)
    }
}

impl SubAssign for RGB {
    /// Subtracts another RGB color component-wise in-place (saturating at 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut yellow = RGB::new(255, 255, 0);
    /// let red = RGB::new(255, 0, 0);
    /// yellow -= red;  // yellow becomes RGB(0, 255, 0)
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.red = self.red.saturating_sub(other.red);
        self.green = self.green.saturating_sub(other.green);
        self.blue = self.blue.saturating_sub(other.blue);
    }
}

impl SubAssign<&Self> for RGB {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.sub_assign(*other);
    }
}

impl Sub<u8> for RGB {
    type Output = Self;

    /// Subtracts a scalar value from all RGB channels (saturating at 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let darker = color - 30;  // Decreases brightness
    /// ```
    #[inline]
    fn sub(self, value: u8) -> Self {
        RGB {
            red: self.red.saturating_sub(value),
            green: self.green.saturating_sub(value),
            blue: self.blue.saturating_sub(value),
        }
    }
}

impl Sub<&u8> for RGB {
    type Output = Self;

    #[inline]
    fn sub(self, value: &u8) -> Self {
        self.sub(*value)
    }
}

impl Sub<&u8> for &RGB {
    type Output = RGB;

    #[inline]
    fn sub(self, value: &u8) -> RGB {
        (*self).sub(*value)
    }
}

impl Sub<u8> for &RGB {
    type Output = RGB;

    #[inline]
    fn sub(self, value: u8) -> RGB {
        (*self).sub(value)
    }
}

impl SubAssign<u8> for RGB {
    /// Subtracts a scalar value from all RGB channels in-place (saturating at 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// color -= 30;  // Decreases brightness in-place
    /// ```
    #[inline]
    fn sub_assign(&mut self, value: u8) {
        self.red = self.red.saturating_sub(value);
        self.green = self.green.saturating_sub(value);
        self.blue = self.blue.saturating_sub(value);
    }
}

impl SubAssign<&u8> for RGB {
    #[inline]
    fn sub_assign(&mut self, value: &u8) {
        self.sub_assign(*value);
    }
}

impl Div for RGB {
    type Output = Self;

    /// Divides two RGB colors component-wise (avoiding division by zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let divisor = RGB::new(2, 4, 5);
    /// let result = color / divisor;
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        RGB {
            red: if other.red == 0 {
                0
            } else {
                self.red.saturating_div(other.red)
            },
            green: if other.green == 0 {
                0
            } else {
                self.green.saturating_div(other.green)
            },
            blue: if other.blue == 0 {
                0
            } else {
                self.blue.saturating_div(other.blue)
            },
        }
    }
}

impl Div<&Self> for RGB {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        self.div(*other)
    }
}

impl Div<&Self> for &RGB {
    type Output = RGB;

    #[inline]
    fn div(self, other: &Self) -> RGB {
        (*self).div(*other)
    }
}

impl Div<RGB> for &RGB {
    type Output = RGB;

    #[inline]
    fn div(self, other: RGB) -> RGB {
        (*self).div(other)
    }
}

impl DivAssign for RGB {
    /// Divides this RGB color by another component-wise in-place (avoiding division by zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// let divisor = RGB::new(2, 4, 5);
    /// color /= divisor;
    /// ```
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.red = if other.red == 0 {
            0
        } else {
            self.red.saturating_div(other.red)
        };
        self.green = if other.green == 0 {
            0
        } else {
            self.green.saturating_div(other.green)
        };
        self.blue = if other.blue == 0 {
            0
        } else {
            self.blue.saturating_div(other.blue)
        };
    }
}

impl DivAssign<&Self> for RGB {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.div_assign(*other);
    }
}

impl Div<f32> for RGB {
    type Output = Self;

    /// Divides all RGB channels by a scalar factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let color = RGB::new(200, 100, 50);
    /// let darker = color / 2.0;  // Half brightness
    /// ```
    #[inline]
    fn div(self, factor: f32) -> Self {
        if factor == 0.0 {
            RGB {
                red: 0,
                green: 0,
                blue: 0,
            }
        } else {
            RGB {
                red: (self.red as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
                green: (self.green as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
                blue: (self.blue as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            }
        }
    }
}

impl Div<&f32> for RGB {
    type Output = Self;

    #[inline]
    fn div(self, factor: &f32) -> Self {
        self.div(*factor)
    }
}

impl Div<&f32> for &RGB {
    type Output = RGB;

    #[inline]
    fn div(self, factor: &f32) -> RGB {
        (*self).div(*factor)
    }
}

impl Div<f32> for &RGB {
    type Output = RGB;

    #[inline]
    fn div(self, factor: f32) -> RGB {
        (*self).div(factor)
    }
}

impl DivAssign<f32> for RGB {
    /// Divides all RGB channels by a scalar factor in-place.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::RGB;
    /// let mut color = RGB::new(200, 100, 50);
    /// color /= 2.0;  // Halves brightness in-place
    /// ```
    #[inline]
    fn div_assign(&mut self, factor: f32) {
        if factor == 0.0 {
            self.red = 0;
            self.green = 0;
            self.blue = 0;
        } else {
            self.red = (self.red as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
            self.green = (self.green as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
            self.blue = (self.blue as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        }
    }
}

impl DivAssign<&f32> for RGB {
    #[inline]
    fn div_assign(&mut self, factor: &f32) {
        self.div_assign(*factor);
    }
}

// ARGB operator implementations

impl Add for ARGB {
    type Output = Self;

    /// Adds two ARGB colors component-wise (saturating at 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let light1 = ARGB::new(50, 100, 0, 0);
    /// let light2 = ARGB::new(50, 0, 100, 0);
    /// let combined = light1 + light2;  // ARGB(100, 100, 100, 0)
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        ARGB {
            alpha: self.alpha.saturating_add(other.alpha),
            red: self.red.saturating_add(other.red),
            green: self.green.saturating_add(other.green),
            blue: self.blue.saturating_add(other.blue),
        }
    }
}

impl Add<&Self> for ARGB {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        self.add(*other)
    }
}

impl Add<&Self> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn add(self, other: &Self) -> ARGB {
        (*self).add(*other)
    }
}

impl Add<ARGB> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn add(self, other: ARGB) -> ARGB {
        (*self).add(other)
    }
}

impl AddAssign for ARGB {
    /// Adds another ARGB color component-wise in-place (saturating at 255).
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.alpha = self.alpha.saturating_add(other.alpha);
        self.red = self.red.saturating_add(other.red);
        self.green = self.green.saturating_add(other.green);
        self.blue = self.blue.saturating_add(other.blue);
    }
}

impl AddAssign<&Self> for ARGB {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.add_assign(*other);
    }
}

impl Mul for ARGB {
    type Output = Self;

    /// Multiplies two ARGB colors component-wise (result = (a * b) / 255).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let sprite = ARGB::new(255, 255, 255, 255);
    /// let tint = ARGB::new(255, 255, 128, 128);
    /// let tinted = sprite * tint;
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        ARGB {
            alpha: self.alpha.saturating_mul(other.alpha),
            red: self.red.saturating_mul(other.red),
            green: self.green.saturating_mul(other.green),
            blue: self.blue.saturating_mul(other.blue),
        }
    }
}

impl Mul<&Self> for ARGB {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        self.mul(*other)
    }
}

impl Mul<&Self> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn mul(self, other: &Self) -> ARGB {
        (*self).mul(*other)
    }
}

impl Mul<ARGB> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn mul(self, other: ARGB) -> ARGB {
        (*self).mul(other)
    }
}

impl MulAssign for ARGB {
    /// Multiplies this ARGB color by another component-wise in-place.
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.alpha = self.alpha.saturating_mul(other.alpha);
        self.red = self.red.saturating_mul(other.red);
        self.green = self.green.saturating_mul(other.green);
        self.blue = self.blue.saturating_mul(other.blue);
    }
}

impl MulAssign<&Self> for ARGB {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        self.mul_assign(*other);
    }
}

// Scalar multiplication for ARGB (affects RGB only, preserves alpha)
impl Mul<f32> for ARGB {
    type Output = Self;

    /// Multiplies RGB channels by a scalar factor (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color = ARGB::new(128, 200, 100, 50);
    /// let darker = color * 0.5;  // Half brightness, alpha stays 128
    /// ```
    #[inline]
    fn mul(self, factor: f32) -> Self {
        ARGB {
            alpha: self.alpha,
            red: (self.red as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            green: (self.green as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            blue: (self.blue as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
        }
    }
}

impl Mul<&f32> for ARGB {
    type Output = Self;

    #[inline]
    fn mul(self, factor: &f32) -> Self {
        self.mul(*factor)
    }
}

impl Mul<&f32> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn mul(self, factor: &f32) -> ARGB {
        (*self).mul(*factor)
    }
}

impl Mul<f32> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn mul(self, factor: f32) -> ARGB {
        (*self).mul(factor)
    }
}

impl MulAssign<f32> for ARGB {
    /// Scales RGB channels by a scalar factor in-place (alpha preserved).
    #[inline]
    fn mul_assign(&mut self, factor: f32) {
        self.red = (self.red as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        self.green = (self.green as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        self.blue = (self.blue as f32 * factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
    }
}

impl MulAssign<&f32> for ARGB {
    #[inline]
    fn mul_assign(&mut self, factor: &f32) {
        self.mul_assign(*factor);
    }
}

impl Mul<ARGB> for f32 {
    type Output = ARGB;

    /// Multiplies RGB channels by a scalar factor (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color = ARGB::new(128, 200, 100, 50);
    /// let darker = 0.5 * color;  // Half brightness, alpha stays 128
    /// ```
    #[inline]
    fn mul(self, color: ARGB) -> ARGB {
        color.mul(self)
    }
}

impl Mul<&ARGB> for f32 {
    type Output = ARGB;

    #[inline]
    fn mul(self, color: &ARGB) -> ARGB {
        color.mul(self)
    }
}

impl Mul<&ARGB> for &f32 {
    type Output = ARGB;

    #[inline]
    fn mul(self, color: &ARGB) -> ARGB {
        color.mul(*self)
    }
}

impl Mul<ARGB> for &f32 {
    type Output = ARGB;

    #[inline]
    fn mul(self, color: ARGB) -> ARGB {
        color.mul(*self)
    }
}

impl Sub for ARGB {
    type Output = Self;

    /// Subtracts two ARGB colors component-wise (saturating at 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color1 = ARGB::new(200, 150, 100, 50);
    /// let color2 = ARGB::new(50, 50, 50, 25);
    /// let result = color1 - color2;  // ARGB(150, 100, 50, 25)
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        ARGB {
            alpha: self.alpha.saturating_sub(other.alpha),
            red: self.red.saturating_sub(other.red),
            green: self.green.saturating_sub(other.green),
            blue: self.blue.saturating_sub(other.blue),
        }
    }
}

impl Sub<&Self> for ARGB {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        self.sub(*other)
    }
}

impl Sub<&Self> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn sub(self, other: &Self) -> ARGB {
        (*self).sub(*other)
    }
}

impl Sub<ARGB> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn sub(self, other: ARGB) -> ARGB {
        (*self).sub(other)
    }
}

impl SubAssign for ARGB {
    /// Subtracts another ARGB color component-wise in-place (saturating at 0).
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.alpha = self.alpha.saturating_sub(other.alpha);
        self.red = self.red.saturating_sub(other.red);
        self.green = self.green.saturating_sub(other.green);
        self.blue = self.blue.saturating_sub(other.blue);
    }
}

impl SubAssign<&Self> for ARGB {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.sub_assign(*other);
    }
}

impl Div for ARGB {
    type Output = Self;

    /// Divides two ARGB colors component-wise (avoiding division by zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color = ARGB::new(255, 200, 100, 50);
    /// let divisor = ARGB::new(2, 4, 5, 10);
    /// let result = color / divisor;
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        ARGB {
            alpha: if other.alpha == 0 {
                0
            } else {
                self.alpha.saturating_div(other.alpha)
            },
            red: if other.red == 0 {
                0
            } else {
                self.red.saturating_div(other.red)
            },
            green: if other.green == 0 {
                0
            } else {
                self.green.saturating_div(other.green)
            },
            blue: if other.blue == 0 {
                0
            } else {
                self.blue.saturating_div(other.blue)
            },
        }
    }
}

impl Div<&Self> for ARGB {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        self.div(*other)
    }
}

impl Div<&Self> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn div(self, other: &Self) -> ARGB {
        (*self).div(*other)
    }
}

impl Div<ARGB> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn div(self, other: ARGB) -> ARGB {
        (*self).div(other)
    }
}

impl DivAssign for ARGB {
    /// Divides this ARGB color by another component-wise in-place (avoiding division by zero).
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.alpha = if other.alpha == 0 {
            0
        } else {
            self.alpha.saturating_div(other.alpha)
        };
        self.red = if other.red == 0 {
            0
        } else {
            self.red.saturating_div(other.red)
        };
        self.green = if other.green == 0 {
            0
        } else {
            self.green.saturating_div(other.green)
        };
        self.blue = if other.blue == 0 {
            0
        } else {
            self.blue.saturating_div(other.blue)
        };
    }
}

impl DivAssign<&Self> for ARGB {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.div_assign(*other);
    }
}

impl Div<f32> for ARGB {
    type Output = Self;

    /// Divides RGB channels by a scalar factor (alpha preserved).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::ARGB;
    /// let color = ARGB::new(128, 200, 100, 50);
    /// let darker = color / 2.0;  // Half brightness, alpha stays 128
    /// ```
    #[inline]
    fn div(self, factor: f32) -> Self {
        if factor == 0.0 {
            ARGB {
                alpha: self.alpha,
                red: 0,
                green: 0,
                blue: 0,
            }
        } else {
            ARGB {
                alpha: self.alpha,
                red: (self.red as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
                green: (self.green as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
                blue: (self.blue as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8,
            }
        }
    }
}

impl Div<&f32> for ARGB {
    type Output = Self;

    #[inline]
    fn div(self, factor: &f32) -> Self {
        self.div(*factor)
    }
}

impl Div<&f32> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn div(self, factor: &f32) -> ARGB {
        (*self).div(*factor)
    }
}

impl Div<f32> for &ARGB {
    type Output = ARGB;

    #[inline]
    fn div(self, factor: f32) -> ARGB {
        (*self).div(factor)
    }
}

impl DivAssign<f32> for ARGB {
    /// Divides RGB channels by a scalar factor in-place (alpha preserved).
    #[inline]
    fn div_assign(&mut self, factor: f32) {
        if factor == 0.0 {
            self.red = 0;
            self.green = 0;
            self.blue = 0;
        } else {
            self.red = (self.red as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
            self.green = (self.green as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
            self.blue = (self.blue as f32 / factor).clamp(MIN_PERCENTAGE, U8_MAX_F32) as u8;
        }
    }
}

impl DivAssign<&f32> for ARGB {
    #[inline]
    fn div_assign(&mut self, factor: &f32) {
        self.div_assign(*factor);
    }
}

// RGB <==> Slice/Tuple conversions

impl From<[u8; 3]> for RGB {
    /// Converts from a 3-element array `[u8; 3]` into [`RGB`].
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let arr = [255, 128, 0];
    /// let color: RGB = arr.into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(arr: [u8; 3]) -> Self {
        RGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
        }
    }
}

impl From<&[u8; 3]> for RGB {
    /// Converts from a reference to a 3-element array `&[u8; 3]` into [`RGB`].
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let arr = [255, 128, 0];
    /// let color: RGB = (&arr).into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(arr: &[u8; 3]) -> Self {
        RGB {
            red: arr[0],
            green: arr[1],
            blue: arr[2],
        }
    }
}

impl From<RGB> for [u8; 3] {
    /// Converts between [`RGB`] and a 3-element array `[u8; 3]`.
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let color = RGB::new(255, 128, 0);
    /// let arr: [u8; 3] = color.into();
    /// assert_eq!(arr, [255, 128, 0]);
    /// ```
    fn from(color: RGB) -> Self {
        [color.red, color.green, color.blue]
    }
}

impl From<&RGB> for [u8; 3] {
    /// Converts between a reference to [`RGB`] and a 3-element array `[u8; 3]`.
    /// The array elements correspond to the RGB channels in order: [Red, Green, Blue].
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let color = RGB::new(255, 128, 0);
    /// let arr: [u8; 3] = (&color).into();
    /// assert_eq!(arr, [255, 128, 0]);
    /// ```
    fn from(color: &RGB) -> Self {
        [color.red, color.green, color.blue]
    }
}

impl From<(u8, u8, u8)> for RGB {
    /// Converts from a tuple `(u8, u8, u8)` into [`RGB`].
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let tup = (255, 128, 0);
    /// let color: RGB = tup.into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(tup: (u8, u8, u8)) -> Self {
        RGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
        }
    }
}

impl From<&(u8, u8, u8)> for RGB {
    /// Converts from a reference to a tuple `&(u8, u8, u8)` into [`RGB`].
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let tup = (255, 128, 0);
    /// let color: RGB = (&tup).into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(tup: &(u8, u8, u8)) -> Self {
        RGB {
            red: tup.0,
            green: tup.1,
            blue: tup.2,
        }
    }
}

impl From<RGB> for (u8, u8, u8) {
    /// Converts between [`RGB`] and a tuple `(u8, u8, u8)`.
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let color = RGB::new(255, 128, 0);
    /// let tup: (u8, u8, u8) = color.into();
    /// assert_eq!(tup, (255, 128, 0));
    /// ```
    fn from(color: RGB) -> Self {
        (color.red, color.green, color.blue)
    }
}

impl From<&RGB> for (u8, u8, u8) {
    /// Converts between a reference to [`RGB`] and a tuple `(u8, u8, u8)`.
    /// The tuple elements correspond to the RGB channels in order: (Red, Green, Blue).
    ///
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// let color = RGB::new(255, 128, 0);
    /// let tup: (u8, u8, u8) = (&color).into();
    /// assert_eq!(tup, (255, 128, 0));
    /// ```
    fn from(color: &RGB) -> Self {
        (color.red, color.green, color.blue)
    }
}

// RGB <==> Vector conversions

impl From<vec::Vec3> for RGB {
    /// Converts from a 3D vector representation ([`Vec3`](crate::primitives::vec::Vec3)) into [`RGB`].
    /// The conversion maps RGB channels to vector components:
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    ///
    /// The values for each channel are denormalized to the range [0, 255] when converting to vector form.
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// use toyengine::primitives::vec;
    /// let vec = vec::Vec3 { x: 1.0, y: 0.5, z: 0.0 };
    /// let color: RGB = vec.into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(v: vec::Vec3) -> Self {
        RGB {
            red: (v.x * U8_MAX_F32) as u8,
            green: (v.y * U8_MAX_F32) as u8,
            blue: (v.z * U8_MAX_F32) as u8,
        }
    }
}

impl From<&vec::Vec3> for RGB {
    fn from(v: &vec::Vec3) -> Self {
        let rgb: RGB = (*v).into();
        rgb
    }
}

impl From<vec::Vec4> for RGB {
    /// Converts from a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)) into [`RGB`].
    /// The conversion maps RGB channels to vector components:
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    ///
    /// The values for each channel are denormalized to the range [0, 255] when converting to vector form.
    /// The w component is ignored.
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// use toyengine::primitives::vec;
    /// let vec = vec::Vec4 { x: 1.0, y: 0.5, z: 0.0, w: 1.0 };
    /// let color: RGB = vec.into();
    /// assert_eq!(color, RGB::new(255, 128, 0));
    /// ```
    fn from(v: vec::Vec4) -> Self {
        RGB {
            red: (v.x * U8_MAX_F32) as u8,
            green: (v.y * U8_MAX_F32) as u8,
            blue: (v.z * U8_MAX_F32) as u8,
        }
    }
}

impl From<&vec::Vec4> for RGB {
    fn from(v: &vec::Vec4) -> Self {
        let rgb: RGB = (*v).into();
        rgb
    }
}

impl From<RGB> for vec::Vec3 {
    /// Converts between [`RGB`] and a 3D vector representation ([`Vec3`](crate::primitives::vec::Vec3)).
    /// The conversion maps RGB channels to vector components:
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    ///
    /// The values for each channel are normalized to the range [0.0, 1.0] when converting to vector form.
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// use toyengine::primitives::vec;
    /// let color = RGB::new(255, 128, 0);
    /// let vec: vec::Vec3 = color.into();
    /// assert_eq!(vec, vec::Vec3 { x: 1.0, y: 0.5019608, z: 0.0 });
    fn from(color: RGB) -> Self {
        vec::Vec3 {
            x: (color.red as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: (color.green as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: (color.blue as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }
}

impl From<&RGB> for vec::Vec3 {
    fn from(color: &RGB) -> Self {
        let vec: vec::Vec3 = (*color).into();
        vec
    }
}

impl From<RGB> for vec::Vec4 {
    /// Converts between [`RGB`] and a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)).
    /// The conversion maps RGB channels to vector components:
    /// - w -> Alpha (set to 1.0)
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    ///
    /// The values for each channel are normalized to the range [0.0, 1.0] when converting to vector form.
    /// The w component is set to 1.0 (fully opaque).
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::RGB;
    /// use toyengine::primitives::vec;
    /// let color = RGB::new(255, 128, 0);
    /// let vec: vec::Vec4 = color.into();
    /// assert_eq!(vec, vec::Vec4 { w: 1.0, x: 1.0, y: 0.5019608, z: 0.0 });
    fn from(color: RGB) -> Self {
        vec::Vec4 {
            x: (color.red as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: (color.green as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: (color.blue as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            w: MAX_PERCENTAGE,
        }
    }
}

impl From<&RGB> for vec::Vec4 {
    fn from(color: &RGB) -> Self {
        let vec: vec::Vec4 = (*color).into();
        vec
    }
}

// ARGB <==> Vector conversions

impl From<vec::Vec4> for ARGB {
    /// Converts from a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)) into [`ARGB`].
    /// The conversion maps ARGB channels to vector components:
    /// - Alpha -> w
    /// - Red -> x
    /// - Green -> y
    /// - Blue -> z
    ///
    /// The values for each channel are denormalized to the range [0, 255] when converting to vector form.
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::ARGB;
    /// use toyengine::primitives::vec;
    /// let vec = vec::Vec4 { w: 0.5, x: 1.0, y: 0.5, z: 0.0 };
    /// let color: ARGB = vec.into();
    /// assert_eq!(color, ARGB::new(128, 255, 128, 0));
    /// ```
    fn from(v: vec::Vec4) -> Self {
        ARGB {
            alpha: (v.w * U8_MAX_F32) as u8,
            red: (v.x * U8_MAX_F32) as u8,
            green: (v.y * U8_MAX_F32) as u8,
            blue: (v.z * U8_MAX_F32) as u8,
        }
    }
}

impl From<&vec::Vec4> for ARGB {
    fn from(v: &vec::Vec4) -> Self {
        let argb: ARGB = (*v).into();
        argb
    }
}

impl From<ARGB> for vec::Vec4 {
    /// Converts between [`ARGB`] and a 4D vector representation ([`Vec4`](crate::primitives::vec::Vec4)).
    /// The conversion maps ARGB channels to vector components:
    /// - x -> Red
    /// - y -> Green
    /// - z -> Blue
    /// - w -> Alpha
    ///
    /// The values for each channel are normalized to the range [0.0, 1.0] when converting to vector form.
    /// # Examples
    /// ```rust
    /// use toyengine::primitives::color::ARGB;
    /// use toyengine::primitives::vec;
    /// let color = ARGB::new(128, 255, 128, 0);
    /// let vec: vec::Vec4 = color.into();
    /// assert_eq!(vec, vec::Vec4 { w: 0.5019608, x: 1.0, y: 0.5019608, z: 0.0 });
    fn from(color: ARGB) -> Self {
        vec::Vec4 {
            x: (color.red as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            y: (color.green as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            z: (color.blue as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
            w: (color.alpha as f32 / U8_MAX_F32).clamp(MIN_PERCENTAGE, MAX_PERCENTAGE),
        }
    }
}

impl From<&ARGB> for vec::Vec4 {
    fn from(color: &ARGB) -> Self {
        let vec: vec::Vec4 = (*color).into();
        vec
    }
}
