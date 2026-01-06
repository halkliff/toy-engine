//! Premultiplied alpha RGBA for GPU-friendly compositing.
//!
//! Premultiplied alpha (also called "associated alpha") stores RGB values
//! already multiplied by their alpha value. This format is standard in:
//! - Modern GPUs
//! - Many image formats (PNG, DDS)
//! - Compositing operations
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::color::{ARGB, PremultipliedARGB};
//!
//! // Convert from standard ARGB to premultiplied
//! let argb = ARGB::new(128, 255, 0, 0); // 50% opaque red
//! let premul: PremultipliedARGB = argb.into();
//!
//! // Direct creation for GPU upload
//! let sprite = PremultipliedARGB::new(0.5, 0.0, 0.0, 0.5);
//!
//! // Fast alpha blending
//! let foreground = PremultipliedARGB::new(0.5, 0.0, 0.0, 0.5);
//! let background = PremultipliedARGB::new(0.2, 0.3, 0.4, 1.0);
//! let blended = foreground.blend(&background);
//! ```

/// Premultiplied RGBA color with floating-point channels in [0.0, 1.0] range.
///
/// In premultiplied format:
/// - R, G, B values are already multiplied by A
/// - RGB values are always ≤ A
/// - Blending is simply: result = foreground + background × (1 - alpha_fg)
///
/// # Benefits
/// - **Faster blending**: No division needed
/// - **Better interpolation**: No color bleeding on transparent edges
/// - **GPU standard**: Direct upload to textures
///
/// # When to Use
/// - GPU texture uploads
/// - Sprite rendering
/// - UI compositing
/// - Canvas-style blending
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::color::{ARGB, PremultipliedRGBA};
/// // Convert from standard ARGB
/// let argb = ARGB::new(128, 255, 0, 0); // 50% opaque red
/// let premul: PremultipliedRGBA = argb.into();
/// // premul.r ≈ 0.5 (was 1.0, now halved by alpha)
///
/// // Fast blending
/// let bg = PremultipliedRGBA::new(0.2, 0.3, 0.4, 1.0);
/// let blended = premul.blend(&bg);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PremultipliedARGB {
    /// Red component (premultiplied) [0.0, 1.0]
    pub r: f32,
    /// Green component (premultiplied) [0.0, 1.0]
    pub g: f32,
    /// Blue component (premultiplied) [0.0, 1.0]
    pub b: f32,
    /// Alpha component [0.0, 1.0]
    pub a: f32,
}

impl PremultipliedARGB {
    /// Creates a new premultiplied RGBA color.
    ///
    /// **Important**: RGB values should already be premultiplied!
    /// Use `from_straight()` if you have straight (non-premultiplied) values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::PremultipliedRGBA;
    /// // 50% red: RGB are halved because alpha is 0.5
    /// let premul = PremultipliedRGBA::new(0.5, 0.0, 0.0, 0.5);
    /// ```
    #[inline]
    pub fn new(a: f32, r: f32, g: f32, b: f32) -> Self {
        PremultipliedARGB {
            r: r.clamp(0.0, a),
            g: g.clamp(0.0, a),
            b: b.clamp(0.0, a),
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Creates premultiplied color from straight (non-premultiplied) RGBA.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::PremultipliedRGBA;
    /// // Input: straight 50% red
    /// let premul = PremultipliedRGBA::from_straight(1.0, 0.0, 0.0, 0.5);
    /// // Output: RGB values multiplied by alpha
    /// assert_eq!(premul.r, 0.5);
    /// ```
    #[inline]
    pub fn from_straight(r: f32, g: f32, b: f32, a: f32) -> Self {
        let a = a.clamp(0.0, 1.0);
        PremultipliedARGB {
            r: (r * a).clamp(0.0, 1.0),
            g: (g * a).clamp(0.0, 1.0),
            b: (b * a).clamp(0.0, 1.0),
            a,
        }
    }

    /// Converts back to straight (non-premultiplied) alpha.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::PremultipliedRGBA;
    /// let premul = PremultipliedRGBA::new(0.5, 0.0, 0.0, 0.5);
    /// let (r, g, b, a) = premul.to_straight();
    /// assert_eq!(r, 1.0); // Un-premultiplied back to full red
    /// ```
    #[inline]
    pub fn to_straight(&self) -> (f32, f32, f32, f32) {
        if self.a == 0.0 {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (self.r / self.a, self.g / self.a, self.b / self.a, self.a)
        }
    }

    /// Fast premultiplied alpha blending (Porter-Duff "over").
    ///
    /// Formula: result = src + dst × (1 - src.alpha)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::PremultipliedRGBA;
    /// let fg = PremultipliedRGBA::new(0.5, 0.0, 0.0, 0.5);
    /// let bg = PremultipliedRGBA::new(0.2, 0.3, 0.4, 1.0);
    /// let result = fg.blend(&bg);
    /// ```
    #[inline]
    pub fn blend(&self, background: &Self) -> Self {
        let inv_a = 1.0 - self.a;
        PremultipliedARGB {
            r: self.r + background.r * inv_a,
            g: self.g + background.g * inv_a,
            b: self.b + background.b * inv_a,
            a: self.a + background.a * inv_a,
        }
    }

    /// Linear interpolation between two premultiplied colors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::color::PremultipliedRGBA;
    /// let c1 = PremultipliedRGBA::new(0.5, 0.0, 0.0, 0.5);
    /// let c2 = PremultipliedRGBA::new(0.0, 0.0, 0.5, 0.5);
    /// let mid = c1.lerp(&c2, 0.5);
    /// ```
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        PremultipliedARGB {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Transparent black (0.0, 0.0, 0.0, 0.0)
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Opaque black (0.0, 0.0, 0.0, 1.0)
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Opaque white (1.0, 1.0, 1.0, 1.0)
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
}

// Note: PremultipliedRGBA cannot implement Color or Solid traits directly
// since it has alpha and different semantics. Use From/Into for conversions.