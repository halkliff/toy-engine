//! Color space conversion implementations.
//!
//! Provides [`From`] trait implementations for converting between color spaces:
//! - [`RGB`] / [`ARGB`]: Standard 8-bit sRGB (gamma-encoded)
//! - [`HSL`] / [`HSLA`]: Hue-Saturation-Lightness (cylindrical color space)
//! - [`HSV`] / [`HSVA`]: Hue-Saturation-Value (cylindrical color space)
//! - [`LinearRGB`] / [`LinearARGB`]: Physically linear RGB for correct lighting calculations
//! - [`PremultipliedARGB`]: GPU-optimized format with alpha premultiplication

use crate::primitives::color::spaces::{
    ARGB, HSL, HSLA, HSV, HSVA, LinearARGB, LinearRGB, PremultipliedARGB, RGB,
};
use crate::primitives::color::traits::*;
use crate::primitives::color::{
    HUE_MAX_DEGREES, HUE_SEGMENT_DEGREES, INV_U8_MAX, SRGB_GAMMA, SRGB_INV_GAMMA, U8_MAX_F32,
};

// ======= RGB Conversions =======

impl From<ARGB> for RGB {
    /// Converts ARGB to RGB, discarding the alpha channel.
    ///
    /// This is a direct channel mapping with no color space transformation.
    fn from(argb: ARGB) -> Self {
        RGB {
            red: argb.red,
            green: argb.green,
            blue: argb.blue,
        }
    }
}

impl From<&ARGB> for RGB {
    fn from(color: &ARGB) -> Self {
        let rgb: RGB = (*color).into();
        rgb
    }
}

impl From<HSL> for RGB {
    /// Converts HSL to RGB color space.
    ///
    /// Uses standard HSL-to-RGB conversion algorithm:
    /// 1. Calculate chroma (color intensity) from saturation and lightness
    /// 2. Determine RGB' values based on hue segment (60° intervals)
    /// 3. Add lightness offset to achieve target brightness
    ///
    /// # Algorithm Details
    /// - Hue segments: Red-Yellow-Green-Cyan-Blue-Magenta (60° each)
    /// - Chroma: `(1 - |2L - 1|) × S`
    /// - Intermediate: `C × (1 - |(H/60) mod 2 - 1|)`
    fn from(hsl: HSL) -> Self {
        let hue = hsl.hue.bounded();
        let saturation = hsl.saturation.bounded();
        let lightness = hsl.lightness.bounded();

        // Calculate chroma (color intensity)
        let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;

        // Calculate intermediate value for the second largest component
        let hue_segment = hue / HUE_SEGMENT_DEGREES; // Which 60° segment of the color wheel
        let intermediate = chroma * (1.0 - ((hue_segment % 2.0) - 1.0).abs());

        // Amount to add to all channels to achieve target lightness
        let lightness_match = lightness - chroma / 2.0;

        // Determine RGB' values based on hue segment
        let (red_prime, green_prime, blue_prime) = if (0.0..HUE_SEGMENT_DEGREES).contains(&hue) {
            // Red to yellow: red=max, green=rising, blue=0
            (chroma, intermediate, 0.0)
        } else if (HUE_SEGMENT_DEGREES..120.0).contains(&hue) {
            // Yellow to green: red=falling, green=max, blue=0
            (intermediate, chroma, 0.0)
        } else if (120.0..180.0).contains(&hue) {
            // Green to cyan: red=0, green=max, blue=rising
            (0.0, chroma, intermediate)
        } else if (180.0..240.0).contains(&hue) {
            // Cyan to blue: red=0, green=falling, blue=max
            (0.0, intermediate, chroma)
        } else if (240.0..300.0).contains(&hue) {
            // Blue to magenta: red=rising, green=0, blue=max
            (intermediate, 0.0, chroma)
        } else {
            // Magenta to red: red=max, green=0, blue=falling
            (chroma, 0.0, intermediate)
        };

        // Convert to final RGB by adding lightness match and scaling to 0-255
        RGB {
            red: ((red_prime + lightness_match) * U8_MAX_F32).round() as u8,
            green: ((green_prime + lightness_match) * U8_MAX_F32).round() as u8,
            blue: ((blue_prime + lightness_match) * U8_MAX_F32).round() as u8,
        }
    }
}

impl From<&HSL> for RGB {
    fn from(color: &HSL) -> Self {
        let rgb: RGB = (*color).into();
        rgb
    }
}

impl From<HSLA> for RGB {
    /// Converts HSLA to RGB, discarding the alpha channel.
    ///
    /// Chains through HSL conversion: HSLA → HSL → RGB.
    fn from(hsla: HSLA) -> Self {
        let hsl: HSL = hsla.into();
        hsl.into()
    }
}

impl From<&HSLA> for RGB {
    fn from(color: &HSLA) -> Self {
        let rgb: RGB = (*color).into();
        rgb
    }
}

impl From<LinearRGB> for RGB {
    /// Converts LinearRGB (physically linear light values) to sRGB (gamma-encoded).
    /// Uses the fast gamma approximation with inverse power of ~0.4545.
    ///
    /// # Examples
    /// ```
    /// let linear = LinearRGB::new(0.5, 0.5, 0.5).unwrap();  // 50% linear light
    /// let rgb: RGB = linear.into();  // ~186 in sRGB (73% gray)
    /// ```
    fn from(linear: LinearRGB) -> Self {
        // Apply inverse gamma and convert back to u8 [0, 255]
        let r = (linear.red.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let g = (linear.green.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let b = (linear.blue.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        RGB {
            red: r,
            green: g,
            blue: b,
        }
    }
}

impl From<&LinearRGB> for RGB {
    fn from(linear: &LinearRGB) -> Self {
        RGB::from(*linear)
    }
}

impl From<PremultipliedARGB> for RGB {
    /// Converts PremultipliedARGB to RGB, discarding alpha.
    /// If alpha is not 1.0, this loses information (colors become darker).
    fn from(premult: PremultipliedARGB) -> Self {
        RGB {
            red: (premult.r * U8_MAX_F32).round() as u8,
            green: (premult.g * U8_MAX_F32).round() as u8,
            blue: (premult.b * U8_MAX_F32).round() as u8,
        }
    }
}

impl From<&PremultipliedARGB> for RGB {
    fn from(color: &PremultipliedARGB) -> Self {
        let rgb: RGB = (*color).into();
        rgb
    }
}

// ====== ARGB Conversions ======

impl From<RGB> for ARGB {
    /// Converts RGB to ARGB, setting alpha to maximum (fully opaque).
    ///
    /// This is a direct channel mapping with no color space transformation.
    ///
    /// # Examples
    /// ```
    /// let rgb = RGB::new(255, 128, 64);
    /// let argb: ARGB = rgb.into();
    /// assert_eq!(argb.alpha, 255);  // Full opacity
    /// ```
    fn from(rgb: RGB) -> Self {
        ARGB {
            alpha: u8::MAX, // Full opacity for solid colors
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
        }
    }
}

impl From<&RGB> for ARGB {
    fn from(color: &RGB) -> Self {
        let argb: ARGB = (*color).into();
        argb
    }
}

impl From<HSL> for ARGB {
    /// Converts HSL to ARGB with full opacity.
    ///
    /// Chains through RGB conversion: HSL → RGB → ARGB.
    fn from(hsl: HSL) -> Self {
        let rgb: RGB = hsl.into();
        ARGB {
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
            alpha: u8::MAX, // Full opacity for solid colors
        }
    }
}

impl From<&HSL> for ARGB {
    fn from(color: &HSL) -> Self {
        let argb: ARGB = (*color).into();
        argb
    }
}

impl From<HSLA> for ARGB {
    /// Converts HSLA to ARGB, preserving the alpha channel.
    ///
    /// Chains through HSL and RGB conversions: HSLA → HSL → RGB → ARGB.
    /// The alpha channel is converted from [0.0, 1.0] to [0, 255].
    fn from(hsla: HSLA) -> Self {
        let hsl: HSL = hsla.into();
        let rgb: RGB = hsl.into();
        ARGB {
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
            alpha: (hsla.alpha.0 * U8_MAX_F32).round() as u8,
        }
    }
}

impl From<&HSLA> for ARGB {
    fn from(color: &HSLA) -> Self {
        let argb: ARGB = (*color).into();
        argb
    }
}

impl From<LinearRGB> for ARGB {
    /// Converts LinearRGB (physically linear light values) to ARGB (gamma-encoded), setting alpha to maximum (opaque).
    /// Uses the fast gamma approximation with inverse power of ~0.4545.
    ///
    /// # Examples
    /// ```
    /// let linear = LinearRGB::new(0.5, 0.5, 0.5).unwrap();  // 50% linear light
    /// let argb: ARGB = linear.into();  // ~186 in sRGB (73% gray), full opacity
    /// ```
    fn from(linear: LinearRGB) -> Self {
        // Apply inverse gamma and convert back to u8 [0, 255]
        let r = (linear.red.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let g = (linear.green.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let b = (linear.blue.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        ARGB {
            alpha: u8::MAX, // Full opacity for solid colors
            red: r,
            green: g,
            blue: b,
        }
    }
}

impl From<&LinearRGB> for ARGB {
    fn from(linear: &LinearRGB) -> Self {
        ARGB::from(*linear)
    }
}

impl From<PremultipliedARGB> for ARGB {
    /// Converts PremultipliedARGB back to ARGB, dividing RGB by alpha to get straight alpha.
    /// Handles fully transparent pixels by returning black with zero alpha.
    ///
    /// # Examples
    /// ```
    /// let premult = PremultipliedARGB { r: 0.5, g: 0.25, b: 0.0, a: 0.5 };
    /// let argb: ARGB = premult.into();
    /// // RGB values are divided by alpha to recover original colors
    /// ```
    fn from(premult: PremultipliedARGB) -> Self {
        if premult.a == 0.0 {
            // Fully transparent - RGB values are undefined, return black
            return ARGB {
                alpha: 0,
                red: 0,
                green: 0,
                blue: 0,
            };
        }

        let inv_alpha = 1.0 / premult.a;
        ARGB {
            alpha: (premult.a * U8_MAX_F32).round() as u8,
            red: ((premult.r * inv_alpha) * U8_MAX_F32).round() as u8,
            green: ((premult.g * inv_alpha) * U8_MAX_F32).round() as u8,
            blue: ((premult.b * inv_alpha) * U8_MAX_F32).round() as u8,
        }
    }
}

impl From<&PremultipliedARGB> for ARGB {
    fn from(color: &PremultipliedARGB) -> Self {
        let argb: ARGB = (*color).into();
        argb
    }
}

// ===== HSL Conversions =====

impl From<RGB> for HSL {
    /// Converts RGB to HSL color space.
    ///
    /// Uses standard HSL conversion algorithm:
    /// - Hue: Based on dominant color channel (0-360°)
    /// - Saturation: Color intensity relative to lightness
    /// - Lightness: Average of min and max RGB channels
    fn from(rgb: RGB) -> Self {
        // Normalize RGB values to [0.0, 1.0] range
        let red_normalized = rgb.red as f32 * INV_U8_MAX;
        let green_normalized = rgb.green as f32 * INV_U8_MAX;
        let blue_normalized = rgb.blue as f32 * INV_U8_MAX;

        // Find maximum and minimum channel values
        let max_channel = red_normalized.max(green_normalized.max(blue_normalized));
        let min_channel = red_normalized.min(green_normalized.min(blue_normalized));
        let chroma = max_channel - min_channel; // Color intensity difference

        // Calculate lightness: average of max and min
        let lightness = (max_channel + min_channel) / 2.0;

        // Calculate saturation based on lightness
        let saturation = if chroma == 0.0 {
            0.0 // No color difference = grayscale
        } else {
            // Use different formula depending on lightness to avoid division issues
            chroma / (1.0 - (2.0 * lightness - 1.0).abs())
        };

        // Calculate hue based on which channel is dominant
        let hue = if chroma == 0.0 {
            0.0 // No color = undefined hue, default to 0
        } else if max_channel == red_normalized {
            // Red is dominant: hue is in red-yellow range
            HUE_SEGMENT_DEGREES * (((green_normalized - blue_normalized) / chroma) % 6.0)
        } else if max_channel == green_normalized {
            // Green is dominant: hue is in yellow-cyan range
            HUE_SEGMENT_DEGREES * (((blue_normalized - red_normalized) / chroma) + 2.0)
        } else {
            // Blue is dominant: hue is in cyan-magenta range
            HUE_SEGMENT_DEGREES * (((red_normalized - green_normalized) / chroma) + 4.0)
        };

        HSL {
            hue: if hue < 0.0 {
                hue + HUE_MAX_DEGREES
            } else {
                hue
            }
            .into(),
            saturation: saturation.into(),
            lightness: lightness.into(),
        }
    }
}

impl From<&RGB> for HSL {
    fn from(color: &RGB) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

impl From<ARGB> for HSL {
    /// Converts ARGB to HSL, discarding the alpha channel.
    ///
    /// Chains through RGB conversion: ARGB → RGB → HSL.
    fn from(argb: ARGB) -> Self {
        let rgb: RGB = argb.into();
        rgb.into()
    }
}

impl From<&ARGB> for HSL {
    fn from(color: &ARGB) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

impl From<HSLA> for HSL {
    /// Converts HSLA to HSL, discarding the alpha channel.
    ///
    /// Direct field mapping with no color space transformation.
    fn from(hsla: HSLA) -> Self {
        HSL {
            hue: hsla.hue,
            saturation: hsla.saturation,
            lightness: hsla.lightness,
        }
    }
}

impl From<&HSLA> for HSL {
    fn from(color: &HSLA) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

impl From<LinearRGB> for HSL {
    /// Converts LinearRGB to HSL by chaining through RGB.
    /// First converts LinearRGB → RGB (gamma-encoded), then RGB → HSL.
    fn from(linear: LinearRGB) -> Self {
        let rgb: RGB = linear.into();
        rgb.into()
    }
}

impl From<&LinearRGB> for HSL {
    fn from(color: &LinearRGB) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

impl From<PremultipliedARGB> for HSL {
    /// Converts PremultipliedARGB to HSL, discarding alpha.
    /// Chains through ARGB → RGB → HSL.
    fn from(premult: PremultipliedARGB) -> Self {
        let argb: ARGB = premult.into();
        let rgb: RGB = argb.into();
        rgb.into()
    }
}

impl From<&PremultipliedARGB> for HSL {
    fn from(color: &PremultipliedARGB) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

// ===== HSLA Conversions =====

impl From<RGB> for HSLA {
    /// Converts RGB to HSLA with full opacity.
    ///
    /// Chains through HSL conversion: RGB → HSL → HSLA.
    /// Alpha is set to 1.0 (fully opaque).
    fn from(rgb: RGB) -> Self {
        let hsl: HSL = rgb.into();
        HSLA {
            hue: hsl.hue,
            saturation: hsl.saturation,
            lightness: hsl.lightness,
            alpha: Percentage::MAX.into(),
        }
    }
}

impl From<&RGB> for HSLA {
    fn from(color: &RGB) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

impl From<ARGB> for HSLA {
    /// Converts ARGB to HSLA, preserving the alpha channel.
    ///
    /// Chains through HSL conversion: ARGB → HSL → HSLA.
    /// The alpha channel is converted from [0, 255] to [0.0, 1.0].
    fn from(argb: ARGB) -> Self {
        let hsl: HSL = argb.into();
        HSLA {
            hue: hsl.hue,
            saturation: hsl.saturation,
            lightness: hsl.lightness,
            alpha: (argb.alpha as f32 * INV_U8_MAX).into(),
        }
    }
}

impl From<&ARGB> for HSLA {
    fn from(color: &ARGB) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

impl From<HSL> for HSLA {
    /// Converts HSL to HSLA with full opacity.
    ///
    /// Direct field mapping with alpha set to 1.0 (fully opaque).
    fn from(hsl: HSL) -> Self {
        HSLA {
            hue: hsl.hue,
            saturation: hsl.saturation,
            lightness: hsl.lightness,
            alpha: Percentage::MAX.into(),
        }
    }
}

impl From<&HSL> for HSLA {
    fn from(color: &HSL) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

impl From<LinearRGB> for HSLA {
    /// Converts LinearRGB to HSLA with full opacity.
    /// Chains through RGB → HSL, then adds alpha=1.0.
    fn from(linear: LinearRGB) -> Self {
        let hsl: HSL = linear.into();
        HSLA {
            hue: hsl.hue,
            saturation: hsl.saturation,
            lightness: hsl.lightness,
            alpha: Percentage(1.0),
        }
    }
}

impl From<&LinearRGB> for HSLA {
    fn from(color: &LinearRGB) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

impl From<PremultipliedARGB> for HSLA {
    /// Converts PremultipliedARGB to HSLA.
    /// Chains through ARGB → HSL, preserving alpha.
    fn from(premult: PremultipliedARGB) -> Self {
        let argb: ARGB = premult.into();
        let hsl: HSL = argb.into();
        HSLA {
            hue: hsl.hue,
            saturation: hsl.saturation,
            lightness: hsl.lightness,
            alpha: Percentage(argb.alpha as f32 * INV_U8_MAX),
        }
    }
}

impl From<&PremultipliedARGB> for HSLA {
    fn from(color: &PremultipliedARGB) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

// ===== LinearRGB Conversions =====

impl From<RGB> for LinearRGB {
    /// Converts sRGB (gamma-encoded) to LinearRGB (physically linear light values).
    /// Uses the fast gamma approximation with power of 2.2.
    ///
    /// # Examples
    /// ```
    /// let rgb = RGB::new(128, 128, 128);  // 50% gray in sRGB
    /// let linear: LinearRGB = rgb.into();  // ~21.8% in linear space
    /// ```
    fn from(rgb: RGB) -> Self {
        // Normalize u8 [0, 255] to f32 [0.0, 1.0] and apply gamma correction
        let r = (rgb.red as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let g = (rgb.green as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let b = (rgb.blue as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        LinearRGB { red: r, green: g, blue: b }
    }
}

impl From<&RGB> for LinearRGB {
    fn from(rgb: &RGB) -> Self {
        LinearRGB::from(*rgb)
    }
}

impl From<ARGB> for LinearRGB {
    /// Converts ARGB (gamma-encoded) to LinearRGB (physically linear light values), discarding the alpha channel.
    /// Uses the fast gamma approximation with power of 2.2.
    ///
    /// # Examples
    /// ```
    /// let argb = ARGB::new(255, 128, 128, 128);  // 50% gray in sRGB with full opacity
    /// let linear: LinearRGB = argb.into();  // ~21.8% in linear space
    /// ```
    fn from(argb: ARGB) -> Self {
        // Normalize u8 [0, 255] to f32 [0.0, 1.0] and apply gamma correction
        let r = (argb.red as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let g = (argb.green as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let b = (argb.blue as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        LinearRGB { red: r, green: g, blue: b }
    }
}

impl From<&ARGB> for LinearRGB {
    fn from(argb: &ARGB) -> Self {
        LinearRGB::from(*argb)
    }
}

impl From<HSL> for LinearRGB {
    /// Converts HSL to LinearRGB by chaining through RGB.
    /// First converts HSL → RGB (gamma-encoded), then RGB → LinearRGB.
    fn from(hsl: HSL) -> Self {
        let rgb: RGB = hsl.into();
        rgb.into()
    }
}

impl From<&HSL> for LinearRGB {
    fn from(color: &HSL) -> Self {
        let linear: LinearRGB = (*color).into();
        linear
    }
}

impl From<HSLA> for LinearRGB {
    /// Converts HSLA to LinearRGB by chaining through HSL, discarding alpha.
    fn from(hsla: HSLA) -> Self {
        let hsl: HSL = hsla.into();
        hsl.into()
    }
}

impl From<&HSLA> for LinearRGB {
    fn from(color: &HSLA) -> Self {
        let linear: LinearRGB = (*color).into();
        linear
    }
}

impl From<PremultipliedARGB> for LinearRGB {
    /// Converts PremultipliedARGB to LinearRGB, discarding alpha.
    ///
    /// **Warning**: If alpha is not 1.0, RGB values will be darker than expected
    /// because they are still premultiplied. Consider converting to ARGB first
    /// to recover the original straight-alpha colors, then to LinearRGB.
    ///
    /// For correct color recovery: `PremultipliedARGB → ARGB → LinearRGB`
    fn from(premult: PremultipliedARGB) -> Self {
        LinearRGB {
            red: premult.r,
            green: premult.g,
            blue: premult.b,
        }
    }
}

impl From<&PremultipliedARGB> for LinearRGB {
    fn from(color: &PremultipliedARGB) -> Self {
        let linear: LinearRGB = (*color).into();
        linear
    }
}

// ===== PremultipliedARGB Conversions =====

impl From<ARGB> for PremultipliedARGB {
    /// Converts ARGB to PremultipliedARGB, multiplying RGB channels by alpha.
    /// This format is optimized for compositing operations.
    ///
    /// # Examples
    /// ```
    /// let argb = ARGB::new(128, 255, 100, 50);  // 50% transparent red-ish
    /// let premult: PremultipliedARGB = argb.into();
    /// // RGB values are multiplied by alpha (128/255 ≈ 0.5)
    /// ```
    fn from(argb: ARGB) -> Self {
        let alpha = argb.alpha as f32 * INV_U8_MAX;
        PremultipliedARGB {
            r: (argb.red as f32 * INV_U8_MAX) * alpha,
            g: (argb.green as f32 * INV_U8_MAX) * alpha,
            b: (argb.blue as f32 * INV_U8_MAX) * alpha,
            a: alpha,
        }
    }
}

impl From<&ARGB> for PremultipliedARGB {
    fn from(color: &ARGB) -> Self {
        let premult: PremultipliedARGB = (*color).into();
        premult
    }
}

impl From<RGB> for PremultipliedARGB {
    /// Converts opaque RGB to PremultipliedARGB with full alpha.
    /// Since alpha is 1.0, RGB values are unchanged.
    fn from(rgb: RGB) -> Self {
        PremultipliedARGB {
            r: rgb.red as f32 * INV_U8_MAX,
            g: rgb.green as f32 * INV_U8_MAX,
            b: rgb.blue as f32 * INV_U8_MAX,
            a: 1.0,
        }
    }
}

impl From<&RGB> for PremultipliedARGB {
    fn from(color: &RGB) -> Self {
        let premult: PremultipliedARGB = (*color).into();
        premult
    }
}

impl From<HSL> for PremultipliedARGB {
    /// Converts HSL to PremultipliedARGB with full opacity.
    /// Chains through RGB → ARGB → PremultipliedARGB.
    fn from(hsl: HSL) -> Self {
        let rgb: RGB = hsl.into();
        let argb: ARGB = rgb.into();
        argb.into()
    }
}

impl From<&HSL> for PremultipliedARGB {
    fn from(color: &HSL) -> Self {
        let premult: PremultipliedARGB = (*color).into();
        premult
    }
}

impl From<HSLA> for PremultipliedARGB {
    /// Converts HSLA to PremultipliedARGB.
    /// Chains through ARGB, multiplying RGB by alpha.
    fn from(hsla: HSLA) -> Self {
        let argb: ARGB = hsla.into();
        argb.into()
    }
}

impl From<&HSLA> for PremultipliedARGB {
    fn from(color: &HSLA) -> Self {
        let premult: PremultipliedARGB = (*color).into();
        premult
    }
}

impl From<LinearRGB> for PremultipliedARGB {
    /// Converts LinearRGB to PremultipliedARGB with full opacity.
    ///
    /// Since LinearRGB has no alpha channel, the result is fully opaque (alpha = 1.0).
    /// RGB channels remain in linear space - no gamma correction applied.
    fn from(linear: LinearRGB) -> Self {
        PremultipliedARGB {
            r: linear.red,
            g: linear.green,
            b: linear.blue,
            a: 1.0,
        }
    }
}

impl From<&LinearRGB> for PremultipliedARGB {
    fn from(color: &LinearRGB) -> Self {
        let premult: PremultipliedARGB = (*color).into();
        premult
    }
}

// ===== LinearARGB Conversions =====

impl From<RGB> for LinearARGB {
    /// Converts sRGB (gamma-encoded) to LinearARGB (physically linear light values) with full opacity.
    /// Uses the fast gamma approximation with power of 2.2.
    ///
    /// # Examples
    /// ```
    /// let rgb = RGB::new(128, 128, 128);  // 50% gray in sRGB
    /// let linear: LinearARGB = rgb.into();  // ~21.8% in linear space, alpha=1.0
    /// ```
    fn from(rgb: RGB) -> Self {
        let r = (rgb.red as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let g = (rgb.green as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let b = (rgb.blue as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        LinearARGB { red: r, green: g, blue: b, alpha: 1.0 }
    }
}

impl From<&RGB> for LinearARGB {
    fn from(rgb: &RGB) -> Self {
        LinearARGB::from(*rgb)
    }
}

impl From<LinearARGB> for RGB {
    /// Converts LinearARGB (physically linear light values) to sRGB (gamma-encoded), discarding alpha.
    /// Uses the fast gamma approximation with inverse power of ~0.4545.
    ///
    /// # Examples
    /// ```
    /// let linear = LinearARGB::new(0.5, 0.5, 0.5, 0.8).unwrap();  // 50% linear light
    /// let rgb: RGB = linear.into();  // ~186 in sRGB (73% gray), alpha discarded
    /// ```
    fn from(linear: LinearARGB) -> Self {
        let r = (linear.red.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let g = (linear.green.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let b = (linear.blue.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        RGB {
            red: r,
            green: g,
            blue: b,
        }
    }
}

impl From<&LinearARGB> for RGB {
    fn from(linear: &LinearARGB) -> Self {
        RGB::from(*linear)
    }
}

impl From<ARGB> for LinearARGB {
    /// Converts ARGB (gamma-encoded) to LinearARGB (physically linear light values).
    /// Uses the fast gamma approximation with power of 2.2.
    /// Alpha is converted from [0, 255] to [0.0, 1.0].
    ///
    /// # Examples
    /// ```
    /// let argb = ARGB::new(128, 128, 128, 128);  // 50% gray in sRGB, 50% alpha
    /// let linear: LinearARGB = argb.into();  // ~21.8% in linear space, alpha=~0.5
    /// ```
    fn from(argb: ARGB) -> Self {
        let r = (argb.red as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let g = (argb.green as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let b = (argb.blue as f32 * INV_U8_MAX).powf(SRGB_GAMMA);
        let a = argb.alpha as f32 * INV_U8_MAX;
        LinearARGB { red: r, green: g, blue: b, alpha: a }
    }
}

impl From<&ARGB> for LinearARGB {
    fn from(argb: &ARGB) -> Self {
        LinearARGB::from(*argb)
    }
}

impl From<LinearARGB> for ARGB {
    /// Converts LinearARGB (physically linear light values) to ARGB (gamma-encoded).
    /// Uses the fast gamma approximation with inverse power of ~0.4545.
    /// Alpha is converted from [0.0, 1.0] to [0, 255].
    ///
    /// # Examples
    /// ```
    /// let linear = LinearARGB::new(0.5, 0.5, 0.5, 0.8).unwrap();  // 50% linear light
    /// let argb: ARGB = linear.into();  // ~186 in sRGB (73% gray), alpha=204
    /// ```
    fn from(linear: LinearARGB) -> Self {
        let r = (linear.red.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let g = (linear.green.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let b = (linear.blue.powf(SRGB_INV_GAMMA) * U8_MAX_F32).round() as u8;
        let a = (linear.alpha * U8_MAX_F32).round() as u8;
        ARGB {
            alpha: a,
            red: r,
            green: g,
            blue: b,
        }
    }
}

impl From<&LinearARGB> for ARGB {
    fn from(linear: &LinearARGB) -> Self {
        ARGB::from(*linear)
    }
}

impl From<HSL> for LinearARGB {
    /// Converts HSL to LinearARGB by chaining through RGB, with full opacity.
    /// First converts HSL → RGB (gamma-encoded), then RGB → LinearARGB.
    fn from(hsl: HSL) -> Self {
        let rgb: RGB = hsl.into();
        rgb.into()
    }
}

impl From<&HSL> for LinearARGB {
    fn from(color: &HSL) -> Self {
        let linear: LinearARGB = (*color).into();
        linear
    }
}

impl From<LinearARGB> for HSL {
    /// Converts LinearARGB to HSL by chaining through RGB, discarding alpha.
    /// First converts LinearARGB → RGB (gamma-encoded), then RGB → HSL.
    fn from(linear: LinearARGB) -> Self {
        let rgb: RGB = linear.into();
        rgb.into()
    }
}

impl From<&LinearARGB> for HSL {
    fn from(color: &LinearARGB) -> Self {
        let hsl: HSL = (*color).into();
        hsl
    }
}

impl From<HSLA> for LinearARGB {
    /// Converts HSLA to LinearARGB by chaining through ARGB.
    fn from(hsla: HSLA) -> Self {
        let argb: ARGB = hsla.into();
        argb.into()
    }
}

impl From<&HSLA> for LinearARGB {
    fn from(color: &HSLA) -> Self {
        let linear: LinearARGB = (*color).into();
        linear
    }
}

impl From<LinearARGB> for HSLA {
    /// Converts LinearARGB to HSLA by chaining through ARGB.
    /// Preserves alpha channel.
    fn from(linear: LinearARGB) -> Self {
        let argb: ARGB = linear.into();
        argb.into()
    }
}

impl From<&LinearARGB> for HSLA {
    fn from(color: &LinearARGB) -> Self {
        let hsla: HSLA = (*color).into();
        hsla
    }
}

impl From<LinearARGB> for LinearRGB {
    /// Converts LinearARGB to LinearRGB, discarding alpha.
    fn from(argb: LinearARGB) -> Self {
        LinearRGB {
            red: argb.red,
            green: argb.green,
            blue: argb.blue,
        }
    }
}

impl From<&LinearARGB> for LinearRGB {
    fn from(argb: &LinearARGB) -> Self {
        LinearRGB::from(*argb)
    }
}

impl From<LinearRGB> for LinearARGB {
    /// Converts LinearRGB to LinearARGB with full opacity.
    fn from(rgb: LinearRGB) -> Self {
        LinearARGB {
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
            alpha: 1.0,
        }
    }
}

impl From<&LinearRGB> for LinearARGB {
    fn from(rgb: &LinearRGB) -> Self {
        LinearARGB::from(*rgb)
    }
}

impl From<PremultipliedARGB> for LinearARGB {
    /// Converts PremultipliedARGB to LinearARGB, dividing RGB by alpha.
    ///
    /// **Note**: This assumes both formats use linear RGB values.
    /// Recovers straight-alpha colors from premultiplied format.
    fn from(premult: PremultipliedARGB) -> Self {
        if premult.a == 0.0 {
            return LinearARGB {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            };
        }

        let inv_alpha = 1.0 / premult.a;
        LinearARGB {
            red: premult.r * inv_alpha,
            green: premult.g * inv_alpha,
            blue: premult.b * inv_alpha,
            alpha: premult.a,
        }
    }
}

impl From<&PremultipliedARGB> for LinearARGB {
    fn from(color: &PremultipliedARGB) -> Self {
        LinearARGB::from(*color)
    }
}

impl From<LinearARGB> for PremultipliedARGB {
    /// Converts LinearARGB to PremultipliedARGB, multiplying RGB by alpha.
    ///
    /// **Note**: This assumes both formats use linear RGB values.
    /// Converts straight-alpha to premultiplied format.
    fn from(linear: LinearARGB) -> Self {
        PremultipliedARGB {
            r: linear.red * linear.alpha,
            g: linear.green * linear.alpha,
            b: linear.blue * linear.alpha,
            a: linear.alpha,
        }
    }
}

impl From<&LinearARGB> for PremultipliedARGB {
    fn from(color: &LinearARGB) -> Self {
        PremultipliedARGB::from(*color)
    }
}

// ===== HSV Conversions =====

impl From<RGB> for HSV {
    /// Converts RGB to HSV color space.
    ///
    /// Uses standard RGB-to-HSV conversion algorithm:
    /// 1. Normalize RGB values to [0, 1]
    /// 2. V = max(R, G, B)
    /// 3. Calculate delta = max - min
    /// 4. S = delta / V (if V > 0, else 0)
    /// 5. H based on which channel is max (0-360°)
    ///
    /// # Algorithm Details
    /// - Value: Maximum of RGB channels
    /// - Saturation: Color intensity relative to value
    /// - Hue: Based on dominant color channel (60° segments)
    fn from(rgb: RGB) -> Self {
        let r = rgb.red as f32 * INV_U8_MAX;
        let g = rgb.green as f32 * INV_U8_MAX;
        let b = rgb.blue as f32 * INV_U8_MAX;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        // Calculate hue (0-360°)
        let hue = if delta == 0.0 {
            0.0 // Grayscale, hue is undefined
        } else if max == r {
            HUE_SEGMENT_DEGREES * (((g - b) / delta) % 6.0)
        } else if max == g {
            HUE_SEGMENT_DEGREES * (((b - r) / delta) + 2.0)
        } else {
            HUE_SEGMENT_DEGREES * (((r - g) / delta) + 4.0)
        };

        // Calculate saturation
        let saturation = if max == 0.0 { 0.0 } else { delta / max };

        HSV::new(hue, saturation, max)
    }
}

impl From<&RGB> for HSV {
    fn from(color: &RGB) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for RGB {
    /// Converts HSV to RGB color space.
    ///
    /// Uses standard HSV-to-RGB conversion algorithm:
    /// 1. Calculate chroma: C = V × S
    /// 2. Determine RGB' values based on hue segment (60° intervals)
    /// 3. Add (V - C) to each component to achieve target brightness
    ///
    /// # Algorithm Details
    /// - Hue segments: Red-Yellow-Green-Cyan-Blue-Magenta (60° each)
    /// - X = C × (1 - |(H/60) mod 2 - 1|)
    /// - m = V - C (brightness offset)
    fn from(hsv: HSV) -> Self {
        let h = hsv.hue.0;
        let s = hsv.saturation.0;
        let v = hsv.value.0;

        let c = v * s; // Chroma
        let h_prime = h / HUE_SEGMENT_DEGREES;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = match h_prime as u8 {
            0 => (c, x, 0.0),       // Red to Yellow (0-60°)
            1 => (x, c, 0.0),       // Yellow to Green (60-120°)
            2 => (0.0, c, x),       // Green to Cyan (120-180°)
            3 => (0.0, x, c),       // Cyan to Blue (180-240°)
            4 => (x, 0.0, c),       // Blue to Magenta (240-300°)
            _ => (c, 0.0, x),       // Magenta to Red (300-360°)
        };

        RGB {
            red: ((r_prime + m) * U8_MAX_F32) as u8,
            green: ((g_prime + m) * U8_MAX_F32) as u8,
            blue: ((b_prime + m) * U8_MAX_F32) as u8,
        }
    }
}

impl From<&HSV> for RGB {
    fn from(color: &HSV) -> Self {
        RGB::from(*color)
    }
}

impl From<ARGB> for HSV {
    /// Converts ARGB to HSV, discarding the alpha channel.
    ///
    /// Chains through RGB conversion: ARGB → RGB → HSV.
    fn from(argb: ARGB) -> Self {
        let rgb: RGB = argb.into();
        rgb.into()
    }
}

impl From<&ARGB> for HSV {
    fn from(color: &ARGB) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for ARGB {
    /// Converts HSV to ARGB with full opacity.
    ///
    /// Chains through RGB conversion: HSV → RGB → ARGB.
    fn from(hsv: HSV) -> Self {
        let rgb: RGB = hsv.into();
        ARGB {
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
            alpha: 255,
        }
    }
}

impl From<&HSV> for ARGB {
    fn from(color: &HSV) -> Self {
        ARGB::from(*color)
    }
}

// ===== HSVA Conversions =====

impl From<RGB> for HSVA {
    /// Converts RGB to HSVA with full opacity.
    ///
    /// Chains through HSV conversion: RGB → HSV → HSVA.
    /// Alpha is set to 1.0 (fully opaque).
    fn from(rgb: RGB) -> Self {
        let hsv: HSV = rgb.into();
        HSVA::new(hsv.hue.0, hsv.saturation.0, hsv.value.0, 1.0)
    }
}

impl From<&RGB> for HSVA {
    fn from(color: &RGB) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for RGB {
    /// Converts HSVA to RGB, discarding the alpha channel.
    ///
    /// Chains through HSV conversion: HSVA → HSV → RGB.
    fn from(hsva: HSVA) -> Self {
        let hsv = HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0);
        hsv.into()
    }
}

impl From<&HSVA> for RGB {
    fn from(color: &HSVA) -> Self {
        RGB::from(*color)
    }
}

impl From<ARGB> for HSVA {
    /// Converts ARGB to HSVA, preserving the alpha channel.
    ///
    /// Chains through RGB and HSV conversions: ARGB → RGB → HSV → HSVA.
    /// The alpha channel is converted from [0, 255] to [0.0, 1.0].
    fn from(argb: ARGB) -> Self {
        let hsv: HSV = RGB::from(argb).into();
        HSVA::new(
            hsv.hue.0,
            hsv.saturation.0,
            hsv.value.0,
            argb.alpha as f32 * INV_U8_MAX,
        )
    }
}

impl From<&ARGB> for HSVA {
    fn from(color: &ARGB) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for ARGB {
    /// Converts HSVA to ARGB, preserving the alpha channel.
    ///
    /// Chains through HSV and RGB conversions: HSVA → HSV → RGB → ARGB.
    /// The alpha channel is converted from [0.0, 1.0] to [0, 255].
    fn from(hsva: HSVA) -> Self {
        let rgb: RGB = HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0).into();
        ARGB {
            red: rgb.red,
            green: rgb.green,
            blue: rgb.blue,
            alpha: (hsva.alpha.0 * U8_MAX_F32) as u8,
        }
    }
}

impl From<&HSVA> for ARGB {
    fn from(color: &HSVA) -> Self {
        ARGB::from(*color)
    }
}

// ===== HSL ↔ HSV Conversions =====

impl From<HSL> for HSV {
    /// Converts HSL to HSV color space.
    ///
    /// Uses the standard HSL-to-HSV transformation:
    /// - V = L + S × min(L, 1 - L)
    /// - S_hsv = if V == 0: 0 else: 2 × (1 - L / V)
    /// - H remains the same
    ///
    /// # Examples
    /// ```rust
    /// let red = HSL::new(0.0, 1.0, 0.5); // HSL(0°, 1.0, 0.5) = Pure red
    /// let hsv_red: HSV = red.into(); // HSV(0°, 1.0, 1.0) = Same pure red
    /// ```
    fn from(hsl: HSL) -> Self {
        let h = hsl.hue.0;
        let s_hsl = hsl.saturation.0;
        let l = hsl.lightness.0;

        // Calculate HSV value
        let v = l + s_hsl * l.min(1.0 - l);

        // Calculate HSV saturation
        let s_hsv = if v == 0.0 { 0.0 } else { 2.0 * (1.0 - l / v) };

        HSV::new(h, s_hsv, v)
    }
}

impl From<&HSL> for HSV {
    fn from(color: &HSL) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for HSL {
    /// Converts HSV to HSL color space.
    ///
    /// Uses the standard HSV-to-HSL transformation:
    /// - L = V × (1 - S / 2)
    /// - S_hsl = if L == 0 or L == 1: 0 else: (V - L) / min(L, 1 - L)
    /// - H remains the same
    ///
    /// # Examples
    /// ```rust
    /// let blue = HSV::new(240.0, 1.0, 1.0); // HSV(240°, 1.0, 1.0) = Pure blue
    /// let hsl_blue: HSL = blue.into(); // HSL(240°, 1.0, 0.5) = Same pure blue
    /// ```
    fn from(hsv: HSV) -> Self {
        let h = hsv.hue.0;
        let s_hsv = hsv.saturation.0;
        let v = hsv.value.0;

        // Calculate HSL lightness
        let l = v * (1.0 - s_hsv / 2.0);

        // Calculate HSL saturation
        let s_hsl = if l == 0.0 || l == 1.0 {
            0.0
        } else {
            (v - l) / l.min(1.0 - l)
        };

        HSL::new(h, s_hsl, l)
    }
}

impl From<&HSV> for HSL {
    fn from(color: &HSV) -> Self {
        HSL::from(*color)
    }
}

impl From<HSLA> for HSV {
    /// Converts HSLA to HSV, discarding the alpha channel.
    ///
    /// Chains through HSL conversion: HSLA → HSL → HSV.
    fn from(hsla: HSLA) -> Self {
        HSL::new(hsla.hue.0, hsla.saturation.0, hsla.lightness.0).into()
    }
}

impl From<&HSLA> for HSV {
    fn from(color: &HSLA) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for HSLA {
    /// Converts HSV to HSLA with full opacity.
    ///
    /// Chains through HSL conversion: HSV → HSL → HSLA.
    /// Alpha is set to 1.0 (fully opaque).
    fn from(hsv: HSV) -> Self {
        let hsl: HSL = hsv.into();
        HSLA::new(hsl.hue.0, hsl.saturation.0, hsl.lightness.0, 1.0)
    }
}

impl From<&HSV> for HSLA {
    fn from(color: &HSV) -> Self {
        HSLA::from(*color)
    }
}

impl From<HSLA> for HSVA {
    /// Converts HSLA to HSVA, preserving the alpha channel.
    ///
    /// Chains through HSL and HSV conversions: HSLA → HSL → HSV → HSVA.
    fn from(hsla: HSLA) -> Self {
        let hsv: HSV = HSL::new(hsla.hue.0, hsla.saturation.0, hsla.lightness.0).into();
        HSVA::new(hsv.hue.0, hsv.saturation.0, hsv.value.0, hsla.alpha.0)
    }
}

impl From<&HSLA> for HSVA {
    fn from(color: &HSLA) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for HSLA {
    /// Converts HSVA to HSLA, preserving the alpha channel.
    ///
    /// Chains through HSV and HSL conversions: HSVA → HSV → HSL → HSLA.
    fn from(hsva: HSVA) -> Self {
        let hsl: HSL = HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0).into();
        HSLA::new(hsl.hue.0, hsl.saturation.0, hsl.lightness.0, hsva.alpha.0)
    }
}

impl From<&HSVA> for HSLA {
    fn from(color: &HSVA) -> Self {
        HSLA::from(*color)
    }
}

impl From<HSL> for HSVA {
    /// Converts HSL to HSVA with full opacity.
    ///
    /// Chains through HSV conversion: HSL → HSV → HSVA.
    fn from(hsl: HSL) -> Self {
        let hsv: HSV = hsl.into();
        HSVA::new(hsv.hue.0, hsv.saturation.0, hsv.value.0, 1.0)
    }
}

impl From<&HSL> for HSVA {
    fn from(color: &HSL) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for HSL {
    /// Converts HSVA to HSL, discarding the alpha channel.
    ///
    /// Chains through HSV conversion: HSVA → HSV → HSL.
    fn from(hsva: HSVA) -> Self {
        HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0).into()
    }
}

impl From<&HSVA> for HSL {
    fn from(color: &HSVA) -> Self {
        HSL::from(*color)
    }
}

// ===== LinearRGB ↔ HSV Conversions =====

impl From<LinearRGB> for HSV {
    /// Converts LinearRGB to HSV by chaining through RGB.
    /// First converts LinearRGB → RGB (gamma-encoded), then RGB → HSV.
    fn from(linear: LinearRGB) -> Self {
        let rgb: RGB = linear.into();
        rgb.into()
    }
}

impl From<&LinearRGB> for HSV {
    fn from(color: &LinearRGB) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for LinearRGB {
    /// Converts HSV to LinearRGB by chaining through RGB.
    /// First converts HSV → RGB (gamma-encoded), then RGB → LinearRGB.
    fn from(hsv: HSV) -> Self {
        let rgb: RGB = hsv.into();
        rgb.into()
    }
}

impl From<&HSV> for LinearRGB {
    fn from(color: &HSV) -> Self {
        LinearRGB::from(*color)
    }
}

impl From<LinearARGB> for HSV {
    /// Converts LinearARGB to HSV, discarding the alpha channel.
    /// Chains through RGB: LinearARGB → RGB → HSV.
    fn from(linear: LinearARGB) -> Self {
        let rgb: RGB = linear.into();
        rgb.into()
    }
}

impl From<&LinearARGB> for HSV {
    fn from(color: &LinearARGB) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for LinearARGB {
    /// Converts HSV to LinearARGB with full opacity.
    /// Chains through ARGB: HSV → ARGB → LinearARGB.
    fn from(hsv: HSV) -> Self {
        let argb: ARGB = hsv.into();
        argb.into()
    }
}

impl From<&HSV> for LinearARGB {
    fn from(color: &HSV) -> Self {
        LinearARGB::from(*color)
    }
}

impl From<LinearARGB> for HSVA {
    /// Converts LinearARGB to HSVA, preserving the alpha channel.
    /// Chains through ARGB: LinearARGB → ARGB → HSVA.
    fn from(linear: LinearARGB) -> Self {
        let argb: ARGB = linear.into();
        argb.into()
    }
}

impl From<&LinearARGB> for HSVA {
    fn from(color: &LinearARGB) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for LinearARGB {
    /// Converts HSVA to LinearARGB, preserving the alpha channel.
    /// Chains through ARGB: HSVA → ARGB → LinearARGB.
    fn from(hsva: HSVA) -> Self {
        let argb: ARGB = hsva.into();
        argb.into()
    }
}

impl From<&HSVA> for LinearARGB {
    fn from(color: &HSVA) -> Self {
        LinearARGB::from(*color)
    }
}

impl From<LinearRGB> for HSVA {
    /// Converts LinearRGB to HSVA with full opacity.
    /// Chains through RGB and HSV: LinearRGB → RGB → HSV → HSVA.
    fn from(linear: LinearRGB) -> Self {
        let hsv: HSV = linear.into();
        HSVA::new(hsv.hue.0, hsv.saturation.0, hsv.value.0, 1.0)
    }
}

impl From<&LinearRGB> for HSVA {
    fn from(color: &LinearRGB) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for LinearRGB {
    /// Converts HSVA to LinearRGB, discarding the alpha channel.
    /// Chains through HSV and RGB: HSVA → HSV → RGB → LinearRGB.
    fn from(hsva: HSVA) -> Self {
        let hsv = HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0);
        hsv.into()
    }
}

impl From<&HSVA> for LinearRGB {
    fn from(color: &HSVA) -> Self {
        LinearRGB::from(*color)
    }
}

// ===== PremultipliedARGB ↔ HSV Conversions =====

impl From<PremultipliedARGB> for HSV {
    /// Converts PremultipliedARGB to HSV, discarding alpha.
    /// Chains through ARGB → RGB → HSV.
    fn from(premult: PremultipliedARGB) -> Self {
        let argb: ARGB = premult.into();
        argb.into()
    }
}

impl From<&PremultipliedARGB> for HSV {
    fn from(color: &PremultipliedARGB) -> Self {
        HSV::from(*color)
    }
}

impl From<HSV> for PremultipliedARGB {
    /// Converts HSV to PremultipliedARGB with full opacity.
    /// Chains through ARGB: HSV → ARGB → PremultipliedARGB.
    /// Since alpha is 1.0, RGB values are unchanged.
    fn from(hsv: HSV) -> Self {
        let argb: ARGB = hsv.into();
        argb.into()
    }
}

impl From<&HSV> for PremultipliedARGB {
    fn from(color: &HSV) -> Self {
        PremultipliedARGB::from(*color)
    }
}

impl From<PremultipliedARGB> for HSVA {
    /// Converts PremultipliedARGB to HSVA, preserving alpha.
    /// Chains through ARGB: PremultipliedARGB → ARGB → HSVA.
    fn from(premult: PremultipliedARGB) -> Self {
        let argb: ARGB = premult.into();
        argb.into()
    }
}

impl From<&PremultipliedARGB> for HSVA {
    fn from(color: &PremultipliedARGB) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for PremultipliedARGB {
    /// Converts HSVA to PremultipliedARGB, preserving alpha.
    /// Chains through ARGB: HSVA → ARGB → PremultipliedARGB.
    fn from(hsva: HSVA) -> Self {
        let argb: ARGB = hsva.into();
        argb.into()
    }
}

impl From<&HSVA> for PremultipliedARGB {
    fn from(color: &HSVA) -> Self {
        PremultipliedARGB::from(*color)
    }
}

// ===== HSV ↔ HSVA Conversions =====

impl From<HSV> for HSVA {
    /// Converts HSV to HSVA with full opacity.
    ///
    /// Direct field mapping with alpha set to 1.0 (fully opaque).
    fn from(hsv: HSV) -> Self {
        HSVA::new(hsv.hue.0, hsv.saturation.0, hsv.value.0, 1.0)
    }
}

impl From<&HSV> for HSVA {
    fn from(color: &HSV) -> Self {
        HSVA::from(*color)
    }
}

impl From<HSVA> for HSV {
    /// Converts HSVA to HSV, discarding the alpha channel.
    ///
    /// Direct field mapping with no color space transformation.
    fn from(hsva: HSVA) -> Self {
        HSV::new(hsva.hue.0, hsva.saturation.0, hsva.value.0)
    }
}

impl From<&HSVA> for HSV {
    fn from(color: &HSVA) -> Self {
        HSV::from(*color)
    }
}
