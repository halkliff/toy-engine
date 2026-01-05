//! Re-exports all color space types.

pub mod rgb;
pub mod hsl;
pub mod hsv;
pub mod linear;
pub mod premult;

// Re-export all color types
pub use rgb::{RGB, ARGB};
pub use hsl::{HSL, HSLA};
pub use hsv::{HSV, HSVA};
pub use linear::{LinearRGB, LinearARGB};
pub use premult::PremultipliedARGB;
