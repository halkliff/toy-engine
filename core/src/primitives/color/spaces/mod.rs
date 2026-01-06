//! Re-exports all color space types.

pub mod rgb;
pub mod hsl;
pub mod hsv;
pub mod linear;
pub mod premult;

// Re-export all color types
pub use rgb::*;
pub use hsl::*;
pub use hsv::*;
pub use linear::*;
pub use premult::*;