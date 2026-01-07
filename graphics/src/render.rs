//! Core rendering types and abstractions.
//!
//! This module provides the fundamental types used throughout the rendering system,
//! including render phases, sort keys, and GPU resource handles. These types form
//! the vocabulary for describing rendering work.
//!
//! # Module Organization
//!
//! - **Render Packets**: See [`render_packet`] module for the main packet type
//! - **Render Phases**: Enum defining the execution phases for rendering
//! - **Sort Keys**: 64-bit keys for ordering render work within phases
//! - **GPU Handles**: Type-safe handles to GPU-resident resources
//!
//! # Design Philosophy
//!
//! The types in this module prioritize:
//!
//! - **Type Safety**: Distinct handle types prevent mixing mesh and material handles
//! - **Performance**: Newtype wrappers have zero runtime cost
//! - **Ergonomics**: Conversion traits and `Deref` make handles easy to work with
//! - **Clarity**: Each type has a clear, single purpose
//!
//! # Render Phases
//!
//! Rendering is divided into distinct phases executed in a specific order.
//! This ensures correct rendering (e.g., shadows before lighting, transparent
//! after opaque) and enables optimizations specific to each phase.
//!
//! # Sort Keys
//!
//! Sort keys are 64-bit unsigned integers that determine the order of render
//! packets within a phase. They can encode multiple sorting criteria in a
//! single value for efficient sorting.
//!
//! # GPU Resource Handles
//!
//! Handles are lightweight identifiers (32-bit integers) that reference
//! GPU-resident resources. They're cheap to copy and compare but don't
//! directly expose the underlying data.
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

pub mod render_packet;
use std::ops::Deref;

pub use render_packet::*;

/// Defines the major phases of rendering execution.
///
/// Each frame is divided into distinct rendering phases that execute in a predefined
/// order. This ordering ensures correct rendering (dependencies between phases) and
/// enables phase-specific optimizations.
///
/// # Phase Execution Order
///
/// Phases execute in the following order (defined in [`FrameManager`]):
///
/// 1. **Shadow**: Render depth from light's perspective for shadow mapping
/// 2. **DepthPrePass**: Populate depth buffer for early-z optimization
/// 3. **Opaque**: Render solid objects with depth testing
/// 4. **Transparent**: Render translucent objects with blending
/// 5. **Overlay**: Render UI and other screen-space elements
/// 6. **Debug**: Render debug visualizations (wireframes, bounding boxes, etc.)
///
/// # Phase Characteristics
///
/// ## Shadow
/// - Renders to shadow map textures
/// - Only depth matters, no color output
/// - Executed once per shadow-casting light
///
/// ## DepthPrePass
/// - Fills depth buffer without color writes
/// - Enables early-z rejection in opaque phase
/// - Optional optimization for complex scenes
///
/// ## Opaque
/// - Main rendering of solid geometry
/// - Sorted front-to-back for early-z
/// - Most rendering work happens here
///
/// ## Transparent
/// - Objects with alpha blending
/// - Sorted back-to-front for correct blending
/// - Typically doesn't write to depth buffer
///
/// ## Overlay
/// - Screen-space UI elements
/// - Fixed depth or depth-disabled
/// - Renders after 3D scene
///
/// ## Debug
/// - Development visualizations
/// - Can be disabled in release builds
/// - Renders last to overlay everything
///
/// [`FrameManager`]: crate::frame::FrameManager
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPhase {
    Shadow,
    DepthPrePass,
    Opaque,
    Transparent,
    Overlay,
    Debug,
}

/// A 64-bit key used for sorting render packets within a phase.
///
/// Sort keys determine the order in which render packets are processed, enabling
/// important optimizations like front-to-back rendering for opaque objects or
/// material batching to reduce state changes.
///
/// # Structure
///
/// The 64-bit key can be divided into bit ranges encoding different criteria:
///
/// ```text
/// Example encoding:
/// Bits 63-48: Material ID (16 bits, 65536 materials)
/// Bits 47-32: Coarse depth bucket (16 bits)
/// Bits 31-0:  Fine depth or other criteria (32 bits)
/// ```
///
/// # Ordering
///
/// `SortKey` implements `Ord`, so packets sort in ascending order by default.
/// For back-to-front sorting (transparent objects), use inverted distance
/// or sort in reverse.
///
/// # Performance
///
/// The transparent newtype has zero runtime cost and sorts as efficiently as a raw `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct SortKey(pub u64);

/// Handle to a GPU-resident mesh resource.
///
/// Meshes contain vertex data (positions, normals, UVs) and index data for
/// drawing geometry. This handle is a lightweight identifier that can be
/// copied freely without duplicating the actual GPU data.
///
/// # Type Safety
///
/// The distinct type prevents accidentally using a material handle where a
/// mesh handle is expected, catching errors at compile time.
///
/// # Implementation
///
/// Currently wraps a 32-bit unsigned integer, allowing up to 4 billion unique
/// mesh handles. In a real implementation, this might include generation counters
/// or other metadata for handle validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GpuMeshHandle(pub u32);

/// Handle to a GPU-resident material resource.
///
/// Materials contain shader programs, texture references, and rendering state
/// (blend modes, depth testing, etc.). This handle is a lightweight identifier
/// that references the actual GPU data.
///
/// # Type Safety
///
/// The distinct type prevents accidentally using a mesh handle where a material
/// handle is expected, catching errors at compile time.
///
/// # Implementation
///
/// Currently wraps a 32-bit unsigned integer, allowing up to 4 billion unique
/// material handles. In a real implementation, this might include generation
/// counters for handle validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GpuMaterialHandle(pub u32);

impl SortKey {
    /// Creates a new sort key from a 64-bit unsigned integer.
    ///
    /// # Arguments
    ///
    /// * `key` - The raw 64-bit value encoding sorting criteria
    ///
    /// # Returns
    ///
    /// A `SortKey` wrapping the provided value.
    pub const fn new(key: u64) -> Self {
        SortKey(key)
    }
}

impl GpuMeshHandle {
    /// Creates a new GPU mesh handle from a raw 32-bit identifier.
    ///
    /// # Arguments
    ///
    /// * `handle` - The raw handle value, typically assigned by a resource manager
    ///
    /// # Returns
    ///
    /// A type-safe `GpuMeshHandle` wrapping the provided value.
    pub const fn new(handle: u32) -> Self {
        GpuMeshHandle(handle)
    }
}

impl GpuMaterialHandle {
    /// Creates a new GPU material handle from a raw 32-bit identifier.
    ///
    /// # Arguments
    ///
    /// * `handle` - The raw handle value, typically assigned by a resource manager
    ///
    /// # Returns
    ///
    /// A type-safe `GpuMaterialHandle` wrapping the provided value.
    pub const fn new(handle: u32) -> Self {
        GpuMaterialHandle(handle)
    }
}

// ============================================================================
// Conversion Traits - SortKey
// ============================================================================

/// Converts a `u64` to a `SortKey`.
impl From<u64> for SortKey {
    fn from(value: u64) -> Self {
        SortKey::new(value)
    }
}

/// Converts a `&u64` to a `SortKey`.
impl From<&u64> for SortKey {
    fn from(value: &u64) -> Self {
        SortKey::new(*value)
    }
}

/// Converts a `SortKey` to a `u64`.
impl From<SortKey> for u64 {
    fn from(value: SortKey) -> Self {
        *value
    }
}

/// Converts a `&SortKey` to a `u64`.
impl From<&SortKey> for u64 {
    fn from(value: &SortKey) -> Self {
        **value
    }
}

/// Implements `Deref` to allow treating `SortKey` as a `u64` reference.
///
/// This enables direct use of the underlying value without explicit unwrapping:
///
/// ```rust,no_run
/// # use toyengine_graphics::render::SortKey;
/// let key = SortKey::new(12345);
/// let value: u64 = *key; // Dereference to get u64
/// ```
impl Deref for SortKey {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ============================================================================
// Conversion Traits - GpuMeshHandle
// ============================================================================

/// Converts a `u32` to a `GpuMeshHandle`.
impl From<u32> for GpuMeshHandle {
    fn from(value: u32) -> Self {
        GpuMeshHandle::new(value)
    }
}

/// Converts a `&u32` to a `GpuMeshHandle`.
impl From<&u32> for GpuMeshHandle {
    fn from(value: &u32) -> Self {
        GpuMeshHandle::new(*value)
    }
}

/// Converts a `GpuMeshHandle` to a `u32`.
impl From<GpuMeshHandle> for u32 {
    fn from(value: GpuMeshHandle) -> Self {
        *value
    }
}

/// Converts a `&GpuMeshHandle` to a `u32`.
impl From<&GpuMeshHandle> for u32 {
    fn from(value: &GpuMeshHandle) -> Self {
        **value
    }
}

/// Implements `Deref` to allow treating `GpuMeshHandle` as a `u32` reference.
impl Deref for GpuMeshHandle {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ============================================================================
// Conversion Traits - GpuMaterialHandle
// ============================================================================

/// Converts a `u32` to a `GpuMaterialHandle`.
impl From<u32> for GpuMaterialHandle {
    fn from(value: u32) -> Self {
        GpuMaterialHandle::new(value)
    }
}

/// Converts a `&u32` to a `GpuMaterialHandle`.
impl From<&u32> for GpuMaterialHandle {
    fn from(value: &u32) -> Self {
        GpuMaterialHandle::new(*value)
    }
}

/// Converts a `GpuMaterialHandle` to a `u32`.
impl From<GpuMaterialHandle> for u32 {
    fn from(value: GpuMaterialHandle) -> Self {
        *value
    }
}

/// Converts a `&GpuMaterialHandle` to a `u32`.
impl From<&GpuMaterialHandle> for u32 {
    fn from(value: &GpuMaterialHandle) -> Self {
        **value
    }
}

/// Implements `Deref` to allow treating `GpuMaterialHandle` as a `u32` reference.
impl Deref for GpuMaterialHandle {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
