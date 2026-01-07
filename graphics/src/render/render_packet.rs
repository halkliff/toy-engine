//! Render packet module for batching and dispatching rendering work.
//!
//! This module defines the [`RenderPacket`] type, which represents a single renderable
//! object with all the data needed to draw it. Render packets are the fundamental unit
//! of work in the rendering system.
//!
//! # Purpose
//!
//! Render packets serve multiple purposes:
//!
//! - **Batching**: Collect rendering commands that can be sorted and grouped for efficiency
//! - **Separation of Concerns**: Decouple high-level scene representation from low-level rendering
//! - **GPU Submission**: Provide a lightweight structure that can be efficiently processed
//! - **Debug Support**: Include optional debug names for profiling and debugging
//!
//! # Design Philosophy
//!
//! The packet design prioritizes:
//!
//! - **Lightweight**: Small enough to create thousands per frame without overhead
//! - **Self-contained**: Contains all data needed to render the object
//! - **Sortable**: Includes phase and sort key for optimal rendering order
//! - **Flexible**: Debug names can be stripped in release builds
//!
//! # Typical Usage Flow
//!
//! 1. Scene traversal creates render packets for visible objects
//! 2. Packets are submitted to the current frame via [`FrameHandle`]
//! 3. Frame manager sorts packets by phase and sort key
//! 4. GPU executes packets in optimized order
//!
//! # Performance Considerations
//!
//! - Packets are cloneable for flexibility but should generally be moved
//! - Debug names use static strings to avoid allocations
//! - Sort keys should be carefully designed for optimal batching
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use toyengine_core::primitives::mat::Matrix4;

use crate::render::{GpuMaterialHandle, GpuMeshHandle, RenderPhase, SortKey};

/// A packet of rendering data for a single drawable object.
///
/// `RenderPacket` encapsulates all the information needed to render one object:
/// transformation, geometry reference, material reference, and metadata for sorting
/// and debugging.
///
/// # Fields
///
/// ## Transformation
///
/// - **world**: 4x4 transformation matrix placing the object in world space
///
/// ## GPU Resources
///
/// - **mesh**: Handle to GPU-resident vertex and index data
/// - **material**: Handle to GPU-resident shader and texture resources
///
/// ## Rendering Control
///
/// - **phase**: Which rendering phase this packet belongs to (opaque, transparent, etc.)
/// - **sort_key**: 64-bit key for sorting packets within a phase
///
/// ## Debug Support
///
/// - **debug_name**: Optional static string for profiling and debugging (can be stripped)
///
/// # Sort Keys
///
/// Sort keys enable important optimizations:
///
/// - **Opaque objects**: Front-to-back sorting for early-z rejection
/// - **Transparent objects**: Back-to-front sorting for correct blending
/// - **Material batching**: Group by material to reduce state changes
/// - **Depth sorting**: Order by distance from camera when needed
///
/// The 64-bit sort key can encode multiple criteria (distance, material ID, etc.)
/// in a single value for efficient sorting.
///
/// # Memory Layout
///
/// The struct is designed to be reasonably small but not micro-optimized.
/// In a high-performance engine, consider:
///
/// - Storing packets in a contiguous array for cache efficiency
/// - Using indices instead of handles for even smaller packets
/// - Separating hot (frequently accessed) and cold (debug) data
///
/// # Cloning
///
/// Packets are `Clone` to allow flexibility in packet generation and processing.
/// However, packets are typically moved rather than cloned for best performance.
#[derive(Debug, Clone)]
pub struct RenderPacket {
    /// World transformation matrix for the object.
    pub world: Matrix4,
    /// Placeholder for mesh data.
    pub mesh: GpuMeshHandle,
    /// Placeholder for material data.
    pub material: GpuMaterialHandle,

    /// The rendering phase this packet belongs to.
    pub phase: RenderPhase,

    /// Sort key for ordering render packets.
    pub sort_key: SortKey,

    /// Optional debug name for the render packet.
    pub debug_name: Option<&'static str>,
}

impl RenderPacket {
    /// Creates a new render packet with all required data.
    ///
    /// This is the primary constructor for render packets. All fields must be
    /// provided explicitly to ensure complete packet configuration.
    ///
    /// # Arguments
    ///
    /// * `world` - World-space transformation matrix for the object
    /// * `mesh` - Handle to the GPU mesh resource containing vertices and indices
    /// * `material` - Handle to the GPU material resource containing shaders and textures
    /// * `phase` - The rendering phase this packet should be executed in
    /// * `sort_key` - 64-bit key used for sorting packets within the phase
    /// * `debug_name` - Optional static string for debugging (can be `None` in release builds)
    ///
    /// # Sort Key Design
    ///
    /// Consider encoding multiple criteria in the sort key:
    ///
    /// ```text
    /// Bits 63-48: Material ID (for batching)
    /// Bits 47-32: Depth bucket (for rough sorting)
    /// Bits 31-0:  Fine-grained distance or other criteria
    /// ```
    ///
    /// # Returns
    ///
    /// A fully initialized `RenderPacket` ready to be submitted to a frame.
    pub fn new(
        world: Matrix4,
        mesh: GpuMeshHandle,
        material: GpuMaterialHandle,
        phase: RenderPhase,
        sort_key: SortKey,
        debug_name: Option<&'static str>,
    ) -> Self {
        RenderPacket {
            world,
            mesh,
            material,
            phase,
            sort_key,
            debug_name,
        }
    }
}
