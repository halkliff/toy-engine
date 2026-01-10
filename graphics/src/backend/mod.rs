//! Rendering backend abstraction layer.
//!
//! This module provides the trait-based abstraction that allows the rendering system
//! to work with different graphics APIs (DirectX, Vulkan, Metal, OpenGL, etc.) through
//! a unified interface.
//!
//! # Architecture
//!
//! The backend system uses a trait-based design with the following components:
//!
//! - [`RenderBackend`]: Core trait defining the rendering API contract
//! - [`BackendState`]: Tracks the current state of a backend instance
//! - Backend implementations: Concrete types implementing the trait for specific APIs
//!
//! # Backend Lifecycle
//!
//! Each frame, the backend goes through this lifecycle:
//!
//! ```text
//! Idle
//!   -> begin_frame() -> InFrame
//!   -> begin_phase() -> InFrame { current_phase: Some(phase) }
//!   -> draw() (called multiple times)
//!   -> end_phase() -> InFrame { current_phase: None }
//!   (repeat begin_phase/end_phase for each render phase)
//!   -> end_frame() -> Idle
//! ```
//!
//! # Thread Safety
//!
//! Backends must implement `Send` to allow ownership transfer between threads,
//! but are not required to be `Sync`. Typical usage is single-threaded with
//! command recording happening on one thread at a time.
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use crate::{
    frame::FrameData,
    render::{RenderPacket, RenderPhase},
};
pub(crate) mod null_render;
use std::fmt::Debug;

/// Represents the current state of a rendering backend.
///
/// Backend state is tracked to validate the calling sequence of backend methods
/// and prevent invalid operations (e.g., drawing outside of a phase, beginning
/// a frame when already in a frame).
///
/// # State Transitions
///
/// Valid state transitions:
/// - `Idle` -> `InFrame` (via `begin_frame`)
/// - `InFrame { current_phase: None }` -> `InFrame { current_phase: Some(phase) }` (via `begin_phase`)
/// - `InFrame { current_phase: Some(phase) }` -> `InFrame { current_phase: None }` (via `end_phase`)
/// - `InFrame { current_phase: None }` -> `Idle` (via `end_frame`)
///
/// # Validation
///
/// Backend implementations use this state to add debug assertions that catch
/// incorrect API usage during development.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum BackendState {
    /// Backend is idle and ready to begin a new frame.
    ///
    /// Valid operations: `begin_frame`
    Idle,

    /// Backend is processing a frame.
    ///
    /// When `current_phase` is `None`, a new phase can be started.
    /// When `current_phase` is `Some(phase)`, draw calls are valid for that phase.
    ///
    /// Valid operations depend on `current_phase`:
    /// - `None`: `begin_phase`, `end_frame`
    /// - `Some(phase)`: `draw`, `end_phase`
    InFrame {
        /// The currently active render phase, or None if between phases.
        current_phase: Option<RenderPhase>,
    },
}

/// Core trait for rendering backend implementations.
///
/// `RenderBackend` defines the contract that all graphics API backends must implement.
/// It provides a high-level, platform-agnostic interface for submitting rendering work.
///
/// # Implementation Requirements
///
/// Implementations must:
/// - Be `Debug` for debugging and logging
/// - Be `Send` to allow moving between threads
/// - Follow the correct calling sequence (validated by state tracking)
/// - Handle all render phases defined in [`RenderPhase`]
///
/// # Method Call Sequence
///
/// For each frame, methods must be called in this order:
///
/// 1. `begin_frame` - Initialize frame state and GPU resources
/// 2. For each render phase:
///    a. `begin_phase` - Set up phase-specific rendering state
///    b. `draw` (called N times) - Submit render packets for this phase
///    c. `end_phase` - Clean up phase-specific state
/// 3. `end_frame` - Finalize frame and submit to GPU
///
/// # Error Handling
///
/// The current design uses assertions for error detection. Future implementations
/// may use `Result` types for recoverable errors.
///
/// # Example Implementation
///
/// ```rust,no_run
/// # use toyengine_graphics::backend::RenderBackend;
/// # use toyengine_graphics::frame::FrameData;
/// # use toyengine_graphics::render::{RenderPacket, RenderPhase};
/// # #[derive(Debug)]
/// # struct MyBackend;
/// impl RenderBackend for MyBackend {
///     fn begin_frame(&mut self, data: &FrameData) {
///         // Initialize frame resources
///         // Set camera uniforms
///     }
///
///     fn begin_phase(&mut self, phase: RenderPhase) {
///         // Configure render targets and state for this phase
///     }
///
///     fn draw(&mut self, packet: &RenderPacket) {
///         // Bind resources and issue draw call
///     }
///
///     fn end_phase(&mut self, phase: RenderPhase) {
///         // Cleanup phase-specific resources
///     }
///
///     fn end_frame(&mut self) {
///         // Submit command buffer to GPU
///     }
/// }
/// ```
pub(crate) trait RenderBackend: Debug + Send {
    /// Called once per frame, before any phases execute.
    ///
    /// Use this to initialize frame-wide resources and upload per-frame data
    /// like camera transforms.
    ///
    /// # Arguments
    ///
    /// * `data` - Complete frame data including camera and other per-frame state
    ///
    /// # Implementation Guidelines
    ///
    /// - Upload camera data to GPU uniform buffers
    /// - Reset any per-frame resource allocators
    /// - Set up frame-wide rendering state
    /// - Begin command buffer recording if using explicit command buffers
    fn begin_frame(&mut self, data: &FrameData);

    /// Called at the start of each render phase.
    ///
    /// Use this to configure rendering state specific to the phase, such as
    /// setting render targets, configuring blend modes, or binding shaders.
    ///
    /// # Arguments
    ///
    /// * `phase` - The render phase being started
    ///
    /// # Implementation Guidelines
    ///
    /// - Set appropriate render targets (shadow maps, G-buffer, backbuffer)
    /// - Configure depth/stencil state
    /// - Set blend modes (opaque vs transparent)
    /// - Clear render targets if needed
    fn begin_phase(&mut self, phase: RenderPhase);

    /// Called for each render packet in the current phase.
    ///
    /// Packets arrive pre-sorted according to the phase's sort key ordering.
    /// Implement the actual draw call submission here.
    ///
    /// # Arguments
    ///
    /// * `packet` - The render packet containing mesh, material, and transform data
    ///
    /// # Implementation Guidelines
    ///
    /// - Bind the mesh's vertex and index buffers
    /// - Bind material textures and set shader parameters
    /// - Upload per-object data (world transform) to uniforms
    /// - Issue the draw call
    /// - Consider batching draws with the same material for efficiency
    fn draw(&mut self, packet: &RenderPacket);

    /// Called at the end of each render phase.
    ///
    /// Use this to clean up phase-specific state or resources.
    ///
    /// # Arguments
    ///
    /// * `phase` - The render phase being ended (for validation)
    ///
    /// # Implementation Guidelines
    ///
    /// - Unbind phase-specific resources
    /// - Resolve MSAA buffers if needed
    /// - Transition resources for next phase
    fn end_phase(&mut self, phase: RenderPhase);

    /// Called once per frame, after all phases execute.
    ///
    /// Use this to finalize the frame and submit work to the GPU.
    ///
    /// # Implementation Guidelines
    ///
    /// - End command buffer recording
    /// - Submit command buffers to GPU queue
    /// - Present the rendered image to the display
    /// - Signal any frame completion fences
    fn end_frame(&mut self);
}