//! Graphics context module for managing rendering state and frame lifecycle.
//!
//! This module provides the core [`GraphicsContext`] type, which serves as the central
//! coordination point for all rendering operations. It manages frame synchronization,
//! render packet submission, and debug state.
//!
//! # Overview
//!
//! The graphics context is responsible for:
//!
//! - **Frame Management**: Coordinates the double-buffered frame system to prevent
//!   GPU stalls and ensure smooth rendering
//! - **Render Packet Batching**: Collects and organizes rendering commands for efficient
//!   GPU submission
//! - **Debug Control**: Provides optional debug annotations and validation that can be
//!   stripped in release builds
//! - **Resource Synchronization**: Ensures proper ordering of GPU operations through
//!   fence-based synchronization
//!
//! # Architecture
//!
//! The context uses a double-buffering strategy with two frame slots:
//!
//! 1. One frame is being recorded (CPU-side)
//! 2. One frame is being processed by the GPU
//!
//! This allows the CPU to prepare the next frame while the GPU processes the current one,
//! maximizing parallelism and preventing pipeline stalls.
//!
//! # Thread Safety
//!
//! `GraphicsContext` is not thread-safe and should only be accessed from a single thread.
//! For multi-threaded rendering, consider using per-thread command buffers that are
//! submitted to the context from the main rendering thread.
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use crate::frame::{FrameHandle, FrameManager};
use crate::render::RenderPacket;

/// The central graphics context for managing rendering operations.
///
/// `GraphicsContext` coordinates all rendering activities, managing the lifecycle
/// of frames and ensuring proper synchronization between the CPU and GPU. It owns
/// the frame manager and controls debug state for the entire rendering pipeline.
///
/// # Fields
///
/// While the internal fields are private, they serve the following purposes:
///
/// - **Frame Manager**: Handles double-buffering, fence synchronization, and frame state
/// - **Debug Mode**: Controls whether debug annotations and names are preserved
///
/// # Lifetime
///
/// The graphics context typically lives for the entire duration of the application
/// or rendering session. Creating and destroying contexts frequently is expensive
/// and should be avoided.
///
/// # Performance Considerations
///
/// - Debug mode (`debug: true`) adds overhead for storing debug names and validation
/// - Release builds should use `debug: false` to minimize memory usage and maximize performance
/// - The frame manager uses a small, fixed-size array for frame storage (no allocations per frame)
#[derive(Debug)]
pub struct GraphicsContext {
    frame_manager: FrameManager,
    debug: bool,
}

impl GraphicsContext {
    /// Creates a new graphics context.
    ///
    /// Initializes the frame management system and sets the debug mode. The context
    /// starts in an idle state with no active frames.
    ///
    /// # Arguments
    ///
    /// * `debug` - Whether to enable debug mode. When `true`, debug names and
    ///   validation are preserved. When `false`, debug information is stripped
    ///   for optimal performance.
    ///
    /// # Performance
    ///
    /// Debug mode adds minimal overhead during packet submission but increases
    /// memory usage for storing debug names. For production builds, use `false`.
    pub fn new(debug: bool) -> Self {
        GraphicsContext {
            frame_manager: FrameManager::new(),
            debug,
        }
    }

    /// Acquires a new frame for rendering.
    ///
    /// This method begins a new frame by:
    /// 1. Selecting the next available frame slot (double-buffered)
    /// 2. Waiting for the GPU to finish any in-flight work on that slot
    /// 3. Returning a [`FrameHandle`] that can be used to submit render packets
    ///
    /// The returned handle provides exclusive access to the frame and ensures
    /// proper cleanup through RAII. When the handle is dropped, the frame is
    /// automatically submitted for GPU processing.
    ///
    /// # Returns
    ///
    /// A [`FrameHandle`] tied to the lifetime of this context. Only one frame
    /// can be active at a time due to the mutable borrow of the context.
    ///
    /// # Blocking Behavior
    ///
    /// If both frame slots are in use (both frames are being processed by the GPU),
    /// this method will block until one becomes available. This prevents the CPU
    /// from getting too far ahead of the GPU.
    ///
    /// # Panics
    ///
    /// May panic in debug builds if called while another frame handle is still active,
    /// though this is prevented by Rust's borrow checker.
    #[inline]
    pub fn acquire_frame(&mut self) -> FrameHandle<'_> {
        // Initialize a new frame handle, and enable manipulation of the frame
        let _ctx = self.frame_manager.begin_frame();
        let frame_index = _ctx.index;
        FrameHandle::new(self, frame_index)
    }

    /// Internal method to release a frame and submit it for GPU processing.
    ///
    /// This is called automatically by [`FrameHandle`] when it is dropped or when
    /// [`FrameHandle::end_frame()`] is called explicitly. It should not be called
    /// directly by user code.
    ///
    /// # Arguments
    ///
    /// * `frame_index` - The index of the frame to release
    ///
    /// # Implementation Note
    ///
    /// This method transfers ownership of the render packets to the frame manager,
    /// which then makes them available for GPU processing. The frame is marked as
    /// in-flight and a fence value is assigned for synchronization.
    #[inline]
    pub(crate) fn _release(&mut self, frame_index: usize) {
        self.frame_manager.end_frame(frame_index);
    }

    /// Internal method to add a validated render packet to the current frame.
    ///
    /// This is called by [`FrameHandle::submit_packet()`] and handles debug name
    /// stripping based on the context's debug mode. It should not be called
    /// directly by user code.
    ///
    /// # Arguments
    ///
    /// * `frame_index` - The index of the frame to add the packet to
    /// * `_packet` - The render packet to add
    ///
    /// # Debug Name Handling
    ///
    /// - **Debug Mode (`debug: true`)**: Preserves all debug names and annotations
    /// - **Release Mode (`debug: false`)**: Strips debug names to reduce memory usage
    ///
    /// This allows developers to use descriptive names during development without
    /// impacting production performance.
    #[inline]
    pub(crate) fn _add_valid_render_packet(&mut self, frame_index: usize, _packet: RenderPacket) {
        // Simple debug stripping of names if not in debug mode
        let packet = if !self.debug {
            RenderPacket {
                debug_name: None,
                .._packet
            }
        } else {
            _packet
        };
        self.frame_manager.push_packet(frame_index, packet);
    }
}
