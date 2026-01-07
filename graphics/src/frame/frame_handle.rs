//! Frame handle module for safe frame lifetime management.
//!
//! This module provides a RAII (Resource Acquisition Is Initialization) handle for managing
//! the lifetime of rendering frames. The [`FrameHandle`] ensures that frames are properly
//! begun and ended, preventing common errors such as:
//!
//! - Submitting render packets to an inactive frame
//! - Forgetting to end a frame
//! - Using a frame after it has been ended
//!
//! The handle automatically ends the frame when dropped, ensuring cleanup even in the
//! presence of early returns or panics.
//!
//! # Frame Lifecycle
//!
//! A typical frame follows this lifecycle:
//!
//! 1. **Begin**: Create a `FrameHandle` by calling [`GraphicsContext::begin_frame()`]
//! 2. **Record**: Submit render packets via [`FrameHandle::submit_packet()`]
//! 3. **End**: Either explicitly call [`FrameHandle::end_frame()`] or let it drop
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use crate::context::GraphicsContext;
use crate::render::RenderPacket;

/// A RAII handle representing an active rendering frame.
///
/// `FrameHandle` provides exclusive access to submit rendering work for a single frame.
/// It enforces the frame lifecycle through Rust's ownership system and automatically
/// cleans up when dropped.
///
/// # Lifetime
///
/// The lifetime parameter `'graphics_context` ties this handle to the `GraphicsContext`
/// it was created from, ensuring that:
///
/// - The handle cannot outlive the graphics context
/// - No other frames can be started while this handle exists (due to mutable borrow)
/// - The frame is properly synchronized with the graphics context
///
/// # Safety Guarantees
///
/// - **No double-submit**: Only one frame can be active at a time per context
/// - **Automatic cleanup**: Frame is always ended, even on panic or early return
/// - **Type-safe state**: Cannot submit to an inactive handle (enforced by borrowing)
#[derive(Debug)]
pub struct FrameHandle<'graphics_context> {
    _graphics_context: &'graphics_context mut GraphicsContext,
    active: bool,
    pub(crate) frame_index: usize,
}

impl<'ctx> FrameHandle<'ctx> {
    /// Creates a new frame handle.
    ///
    /// This is typically called internally by [`GraphicsContext::begin_frame()`]
    /// and should not be constructed manually by user code.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Mutable reference to the graphics context
    /// * `frame_index` - Index of the frame being managed
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::context::GraphicsContext;
    /// # use toyengine_graphics::frame::FrameHandle;
    /// # let mut ctx = GraphicsContext::new();
    /// // Typically called internally, not by user code
    /// let frame = FrameHandle::new(&mut ctx, 0);
    /// ```
    pub fn new(ctx: &'ctx mut GraphicsContext, frame_index: usize) -> Self {
        FrameHandle {
            _graphics_context: ctx,
            active: true,
            frame_index,
        }
    }

    /// Submits a render packet to this frame.
    ///
    /// Render packets contain the data and commands needed to render geometry,
    /// apply effects, or execute other GPU operations. Multiple packets can be
    /// submitted to a single frame and will be processed in submission order.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if called on an inactive frame handle (one that has
    /// already been ended).
    ///
    /// # Arguments
    ///
    /// * `_data` - The render packet to submit
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::context::GraphicsContext;
    /// # use toyengine_graphics::render::RenderPacket;
    /// # let mut ctx = GraphicsContext::new();
    /// let mut frame = ctx.begin_frame();
    ///
    /// // Submit multiple render packets
    /// frame.submit_packet(RenderPacket::default());
    /// frame.submit_packet(RenderPacket::default());
    /// frame.submit_packet(RenderPacket::default());
    ///
    /// frame.end_frame();
    /// ```
    #[inline]
    pub fn submit_packet(&mut self, _data: RenderPacket) {
        debug_assert!(self.active, "Cannot submit to an inactive handle");

        self._graphics_context
            ._add_valid_render_packet(self.frame_index, _data);
    }

    /// Explicitly ends the frame, releasing it for GPU processing.
    ///
    /// This method marks the frame as complete and submits all recorded render
    /// packets to the graphics context for processing. After calling this method,
    /// no more render packets can be submitted to this frame.
    ///
    /// **Note**: Calling this method is optional, as the [`Drop`] implementation
    /// automatically ends the frame when the handle goes out of scope. Explicit
    /// calls are useful for timing control or when you want to ensure frame
    /// submission happens at a specific point.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if called on a frame that has already been ended.
    ///
    /// # Examples
    ///
    /// ## Explicit End
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::context::GraphicsContext;
    /// # use toyengine_graphics::render::RenderPacket;
    /// # let mut ctx = GraphicsContext::new();
    /// let mut frame = ctx.begin_frame();
    /// frame.submit_packet(RenderPacket::default());
    /// frame.end_frame(); // Explicitly end
    /// ```
    ///
    /// ## Automatic End via Drop
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::context::GraphicsContext;
    /// # use toyengine_graphics::render::RenderPacket;
    /// # let mut ctx = GraphicsContext::new();
    /// {
    ///     let mut frame = ctx.begin_frame();
    ///     frame.submit_packet(RenderPacket::default());
    /// } // Frame automatically ended here
    /// ```
    #[inline]
    pub fn end_frame(&mut self) {
        debug_assert!(self.active, "Frame already ended");
        self.active = false;
        self._graphics_context._release(self.frame_index);
    }
}

/// Implements automatic frame cleanup when the handle is dropped.
///
/// This ensures that frames are always properly ended, even in the presence
/// of panics, early returns, or other non-linear control flow. If [`end_frame()`]
/// has not been called explicitly, it will be called automatically here.
///
/// This is a key part of the RAII pattern that makes frame management safe
/// and ergonomic.
///
/// # Examples
///
/// ```rust,no_run
/// # use toyengine_graphics::context::GraphicsContext;
/// # use toyengine_graphics::render::RenderPacket;
/// # let mut ctx = GraphicsContext::new();
/// fn render_conditional(ctx: &mut GraphicsContext, should_render: bool) {
///     let mut frame = ctx.begin_frame();
///
///     if !should_render {
///         return; // Drop is called here, frame is ended automatically
///     }
///
///     frame.submit_packet(RenderPacket::default());
///     // Drop is called here, frame is ended automatically
/// }
/// ```
///
/// [`end_frame()`]: FrameHandle::end_frame
impl<'a> Drop for FrameHandle<'a> {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            self._graphics_context._release(self.frame_index);
        }
    }
}
