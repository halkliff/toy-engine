//! Frame data module for per-frame rendering state.
//!
//! This module provides data structures for managing per-frame rendering state that
//! needs to be communicated from the application to the rendering backend. This includes
//! camera information, timing data, and environment settings.
//!
//! # Architecture
//!
//! [`FrameData`] is the primary container for all per-frame state, while [`FrameDataProp`]
//! provides a way to set individual properties incrementally during frame setup.
//!
//! # Usage
//!
//! Frame data is typically set once per frame through the [`FrameHandle`] API:
//!
//! ```rust,no_run
//! # use toyengine_graphics::context::GraphicsContext;
//! # use toyengine_graphics::camera::CameraData;
//! # let mut ctx = GraphicsContext::new(false);
//! let mut frame = ctx.acquire_frame();
//! frame.set_camera(CameraData::default());
//! ```
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**
//!
//! [`FrameHandle`]: crate::frame::FrameHandle

use crate::camera::CameraData;

/// Individual frame data properties that can be set independently.
///
/// This enum allows for granular updates to frame state without requiring
/// a complete [`FrameData`] structure. It's primarily used internally for
/// incremental frame setup.
///
/// # Variants
///
/// - **CameraData**: Camera view and projection information
/// - **TimeData**: Timing information (delta time, elapsed time) - not yet implemented
/// - **EnvironmentData**: Scene environment (lighting, fog) - not yet implemented
///
/// # Future Expansion
///
/// Additional variants will be added as needed for:
/// - Global illumination state
/// - Weather/atmospheric effects
/// - Post-processing parameters
/// - Debug visualization settings
#[derive(Debug, Clone)]
pub enum FrameDataProp {
    /// Camera transformation and projection data.
    CameraData(CameraData),

    /// Timing data for animations and frame-rate dependent effects.
    ///
    /// **TODO**: Implement with delta time, total elapsed time, frame number
    TimeData(),

    /// Environment and lighting data for the scene.
    ///
    /// **TODO**: Implement with ambient color, fog parameters, skybox references
    EnvironmentData(),
}

/// Complete per-frame rendering state.
///
/// `FrameData` encapsulates all the information that needs to be provided to the
/// rendering backend once per frame. This data is used by shaders, culling systems,
/// and other rendering algorithms.
///
/// # Fields
///
/// - **camera**: Camera position, orientation, and projection parameters
///
/// # Lifetime
///
/// Frame data is valid only for the duration of a single frame. It's copied into
/// the frame context during frame submission and remains available until GPU
/// processing completes.
///
/// # Future Expansion
///
/// Additional fields will be added for:
/// - Global timing information (delta time, elapsed time)
/// - Environment state (lighting, atmosphere)
/// - Debug settings (wireframe, bounding boxes)
/// - Post-processing parameters
///
/// # Size Considerations
///
/// Keep this structure reasonably small as it's copied per frame. For large
/// per-frame data (e.g., arrays of lights), consider using GPU buffers referenced
/// by handles instead of inline data.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Camera data including view/projection matrices and parameters.
    pub camera: CameraData,
}

impl FrameData {
    /// Creates default frame data with identity camera transforms.
    ///
    /// The default frame data has:
    /// - Identity view and projection matrices
    /// - Camera at origin
    /// - Default near/far planes (0.1 to 1000.0)
    /// - 16:9 aspect ratio
    /// - No valid FOV (marked as uninitialized)
    ///
    /// # Returns
    ///
    /// A `FrameData` instance with default values suitable for initialization.
    ///
    /// # Usage
    ///
    /// This is typically used as a starting point before setting actual camera data:
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::frame::FrameData;
    /// let mut frame_data = FrameData::default();
    /// // Set camera data before use
    /// ```
    pub const fn default() -> Self {
        Self {
            camera: CameraData::default(),
        }
    }

    /// Checks if the frame data has been properly initialized.
    ///
    /// Frame data is considered "none" (uninitialized) if the camera data
    /// has not been set to valid values. This is used internally to validate
    /// that required frame data has been provided before rendering.
    ///
    /// # Returns
    ///
    /// `true` if the frame data is uninitialized, `false` if it has valid data.
    ///
    /// # Implementation Note
    ///
    /// Currently only checks camera data. Future implementations may check
    /// additional required fields.
    pub fn is_none(&self) -> bool {
        self.camera.is_none()
    }
}
