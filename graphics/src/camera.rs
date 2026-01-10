//! Camera module for view and projection transformations.
//!
//! This module provides camera data structures and utilities for 3D rendering.
//! The camera defines how the 3D world is viewed and projected onto the 2D screen.
//!
//! # Camera Types
//!
//! Two primary camera types are supported:
//!
//! - **Perspective**: Objects appear smaller with distance (realistic 3D)
//! - **Orthographic**: Objects maintain size regardless of distance (technical drawings, UI)
//!
//! # Camera Data
//!
//! [`CameraData`] contains all the information needed to render from a camera's viewpoint:
//! - View matrix (camera position and orientation)
//! - Projection matrix (how 3D space maps to 2D screen)
//! - Near and far clipping planes
//! - Field of view and aspect ratio (for perspective cameras)
//!
//! # Memory Layout
//!
//! `CameraData` is carefully laid out to be GPU-friendly with proper alignment and
//! padding for use in uniform buffers. The `#[repr(C)]` attribute ensures consistent
//! memory layout across platforms.
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use bitflags::bitflags;
use bytemuck::Zeroable;
use toyengine_core::primitives::{mat::Matrix4, vec::Vec3};

bitflags! {
    /// Flags indicating camera configuration and type.
    ///
    /// These flags control camera behavior and indicate which parameters are valid.
    /// Multiple flags can be combined to describe the complete camera configuration.
    ///
    /// # Usage
    ///
    /// Flags are typically set when configuring a camera:
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::camera::{CameraData, CameraFlags};
    /// let mut camera = CameraData::default();
    /// camera.flags = CameraFlags::PERSPECTIVE | CameraFlags::HAS_FOV | CameraFlags::HAS_ASPECT_RATIO;
    /// ```
    ///
    /// # Validation
    ///
    /// The rendering system uses these flags to validate camera setup:
    /// - Perspective cameras should have `HAS_FOV` and `HAS_ASPECT_RATIO` set
    /// - Orthographic cameras typically don't use FOV
    /// - Exactly one of `PERSPECTIVE` or `ORTHOGRAPHIC` should be set
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable)]
    #[repr(transparent)]
    pub struct CameraFlags: u32 {
      /// No flags set - camera is uninitialized or invalid.
      const NONE = 0;

      /// Camera has a valid field of view (FOV) value.
      ///
      /// This flag should be set for perspective cameras where the `fov_y` field is valid.
      const HAS_FOV = 1;

      /// Camera has a valid aspect ratio value.
      ///
      /// This flag indicates the `aspect_ratio` field contains valid data.
      /// Typically set for perspective cameras.
      const HAS_ASPECT_RATIO = 1 << 1;

      /// Camera uses perspective projection.
      ///
      /// Objects appear smaller with distance. This is the standard projection
      /// for realistic 3D rendering.
      const PERSPECTIVE = 1 << 2;

      /// Camera uses orthographic projection.
      ///
      /// Objects maintain their size regardless of distance. Useful for technical
      /// drawings, 2D games, and UI rendering.
      const ORTHOGRAPHIC = 1 << 3;
    }
}

/// Complete camera data for rendering.
///
/// `CameraData` contains all the information needed to render from a camera's viewpoint.
/// It includes transformation matrices, projection parameters, and configuration flags.
///
/// # Memory Layout
///
/// This structure uses `#[repr(C)]` to ensure consistent memory layout for GPU usage.
/// Padding fields (`__pad0`, `__pad1`) ensure proper alignment for use in uniform buffers,
/// particularly for std140 and std430 GLSL layouts.
///
/// # Fields
///
/// ## Transformation Matrices
/// - **view**: Transforms from world space to camera space (camera transform)
/// - **projection**: Transforms from camera space to clip space (perspective or ortho)
///
/// ## Position and Parameters
/// - **position**: Camera position in world space
/// - **near**: Near clipping plane distance
/// - **far**: Far clipping plane distance
/// - **fov_y**: Vertical field of view in degrees (perspective only)
/// - **aspect_ratio**: Width/height ratio of the viewport
///
/// ## Configuration
/// - **flags**: Configuration flags indicating camera type and valid parameters
///
/// # GPU Usage
///
/// This structure is designed to be directly uploaded to GPU uniform buffers.
/// The alignment and padding follow std140 rules for maximum compatibility.
///
/// # Example
///
/// ```rust,no_run
/// # use toyengine_graphics::camera::{CameraData, CameraFlags};
/// # use toyengine_core::primitives::{mat::Matrix4, vec::Vec3};
/// let mut camera = CameraData::default();
/// camera.position = Vec3::new(0.0, 5.0, 10.0);
/// camera.fov_y = 45.0;
/// camera.aspect_ratio = 16.0 / 9.0;
/// camera.flags = CameraFlags::PERSPECTIVE | CameraFlags::HAS_FOV | CameraFlags::HAS_ASPECT_RATIO;
/// camera.view = Matrix4::look_at(
///     camera.position,
///     Vec3::new(0.0, 0.0, 0.0),
///     Vec3::new(0.0, 1.0, 0.0)
/// );
/// camera.projection = Matrix4::perspective(
///     camera.fov_y.to_radians(),
///     camera.aspect_ratio,
///     camera.near,
///     camera.far
/// );
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CameraData {
    /// View matrix transforming from world space to camera space.
    pub view: Matrix4,

    /// Projection matrix transforming from camera space to clip space.
    pub projection: Matrix4,

    /// Camera position in world space.
    pub position: Vec3,
    /// Padding for alignment (16-byte boundary for GPU).
    __pad0: f32,

    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,

    /// Vertical field of view in degrees (valid when HAS_FOV flag is set).
    pub fov_y: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect_ratio: f32,

    /// Configuration flags indicating camera type and parameters.
    pub flags: CameraFlags,
    /// Padding for alignment (16-byte boundary for GPU).
    __pad1: [u32; 3],
}

impl CameraData {
    /// Identity matrix constant for default initialization.
    const IDENTITY_MATRIX4: Matrix4 = Matrix4::identity();
    /// Zero vector constant for default initialization.
    const EMPTY_VEC3: Vec3 = Vec3::zero();

    /// Creates default camera data with identity transforms.
    ///
    /// The default camera has:
    /// - Identity view and projection matrices
    /// - Position at world origin (0, 0, 0)
    /// - Near plane at 0.1 units
    /// - Far plane at 1000.0 units
    /// - Invalid FOV (-1.0) to indicate uninitialized state
    /// - 16:9 aspect ratio
    /// - No flags set
    ///
    /// # Returns
    ///
    /// A `CameraData` instance with default values. The camera is not ready for
    /// rendering until proper view, projection, and flags are set.
    ///
    /// # Usage
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::camera::CameraData;
    /// let camera = CameraData::default();
    /// // Configure camera before use
    /// ```
    pub const fn default() -> Self {
        Self {
            view: Self::IDENTITY_MATRIX4,
            projection: Self::IDENTITY_MATRIX4,
            position: Self::EMPTY_VEC3,
            __pad0: 0.0,
            near: 0.1,
            far: 1000.0,
            fov_y: -1.0,  // Negative indicates uninitialized
            aspect_ratio: 16.0 / 9.0,
            flags: CameraFlags::NONE,
            __pad1: [0; 3],
        }
    }

    /// Checks if the camera data is uninitialized.
    ///
    /// Camera data is considered "none" (uninitialized) if:
    /// - Position is at origin
    /// - View and projection matrices are identity
    /// - FOV is negative (invalid)
    ///
    /// # Returns
    ///
    /// `true` if the camera is uninitialized, `false` if it has been configured.
    ///
    /// # Usage
    ///
    /// This is used internally to validate that camera data has been properly
    /// set before rendering:
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::camera::CameraData;
    /// let camera = CameraData::default();
    /// assert!(camera.is_none());  // Default camera is uninitialized
    /// ```
    pub fn is_none(&self) -> bool {
        self.position == Self::EMPTY_VEC3
            && self.view == Self::IDENTITY_MATRIX4
            && self.projection == Self::IDENTITY_MATRIX4
            && self.fov_y < 0.0
    }
}
