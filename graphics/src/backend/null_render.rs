//! Null rendering backend for testing and debugging.
//!
//! This module provides a [`NullRenderBackend`] implementation that doesn't perform
//! actual GPU rendering. Instead, it prints diagnostic information to stdout, making
//! it useful for:
//!
//! - Testing the frame system without GPU dependencies
//! - Debugging render packet ordering and phase execution
//! - Verifying API usage patterns
//! - Running in headless environments (CI, servers)
//!
//! # Output Format
//!
//! The null backend prints a hierarchical view of the rendering process:
//!
//! ```text
//! === BEGIN FRAME 0 ===
//! └─ Camera Projection Matrix: ...
//! └─ Camera View Matrix: ...
//! └──── BEGIN PHASE Opaque ───
//!   └─ Packet Name
//! └──── END PHASE Opaque ───
//! === END FRAME 0 ===
//! ```
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use super::{BackendState, RenderBackend};
use crate::{
    camera::CameraFlags,
    frame::FrameData,
    render::{RenderPacket, RenderPhase},
};

/// A rendering backend that prints diagnostic information instead of rendering.
///
/// `NullRenderBackend` implements the [`RenderBackend`] trait but doesn't interact
/// with any actual graphics API. It validates the calling sequence and prints
/// detailed information about each frame, phase, and draw call.
///
/// # Use Cases
///
/// - **Testing**: Validate frame system logic without GPU setup
/// - **Debugging**: Visualize render packet submission order and content
/// - **CI/CD**: Run graphics code in environments without GPU support
/// - **Development**: Prototype rendering logic before implementing GPU backend
///
/// # State Tracking
///
/// The backend maintains state to validate correct API usage:
/// - Tracks frame count for identifying frames in output
/// - Validates phase and frame lifecycle (prevents invalid call sequences)
/// - Asserts correct state transitions
///
/// # Example Output
///
/// ```text
/// === BEGIN FRAME 0 ===
/// └─ Camera Position: Vec3(0.0, 0.0, -5.0)
/// └─ Camera Is Perspective: true
/// └──── BEGIN PHASE Opaque ───
///   └─ "Cube Mesh"
///   └─ "Floor Plane"
/// └──── END PHASE Opaque ───
/// === END FRAME 0 ===
/// ```
#[derive(Debug)]
pub struct NullRenderBackend {
    /// Counter tracking how many frames have been rendered.
    ///
    /// Used to identify frames in the diagnostic output and help correlate
    /// output with application behavior.
    frame_counter: u64,

    /// Current state of the backend for validation.
    ///
    /// Used to assert that methods are called in the correct order and
    /// prevent invalid operations like drawing outside a phase.
    state: BackendState,
}

impl RenderBackend for NullRenderBackend {
    fn begin_frame(&mut self, data: &FrameData) {
        assert!(
            self.state == BackendState::Idle,
            "Cannot begin frame: backend not idle"
        );
        self.state = BackendState::InFrame {
            current_phase: None,
        };

        // Print frame header
        println!("\n=== BEGIN FRAME {} ===", self.frame_counter);

        // Print camera matrices
        println!("└─ Camera Projection Matrix: {:?}", data.camera.projection);
        println!("└─ Camera View Matrix: {:?}", data.camera.view);

        // Print camera transform data
        println!("└─ Camera Position: {:?}", data.camera.position);
        println!("└─ Camera Flags: {:?}", data.camera.flags);

        // Print clipping planes
        println!("└─ Camera Near Plane: {:?}", data.camera.near);
        println!("└─ Camera Far Plane: {:?}", data.camera.far);

        // Print optional parameters if they're valid
        if data.camera.flags.contains(CameraFlags::HAS_FOV) {
            println!("└─ Camera FOV Y: {:?}", data.camera.fov_y);
        }
        if data.camera.flags.contains(CameraFlags::HAS_ASPECT_RATIO) {
            println!("└─ Camera Aspect Ratio: {:?}", data.camera.aspect_ratio);
        }

        // Print camera projection type
        println!(
            "└─ Camera Is Orthographic: {:?}",
            data.camera.flags.contains(CameraFlags::ORTHOGRAPHIC)
        );

        println!(
            "└─ Camera Is Perspective: {:?}",
            data.camera.flags.contains(CameraFlags::PERSPECTIVE)
        );
    }

    fn begin_phase(&mut self, phase: RenderPhase) {
        assert!(
            matches!(
                self.state,
                BackendState::InFrame {
                    current_phase: None
                }
            ),
            "Cannot begin phase: backend not in frame or phase already started"
        );
        self.state = BackendState::InFrame {
            current_phase: Some(phase),
        };
        println!("└──── BEGIN PHASE {:?} ───", phase);
    }

    fn draw(&mut self, packet: &RenderPacket) {
        assert!(
            matches!(
                self.state,
                BackendState::InFrame {
                    current_phase: Some(_)
                }
            ),
            "Cannot draw: backend not in frame or phase not started"
        );
        // Print packet information with debug name if available
        println!("  └─ {:?}", packet.debug_name.unwrap_or("Unnamed Packet"));
    }

    fn end_phase(&mut self, phase: RenderPhase) {
        println!("└──── END PHASE {:?} ───", phase);
        assert!(
            matches!(
                self.state,
                BackendState::InFrame {
                    current_phase: Some(current_phase)
                } if current_phase == phase
            ),
            "Cannot end phase: backend not in frame or phase not started"
        );
        self.state = BackendState::InFrame {
            current_phase: None,
        };
    }

    fn end_frame(&mut self) {
        assert!(
            matches!(
                self.state,
                BackendState::InFrame {
                    current_phase: None
                }
            ),
            "Cannot end frame: backend not in frame or phase still active"
        );
        println!("=== END FRAME {} ===\n", self.frame_counter);
        self.frame_counter += 1;
        self.state = BackendState::Idle;
    }
}

impl Default for NullRenderBackend {
    /// Creates a new null render backend in the idle state.
    ///
    /// # Returns
    ///
    /// A `NullRenderBackend` ready to begin its first frame.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use toyengine_graphics::backend::null_render::NullRenderBackend;
    /// let backend = NullRenderBackend::default();
    /// // Backend is ready to use
    /// ```
    fn default() -> Self {
        Self {
            frame_counter: 0,
            state: BackendState::Idle,
        }
    }
}
