//! Example demonstrating the null renderer backend.
//!
//! This example shows how to:
//! - Initialize a graphics context with the null renderer
//! - Configure a perspective camera
//! - Submit render packets to multiple frames
//! - Work with different render phases
//!
//! The null renderer doesn't perform actual GPU rendering but prints
//! detailed diagnostic information about the rendering process.
//!
//! # Running
//!
//! ```bash
//! cargo run --package graphics --example null_renderer_example
//! ```
//!
//! # Expected Output
//!
//! The example will print a hierarchical view of two frames being rendered,
//! showing camera setup and render packet submission.

use graphics::{
    camera::{CameraData, CameraFlags},
    context::GraphicsContext,
    render::{GpuMaterialHandle, GpuMeshHandle, RenderPacket, RenderPhase, SortKey},
};
use toyengine_core::primitives::{mat::Matrix4, vec::Vec3};

fn main() {
    // Initialize graphics context with null renderer and debug mode enabled
    let mut context = GraphicsContext::new_null_renderer(true);
    println!("Render context initialized: {:?}", context);

    // Configure a perspective camera
    let mut camera_data = CameraData::default();

    camera_data.aspect_ratio = 16.0 / 9.0;
    camera_data.fov_y = 45.0;
    camera_data.flags =
        CameraFlags::HAS_FOV | CameraFlags::HAS_ASPECT_RATIO | CameraFlags::PERSPECTIVE;
    camera_data.position = Vec3::new(0.0, 0.0, -5.0);
    camera_data.projection = Matrix4::perspective(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    camera_data.view = Matrix4::look_at(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    camera_data.near = 0.1;
    camera_data.far = 1000.0;

    // First frame: Submit an opaque render packet
    {
        let mut frame = context.acquire_frame();

        frame.set_camera(camera_data);
        frame.submit_packet(RenderPacket {
            world: Matrix4::identity(),
            debug_name: Some("Test Packet"),
            phase: RenderPhase::Opaque,
            material: GpuMaterialHandle(0),
            mesh: GpuMeshHandle(0),
            sort_key: SortKey(1),
        });
    }

    // Second frame: Submit a transparent render packet
    {
        let mut frame2 = context.acquire_frame();
        frame2.set_camera(camera_data);
        frame2.submit_packet(RenderPacket {
            world: Matrix4::identity(),
            debug_name: Some("Second Test Packet"),
            phase: RenderPhase::Transparent,
            material: GpuMaterialHandle(1),
            mesh: GpuMeshHandle(1),
            sort_key: SortKey(2),
        });
    }
}
