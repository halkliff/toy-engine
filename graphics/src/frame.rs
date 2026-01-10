//! Frame management module for double-buffered rendering.
//!
//! This module provides the internal frame management infrastructure that coordinates
//! CPU-side rendering command recording with GPU execution. It implements a double-buffering
//! strategy to maximize parallelism between the CPU and GPU.
//!
//! # Architecture
//!
//! The frame system consists of several key components:
//!
//! - [`FrameManager`]: Orchestrates the frame lifecycle and manages synchronization
//! - [`FrameContext`]: Represents a single frame slot with its associated state and packets
//! - [`FrameHandle`]: Public RAII handle for safe frame manipulation (see [`frame_handle`] module)
//! - [`FrameState`]: Tracks the current state of a frame in its lifecycle
//!
//! # Frame Lifecycle
//!
//! Each frame progresses through the following states:
//!
//! 1. **Idle**: Frame slot is available and ready to be used
//! 2. **Recording**: CPU is actively recording render commands to this frame
//! 3. **Submitted**: Frame has been submitted and is being processed by the GPU
//! 4. **Idle** (again): GPU processing complete, frame slot is available again
//!
//! # Double Buffering
//!
//! The system maintains exactly two frame slots, allowing:
//!
//! - Frame N to be processed by the GPU
//! - Frame N+1 to be recorded by the CPU simultaneously
//!
//! This prevents GPU stalls and maximizes throughput by keeping both the CPU and GPU busy.
//!
//! # Synchronization
//!
//! Frame synchronization uses fence values to track GPU progress. When a frame is submitted,
//! it receives a monotonically increasing fence value. Before reusing a frame slot, the
//! manager waits until the GPU has completed processing that fence value.
//!
//! # Render Phase Ordering
//!
//! Render packets within a frame are executed in a predefined phase order to ensure
//! correct rendering results:
//!
//! 1. Shadow mapping
//! 2. Depth pre-pass
//! 3. Opaque geometry
//! 4. Transparent geometry (sorted)
//! 5. Overlay elements
//! 6. Debug visualization
//!
//! Within each phase, packets are sorted by their sort key for optimal performance.
//!
//! # API Status
//!
//! **⚠️ This API is not final and is actively being developed.**

use crate::backend::RenderBackend;
use crate::camera::CameraData;
use crate::render::{RenderPacket, RenderPhase};

pub mod frame_handle;
pub use frame_handle::*;

pub mod frame_data;
pub use frame_data::*;

use hashbrown::HashMap;

/// Represents the current state of a frame in its lifecycle.
///
/// Each frame transitions through these states as it progresses from CPU recording
/// to GPU execution and back to being available for reuse.
///
/// # State Transitions
///
/// ```text
/// Idle -> Recording -> Submitted -> Idle
///   ^                                 |
///   |_________________________________|
/// ```
///
/// - **Idle → Recording**: When [`FrameManager::begin_frame()`] selects this frame
/// - **Recording → Submitted**: When [`FrameManager::end_frame()`] is called
/// - **Submitted → Idle**: When GPU completes processing (fence signaled)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FrameState {
    /// Frame is available and ready to be used for recording.
    Idle,
    /// Frame is actively recording render commands on the CPU.
    Recording,
    /// Frame has been submitted and is being processed by the GPU.
    Submitted,
}

/// Internal representation of a single frame slot.
///
/// Each `FrameContext` owns the render packets for its frame and tracks the state
/// and synchronization information needed to coordinate with the GPU.
///
/// # Fields
///
/// - **fence_value**: The GPU fence value assigned when this frame was submitted.
///   Used to determine when the GPU has finished processing this frame.
/// - **in_flight**: Whether this frame is currently being processed by the GPU.
///   Prevents reusing the frame slot until GPU work completes.
/// - **packets**: Collection of render packets to be executed for this frame.
///   Owned by the frame context to ensure they live as long as needed.
/// - **frame_state**: Current lifecycle state of this frame.
/// - **index**: The slot index (0 or 1) for this frame in the double-buffer array.
///
/// # Ownership
///
/// The frame context owns its render packets, ensuring they remain valid for the
/// entire duration of GPU processing. This prevents use-after-free issues and
/// ensures proper cleanup.
#[derive(Debug)]
pub(crate) struct FrameContext {
    pub(crate) fence_value: u64,
    pub(crate) in_flight: bool,
    pub(crate) packets: Vec<RenderPacket>,
    pub(crate) frame_state: FrameState,
    pub(crate) index: usize,
    pub(crate) frame_data: FrameData,
}

impl FrameContext {
    /// Creates a new frame context initialized to the idle state.
    ///
    /// # Arguments
    ///
    /// * `index` - The slot index (0 or 1) for this frame in the double-buffer array
    ///
    /// # Initial State
    ///
    /// - Fence value: 0 (no GPU work submitted yet)
    /// - In flight: false (not being processed)
    /// - Packets: Empty vector (no rendering work)
    /// - Frame state: Idle (ready to use)
    pub(crate) const fn new(index: usize) -> Self {
        FrameContext {
            fence_value: 0,
            in_flight: false,
            packets: Vec::new(),
            frame_state: FrameState::Idle,
            index,
            frame_data: FrameData::default(),
        }
    }
}

/// Manages the frame lifecycle and synchronization for double-buffered rendering.
///
/// The frame manager is the core of the frame system, coordinating between CPU-side
/// command recording and GPU-side execution. It maintains exactly two frame slots
/// and ensures proper synchronization to prevent data races and maximize throughput.
///
/// # Responsibilities
///
/// - **Frame Selection**: Chooses which frame slot to use for the next frame
/// - **Synchronization**: Waits for GPU completion before reusing frame slots
/// - **Packet Management**: Collects and organizes render packets within frames
/// - **Phase Ordering**: Ensures render packets execute in the correct order
/// - **Fence Management**: Tracks GPU progress using monotonically increasing fence values
///
/// # Double Buffering Strategy
///
/// With two frame slots, the typical execution pattern is:
///
/// ```text
/// Time ->
/// Frame 0: [Recording] [GPU Processing] [Idle] [Recording] ...
/// Frame 1: [Idle] [Recording] [GPU Processing] [Idle] ...
/// ```
///
/// This overlapping ensures neither the CPU nor GPU is idle waiting for the other.
///
/// # Synchronization
///
/// The manager uses fence values for GPU synchronization. Each submitted frame
/// receives a unique, incrementing fence value. Before reusing a frame slot,
/// the manager waits until the GPU signals completion of that fence value.
#[derive(Debug)]
pub(crate) struct FrameManager {
    /// The two frame slots for double buffering.
    frames: [FrameContext; 2],

    /// Index of the currently active frame slot.
    current: usize,

    /// The last fence value that the GPU has completed processing.
    gpu_completed_fence: u64,

    /// The next fence value to assign to a submitted frame.
    next_fence_value: u64,

    backend: Box<dyn RenderBackend>,

    phase_order: HashMap<RenderPhase, usize>,
}

impl FrameManager {
    /// Number of frame slots in the double-buffer system.
    const FRAME_LENGTH: usize = 2;

    /// The order in which render phases are executed.
    ///
    /// This ordering ensures correct rendering by executing phases in the proper sequence:
    /// shadow maps first, then depth pre-pass, then opaque geometry, etc.
    const PHASE_ORDER: [RenderPhase; 6] = [
        RenderPhase::Shadow,
        RenderPhase::DepthPrePass,
        RenderPhase::Opaque,
        RenderPhase::Transparent,
        RenderPhase::Overlay,
        RenderPhase::Debug,
    ];

    /// Creates a new frame manager with initialized frame slots.
    ///
    /// Both frame slots are created in the idle state, ready to be used.
    /// The manager starts with frame 0 as current and the next fence value at 1.
    pub(crate) fn new(backend: Box<dyn RenderBackend>) -> Self {
        let mut phase_order = HashMap::new();
        for (i, phase) in Self::PHASE_ORDER.iter().enumerate() {
            phase_order.insert(*phase, i);
        }
        FrameManager {
            frames: [FrameContext::new(0), FrameContext::new(1)],
            current: 0,
            gpu_completed_fence: 0,
            next_fence_value: 1,
            backend,
            phase_order,
        }
    }

    /// Adds a render packet to the specified frame.
    ///
    /// This method is called internally when render packets are submitted through
    /// a [`FrameHandle`]. The packet is appended to the frame's packet collection.
    ///
    /// # Arguments
    ///
    /// * `index` - The frame slot index to add the packet to
    /// * `packet` - The render packet to add
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the specified frame is not in the Recording state.
    pub(crate) fn push_packet(&mut self, index: usize, packet: RenderPacket) {
        assert!(
            self.frames[index].frame_state == FrameState::Recording,
            "Frame index {index} is not in recording state"
        );

        self.frames[index].packets.push(packet);
    }

    pub(crate) fn set_frame_camera_data(&mut self, index: usize, data: CameraData) {
        assert!(
            self.frames[index].frame_state == FrameState::Recording,
            "Frame index {index} is not in recording state"
        );

        let frame_data = &mut self.frames[index].frame_data;

        let frame_data_is_none = frame_data.camera.is_none();

        assert!(
            frame_data_is_none,
            "camera data already set for frame index {index}"
        );

        if frame_data_is_none {
            frame_data.camera = data;
        }
    }

    pub(crate) fn set_frame_data(&mut self, index: usize, data: FrameData) {
        let frame = &mut self.frames[index];
        assert!(
            frame.frame_state == FrameState::Recording,
            "Frame index {index} is not in recording state"
        );

        let curr_frame_camera_data_is_none = frame.frame_data.is_none();
        let frame_data_camera_data_is_none = data.is_none();

        assert!(
            curr_frame_camera_data_is_none || frame_data_camera_data_is_none,
            "frame data already set for frame index {index}, and cannot be set more than once",
        );

        if curr_frame_camera_data_is_none || frame_data_camera_data_is_none {
            frame.frame_data = data;
        }
    }

    pub(crate) fn set_frame_data_prop(&mut self, index: usize, data: FrameDataProp) {
        let frame = &mut self.frames[index];
        assert!(
            frame.frame_state == FrameState::Recording,
            "Frame index {index} is not in recording state"
        );

        match data {
            FrameDataProp::CameraData(cam_data) => self.set_frame_camera_data(index, cam_data),
            _ => todo!("Other FrameDataProp variants not implemented yet"),
        };
    }

    /// Begins a new frame, returning a mutable reference to the selected frame context.
    ///
    /// This method implements the frame selection and synchronization logic:
    ///
    /// 1. Advances to the next frame slot (wraps around after frame 1)
    /// 2. Waits if that slot is still being processed by the GPU
    /// 3. Transitions the frame from Idle to Recording state
    /// 4. Returns a mutable reference to the frame context
    ///
    /// # Returns
    ///
    /// A mutable reference to the [`FrameContext`] that is now ready for recording.
    ///
    /// # Blocking Behavior
    ///
    /// If the selected frame slot is still in flight (being processed by the GPU),
    /// this method blocks until the GPU completes processing. This prevents the CPU
    /// from getting more than one frame ahead of the GPU.
    ///
    /// # Implementation Note
    ///
    /// In the current implementation, GPU progress is simulated. In a real GPU backend,
    /// this would wait on an actual fence or event from the graphics API.
    #[inline]
    pub(crate) fn begin_frame(&mut self) -> &mut FrameContext {
        // 1. Select next frame by advancing the frame index
        let next_index = (self.current + 1) % Self::FRAME_LENGTH;

        // 2. Wait (or simulate waiting) until it's safe to reuse
        if self.frames[next_index].in_flight {
            // This error is safe to ignore, because this condition doesn't happen synchronously.
            // It should, however, be fixed in a real implementation.
            #[allow(clippy::while_immutable_condition)]
            while self.frames[next_index].fence_value > self.gpu_completed_fence {
                // In a real implementation, we would wait here.
            }
            self.frames[next_index].in_flight = false;
        }

        assert!(
            self.frames[next_index].frame_state == FrameState::Idle,
            "Frame index {next_index} is not idle",
        );

        // Update to the next frame
        self.current = next_index;
        self.frames[self.current].frame_state = FrameState::Recording;

        // 3. Return mutable access
        &mut self.frames[self.current]
    }

    /// Ends the current frame and submits it for GPU processing.
    ///
    /// This method transitions the frame from Recording to Submitted state and assigns
    /// a fence value for synchronization. The render packets remain owned by the frame
    /// until GPU processing completes.
    ///
    /// # Arguments
    ///
    /// * `frame_index` - The index of the frame to end (must be currently recording)
    ///
    /// # State Transitions
    ///
    /// - **in_flight**: Set to `true` to indicate GPU is processing this frame
    /// - **fence_value**: Assigned the next sequential fence value
    /// - **frame_state**: Transitioned from Recording to Submitted
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - No render packets were submitted to the frame
    /// - The frame is not in the Recording state
    ///
    /// # Implementation Note
    ///
    /// The frame index is not advanced here; it's advanced in [`begin_frame()`]
    /// to keep the current frame index valid until the next frame begins.
    ///
    /// [`begin_frame()`]: Self::begin_frame
    #[inline]
    pub(crate) fn end_frame(&mut self, frame_index: usize) {
        assert!(
            !self.frames[frame_index].packets.is_empty(),
            "No render packets submitted for this frame"
        );
        assert!(
            self.frames[frame_index].frame_state == FrameState::Recording,
            "Current frame in index {} is not in recording state",
            frame_index
        );

        assert!(
            !self.frames[frame_index].frame_data.camera.is_none(),
            "Camera data not set for frame index {frame_index}",
        );

        // 1. Signal fence for this frame
        self.frames[frame_index].in_flight = true;
        self.frames[frame_index].fence_value = self.next_fence_value;
        self.next_fence_value = self.next_fence_value.wrapping_add(1);
        self.frames[frame_index].frame_state = FrameState::Submitted;

        self.execute_frame(frame_index);
        // Note: frame index is advanced in begin_frame, not here
        // This keeps the current frame index valid until the next begin_frame call
    }

    /// Executes the rendering commands for the specified frame.
    ///
    /// This method processes all render packets in the frame according to the
    /// predefined phase order. It handles phase transitions and packet sorting
    /// within each phase.
    ///
    /// # Arguments
    ///
    /// * `frame_index` - The index of the frame to process
    ///
    /// # Processing Steps
    ///
    /// 1. Takes ownership of the frame's render packets
    /// 2. Groups packets by render phase
    /// 3. Sorts packets within each phase by their sort key
    /// 4. Executes each phase in the predefined order
    /// 5. Transitions the frame back to Idle state
    /// 6. Increments the GPU completed fence counter
    ///
    /// # Phase Execution Order
    ///
    /// Phases are executed in the order defined by [`PHASE_ORDER`]:
    /// Shadow → DepthPrePass → Opaque → Transparent → Overlay → Debug
    ///
    /// # Sort Key Ordering
    ///
    /// Within each phase, packets are sorted by their sort key. This allows for:
    /// - Front-to-back rendering for opaque objects (early-z optimization)
    /// - Back-to-front rendering for transparent objects (correct blending)
    /// - Material batching to minimize state changes
    ///
    /// [`PHASE_ORDER`]: Self::PHASE_ORDER
    fn execute_frame(&mut self, frame_index: usize) {
        let frame = &mut self.frames[frame_index];
        let mut packets = std::mem::take(&mut frame.packets);

        self.backend.begin_frame(&frame.frame_data);

        let mut curr_phase: Option<RenderPhase> = None;

        packets.sort_by(|a, b| {
            let a_phase_order = self.phase_order.get(&a.phase).unwrap_or(&usize::MAX);
            let b_phase_order = self.phase_order.get(&b.phase).unwrap_or(&usize::MAX);
            a_phase_order
                .cmp(b_phase_order)
                .then_with(|| a.sort_key.cmp(&b.sort_key))
        });

        for packet in packets.iter() {
            if curr_phase != Some(packet.phase) {
                if let Some(phase) = curr_phase {
                    self.backend.end_phase(phase);
                }
                self.backend.begin_phase(packet.phase);
                curr_phase = Some(packet.phase);
            }
            self.backend.draw(packet);
        }

        if let Some(phase) = curr_phase {
            self.backend.end_phase(phase);
        }

        self.backend.end_frame();

        frame.frame_state = FrameState::Idle;
        // Pretend the GPU finished some work
        self.gpu_completed_fence += 1;
    }
}
