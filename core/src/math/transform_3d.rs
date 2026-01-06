//! 3D transformation utilities.
//!
//! Provides structures and functions for 3D translation, rotation (using quaternions),
//! and scaling. These are the foundation of 3D game engines, physics systems, and
//! graphics applications.
//!
//! # Overview
//!
//! The [`Transform3D`] struct represents a 3D affine transformation consisting of:
//! - **Position**: Translation in 3D space (Vec3)
//! - **Rotation**: Rotation as a quaternion (Quaternion) - authoritative representation
//! - **Scale**: Non-uniform scaling factors (Vec3)
//!
//! Transformations are applied in TRS order: Scale → Rotate → Translate
//!
//! # Coordinate System
//!
//! This implementation uses a **right-handed coordinate system**:
//! - **+X**: Right
//! - **+Y**: Up
//! - **-Z**: Forward (toward camera)
//! - **+Z**: Back (away from camera)
//!
//! # Rotation Representation
//!
//! Rotations are stored as **quaternions** (authoritative) because:
//! - No gimbal lock
//! - Easy composition
//! - Smooth interpolation (SLERP)
//! - Matches modern engines (Unity, Unreal, Godot)
//!
//! Euler angles are supported for convenience but converted to quaternions internally.
//!
//! # Examples
//!
//! ```rust
//! # use toyengine::math::transform_3d::Transform3D;
//! # use toyengine::primitives::vec::Vec3;
//! # use toyengine::primitives::quat::Quaternion;
//! // Create a transform
//! let transform = Transform3D::new(
//!     Vec3::new(10.0, 5.0, -20.0),           // position
//!     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4), // 45° Y rotation
//!     Vec3::one()                             // uniform scale
//! );
//!
//! // Transform a point
//! let point = Vec3::new(1.0, 0.0, 0.0);
//! let transformed = transform.transform_point(point);
//!
//! // Transform a direction vector (no translation)
//! let direction = Vec3::new(0.0, 0.0, -1.0); // Forward
//! let rotated_direction = transform.transform_vector(direction);
//!
//! // Convert to matrix for GPU rendering
//! let matrix = transform.to_matrix4();
//! ```

use crate::primitives::{mat::Matrix4, quat::Quaternion, vec::{Vec3, Vec4}};
use std::ops::Mul;

/// A 3D transformation consisting of position, rotation (quaternion), and scale.
///
/// Represents an affine transformation in 3D space that can translate, rotate,
/// and scale points. The transformation is applied in TRS order (Scale, then Rotate,
/// then Translate).
///
/// # Coordinate System
///
/// - **Position**: Translation offset in world space
/// - **Rotation**: Stored as a quaternion (no gimbal lock, easy composition)
/// - **Scale**: Non-uniform scaling factors along X, Y, and Z axes
/// - **Handedness**: Right-handed (+Y up, +X right, -Z forward)
///
/// # Transformation Order
///
/// When applying to a point: `point' = point * S * R * T`
/// 1. Scale is applied first
/// 2. Rotation is applied second
/// 3. Translation is applied last
///
/// # Use Cases
///
/// - 3D game object transformations (characters, props, cameras)
/// - Scene graph hierarchies (parent-child relationships)
/// - Physics body transforms
/// - Animation systems
/// - Camera transformations
///
/// # Examples
///
/// ```rust
/// # use toyengine::math::transform_3d::Transform3D;
/// # use toyengine::primitives::vec::Vec3;
/// # use toyengine::primitives::quat::Quaternion;
/// // Create a game object transform
/// let mut player = Transform3D::from_position(Vec3::new(0.0, 0.0, -10.0));
///
/// // Move the player forward
/// player.translate(Vec3::new(0.0, 0.0, -1.0));
///
/// // Rotate 45 degrees around Y-axis
/// player.rotate(Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4));
///
/// // Get the forward direction
/// let forward = player.forward();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3D {
    /// Translation in world space
    pub position: Vec3,

    /// Rotation stored as a quaternion (authoritative)
    pub rotation: Quaternion,

    /// Non-uniform scale
    pub scale: Vec3,
}

// ============================================================================
// Transform3D Implementation
// ============================================================================

impl Transform3D {
    // Constructors

    /// Creates a new `Transform3D` with the specified position, rotation, and scale.
    ///
    /// # Arguments
    ///
    /// * `position` - Translation offset in 3D space
    /// * `rotation` - Rotation as a quaternion
    /// * `scale` - Non-uniform scale factors for X, Y, and Z axes
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::new(
    ///     Vec3::new(10.0, 5.0, -20.0),
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
    ///     Vec3::new(2.0, 2.0, 2.0)
    /// );
    /// ```
    #[inline]
    pub const fn new(position: Vec3, rotation: Quaternion, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Creates an identity transform.
    ///
    /// The identity transform has no translation, no rotation (identity quaternion),
    /// and uniform scale of 1.0. Applying this transform to a point leaves it unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let identity = Transform3D::identity();
    /// let point = Vec3::new(5.0, 10.0, -3.0);
    /// let result = identity.transform_point(point);
    /// assert_eq!(result, point);
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform from only a position (translation).
    ///
    /// Rotation is set to identity and scale to 1.
    ///
    /// # Arguments
    ///
    /// * `position` - Translation offset
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::from_position(Vec3::new(10.0, 5.0, -20.0));
    /// ```
    #[inline]
    pub const fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quaternion::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform from only a rotation (quaternion).
    ///
    /// Position is set to origin and scale to 1.
    ///
    /// # Arguments
    ///
    /// * `rotation` - Rotation as a quaternion
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let rotation = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::PI);
    /// let transform = Transform3D::from_rotation(rotation);
    /// ```
    #[inline]
    pub const fn from_rotation(rotation: Quaternion) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform from only a scale factor.
    ///
    /// Position is set to origin and rotation to identity.
    ///
    /// # Arguments
    ///
    /// * `scale` - Non-uniform scale factors
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::from_scale(Vec3::new(2.0, 3.0, 1.5));
    /// ```
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale,
        }
    }

    /// Creates a transform from a uniform scale factor.
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor applied to X, Y, and Z
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::from_scale_uniform(2.0);
    /// ```
    #[inline]
    pub const fn from_scale_uniform(scale: f32) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vec3::splat(scale),
        }
    }

    /// Creates a transform from position, Euler angles, and scale.
    ///
    /// The Euler angles are converted to a quaternion using the ZXY rotation order:
    /// 1. First rotate around Z-axis (roll)
    /// 2. Then rotate around X-axis (pitch)
    /// 3. Finally rotate around Y-axis (yaw)
    ///
    /// # Arguments
    ///
    /// * `position` - Translation offset
    /// * `euler` - Euler angles in radians (x=pitch, y=yaw, z=roll)
    /// * `scale` - Non-uniform scale factors
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::from_euler_zxy(
    ///     Vec3::new(0.0, 0.0, -10.0),
    ///     Vec3::new(0.0, std::f32::consts::FRAC_PI_4, 0.0), // 45° yaw
    ///     Vec3::one()
    /// );
    /// ```
    #[inline]
    pub fn from_euler_zxy(position: Vec3, euler: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation: Quaternion::from_euler_zxy(euler),
            scale,
        }
    }

    // Constants

    /// Identity transform (no translation, rotation, or scaling)
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quaternion::IDENTITY,
        scale: Vec3::ONE,
    };

    // Core Transformation Operations

    /// Transforms a point from local space to world space.
    ///
    /// Applies the full transformation (scale, rotation, translation) to a point.
    /// The transformation order is: Scale → Rotate → Translate (TRS).
    ///
    /// # Arguments
    ///
    /// * `point` - Point in local space
    ///
    /// # Returns
    ///
    /// The point transformed to world space
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let local = Vec3::new(5.0, 0.0, 0.0);
    /// let world = transform.transform_point(local);
    /// assert_eq!(world, Vec3::new(15.0, 0.0, 0.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        // Apply scale
        let scaled = Vec3::new(
            point.x * self.scale.x,
            point.y * self.scale.y,
            point.z * self.scale.z,
        );

        // Apply rotation
        let rotated = self.rotation.rotate_vec3(scaled);

        // Apply translation
        rotated + self.position
    }

    /// Transforms a vector (direction) from local space to world space.
    ///
    /// Unlike [`transform_point`](Self::transform_point), this does not apply translation.
    /// Use this for transforming directions, normals, velocities, or any vector quantity
    /// that represents a direction rather than a position.
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector in local space
    ///
    /// # Returns
    ///
    /// The vector transformed to world space (without translation)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::new(
    ///     Vec3::new(100.0, 50.0, 0.0),  // Translation ignored for vectors
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2),
    ///     Vec3::one()
    /// );
    /// let forward = Vec3::new(0.0, 0.0, -1.0);
    /// let transformed = transform.transform_vector(forward);
    /// // Result is rotated but not translated
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        // Apply scale
        let scaled = Vec3::new(
            vector.x * self.scale.x,
            vector.y * self.scale.y,
            vector.z * self.scale.z,
        );

        // Apply rotation (no translation)
        self.rotation.rotate_vec3(scaled)
    }

    /// Transforms a point from world space back to local space (inverse transformation).
    ///
    /// This is the inverse of [`transform_point`](Self::transform_point).
    /// Returns `None` if any scale component is too close to zero (within epsilon).
    ///
    /// The inverse transformation order is: Inverse Translate → Inverse Rotate → Inverse Scale
    ///
    /// # Arguments
    ///
    /// * `point` - Point in world space
    ///
    /// # Returns
    ///
    /// `Some(point)` if the transform is invertible, `None` if scale is near zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::new(
    ///     Vec3::new(10.0, 5.0, -3.0),
    ///     Quaternion::identity(),
    ///     Vec3::new(2.0, 2.0, 2.0)
    /// );
    ///
    /// let world_point = Vec3::new(12.0, 7.0, -1.0);
    /// let local_point = transform.inverse_transform_point(world_point).unwrap();
    /// // Verify round-trip
    /// let recovered = transform.transform_point(local_point);
    /// assert!((recovered - world_point).length() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, point: Vec3) -> Option<Vec3> {
        const EPSILON: f32 = 1e-10;

        // Check for degenerate scale
        if self.scale.x.abs() < EPSILON
            || self.scale.y.abs() < EPSILON
            || self.scale.z.abs() < EPSILON
        {
            return None;
        }

        // Remove translation
        let p = point - self.position;

        // Inverse rotation (conjugate for unit quaternions)
        let p = self.rotation.conjugate().rotate_vec3(p);

        // Inverse scale
        Some(Vec3::new(
            p.x / self.scale.x,
            p.y / self.scale.y,
            p.z / self.scale.z,
        ))
    }

    /// Transforms a vector from world space back to local space (inverse transformation).
    ///
    /// This is the inverse of [`transform_vector`](Self::transform_vector).
    /// Returns `None` if any scale component is too close to zero (within epsilon).
    ///
    /// Unlike [`inverse_transform_point`](Self::inverse_transform_point), this does not
    /// apply inverse translation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Vector in world space
    ///
    /// # Returns
    ///
    /// `Some(vector)` if the transform is invertible, `None` if scale is near zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::from_rotation(
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2)
    /// );
    ///
    /// let world_vec = Vec3::new(1.0, 0.0, 0.0);
    /// let local_vec = transform.inverse_transform_vector(world_vec).unwrap();
    /// // Verify round-trip
    /// let recovered = transform.transform_vector(local_vec);
    /// assert!((recovered - world_vec).length() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, vector: Vec3) -> Option<Vec3> {
        const EPSILON: f32 = 1e-10;

        // Check for degenerate scale
        if self.scale.x.abs() < EPSILON
            || self.scale.y.abs() < EPSILON
            || self.scale.z.abs() < EPSILON
        {
            return None;
        }

        // Inverse rotation (conjugate for unit quaternions)
        let v = self.rotation.conjugate().rotate_vec3(vector);

        // Inverse scale
        Some(Vec3::new(
            v.x / self.scale.x,
            v.y / self.scale.y,
            v.z / self.scale.z,
        ))
    }

    // Composition and Matrix Conversion

    /// Composes two transforms: applies `self` first, then `other`.
    ///
    /// This creates a new transform equivalent to applying `self` and then `other`.
    /// The composition order follows the mathematical convention: `result = self.then(other)`
    /// means "do self, then do other".
    ///
    /// For quaternion composition, note that the order is reversed compared to matrix
    /// multiplication: `q_result = other.rotation * self.rotation`
    ///
    /// # Arguments
    ///
    /// * `other` - The transform to apply after `self`
    ///
    /// # Returns
    ///
    /// A new transform representing the composition
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let translate = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let rotate = Transform3D::from_rotation(
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2)
    /// );
    ///
    /// // First translate, then rotate
    /// let combined = translate.then(&rotate);
    ///
    /// // Verify composition
    /// let point = Vec3::new(1.0, 0.0, 0.0);
    /// let result1 = combined.transform_point(point);
    /// let result2 = rotate.transform_point(translate.transform_point(point));
    /// assert!((result1 - result2).length() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        Self {
            // Transform self's position by other's transform
            position: other.transform_point(self.position),
            // Quaternion composition: other * self (reverse of matrix order)
            rotation: other.rotation * self.rotation,
            // Component-wise scale multiplication
            scale: Vec3::new(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
                self.scale.z * other.scale.z,
            ),
        }
    }

    /// Converts the transform to a 4x4 homogeneous transformation matrix.
    ///
    /// The resulting matrix is in row-major order and can be used with row vectors.
    /// The matrix includes scale, rotation, and translation and is suitable for
    /// GPU rendering pipelines.
    ///
    /// # Matrix Layout (Row-Major)
    /// ```text
    /// [ sx*r00  sy*r01  sz*r02  0 ]
    /// [ sx*r10  sy*r11  sz*r12  0 ]
    /// [ sx*r20  sy*r21  sz*r22  0 ]
    /// [ tx      ty      tz      1 ]
    /// ```
    ///
    /// where r_ij are rotation matrix elements, s_i are scale factors, and t_i are translation.
    ///
    /// # Transformation Order
    ///
    /// The matrix applies transformations in TRS order: Scale → Rotate → Translate
    ///
    /// For a row vector: `result = vector * matrix`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::new(
    ///     Vec3::new(10.0, 5.0, -20.0),
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4),
    ///     Vec3::new(2.0, 2.0, 2.0)
    /// );
    ///
    /// let matrix = transform.to_matrix4();
    ///
    /// // Use matrix in rendering pipeline
    /// // Pass to shader as uniform, etc.
    /// ```
    #[inline]
    #[must_use]
    pub fn to_matrix4(&self) -> Matrix4 {
        // Get rotation matrix (3x3 as Matrix4 without translation)
        let r = self.rotation.to_matrix4();

        // Build combined transformation matrix with scale and translation
        // With row vectors, each entire row is scaled by its corresponding scale factor.
        // This ensures v * M transforms correctly: scale is applied per-axis, not per-column.
        Matrix4::new(
            Vec4::new(
                r.row_x.x * self.scale.x,
                r.row_x.y * self.scale.x,
                r.row_x.z * self.scale.x,
                0.0,
            ),
            Vec4::new(
                r.row_y.x * self.scale.y,
                r.row_y.y * self.scale.y,
                r.row_y.z * self.scale.y,
                0.0,
            ),
            Vec4::new(
                r.row_z.x * self.scale.z,
                r.row_z.y * self.scale.z,
                r.row_z.z * self.scale.z,
                0.0,
            ),
            Vec4::new(self.position.x, self.position.y, self.position.z, 1.0),
        )
    }

    // Mutation Methods

    /// Adds a translation offset to the current position (mutating).
    ///
    /// # Arguments
    ///
    /// * `offset` - Translation offset to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut transform = Transform3D::identity();
    /// transform.translate(Vec3::new(5.0, 10.0, -3.0));
    /// assert_eq!(transform.position, Vec3::new(5.0, 10.0, -3.0));
    /// ```
    #[inline]
    pub fn translate(&mut self, offset: Vec3) {
        self.position += offset;
    }

    /// Adds a translation offset and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `offset` - Translation offset to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::identity()
    ///     .with_translate(Vec3::new(5.0, 10.0, -3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_translate(mut self, offset: Vec3) -> Self {
        self.translate(offset);
        self
    }

    /// Applies an additional rotation (mutating).
    ///
    /// The new rotation is composed with the existing rotation.
    /// The operation is: `self.rotation = rotation * self.rotation`
    ///
    /// # Arguments
    ///
    /// * `rotation` - Quaternion rotation to apply
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut transform = Transform3D::identity();
    /// transform.rotate(Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2));
    /// ```
    #[inline]
    pub fn rotate(&mut self, rotation: Quaternion) {
        self.rotation = rotation * self.rotation;
    }

    /// Applies an additional rotation and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `rotation` - Quaternion rotation to apply
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::identity()
    ///     .with_rotate(Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotate(mut self, rotation: Quaternion) -> Self {
        self.rotate(rotation);
        self
    }

    /// Applies an additional Euler angle rotation (mutating).
    ///
    /// The Euler angles are converted to a quaternion using ZXY order and composed
    /// with the existing rotation.
    ///
    /// # Arguments
    ///
    /// * `euler` - Euler angles in radians (x=pitch, y=yaw, z=roll)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut transform = Transform3D::identity();
    /// transform.rotate_euler(Vec3::new(0.0, std::f32::consts::FRAC_PI_4, 0.0));
    /// ```
    #[inline]
    pub fn rotate_euler(&mut self, euler: Vec3) {
        self.rotation = Quaternion::from_euler_zxy(euler) * self.rotation;
    }

    /// Applies an additional Euler angle rotation and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `euler` - Euler angles in radians (x=pitch, y=yaw, z=roll)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::identity()
    ///     .with_rotate_euler(Vec3::new(0.0, std::f32::consts::FRAC_PI_4, 0.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotate_euler(mut self, euler: Vec3) -> Self {
        self.rotate_euler(euler);
        self
    }

    /// Rotates around the X-axis (mutating).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let mut transform = Transform3D::identity();
    /// transform.rotate_x(std::f32::consts::FRAC_PI_2); // 90 degrees
    /// ```
    #[inline]
    pub fn rotate_x(&mut self, angle: f32) {
        self.rotation = Quaternion::from_axis_angle(Vec3::X, angle) * self.rotation;
    }

    /// Rotates around the X-axis and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::identity()
    ///     .with_rotate_x(std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotate_x(mut self, angle: f32) -> Self {
        self.rotate_x(angle);
        self
    }

    /// Rotates around the Y-axis (mutating).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let mut transform = Transform3D::identity();
    /// transform.rotate_y(std::f32::consts::FRAC_PI_2); // 90 degrees
    /// ```
    #[inline]
    pub fn rotate_y(&mut self, angle: f32) {
        self.rotation = Quaternion::from_axis_angle(Vec3::Y, angle) * self.rotation;
    }

    /// Rotates around the Y-axis and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::identity()
    ///     .with_rotate_y(std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotate_y(mut self, angle: f32) -> Self {
        self.rotate_y(angle);
        self
    }

    /// Rotates around the Z-axis (mutating).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let mut transform = Transform3D::identity();
    /// transform.rotate_z(std::f32::consts::FRAC_PI_2); // 90 degrees
    /// ```
    #[inline]
    pub fn rotate_z(&mut self, angle: f32) {
        self.rotation = Quaternion::from_axis_angle(Vec3::Z, angle) * self.rotation;
    }

    /// Rotates around the Z-axis and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::identity()
    ///     .with_rotate_z(std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotate_z(mut self, angle: f32) -> Self {
        self.rotate_z(angle);
        self
    }

    /// Multiplies the current scale by additional scale factors (mutating).
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut transform = Transform3D::from_scale(Vec3::new(2.0, 2.0, 2.0));
    /// transform.scale_by(Vec3::new(3.0, 1.0, 1.5));
    /// assert_eq!(transform.scale, Vec3::new(6.0, 2.0, 3.0));
    /// ```
    #[inline]
    pub fn scale_by(&mut self, scale: Vec3) {
        self.scale.x *= scale.x;
        self.scale.y *= scale.y;
        self.scale.z *= scale.z;
    }

    /// Multiplies the current scale and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::identity()
    ///     .with_scale_by(Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_scale_by(mut self, scale: Vec3) -> Self {
        self.scale_by(scale);
        self
    }

    /// Multiplies the current scale by a uniform factor (mutating).
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut transform = Transform3D::from_scale(Vec3::new(2.0, 3.0, 4.0));
    /// transform.scale_by_uniform(2.0);
    /// assert_eq!(transform.scale, Vec3::new(4.0, 6.0, 8.0));
    /// ```
    #[inline]
    pub fn scale_by_uniform(&mut self, scale: f32) {
        self.scale *= scale;
    }

    /// Multiplies the current scale by a uniform factor and returns the modified transform (chaining).
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::identity()
    ///     .with_scale_by_uniform(2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_scale_by_uniform(mut self, scale: f32) -> Self {
        self.scale_by_uniform(scale);
        self
    }

    // Helper Methods

    /// Returns the forward direction vector of this transform.
    ///
    /// In a right-handed coordinate system, forward is **-Z** (negative Z-axis).
    /// This method transforms the forward vector by the current rotation and scale.
    ///
    /// # Returns
    ///
    /// The forward direction vector in world space
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::from_rotation(
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2)
    /// );
    /// let forward = transform.forward();
    /// // After 90° Y rotation, forward (-Z) becomes +X
    /// assert!((forward.x - 1.0).abs() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        self.transform_vector(Vec3::new(0.0, 0.0, -1.0))
    }

    /// Returns the up direction vector of this transform.
    ///
    /// In a right-handed coordinate system, up is **+Y** (positive Y-axis).
    /// This method transforms the up vector by the current rotation and scale.
    ///
    /// # Returns
    ///
    /// The up direction vector in world space
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::from_rotation(
    ///     Quaternion::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2)
    /// );
    /// let up = transform.up();
    /// // After 90° Z rotation, up (+Y) becomes -X
    /// assert!((up.x + 1.0).abs() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn up(&self) -> Vec3 {
        self.transform_vector(Vec3::new(0.0, 1.0, 0.0))
    }

    /// Returns the right direction vector of this transform.
    ///
    /// In a right-handed coordinate system, right is **+X** (positive X-axis).
    /// This method transforms the right vector by the current rotation and scale.
    ///
    /// # Returns
    ///
    /// The right direction vector in world space
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let transform = Transform3D::from_rotation(
    ///     Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_2)
    /// );
    /// let right = transform.right();
    /// // After 90° Y rotation, right (+X) becomes -Z
    /// assert!((right.z + 1.0).abs() < 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.transform_vector(Vec3::new(1.0, 0.0, 0.0))
    }
}

// ============================================================================
// Operator Overloading - Transform3D
// ============================================================================

impl Mul for Transform3D {
    type Output = Self;

    /// Composes two transforms using the `*` operator.
    ///
    /// Equivalent to [`then`](Self::then): `a * b` means "apply `a` first, then `b`".
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let b = Transform3D::from_position(Vec3::new(0.0, 5.0, 0.0));
    /// let combined = a * b; // Apply a, then b
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.then(&rhs)
    }
}

impl Mul<&Transform3D> for Transform3D {
    type Output = Transform3D;

    /// Composes two transforms using the `*` operator (reference variant).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let b = Transform3D::from_position(Vec3::new(0.0, 5.0, 0.0));
    /// let combined = a * &b;
    /// ```
    #[inline]
    fn mul(self, rhs: &Transform3D) -> Self::Output {
        self.then(rhs)
    }
}

impl Mul<Transform3D> for &Transform3D {
    type Output = Transform3D;

    /// Composes two transforms using the `*` operator (reference variant).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let b = Transform3D::from_position(Vec3::new(0.0, 5.0, 0.0));
    /// let combined = &a * b;
    /// ```
    #[inline]
    fn mul(self, rhs: Transform3D) -> Self::Output {
        self.then(&rhs)
    }
}

impl Mul<&Transform3D> for &Transform3D {
    type Output = Transform3D;

    /// Composes two transforms using the `*` operator (reference variant).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
    /// let b = Transform3D::from_position(Vec3::new(0.0, 5.0, 0.0));
    /// let combined = &a * &b;
    /// ```
    #[inline]
    fn mul(self, rhs: &Transform3D) -> Self::Output {
        self.then(rhs)
    }
}

// ============================================================================
// Default Trait
// ============================================================================

impl Default for Transform3D {
    /// Returns the identity transform as the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::default();
    /// assert_eq!(transform, Transform3D::identity());
    /// ```
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<(Vec3, Quaternion, Vec3)> for Transform3D {
    /// Converts a tuple of (position, rotation, scale) to a Transform3D.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let tuple = (
    ///     Vec3::new(10.0, 5.0, -3.0),
    ///     Quaternion::identity(),
    ///     Vec3::one()
    /// );
    /// let transform = Transform3D::from(tuple);
    /// ```
    #[inline]
    fn from(tuple: (Vec3, Quaternion, Vec3)) -> Self {
        Self {
            position: tuple.0,
            rotation: tuple.1,
            scale: tuple.2,
        }
    }
}

impl From<&(Vec3, Quaternion, Vec3)> for Transform3D {
    /// Converts a reference tuple to a Transform3D.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::quat::Quaternion;
    /// let tuple = (Vec3::zero(), Quaternion::identity(), Vec3::one());
    /// let transform = Transform3D::from(&tuple);
    /// ```
    #[inline]
    fn from(tuple: &(Vec3, Quaternion, Vec3)) -> Self {
        Self {
            position: tuple.0,
            rotation: tuple.1,
            scale: tuple.2,
        }
    }
}

impl From<Transform3D> for (Vec3, Quaternion, Vec3) {
    /// Converts a Transform3D to a tuple of (position, rotation, scale).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let transform = Transform3D::identity();
    /// let tuple: (Vec3, _, Vec3) = transform.into();
    /// ```
    #[inline]
    fn from(transform: Transform3D) -> Self {
        (transform.position, transform.rotation, transform.scale)
    }
}

impl From<&Transform3D> for (Vec3, Quaternion, Vec3) {
    /// Converts a Transform3D reference to a tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// let transform = Transform3D::identity();
    /// let tuple = <(_, _, _)>::from(&transform);
    /// ```
    #[inline]
    fn from(transform: &Transform3D) -> Self {
        (transform.position, transform.rotation, transform.scale)
    }
}

impl From<Vec3> for Transform3D {
    /// Converts a Vec3 position to a Transform3D with only translation.
    ///
    /// Rotation is set to identity and scale to 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_3d::Transform3D;
    /// # use toyengine::primitives::vec::Vec3;
    /// let position = Vec3::new(10.0, 5.0, -3.0);
    /// let transform = Transform3D::from(position);
    /// assert_eq!(transform.position, position);
    /// ```
    #[inline]
    fn from(position: Vec3) -> Self {
        Self::from_position(position)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    const EPSILON: f32 = 1e-5;

    fn vec3_approx_eq(a: Vec3, b: Vec3, epsilon: f32) -> bool {
        (a - b).length() < epsilon
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_identity() {
        let identity = Transform3D::identity();
        assert_eq!(identity.position, Vec3::ZERO);
        assert_eq!(identity.rotation, Quaternion::IDENTITY);
        assert_eq!(identity.scale, Vec3::ONE);
    }

    #[test]
    fn test_new() {
        let pos = Vec3::new(10.0, 5.0, -3.0);
        let rot = Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4);
        let scale = Vec3::new(2.0, 3.0, 4.0);

        let transform = Transform3D::new(pos, rot, scale);

        assert_eq!(transform.position, pos);
        assert_eq!(transform.rotation, rot);
        assert_eq!(transform.scale, scale);
    }

    #[test]
    fn test_from_position() {
        let pos = Vec3::new(10.0, 5.0, -3.0);
        let transform = Transform3D::from_position(pos);

        assert_eq!(transform.position, pos);
        assert_eq!(transform.rotation, Quaternion::IDENTITY);
        assert_eq!(transform.scale, Vec3::ONE);
    }

    #[test]
    fn test_from_rotation() {
        let rot = Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2);
        let transform = Transform3D::from_rotation(rot);

        assert_eq!(transform.position, Vec3::ZERO);
        assert_eq!(transform.rotation, rot);
        assert_eq!(transform.scale, Vec3::ONE);
    }

    #[test]
    fn test_from_scale() {
        let scale = Vec3::new(2.0, 3.0, 4.0);
        let transform = Transform3D::from_scale(scale);

        assert_eq!(transform.position, Vec3::ZERO);
        assert_eq!(transform.rotation, Quaternion::IDENTITY);
        assert_eq!(transform.scale, scale);
    }

    #[test]
    fn test_from_scale_uniform() {
        let transform = Transform3D::from_scale_uniform(2.0);

        assert_eq!(transform.position, Vec3::ZERO);
        assert_eq!(transform.rotation, Quaternion::IDENTITY);
        assert_eq!(transform.scale, Vec3::splat(2.0));
    }

    #[test]
    fn test_from_euler_zxy() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let euler = Vec3::new(0.1, 0.2, 0.3);
        let scale = Vec3::new(2.0, 2.0, 2.0);

        let transform = Transform3D::from_euler_zxy(pos, euler, scale);

        assert_eq!(transform.position, pos);
        assert_eq!(transform.scale, scale);
        // Rotation should match Quaternion::from_euler_zxy
        assert_eq!(transform.rotation, Quaternion::from_euler_zxy(euler));
    }

    // ========================================================================
    // Transform Point/Vector Tests
    // ========================================================================

    #[test]
    fn test_transform_point_identity() {
        let transform = Transform3D::identity();
        let point = Vec3::new(1.0, 2.0, 3.0);
        let result = transform.transform_point(point);

        assert!(vec3_approx_eq(result, point, EPSILON));
    }

    #[test]
    fn test_transform_point_translation() {
        let transform = Transform3D::from_position(Vec3::new(10.0, 5.0, -3.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let result = transform.transform_point(point);

        assert!(vec3_approx_eq(result, Vec3::new(11.0, 7.0, 0.0), EPSILON));
    }

    #[test]
    fn test_transform_point_scale() {
        let transform = Transform3D::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let result = transform.transform_point(point);

        assert!(vec3_approx_eq(result, Vec3::new(2.0, 6.0, 12.0), EPSILON));
    }

    #[test]
    fn test_transform_point_rotation_y_90() {
        let transform = Transform3D::from_rotation(
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2)
        );
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = transform.transform_point(point);

        // After 90° Y rotation, X becomes -Z
        assert!(vec3_approx_eq(result, Vec3::new(0.0, 0.0, -1.0), EPSILON));
    }

    #[test]
    fn test_transform_vector_no_translation() {
        let transform = Transform3D::new(
            Vec3::new(100.0, 50.0, -20.0), // Should be ignored
            Quaternion::identity(),
            Vec3::one()
        );
        let vector = Vec3::new(1.0, 2.0, 3.0);
        let result = transform.transform_vector(vector);

        // Translation should not affect vectors
        assert!(vec3_approx_eq(result, vector, EPSILON));
    }

    #[test]
    fn test_transform_vector_rotation() {
        let transform = Transform3D::from_rotation(
            Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_2)
        );
        let vector = Vec3::new(1.0, 0.0, 0.0);
        let result = transform.transform_vector(vector);

        // After 90° Z rotation, X becomes Y
        assert!(vec3_approx_eq(result, Vec3::new(0.0, 1.0, 0.0), EPSILON));
    }

    // ========================================================================
    // Inverse Transform Tests
    // ========================================================================

    #[test]
    fn test_inverse_transform_point_round_trip() {
        let transform = Transform3D::new(
            Vec3::new(10.0, 5.0, -3.0),
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4),
            Vec3::new(2.0, 3.0, 4.0)
        );

        let point = Vec3::new(1.0, 2.0, 3.0);
        let world_point = transform.transform_point(point);
        let recovered = transform.inverse_transform_point(world_point).unwrap();

        assert!(vec3_approx_eq(recovered, point, EPSILON));
    }

    #[test]
    fn test_inverse_transform_vector_round_trip() {
        let transform = Transform3D::new(
            Vec3::new(10.0, 5.0, -3.0),
            Quaternion::from_axis_angle(Vec3::X, FRAC_PI_4),
            Vec3::new(2.0, 3.0, 4.0)
        );

        let vector = Vec3::new(1.0, 2.0, 3.0);
        let world_vec = transform.transform_vector(vector);
        let recovered = transform.inverse_transform_vector(world_vec).unwrap();

        assert!(vec3_approx_eq(recovered, vector, EPSILON));
    }

    #[test]
    fn test_inverse_transform_zero_scale() {
        let transform = Transform3D::from_scale(Vec3::new(0.0, 1.0, 1.0));
        let point = Vec3::new(1.0, 2.0, 3.0);

        assert!(transform.inverse_transform_point(point).is_none());
        assert!(transform.inverse_transform_vector(point).is_none());
    }

    // ========================================================================
    // Composition Tests
    // ========================================================================

    #[test]
    fn test_then_composition() {
        let translate = Transform3D::from_position(Vec3::new(10.0, 0.0, 0.0));
        let rotate = Transform3D::from_rotation(
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2)
        );

        let combined = translate.then(&rotate);

        let point = Vec3::new(1.0, 0.0, 0.0);
        let result1 = combined.transform_point(point);
        let result2 = rotate.transform_point(translate.transform_point(point));

        assert!(vec3_approx_eq(result1, result2, EPSILON));
    }

    #[test]
    fn test_mul_operator_composition() {
        let a = Transform3D::from_position(Vec3::new(5.0, 0.0, 0.0));
        let b = Transform3D::from_scale(Vec3::new(2.0, 2.0, 2.0));

        let combined = a * b;
        let point = Vec3::new(1.0, 1.0, 1.0);

        let result1 = combined.transform_point(point);
        let result2 = b.transform_point(a.transform_point(point));

        assert!(vec3_approx_eq(result1, result2, EPSILON));
    }

    // ========================================================================
    // Matrix Conversion Tests
    // ========================================================================

    #[test]
    fn test_to_matrix4_identity() {
        let transform = Transform3D::identity();
        let matrix = transform.to_matrix4();
        let identity_matrix = Matrix4::identity();

        // Check that identity transform produces identity matrix
        assert_eq!(matrix, identity_matrix);
    }

    #[test]
    fn test_to_matrix4_translation() {
        let transform = Transform3D::from_position(Vec3::new(10.0, 5.0, -3.0));
        let matrix = transform.to_matrix4();

        // Translation should be in the last row (row_w)
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_w.x, matrix.row_w.y, matrix.row_w.z),
            Vec3::new(10.0, 5.0, -3.0),
            EPSILON
        ));
    }

    #[test]
    fn test_to_matrix4_non_uniform_scale() {
        // Critical test: verify row-vector scaling is correct
        let transform = Transform3D::new(
            Vec3::ZERO,
            Quaternion::identity(),
            Vec3::new(2.0, 3.0, 4.0)  // Non-uniform scale
        );
        let matrix = transform.to_matrix4();
        
        // With row vectors and identity rotation, the matrix should be:
        // [ 2  0  0  0 ]
        // [ 0  3  0  0 ]
        // [ 0  0  4  0 ]
        // [ 0  0  0  1 ]
        
        // Each row should be scaled by its corresponding factor
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_x.x, matrix.row_x.y, matrix.row_x.z),
            Vec3::new(2.0, 0.0, 0.0),
            EPSILON
        ));
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_y.x, matrix.row_y.y, matrix.row_y.z),
            Vec3::new(0.0, 3.0, 0.0),
            EPSILON
        ));
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_z.x, matrix.row_z.y, matrix.row_z.z),
            Vec3::new(0.0, 0.0, 4.0),
            EPSILON
        ));
    }

    #[test]
    fn test_to_matrix4_scale_rotation_consistency() {
        // Verify that to_matrix4 with scale matches transform_point behavior
        let transform = Transform3D::new(
            Vec3::ZERO,
            Quaternion::from_axis_angle(Vec3::Z, FRAC_PI_2),
            Vec3::new(2.0, 3.0, 1.0)
        );
        
        let point = Vec3::new(1.0, 1.0, 1.0);
        
        // Transform via direct method (verify it doesn't error)
        let _direct = transform.transform_point(point);
        
        // Transform via matrix (point is implicitly [x, y, z, 1] when multiplied by matrix)
        // For row vectors: result = point * matrix
        // We can verify by checking the scaled basis vectors
        let matrix = transform.to_matrix4();
        
        // The rotation part (row_x, row_y, row_z) should represent:
        // - row_x: X-axis rotated and scaled
        // - row_y: Y-axis rotated and scaled  
        // - row_z: Z-axis rotated and scaled
        
        // For a 90° Z rotation with scale (2, 3, 1):
        // - X-axis (1,0,0) → (0,1,0) → scaled by 2 → (0,2,0)
        // - Y-axis (0,1,0) → (-1,0,0) → scaled by 3 → (-3,0,0)
        // - Z-axis (0,0,1) → (0,0,1) → scaled by 1 → (0,0,1)
        
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_x.x, matrix.row_x.y, matrix.row_x.z),
            Vec3::new(0.0, 2.0, 0.0),
            EPSILON
        ));
        assert!(vec3_approx_eq(
            Vec3::new(matrix.row_y.x, matrix.row_y.y, matrix.row_y.z),
            Vec3::new(-3.0, 0.0, 0.0),
            EPSILON
        ));
    }

    #[test]
    fn test_with_translate_chaining() {
        let transform = Transform3D::identity()
            .with_translate(Vec3::new(5.0, 10.0, -3.0));

        assert_eq!(transform.position, Vec3::new(5.0, 10.0, -3.0));
    }

    #[test]
    fn test_rotate_mutation() {
        let mut transform = Transform3D::identity();
        let rot = Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4);
        transform.rotate(rot);

        assert_eq!(transform.rotation, rot);
    }

    #[test]
    fn test_scale_by_mutation() {
        let mut transform = Transform3D::from_scale(Vec3::new(2.0, 3.0, 4.0));
        transform.scale_by(Vec3::new(2.0, 1.0, 0.5));

        assert_eq!(transform.scale, Vec3::new(4.0, 3.0, 2.0));
    }

    #[test]
    fn test_scale_by_uniform() {
        let mut transform = Transform3D::from_scale(Vec3::new(2.0, 3.0, 4.0));
        transform.scale_by_uniform(2.0);

        assert_eq!(transform.scale, Vec3::new(4.0, 6.0, 8.0));
    }

    // ========================================================================
    // Helper Method Tests
    // ========================================================================

    #[test]
    fn test_forward_no_rotation() {
        let transform = Transform3D::identity();
        let forward = transform.forward();

        // Forward should be -Z
        assert!(vec3_approx_eq(forward, Vec3::new(0.0, 0.0, -1.0), EPSILON));
    }

    #[test]
    fn test_forward_y_rotation_90() {
        let transform = Transform3D::from_rotation(
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2)
        );
        let forward = transform.forward();

        // After 90° Y rotation (counterclockwise looking down from +Y),
        // forward (-Z) rotates to -X
        assert!(vec3_approx_eq(forward, Vec3::new(-1.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_up_no_rotation() {
        let transform = Transform3D::identity();
        let up = transform.up();

        // Up should be +Y
        assert!(vec3_approx_eq(up, Vec3::new(0.0, 1.0, 0.0), EPSILON));
    }

    #[test]
    fn test_right_no_rotation() {
        let transform = Transform3D::identity();
        let right = transform.right();

        // Right should be +X
        assert!(vec3_approx_eq(right, Vec3::new(1.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_right_y_rotation_90() {
        let transform = Transform3D::from_rotation(
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2)
        );
        let right = transform.right();

        // After 90° Y rotation, right (+X) becomes -Z
        assert!(vec3_approx_eq(right, Vec3::new(0.0, 0.0, -1.0), EPSILON));
    }

    // ========================================================================
    // Conversion Trait Tests
    // ========================================================================

    #[test]
    fn test_default() {
        let transform = Transform3D::default();
        assert_eq!(transform, Transform3D::identity());
    }

    #[test]
    fn test_from_tuple() {
        let tuple = (
            Vec3::new(10.0, 5.0, -3.0),
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4),
            Vec3::new(2.0, 3.0, 4.0)
        );

        let transform = Transform3D::from(tuple);

        assert_eq!(transform.position, tuple.0);
        assert_eq!(transform.rotation, tuple.1);
        assert_eq!(transform.scale, tuple.2);
    }

    #[test]
    fn test_into_tuple() {
        let transform = Transform3D::new(
            Vec3::new(10.0, 5.0, -3.0),
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4),
            Vec3::new(2.0, 3.0, 4.0)
        );

        let tuple: (Vec3, Quaternion, Vec3) = transform.into();

        assert_eq!(tuple.0, transform.position);
        assert_eq!(tuple.1, transform.rotation);
        assert_eq!(tuple.2, transform.scale);
    }

    #[test]
    fn test_from_vec3() {
        let pos = Vec3::new(10.0, 5.0, -3.0);
        let transform = Transform3D::from(pos);

        assert_eq!(transform.position, pos);
        assert_eq!(transform.rotation, Quaternion::IDENTITY);
        assert_eq!(transform.scale, Vec3::ONE);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_chaining_operations() {
        let transform = Transform3D::identity()
            .with_translate(Vec3::new(10.0, 0.0, 0.0))
            .with_rotate(Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_4))
            .with_scale_by(Vec3::new(2.0, 2.0, 2.0));

        assert_eq!(transform.position, Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(transform.scale, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_trs_order() {
        // Verify that transform order is Scale -> Rotate -> Translate
        let transform = Transform3D::new(
            Vec3::new(10.0, 0.0, 0.0),
            Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2),
            Vec3::new(2.0, 1.0, 1.0)
        );

        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = transform.transform_point(point);

        // Manual TRS application
        let scaled = Vec3::new(2.0, 0.0, 0.0);
        let rotated = Quaternion::from_axis_angle(Vec3::Y, FRAC_PI_2).rotate_vec3(scaled);
        let translated = rotated + Vec3::new(10.0, 0.0, 0.0);

        assert!(vec3_approx_eq(result, translated, EPSILON));
    }
}
