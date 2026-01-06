//! 2D transformation utilities.
//!
//! Provides structures and functions for 2D translation, rotation, and scaling.
//! These are commonly used in 2D games, UI systems, and graphics applications.
//!
//! # Overview
//!
//! The [`Transform2D`] struct represents a 2D affine transformation consisting of:
//! - **Position**: Translation in 2D space (Vec2)
//! - **Rotation**: Rotation angle in radians (f32)
//! - **Scale**: Non-uniform scaling factors (Vec2)
//!
//! Transformations are applied in TRS order: Scale → Rotate → Translate
//!
//! # Examples
//!
//! ```rust
//! # use toyengine::math::transform_2d::Transform2D;
//! # use toyengine::primitives::vec::Vec2;
//! // Create a transform
//! let transform = Transform2D::new(
//!     Vec2::new(100.0, 50.0),              // position
//!     std::f32::consts::FRAC_PI_4,         // 45 degree rotation
//!     Vec2::new(2.0, 2.0)                  // 2x scale
//! );
//!
//! // Apply to a point
//! let point = Vec2::new(10.0, 0.0);
//! let transformed = transform.transform_point(point);
//!
//! // Convert to a rotation/scale matrix for batch linear operations (no translation)
//! let matrix = transform.to_rotation_scale_matrix();
//! ```

use crate::primitives::{mat::Matrix3, vec::Vec2};

/// A 2D transformation consisting of position, rotation, and scale.
///
/// Represents an affine transformation in 2D space that can translate, rotate,
/// and scale points. The transformation is applied in TRS order (Scale, then Rotate,
/// then Translate).
///
/// # Coordinate System
///
/// - **Position**: Translation offset in world space
/// - **Rotation**: Counter-clockwise rotation in radians (right-hand rule around Z-axis)
/// - **Scale**: Non-uniform scaling factors along X and Y axes
///
/// # Transformation Order
///
/// When applying to a point: `point' = T * R * S * point`
/// 1. Scale is applied first
/// 2. Rotation is applied second
/// 3. Translation is applied last
///
/// # Use Cases
///
/// - 2D game object transformations (sprites, entities)
/// - UI element positioning and layout
/// - Camera transformations in 2D
/// - Particle system transformations
/// - 2D physics body transforms
///
/// # Examples
///
/// ```rust
/// # use toyengine::math::transform_2d::Transform2D;
/// # use toyengine::primitives::vec::Vec2;
/// // Create a game object transform
/// let mut player = Transform2D::new(
///     Vec2::new(100.0, 200.0),           // screen position
///     0.0,                                // no rotation
///     Vec2::one()                         // normal size
/// );
///
/// // Move the player
/// player.translate(Vec2::new(10.0, 0.0));
///
/// // Rotate 45 degrees
/// player.rotate(std::f32::consts::FRAC_PI_4);
///
/// // Apply transform to local space vertices
/// let local_vertex = Vec2::new(5.0, 0.0);
/// let world_vertex = player.transform_point(local_vertex);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    /// Position (translation) in 2D space
    pub position: Vec2,
    /// Rotation angle in radians (counter-clockwise)
    pub rotation: f32,
    /// Scale factors along X and Y axes
    pub scale: Vec2,
}

// ============================================================================
// Transform2D Implementation
// ============================================================================

impl Transform2D {
    // Constructors

    /// Creates a new `Transform2D` with the specified position, rotation, and scale.
    ///
    /// # Arguments
    ///
    /// * `position` - Translation offset in 2D space
    /// * `rotation` - Rotation angle in radians (counter-clockwise)
    /// * `scale` - Non-uniform scale factors for X and Y axes
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_2,      // 90 degrees
    ///     Vec2::new(2.0, 3.0)
    /// );
    /// ```
    #[inline]
    pub const fn new(position: Vec2, rotation: f32, scale: Vec2) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Creates an identity transform.
    ///
    /// The identity transform has no translation, no rotation, and uniform scale of 1.0.
    /// Applying this transform to a point leaves it unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let identity = Transform2D::identity();
    /// let point = Vec2::new(5.0, 10.0);
    /// let result = identity.transform_point(point);
    /// assert_eq!(result, point);
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Self {
            position: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }

    /// Creates a transform from only a position (translation).
    ///
    /// Rotation is set to 0 and scale to 1.
    ///
    /// # Arguments
    ///
    /// * `position` - Translation offset
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::from_position(Vec2::new(100.0, 50.0));
    /// ```
    #[inline]
    pub const fn from_position(position: Vec2) -> Self {
        Self {
            position,
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }

    /// Creates a transform from only a rotation angle.
    ///
    /// Position is set to origin and scale to 1.
    ///
    /// # Arguments
    ///
    /// * `rotation` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let transform = Transform2D::from_rotation(std::f32::consts::PI);
    /// ```
    #[inline]
    pub const fn from_rotation(rotation: f32) -> Self {
        Self {
            position: Vec2::ZERO,
            rotation,
            scale: Vec2::ONE,
        }
    }

    /// Creates a transform from only a scale factor.
    ///
    /// Position is set to origin and rotation to 0.
    ///
    /// # Arguments
    ///
    /// * `scale` - Non-uniform scale factors
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::from_scale(Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    pub const fn from_scale(scale: Vec2) -> Self {
        Self {
            position: Vec2::ZERO,
            rotation: 0.0,
            scale,
        }
    }

    /// Creates a transform from a uniform scale factor.
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor applied to both X and Y
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let transform = Transform2D::from_scale_uniform(2.0);
    /// ```
    #[inline]
    pub const fn from_scale_uniform(scale: f32) -> Self {
        Self {
            position: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::new(scale, scale),
        }
    }

    // Constants

    /// Identity transform (no translation, rotation, or scaling)
    pub const IDENTITY: Self = Self {
        position: Vec2::ZERO,
        rotation: 0.0,
        scale: Vec2::ONE,
    };

    // Core Operations

    /// Returns the 2x2 rotation and scale matrix (linear part only, no translation).
    ///
    /// This extracts just the rotation and scale components as a Matrix3.
    /// Note: The returned matrix does NOT include translation, as Matrix3 is designed
    /// for 3D linear transformations without translation support.
    ///
    /// For full transformation including translation, use [`transform_point`](Self::transform_point)
    /// or [`transform_vector`](Self::transform_vector).
    ///
    /// # Matrix Layout
    ///
    /// ```text
    /// [ cos(θ)*sx  -sin(θ)*sy   0 ]
    /// [ sin(θ)*sx   cos(θ)*sy   0 ]
    /// [     0            0       1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(100.0, 200.0),  // Translation ignored in matrix
    ///     std::f32::consts::FRAC_PI_4,
    ///     Vec2::new(2.0, 2.0)
    /// );
    /// let matrix = transform.to_rotation_scale_matrix();
    /// // This matrix only contains rotation and scale, not translation
    /// ```
    #[inline]
    #[must_use]
    pub fn to_rotation_scale_matrix(&self) -> Matrix3 {
      let (rotation_sin, rotation_cos) = self.rotation.sin_cos();

        // Combined rotation and scale (no translation)
        let m00 = rotation_cos * self.scale.x;
        let m01 = -rotation_sin * self.scale.y;
        let m10 = rotation_sin * self.scale.x;
        let m11 = rotation_cos * self.scale.y;

        Matrix3::new(
            Vec2::new(m00, m01).extend(0.0),
            Vec2::new(m10, m11).extend(0.0),
            Vec2::new(0.0, 0.0).extend(1.0),
        )
    }

    /// Transforms a point from local space to world space.
    ///
    /// Applies the full transformation (scale, rotation, translation) to a point.
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
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::from_position(Vec2::new(10.0, 20.0));
    /// let local = Vec2::new(5.0, 0.0);
    /// let world = transform.transform_point(local);
    /// assert_eq!(world, Vec2::new(15.0, 20.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        let (rotation_sin, rotation_cos) = self.rotation.sin_cos();

        // Apply scale
        let scaled = Vec2::new(point.x * self.scale.x, point.y * self.scale.y);

        // Apply rotation
        let rotated = Vec2::new(
            scaled.x * rotation_cos - scaled.y * rotation_sin,
            scaled.x * rotation_sin + scaled.y * rotation_cos,
        );

        // Apply translation
        rotated + self.position
    }

    /// Transforms a vector (direction) from local space to world space.
    ///
    /// Unlike [`transform_point`](Self::transform_point), this does not apply translation.
    /// Useful for transforming directions, normals, or velocities.
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
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(100.0, 100.0),           // position ignored for vectors
    ///     std::f32::consts::FRAC_PI_2,       // 90 degree rotation
    ///     Vec2::one()
    /// );
    /// let forward = Vec2::new(1.0, 0.0);
    /// let rotated = transform.transform_vector(forward);
    /// // Result is approximately (0, 1) - rotated 90 degrees
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        let (rotation_sin, rotation_cos) = self.rotation.sin_cos();

        // Apply scale
        let scaled = Vec2::new(vector.x * self.scale.x, vector.y * self.scale.y);

        // Apply rotation (no translation)
        Vec2::new(
            scaled.x * rotation_cos - scaled.y * rotation_sin,
            scaled.x * rotation_sin + scaled.y * rotation_cos,
        )
    }

    /// Computes the inverse transform.
    ///
    /// The returned transform matches the analytic inverse when scale is uniform (S commutes with R).
    /// For non-uniform scale, use [`inverse_transform_point`](Self::inverse_transform_point) because
    /// `transform_point` applies operations in fixed S→R→T order (yielding `R⁻¹S⁻¹` instead of
    /// `S⁻¹R⁻¹`).
    ///
    /// The translation is set to `-S⁻¹(R⁻¹(position))`, which is the affine inverse translation for
    /// `p' = R(S(p)) + T` under the `R⁻¹S⁻¹` linear part implemented by [`transform_point`](Self::transform_point).
    ///
    /// # Mathematics
    ///
    /// For a TRS (Translate-Rotate-Scale) transform where `transform_point` applies:
    /// `p' = R(S(p)) + T`
    ///
    /// The inverse components are:
    /// - `scale_inv = 1 / scale`
    /// - `rotation_inv = -rotation`
    /// - `position_inv = -position` (simplified for the inverse transform representation)
    ///
    /// # Returns
    ///
    /// `Some(Transform2D)` if invertible, `None` if singular (zero scale)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_4,
    ///     Vec2::new(2.0, 2.0)
    /// );
    ///
    /// // To invert a point, use inverse_transform_point:
    /// let point = Vec2::new(5.0, 5.0);
    /// let transformed = transform.transform_point(point);
    /// let original = transform.inverse_transform_point(transformed);
    /// // original ≈ point (within floating-point precision)
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        const EPSILON: f32 = 1e-10;

        if self.scale.x.abs() < EPSILON || self.scale.y.abs() < EPSILON {
            return None;
        }

        let inv_scale = Vec2::new(1.0 / self.scale.x, 1.0 / self.scale.y);
        let inv_rotation = -self.rotation;

        // Translation of the affine inverse: -S⁻¹ R⁻¹ T
        let (inv_rot_sin, inv_rot_cos) = inv_rotation.sin_cos();
        let rotated_pos = Vec2::new(
            self.position.x * inv_rot_cos - self.position.y * inv_rot_sin,
            self.position.x * inv_rot_sin + self.position.y * inv_rot_cos,
        );
        let inv_position = Vec2::new(-rotated_pos.x * inv_scale.x, -rotated_pos.y * inv_scale.y);

        Some(Self {
            position: inv_position,
            rotation: inv_rotation,
            scale: inv_scale,
        })
    }

    /// Transforms a point from world space back to local space.
    ///
    /// This is the inverse operation of [`transform_point`](Self::transform_point).
    /// It applies the inverse transformations in the correct order: T⁻¹ → R⁻¹ → S⁻¹
    ///
    /// # Mathematics
    ///
    /// For a forward transform `p' = R(S(p)) + T`, the inverse is:
    /// `p = S⁻¹(R⁻¹(p' - T))`
    ///
    /// # Arguments
    ///
    /// * `point` - Point in world space
    ///
    /// # Returns
    ///
    /// `Some(Vec2)` with the point transformed to local space, or `None` if the transform
    /// is not invertible (zero scale)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_4,
    ///     Vec2::new(2.0, 3.0)
    /// );
    ///
    /// let point = Vec2::new(5.0, 10.0);
    /// let world = transform.transform_point(point);
    /// if let Some(local) = transform.inverse_transform_point(world) {
    ///     // local ≈ point (within floating-point precision)
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, point: Vec2) -> Option<Vec2> {
        const EPSILON: f32 = 1e-10;

        if self.scale.x.abs() < EPSILON || self.scale.y.abs() < EPSILON {
            return None;
        }

        let inv_scale = Vec2::new(1.0 / self.scale.x, 1.0 / self.scale.y);
        let inv_rotation = -self.rotation;

        // Step 1: Remove translation (T⁻¹)
        let after_trans = point - self.position;

        // Step 2: Apply inverse rotation (R⁻¹)
        let (inv_rot_sin, inv_rot_cos) = inv_rotation.sin_cos();
        let after_rot = Vec2::new(
            after_trans.x * inv_rot_cos - after_trans.y * inv_rot_sin,
            after_trans.x * inv_rot_sin + after_trans.y * inv_rot_cos,
        );

        // Step 3: Apply inverse scale (S⁻¹)
        Some(Vec2::new(
            after_rot.x * inv_scale.x,
            after_rot.y * inv_scale.y,
        ))
    }

    /// Transforms a vector (direction) from world space back to local space.
    ///
    /// This is the inverse operation of [`transform_vector`](Self::transform_vector).
    /// It applies the inverse transformations (scale and rotation) in the correct order: R⁻¹ → S⁻¹
    ///
    /// Unlike [`inverse_transform_point`](Self::inverse_transform_point), this does not apply translation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Direction vector in world space
    ///
    /// # Returns
    ///
    /// `Some(Vec2)` with the vector transformed to local space, or `None` if the transform
    /// is not invertible (zero scale)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(10.0, 20.0),  // Translation is ignored for vectors
    ///     std::f32::consts::FRAC_PI_2,
    ///     Vec2::new(2.0, 3.0)
    /// );
    ///
    /// let direction = Vec2::new(1.0, 0.0);
    /// let world = transform.transform_vector(direction);
    /// if let Some(local) = transform.inverse_transform_vector(world) {
    ///     // local ≈ direction (within floating-point precision)
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, vector: Vec2) -> Option<Vec2> {
        const EPSILON: f32 = 1e-10;

        if self.scale.x.abs() < EPSILON || self.scale.y.abs() < EPSILON {
            return None;
        }

        let inv_scale = Vec2::new(1.0 / self.scale.x, 1.0 / self.scale.y);
        let inv_rotation = -self.rotation;

        // Step 1: Apply inverse rotation (R⁻¹) - no translation for vectors
        let (inv_rot_sin, inv_rot_cos) = inv_rotation.sin_cos();
        let after_rot = Vec2::new(
            vector.x * inv_rot_cos - vector.y * inv_rot_sin,
            vector.x * inv_rot_sin + vector.y * inv_rot_cos,
        );

        // Step 2: Apply inverse scale (S⁻¹)
        Some(Vec2::new(
            after_rot.x * inv_scale.x,
            after_rot.y * inv_scale.y,
        ))
    }

    // Mutation Methods

    /// Translates (moves) this transform by the given offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - Translation offset to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut transform = Transform2D::identity();
    /// transform.translate(Vec2::new(10.0, 5.0));
    /// assert_eq!(transform.position, Vec2::new(10.0, 5.0));
    /// ```
    #[inline]
    pub fn translate(&mut self, offset: Vec2) {
        self.position += offset;
    }

    /// Rotates this transform by the given angle.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let mut transform = Transform2D::identity();
    /// transform.rotate(std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    pub fn rotate(&mut self, angle: f32) {
        self.rotation += angle;
    }

    /// Scales this transform by the given factors.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut transform = Transform2D::identity();
    /// transform.scale_by(Vec2::new(2.0, 2.0));
    /// ```
    #[inline]
    pub fn scale_by(&mut self, scale: Vec2) {
        self.scale.x *= scale.x;
        self.scale.y *= scale.y;
    }

    /// Uniformly scales this transform by the given factor.
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let mut transform = Transform2D::identity();
    /// transform.scale_uniform(2.0);
    /// ```
    #[inline]
    pub fn scale_uniform(&mut self, scale: f32) {
        self.scale.x *= scale;
        self.scale.y *= scale;
    }

    // Chaining Methods

    /// Returns a new transform with the given translation added.
    ///
    /// # Arguments
    ///
    /// * `offset` - Translation offset to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::identity()
    ///     .with_translation(Vec2::new(10.0, 20.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_translation(mut self, offset: Vec2) -> Self {
        self.translate(offset);
        self
    }

    /// Returns a new transform with the given rotation added.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let transform = Transform2D::identity()
    ///     .with_rotation(std::f32::consts::FRAC_PI_4);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rotation(mut self, angle: f32) -> Self {
        self.rotate(angle);
        self
    }

    /// Returns a new transform with the given scale multiplied.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::identity()
    ///     .with_scale(Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_scale(mut self, scale: Vec2) -> Self {
        self.scale_by(scale);
        self
    }

    /// Returns a new transform with the given uniform scale multiplied.
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor to multiply by
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let transform = Transform2D::identity()
    ///     .with_scale_uniform(2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_scale_uniform(mut self, scale: f32) -> Self {
        self.scale_uniform(scale);
        self
    }

    // Combination Operations

    /// Combines this transform with another transform.
    ///
    /// The result is equivalent to applying `self` first, then `other`.
    /// In matrix terms: `result = other * self`
    ///
    /// # Arguments
    ///
    /// * `other` - Transform to apply after this one
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let translate = Transform2D::from_position(Vec2::new(10.0, 0.0));
    /// let rotate = Transform2D::from_rotation(std::f32::consts::FRAC_PI_2);
    /// let combined = translate.then(&rotate);
    /// ```
    #[inline]
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        // Transform self's position through other's transform
        let new_position = other.transform_point(self.position);

        // Combine rotations
        let new_rotation = self.rotation + other.rotation;

        // Combine scales
        let new_scale = Vec2::new(self.scale.x * other.scale.x, self.scale.y * other.scale.y);

        Self {
            position: new_position,
            rotation: new_rotation,
            scale: new_scale,
        }
    }

    /// Linearly interpolates between this transform and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Target transform to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let start = Transform2D::from_position(Vec2::new(0.0, 0.0));
    /// let end = Transform2D::from_position(Vec2::new(100.0, 100.0));
    /// let mid = start.lerp(&end, 0.5);
    /// assert_eq!(mid.position, Vec2::new(50.0, 50.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            rotation: self.rotation + (other.rotation - self.rotation) * t,
            scale: self.scale.lerp(&other.scale, t),
        }
    }
}

// ============================================================================
// Default Trait
// ============================================================================

impl Default for Transform2D {
    /// Returns the identity transform as the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// let transform = Transform2D::default();
    /// assert_eq!(transform, Transform2D::IDENTITY);
    /// ```
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<(Vec2, f32, Vec2)> for Transform2D {
    /// Converts a tuple of (position, rotation, scale) into a Transform2D.
    ///
    /// # Arguments
    ///
    /// * `tuple` - A tuple containing (position, rotation_radians, scale)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform: Transform2D = (
    ///     Vec2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_2,
    ///     Vec2::new(2.0, 2.0)
    /// ).into();
    /// ```
    fn from((position, rotation, scale): (Vec2, f32, Vec2)) -> Self {
        Self::new(position, rotation, scale)
    }
}

impl From<&(Vec2, f32, Vec2)> for Transform2D {
    /// Converts a reference to a tuple into a Transform2D.
    fn from(&(position, rotation, scale): &(Vec2, f32, Vec2)) -> Self {
        Self::new(position, rotation, scale)
    }
}

impl From<Transform2D> for (Vec2, f32, Vec2) {
    /// Converts a Transform2D into a tuple of (position, rotation, scale).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform = Transform2D::new(
    ///     Vec2::new(10.0, 20.0),
    ///     std::f32::consts::FRAC_PI_2,
    ///     Vec2::new(2.0, 2.0)
    /// );
    /// let (pos, rot, scale): (Vec2, f32, Vec2) = transform.into();
    /// ```
    fn from(transform: Transform2D) -> Self {
        (transform.position, transform.rotation, transform.scale)
    }
}

impl From<&Transform2D> for (Vec2, f32, Vec2) {
    /// Converts a reference to a Transform2D into a tuple.
    fn from(transform: &Transform2D) -> Self {
        (transform.position, transform.rotation, transform.scale)
    }
}

impl From<Vec2> for Transform2D {
    /// Converts a Vec2 position into a Transform2D with identity rotation and scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::math::transform_2d::Transform2D;
    /// # use toyengine::primitives::vec::Vec2;
    /// let transform: Transform2D = Vec2::new(10.0, 20.0).into();
    /// assert_eq!(transform.position, Vec2::new(10.0, 20.0));
    /// assert_eq!(transform.rotation, 0.0);
    /// assert_eq!(transform.scale, Vec2::one());
    /// ```
    fn from(position: Vec2) -> Self {
        Self::from_position(position)
    }
}

// Note: No From<Matrix3> conversion provided because Matrix3 is a 3D linear
// transformation matrix that doesn't encode 2D affine translation.
// For proper 2D affine matrix support with translation, a dedicated Affine2D
// or similar type would be needed in the future.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let identity = Transform2D::identity();
        let point = Vec2::new(5.0, 10.0);
        let result = identity.transform_point(point);
        assert_eq!(result, point);
    }

    #[test]
    fn test_translation_only() {
        let translation = Transform2D::from_position(Vec2::new(10.0, 20.0));
        let point = Vec2::new(5.0, 5.0);
        let result = translation.transform_point(point);
        let expected = Vec2::new(15.0, 25.0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_rotation_only() {
        let rotation = Transform2D::from_rotation(std::f32::consts::FRAC_PI_2);
        let point = Vec2::new(1.0, 0.0);
        let result = rotation.transform_point(point);
        // Should rotate (1,0) to approximately (0,1)
        assert!(
            (result.x - 0.0).abs() < 0.0001,
            "X should be ~0, got {}",
            result.x
        );
        assert!(
            (result.y - 1.0).abs() < 0.0001,
            "Y should be ~1, got {}",
            result.y
        );
    }

    #[test]
    fn test_scale_only() {
        let scale = Transform2D::from_scale(Vec2::new(2.0, 3.0));
        let point = Vec2::new(5.0, 10.0);
        let result = scale.transform_point(point);
        let expected = Vec2::new(10.0, 30.0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_combined_trs_transform() {
        // Test that transformations apply in correct order: Scale → Rotate → Translate
        let transform = Transform2D::new(
            Vec2::new(100.0, 200.0),     // translation
            std::f32::consts::FRAC_PI_2, // 90° rotation
            Vec2::new(2.0, 2.0),         // 2x scale
        );
        let point = Vec2::new(1.0, 0.0);
        let result = transform.transform_point(point);
        // Scale (1,0) → (2,0), Rotate 90° → (0,2), Translate → (100,202)
        assert!(
            (result.x - 100.0).abs() < 0.0001,
            "X should be ~100, got {}",
            result.x
        );
        assert!(
            (result.y - 202.0).abs() < 0.0001,
            "Y should be ~202, got {}",
            result.y
        );
    }

    #[test]
    fn test_rotation_scale_matrix_extraction() {
        let original = Transform2D::new(
            Vec2::new(50.0, 75.0),
            std::f32::consts::FRAC_PI_4, // 45°
            Vec2::new(1.5, 2.0),
        );
        let matrix = original.to_rotation_scale_matrix();

        // The matrix should only contain rotation and scale, not translation
        // So transforming (0,0) through the matrix should give (0,0)
        let origin = Vec2::ZERO.extend(1.0);
        let result = origin * matrix;
        assert!((result.x).abs() < 0.0001);
        assert!((result.y).abs() < 0.0001);

        // Test that matrix contains the rotation and scale components
        // For a 45° rotation with scale (1.5, 2.0):
        // cos(45°) ≈ 0.707, sin(45°) ≈ 0.707
        // m00 = cos*sx ≈ 0.707*1.5 ≈ 1.06
        // m01 = -sin*sy ≈ -0.707*2.0 ≈ -1.414
        // m10 = sin*sx ≈ 0.707*1.5 ≈ 1.06
        // m11 = cos*sy ≈ 0.707*2.0 ≈ 1.414
        let cos = std::f32::consts::FRAC_1_SQRT_2; // cos(45°)
        let sin = std::f32::consts::FRAC_1_SQRT_2; // sin(45°)
        assert!((matrix.row_x.x - cos * 1.5).abs() < 0.0001);
        assert!((matrix.row_x.y - (-sin * 2.0)).abs() < 0.0001);
        assert!((matrix.row_y.x - sin * 1.5).abs() < 0.0001);
        assert!((matrix.row_y.y - cos * 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_inverse_transform_point() {
        let transform = Transform2D::new(
            Vec2::new(25.0, 50.0),
            std::f32::consts::FRAC_PI_4,
            Vec2::new(2.0, 3.0),
        );

        let point = Vec2::new(10.0, 20.0);
        let transformed = transform.transform_point(point);
        let back = transform
            .inverse_transform_point(transformed)
            .expect("Failed to invert point");

        assert!(
            (back.x - point.x).abs() < 0.001,
            "X mismatch: {} vs {}",
            back.x,
            point.x
        );
        assert!(
            (back.y - point.y).abs() < 0.001,
            "Y mismatch: {} vs {}",
            back.y,
            point.y
        );
    }

    #[test]
    fn test_inverse_transform_point_via_inverse_transform() {
        let transform = Transform2D::new(
            Vec2::new(25.0, 50.0),
            std::f32::consts::FRAC_PI_4,
            Vec2::new(2.0, 2.0), // uniform scale so inverse composition through transform_point is valid
        );

        let point = Vec2::new(10.0, 20.0);
        let forward = transform.transform_point(point);
        let inverse = transform.inverse().expect("Invertible");
        let back = inverse.transform_point(forward);

        assert!(
            (back.x - point.x).abs() < 0.001,
            "X mismatch: {} vs {}",
            back.x,
            point.x
        );
        assert!(
            (back.y - point.y).abs() < 0.001,
            "Y mismatch: {} vs {}",
            back.y,
            point.y
        );
    }

    #[test]
    fn test_inverse_transform_multiple_points() {
        let transform = Transform2D::new(
            Vec2::new(25.0, 50.0),
            std::f32::consts::FRAC_PI_4,
            Vec2::new(2.0, 3.0),
        );

        let test_points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(-3.0, 7.0),
        ];

        for p in test_points {
            let fwd = transform.transform_point(p);
            let inv = transform
                .inverse_transform_point(fwd)
                .expect("Failed to invert");
            assert!(
                (p.x - inv.x).abs() < 0.001,
                "Point {:?} failed round-trip",
                p
            );
            assert!(
                (p.y - inv.y).abs() < 0.001,
                "Point {:?} failed round-trip",
                p
            );
        }
    }

    #[test]
    fn test_transform_vector_ignores_translation() {
        let transform = Transform2D::new(
            Vec2::new(1000.0, 1000.0),   // large translation
            std::f32::consts::FRAC_PI_2, // 90° rotation
            Vec2::new(2.0, 2.0),
        );
        let vector = Vec2::new(1.0, 0.0);
        let result = transform.transform_vector(vector);

        // Should scale to (2,0) then rotate to (0,2), NO translation
        assert!(
            (result.x - 0.0).abs() < 0.0001,
            "X should be ~0, got {}",
            result.x
        );
        assert!(
            (result.y - 2.0).abs() < 0.0001,
            "Y should be ~2, got {}",
            result.y
        );
    }

    #[test]
    fn test_chaining_methods() {
        let transform = Transform2D::identity()
            .with_translation(Vec2::new(10.0, 20.0))
            .with_rotation(std::f32::consts::FRAC_PI_4)
            .with_scale(Vec2::new(2.0, 2.0));

        assert_eq!(transform.position, Vec2::new(10.0, 20.0));
        assert_eq!(transform.rotation, std::f32::consts::FRAC_PI_4);
        assert_eq!(transform.scale, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_transform_combination() {
        let t1 = Transform2D::from_position(Vec2::new(10.0, 0.0));
        let t2 = Transform2D::from_rotation(std::f32::consts::FRAC_PI_2);
        let combined = t1.then(&t2);

        let point = Vec2::new(0.0, 0.0);
        let result1 = t2.transform_point(t1.transform_point(point));
        let result2 = combined.transform_point(point);

        assert!((result1.x - result2.x).abs() < 0.0001);
        assert!((result1.y - result2.y).abs() < 0.0001);
    }

    #[test]
    fn test_lerp_interpolation() {
        let start = Transform2D::from_position(Vec2::new(0.0, 0.0));
        let end = Transform2D::from_position(Vec2::new(100.0, 100.0));
        let mid = start.lerp(&end, 0.5);

        assert_eq!(mid.position, Vec2::new(50.0, 50.0));
    }

    #[test]
    fn test_lerp_full_transform() {
        let start = Transform2D::new(Vec2::new(0.0, 0.0), 0.0, Vec2::new(1.0, 1.0));
        let end = Transform2D::new(
            Vec2::new(100.0, 200.0),
            std::f32::consts::PI,
            Vec2::new(3.0, 4.0),
        );
        let mid = start.lerp(&end, 0.5);

        assert_eq!(mid.position, Vec2::new(50.0, 100.0));
        assert!((mid.rotation - std::f32::consts::FRAC_PI_2).abs() < 0.0001);
        assert_eq!(mid.scale, Vec2::new(2.0, 2.5));
    }

    #[test]
    fn test_tuple_conversion() {
        let tuple = (
            Vec2::new(5.0, 10.0),
            std::f32::consts::FRAC_PI_4,
            Vec2::new(2.0, 2.0),
        );
        let transform: Transform2D = tuple.into();

        assert_eq!(transform.position, Vec2::new(5.0, 10.0));
        assert_eq!(transform.rotation, std::f32::consts::FRAC_PI_4);
        assert_eq!(transform.scale, Vec2::new(2.0, 2.0));

        // Test reverse conversion
        let tuple_back: (Vec2, f32, Vec2) = transform.into();
        assert_eq!(tuple_back.0, Vec2::new(5.0, 10.0));
        assert_eq!(tuple_back.1, std::f32::consts::FRAC_PI_4);
        assert_eq!(tuple_back.2, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_inverse_vector_transform() {
        let transform = Transform2D::new(
            Vec2::new(100.0, 200.0), // Translation should be ignored
            std::f32::consts::FRAC_PI_4,
            Vec2::new(2.0, 3.0),
        );

        let direction = Vec2::new(1.0, 1.0);
        let world_dir = transform.transform_vector(direction);
        let local_dir = transform
            .inverse_transform_vector(world_dir)
            .expect("Failed to invert vector");

        assert!(
            (local_dir.x - direction.x).abs() < 0.001,
            "X mismatch: {} vs {}",
            local_dir.x,
            direction.x
        );
        assert!(
            (local_dir.y - direction.y).abs() < 0.001,
            "Y mismatch: {} vs {}",
            local_dir.y,
            direction.y
        );
    }

    #[test]
    fn test_inverse_returns_none_for_zero_scale() {
        let transform = Transform2D::new(
            Vec2::new(10.0, 20.0),
            std::f32::consts::FRAC_PI_4,
            Vec2::new(0.0, 1.0), // Zero scale on X
        );

        assert!(transform.inverse().is_none());
        assert!(transform.inverse_transform_point(Vec2::ZERO).is_none());
        assert!(
            transform
                .inverse_transform_vector(Vec2::new(1.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn test_mutation_methods() {
        let mut transform = Transform2D::identity();

        transform.translate(Vec2::new(10.0, 5.0));
        assert_eq!(transform.position, Vec2::new(10.0, 5.0));

        transform.rotate(std::f32::consts::FRAC_PI_2);
        assert!((transform.rotation - std::f32::consts::FRAC_PI_2).abs() < 0.0001);

        transform.scale_by(Vec2::new(2.0, 3.0));
        assert_eq!(transform.scale, Vec2::new(2.0, 3.0));

        transform.scale_uniform(0.5);
        assert_eq!(transform.scale, Vec2::new(1.0, 1.5));
    }

    #[test]
    fn test_from_vec2_conversion() {
        let transform: Transform2D = Vec2::new(10.0, 20.0).into();
        assert_eq!(transform.position, Vec2::new(10.0, 20.0));
        assert_eq!(transform.rotation, 0.0);
        assert_eq!(transform.scale, Vec2::one());
    }

    #[test]
    fn test_default_trait() {
        let transform = Transform2D::default();
        assert_eq!(transform.position, Vec2::ZERO);
        assert_eq!(transform.rotation, 0.0);
        assert_eq!(transform.scale, Vec2::one());
    }

    #[test]
    fn test_negative_scale() {
        // Negative scale should flip/mirror
        let transform = Transform2D::from_scale(Vec2::new(-1.0, 1.0));
        let point = Vec2::new(5.0, 10.0);
        let result = transform.transform_point(point);
        assert_eq!(result, Vec2::new(-5.0, 10.0));
    }

    #[test]
    fn test_non_uniform_scale() {
        let transform =
            Transform2D::new(Vec2::ZERO, std::f32::consts::FRAC_PI_4, Vec2::new(2.0, 0.5));
        let point = Vec2::new(1.0, 1.0);
        let result = transform.transform_point(point);

        // Verify non-uniform scaling works correctly
        let back = transform.inverse_transform_point(result).unwrap();
        assert!((back.x - point.x).abs() < 0.001);
        assert!((back.y - point.y).abs() < 0.001);
    }
}
