//! Quaternions primitives for 3D rotations and orientations.
//!
//! This module provides a high-performance quaternion type used for representing
//! rotations in 3D space. Quaternions avoid issues like gimbal lock and provide
//! smooth interpolation between orientations.
//!
//! In this implementation, Quaternions are represented as four-dimensional vectors
//! with components (x, y, z, w), where (x, y, z) represent the vector part
//! and w represents the scalar part. The module includes common operations such as
//! multiplication, normalization, conjugation, and conversion to/from rotation matrices
//! and Euler angles.
//!
//! # Examples
//!
//! ```rust
//! use toyengine::primitives::quat::Quaternion;
//! use toyengine::primitives::vec::Vec3;
//!
//! // Create a rotation around the Y-axis (45 degrees)
//! let rotation = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
//!
//! // Apply rotation to a vector
//! let point = Vec3::new(1.0, 0.0, 0.0);
//! let rotated = rotation * point;
//!
//! // Combine rotations
//! let rot1 = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
//! let rot2 = Quaternion::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_6);
//! let combined = rot1 * rot2;
//!
//! // Interpolate between rotations (SLERP)
//! let start = Quaternion::identity();
//! let end = Quaternion::from_axis_angle(Vec3::Y, std::f32::consts::PI);
//! let halfway = start.slerp(end, 0.5);
//! ```

use super::mat::Matrix4;
use super::vec::{Vec3, Vec4};

use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// ============================================================================
// Module Constants
// ============================================================================

/// Threshold for SLERP to fall back to linear interpolation when quaternions are very close.
/// When the dot product between two quaternions exceeds this value, linear interpolation
/// is used instead of spherical interpolation to avoid numerical instability.
const SLERP_DOT_THRESHOLD: f32 = 0.9995;

/// Threshold for detecting gimbal lock in Euler angle conversions.
/// When the sine of the pitch angle exceeds this value, the quaternion is considered
/// to be in a gimbal lock configuration.
const GIMBAL_LOCK_THRESHOLD: f32 = 0.99999;

/// A quaternion represented by a [`Vec4`] with four components: x, y, z, w.
/// The quaternion is typically used to represent rotations in 3D space.
///
/// The components are defined as follows:
/// - x, y, z: The vector part of the quaternion.
/// - w: The scalar part of the quaternion.
/// # Example
///
/// ```rust
/// # use toyengine::primitives::quat::Quaternion;
/// let q = Quaternion::new(0.0, 0.0, 0.0, 1.0); // Identity quaternion
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Quaternion(pub Vec4);

impl Quaternion {
    /// Creates a new quaternion with the given components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Quaternion(Vec4::new(x, y, z, w))
    }

    /// Returns the identity quaternion (no rotation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let identity = Quaternion::identity(); // Quaternion(0.0, 0.0, 0.0, 1.0)
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Quaternion(Vec4::new(0.0, 0.0, 0.0, 1.0))
    }

    /// Creates a quaternion representing a rotation around a given axis by a specified angle (in radians).
    /// # Panics / Undefined behavior
    /// `axis` must be normalized.
    ///
    /// # Arguments
    /// * `axis` - The axis of rotation (should be a normalized vector).
    /// * `angle_rad` - The rotation angle in radians.
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec3;
    /// let axis = Vec3::new(0.0, 1.0, 0.0); // Y-axis
    /// let angle = std::f32::consts::FRAC_PI_2; // 90 degrees in radians
    /// let q = Quaternion::from_axis_angle(axis, angle);
    /// ```
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let half_angle = angle_rad * 0.5;
        let (sin_half, cos_half) = half_angle.sin_cos();
        let normalized_axis = axis.with_normalize();
        let scaled_axis = normalized_axis * sin_half;

        Quaternion::new(scaled_axis.x, scaled_axis.y, scaled_axis.z, cos_half)
    }

    /// Creates a new quaternion from an Euler angle representation (pitch, yaw, roll).
    ///
    /// **Note**: This implementation uses **ZXY rotation order**:
    /// 1. First rotate around Z-axis (roll)
    /// 2. Then rotate around X-axis (pitch)
    /// 3. Finally rotate around Y-axis (yaw)
    ///
    /// This uses intrinsic (local-axis) rotations.
    /// Equivalent to extrinsic Y → X → Z in world space.
    ///
    /// # Arguments
    /// * `euler` - Vec3 where x=pitch, y=yaw, z=roll (all in radians)
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec3;
    /// let euler = Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2); // 90 degrees roll
    /// let q = Quaternion::from_euler_zxy(euler);
    /// ```
    pub fn from_euler_zxy(euler: Vec3) -> Self {
        let (pitch, yaw, roll) = (euler.x, euler.y, euler.z);

        // Half angles for quaternion conversion
        let (sin_pitch, cos_pitch) = (pitch * 0.5).sin_cos();
        let (sin_yaw, cos_yaw) = (yaw * 0.5).sin_cos();
        let (sin_roll, cos_roll) = (roll * 0.5).sin_cos();

        // Precompute products
        let cos_pitch_cos_yaw = cos_pitch * cos_yaw;
        let sin_pitch_sin_yaw = sin_pitch * sin_yaw;
        let sin_pitch_cos_yaw = sin_pitch * cos_yaw;
        let cos_pitch_sin_yaw = cos_pitch * sin_yaw;

        // ZXY rotation order: Q = Qy * Qx * Qz
        let w = cos_pitch_cos_yaw * cos_roll + sin_pitch_sin_yaw * sin_roll;
        let x = sin_pitch_cos_yaw * cos_roll - cos_pitch_sin_yaw * sin_roll;
        let y = cos_pitch_sin_yaw * cos_roll + sin_pitch_cos_yaw * sin_roll;
        let z = cos_pitch_cos_yaw * sin_roll - sin_pitch_sin_yaw * cos_roll;

        Quaternion::new(x, y, z, w)
    }

    // Constants

    /// Identity quaternion representing no rotation (0, 0, 0, 1).
    pub const IDENTITY: Self = Self(Vec4::new(0.0, 0.0, 0.0, 1.0));

    // Core Operations

    /// Computes the dot product (scalar product) of two quaternions.
    ///
    /// The dot product measures the similarity between two quaternions.
    /// For unit quaternions, the dot product is the cosine of half the angle between them.
    ///
    /// # Formula
    /// dot(q1, q2) = q1.x × q2.x + q1.y × q2.y + q1.z × q2.z + q1.w × q2.w
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// assert_eq!(q1.dot(&q2), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.0.dot(&other.0)
    }

    /// Computes the squared length (magnitude squared) of the quaternion.
    ///
    /// This is cheaper than [`length`](Self::length) as it avoids the square root.
    /// Useful for comparing lengths or when only relative magnitudes matter.
    ///
    /// # Formula
    /// length_squared = x² + y² + z² + w²
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(q.length_squared(), 25.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> f32 {
        self.0.length_squared()
    }

    /// Computes the length (magnitude) of the quaternion.
    ///
    /// For unit quaternions (used for rotations), the length should be 1.0.
    ///
    /// # Formula
    /// length = √(x² + y² + z² + w²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// assert_eq!(q.length(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn length(&self) -> f32 {
        self.0.length()
    }

    /// Normalizes the quaternion to unit length in place.
    ///
    /// Unit quaternions represent valid rotations. If the quaternion has zero length,
    /// it remains unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// q.normalize();
    /// assert!((q.length() - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn normalize(&mut self) {
        self.0.normalize();
    }

    /// Checks if the quaternion is normalized (unit length).
    ///
    /// Uses precision threshold to account for floating-point errors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::identity();
    /// assert!(q.is_normalized());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.0.is_normalized()
    }

    /// Returns the conjugate of the quaternion.
    ///
    /// The conjugate negates the vector part (x, y, z) while keeping the scalar part (w).
    /// For unit quaternions, the conjugate represents the inverse rotation.
    ///
    /// # Formula
    /// conjugate(x, y, z, w) = (-x, -y, -z, w)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let conj = q.conjugate();
    /// assert_eq!(conj, Quaternion::new(-1.0, -2.0, -3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Quaternion::new(-self.0.x, -self.0.y, -self.0.z, self.0.w)
    }

    /// Returns the inverse of the quaternion.
    ///
    /// The inverse quaternion represents the reverse rotation.
    /// For unit quaternions, the inverse is equal to the conjugate.
    ///
    /// # Formula
    /// inverse(q) = conjugate(q) / length_squared(q)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::identity();
    /// let inv = q.inverse();
    /// assert_eq!(inv, Quaternion::identity());
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len_sq = 1.0 / len_sq;
            Quaternion::new(
                -self.0.x * inv_len_sq,
                -self.0.y * inv_len_sq,
                -self.0.z * inv_len_sq,
                self.0.w * inv_len_sq,
            )
        } else {
            *self
        }
    }

    /// Rotates a 3D vector by this quaternion.
    ///
    /// Uses the formula: v' = q * v * q^(-1)
    /// where v is treated as a quaternion (v.x, v.y, v.z, 0).
    ///
    /// # Requirements
    /// This quaternion must be normalized.
    ///
    /// # Arguments
    /// * `vec` - The vector to rotate
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec3;
    /// let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
    /// let v = Vec3::new(1.0, 0.0, 0.0);
    /// let rotated = q.rotate_vec3(v);
    /// assert!((rotated.y - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn rotate_vec3(&self, vec: Vec3) -> Vec3 {
        debug_assert!(self.is_normalized());

        // Optimized formula: v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        let qvec = Vec3::new(self.0.x, self.0.y, self.0.z);
        let uv = qvec.cross(&vec);
        let uuv = qvec.cross(&uv);
        vec + ((uv * self.0.w) + uuv) * 2.0
    }

    /// Converts the quaternion to Euler angles (pitch, yaw, roll) in ZXY rotation order.
    ///
    /// Returns a Vec3 where x=pitch, y=yaw, z=roll (all in radians).
    ///
    /// **Note**: This is the inverse of [`from_euler_zxy`](Self::from_euler_zxy).
    /// Gimbal lock may occur at pitch = ±90 degrees.
    ///
    /// In a gimbal lock situation, yaw is set to zero by convention.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec3;
    /// let euler = Vec3::new(0.5, 0.3, 0.7);
    /// let q = Quaternion::from_euler_zxy(euler);
    /// let recovered = q.to_euler_zxy();
    /// // Due to floating point, check approximate equality
    /// ```
    #[inline]
    #[must_use]
    pub fn to_euler_zxy(&self) -> Vec3 {
        let (x, y, z, w) = (self.0.x, self.0.y, self.0.z, self.0.w);

        // Extract angles from quaternion for ZXY order
        let sin_x = 2.0 * (w * x - y * z);
        let sin_x_clamped = sin_x.clamp(-1.0, 1.0);

        // Check for gimbal lock
        if sin_x.abs() >= GIMBAL_LOCK_THRESHOLD {
            // Gimbal lock case: pitch is ±90 degrees
            let pitch = sin_x_clamped.asin();
            let yaw = 0.0; // Set yaw to zero by convention
            let roll = (2.0 * (x * y - w * z)).atan2(2.0 * (w * w + z * z) - 1.0);
            Vec3::new(pitch, yaw, roll)
        } else {
            // Standard case
            let pitch = sin_x_clamped.asin();
            let yaw = (2.0 * (w * y + x * z)).atan2(2.0 * (w * w + z * z) - 1.0);
            let roll = (2.0 * (w * z + x * y)).atan2(2.0 * (w * w + y * y) - 1.0);
            Vec3::new(pitch, yaw, roll)
        }
    }

    /// Converts the quaternion to a 4x4 rotation matrix.
    ///
    /// Returns a row-major 4x4 matrix representing the same rotation as this quaternion.
    /// The resulting matrix is a pure rotation matrix with no translation or scale.
    ///
    /// # Matrix Layout (Row-Major)
    /// ```text
    /// [ 1-2(y²+z²)   2(xy-wz)     2(xz+wy)     0 ]
    /// [ 2(xy+wz)     1-2(x²+z²)   2(yz-wx)     0 ]
    /// [ 2(xz-wy)     2(yz+wx)     1-2(x²+y²)   0 ]
    /// [ 0            0            0            1 ]
    /// ```
    ///
    /// # Coordinate System
    /// - Right-handed coordinate system
    /// - Forward direction: -Z axis
    /// - Up direction: +Y axis
    /// - Right direction: +X axis
    ///
    /// # Requirements
    /// This quaternion should be normalized (unit quaternion) for correct results.
    /// The method includes a debug assertion to verify normalization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec3;
    /// # use toyengine::primitives::mat::Matrix4;
    ///
    /// // 90-degree rotation around Y-axis
    /// let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
    /// let matrix = q.to_matrix4();
    ///
    /// // Transform a point using the matrix
    /// let point = Vec3::new(1.0, 0.0, 0.0);
    /// let transformed = matrix.transform_point(point);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_matrix4(&self) -> Matrix4 {
        debug_assert!(self.is_normalized(), "Quaternion must be normalized");

        let (x, y, z, w) = (self.0.x, self.0.y, self.0.z, self.0.w);

        // Precompute repeated values for efficiency
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        // Build rotation matrix
        // Note: Despite the "row_*" naming, Matrix4 stores columns
        // Each Vec4 represents a transformed basis vector (column)
        Matrix4::new(
            Vec4::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0),
            Vec4::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0),
            Vec4::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Spherical linear interpolation between two quaternions.
    ///
    /// SLERP provides smooth interpolation between rotations at a constant angular velocity.
    /// This is the standard method for animating rotations.
    ///
    /// # Arguments
    /// * `other` - Target quaternion to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// let mid = q1.slerp(&q2, 0.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        let mut dot = self.dot(other);

        // If the dot product is negative, slerp won't take the shorter path.
        // Fix by reversing one quaternion.
        let other = if dot < 0.0 {
            dot = -dot;
            Quaternion::new(-other.0.x, -other.0.y, -other.0.z, -other.0.w)
        } else {
            *other
        };

        // If quaternions are very close, use linear interpolation
        if dot > SLERP_DOT_THRESHOLD {
            return Quaternion(self.0.lerp(&other.0, t)).with_normalize();
        }

        // Calculate coefficients
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();

        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Quaternion(self.0 * a + other.0 * b).with_normalize()
    }

    /// Computes the angle (in radians) between two quaternions.
    ///
    /// Returns the angle of the shortest rotation between the two orientations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// assert!((q1.angle_between(&q2)).abs() < 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn angle_between(&self, other: &Self) -> f32 {
        let dot = self.dot(other).abs().min(1.0);
        2.0 * dot.acos()
    }

    // Chainable Methods

    /// Returns a normalized (unit length) version of this quaternion.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).with_normalize();
    /// assert!((q.length() - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_normalize(self) -> Self {
        Quaternion(self.0.with_normalize())
    }

    /// Returns the conjugate of this quaternion with method chaining.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).with_conjugate();
    /// assert_eq!(q, Quaternion::new(-1.0, -2.0, -3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_conjugate(self) -> Self {
        self.conjugate()
    }

    /// Returns the inverse of this quaternion with method chaining.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::identity().with_inverse();
    /// assert_eq!(q, Quaternion::identity());
    /// ```
    #[inline]
    #[must_use]
    pub fn with_inverse(self) -> Self {
        self.inverse()
    }

    /// Spherical linear interpolation with method chaining.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// let result = q1.with_slerp(&q2, 0.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_slerp(self, other: &Self, t: f32) -> Self {
        self.slerp(other, t)
    }
}

// ============================================================================
// Operator Overloading - Quaternion
// ============================================================================

impl Add for Quaternion {
    type Output = Self;

    /// Component-wise addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let result = q1 + q2;
    /// assert_eq!(result, Quaternion::new(6.0, 8.0, 10.0, 12.0));
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Quaternion(*self + *other)
    }
}

impl Add<&Self> for Quaternion {
    type Output = Self;
    #[inline]
    fn add(self, other: &Self) -> Self {
        Quaternion(*self + **other)
    }
}

impl Add<&Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn add(self, other: &Quaternion) -> Quaternion {
        Quaternion(**self + **other)
    }
}

impl Add<Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn add(self, other: Quaternion) -> Quaternion {
        Quaternion(**self + *other)
    }
}

impl Sub for Quaternion {
    type Output = Self;

    /// Component-wise subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::new(10.0, 12.0, 14.0, 16.0);
    /// let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let result = q1 - q2;
    /// assert_eq!(result, Quaternion::new(5.0, 6.0, 7.0, 8.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Quaternion(*self - *other)
    }
}

impl Sub<&Self> for Quaternion {
    type Output = Self;
    #[inline]
    fn sub(self, other: &Self) -> Self {
        Quaternion(*self - **other)
    }
}

impl Sub<&Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn sub(self, other: &Quaternion) -> Quaternion {
        Quaternion(**self - **other)
    }
}

impl Sub<Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn sub(self, other: Quaternion) -> Quaternion {
        Quaternion(**self - *other)
    }
}

impl Mul for Quaternion {
    type Output = Self;

    /// Quaternion multiplication (Hamilton product) using the `*` operator.
    ///
    /// This represents composition of rotations: q1 * q2 applies q2 first, then q1.
    ///
    /// # Formula
    /// q1 * q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ///            w1*x2 + x1*w2 + y1*z2 - z1*y2,
    ///            w1*y2 - x1*z2 + y1*w2 + z1*x2,
    ///            w1*z2 + x1*y2 - y1*x2 + z1*w2)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q1 = Quaternion::identity();
    /// let q2 = Quaternion::identity();
    /// let result = q1 * q2;
    /// assert_eq!(result, Quaternion::identity());
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        let (x1, y1, z1, w1) = (self.x, self.y, self.z, self.w);
        let (x2, y2, z2, w2) = (other.x, other.y, other.z, other.w);

        Quaternion::new(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    }
}

impl Mul<&Self> for Quaternion {
    type Output = Self;
    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, other: &Quaternion) -> Quaternion {
        *self * *other
    }
}

impl Mul<Quaternion> for &Quaternion {
    type Output = Quaternion;
    #[inline]
    fn mul(self, other: Quaternion) -> Quaternion {
        *self * other
    }
}

impl Mul<f32> for Quaternion {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let result = q * 2.0;
    /// assert_eq!(result, Quaternion::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Quaternion(*self * scalar)
    }
}

impl Mul<&f32> for Quaternion {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        Quaternion(*self * scalar)
    }
}

impl Mul<Quaternion> for f32 {
    type Output = Quaternion;

    /// Scalar multiplication (reversed operand order) using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let result = 2.0 * q;
    /// assert_eq!(result, Quaternion::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul(self, quat: Quaternion) -> Quaternion {
        Quaternion(self * *quat)
    }
}

impl Mul<&Quaternion> for f32 {
    type Output = Quaternion;
    #[inline]
    fn mul(self, quat: &Quaternion) -> Quaternion {
        Quaternion(self * **quat)
    }
}

impl Div<f32> for Quaternion {
    type Output = Self;

    /// Scalar division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::new(6.0, 9.0, 12.0, 15.0);
    /// let result = q / 3.0;
    /// assert_eq!(result, Quaternion::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Quaternion(*self / scalar)
    }
}

impl Div<&f32> for Quaternion {
    type Output = Self;
    #[inline]
    fn div(self, scalar: &f32) -> Self {
        Quaternion(*self / scalar)
    }
}

impl AddAssign for Quaternion {
    /// Component-wise addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// q += Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(q, Quaternion::new(6.0, 8.0, 10.0, 12.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl AddAssign<&Self> for Quaternion {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.0 += other.0;
    }
}

impl SubAssign for Quaternion {
    /// Component-wise subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::new(10.0, 12.0, 14.0, 16.0);
    /// q -= Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(q, Quaternion::new(5.0, 6.0, 7.0, 8.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl SubAssign<&Self> for Quaternion {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for Quaternion {
    /// Quaternion multiplication assignment (Hamilton product) using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::identity();
    /// q *= Quaternion::identity();
    /// assert_eq!(q, Quaternion::identity());
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<&Self> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self = *self * *other;
    }
}

impl MulAssign<f32> for Quaternion {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// q *= 2.0;
    /// assert_eq!(q, Quaternion::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.0 *= scalar;
    }
}

impl MulAssign<&f32> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        self.0 *= scalar;
    }
}

impl DivAssign<f32> for Quaternion {
    /// Scalar division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let mut q = Quaternion::new(6.0, 9.0, 12.0, 15.0);
    /// q /= 3.0;
    /// assert_eq!(q, Quaternion::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        self.0 /= scalar;
    }
}

impl DivAssign<&f32> for Quaternion {
    #[inline]
    fn div_assign(&mut self, scalar: &f32) {
        self.0 /= scalar;
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<Vec4> for Quaternion {
    /// Converts a Vec4 to a Quaternion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(0.0, 0.0, 0.0, 1.0);
    /// let q: Quaternion = v.into();
    /// assert_eq!(q, Quaternion::identity());
    /// ```
    #[inline]
    fn from(vec: Vec4) -> Self {
        Quaternion(vec)
    }
}

impl From<&Vec4> for Quaternion {
    #[inline]
    fn from(vec: &Vec4) -> Self {
        Quaternion(*vec)
    }
}

impl From<Quaternion> for Vec4 {
    /// Converts a Quaternion to a Vec4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// # use toyengine::primitives::vec::Vec4;
    /// let q = Quaternion::identity();
    /// let v: Vec4 = q.into();
    /// assert_eq!(v, Vec4::new(0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline]
    fn from(quat: Quaternion) -> Self {
        *quat
    }
}

impl From<&Quaternion> for Vec4 {
    #[inline]
    fn from(quat: &Quaternion) -> Self {
        *(*quat)
    }
}

impl From<[f32; 4]> for Quaternion {
    /// Converts an array [x, y, z, w] to a Quaternion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q: Quaternion = [0.0, 0.0, 0.0, 1.0].into();
    /// assert_eq!(q, Quaternion::identity());
    /// ```
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Quaternion(Vec4::from(arr))
    }
}

impl From<&[f32; 4]> for Quaternion {
    #[inline]
    fn from(arr: &[f32; 4]) -> Self {
        Quaternion(Vec4::from(arr))
    }
}

impl From<Quaternion> for [f32; 4] {
    /// Converts a Quaternion to an array [x, y, z, w].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::identity();
    /// let arr: [f32; 4] = q.into();
    /// assert_eq!(arr, [0.0, 0.0, 0.0, 1.0]);
    /// ```
    #[inline]
    fn from(quat: Quaternion) -> Self {
        [quat.x, quat.y, quat.z, quat.w]
    }
}

impl From<&Quaternion> for [f32; 4] {
    #[inline]
    fn from(quat: &Quaternion) -> Self {
        [quat.x, quat.y, quat.z, quat.w]
    }
}

// Deref implemented to access underlying Vec4
impl Deref for Quaternion {
    type Target = Vec4;

    /// Dereferences the Quaternion to access its underlying Vec4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::quat::Quaternion;
    /// let q = Quaternion::identity();
    /// let v: &toyengine::primitives::vec::Vec4 = &*q;
    /// assert_eq!(*v, toyengine::primitives::vec::Vec4::new(0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPSILON: f32 = 1e-5;
    const EPSILON_ROUGH: f32 = 1e-4;

    #[test]
    fn test_identity() {
        let identity = Quaternion::identity();
        assert_eq!(identity, Quaternion::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(identity, Quaternion::IDENTITY);
        assert!(identity.is_normalized());
    }

    #[test]
    fn test_identity_multiplication() {
        let q = Quaternion::new(0.5, 0.5, 0.5, 0.5);
        let result1 = q * Quaternion::identity();
        let result2 = Quaternion::identity() * q;
        assert_eq!(result1, q);
        assert_eq!(result2, q);
    }

    #[test]
    fn test_normalization() {
        let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let original_length = q.length();
        assert!((original_length - 5.477226).abs() < EPSILON);

        q.normalize();
        let normalized_length = q.length();
        assert!((normalized_length - 1.0).abs() < EPSILON);
        assert!(q.is_normalized());
    }

    #[test]
    fn test_with_normalize() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).with_normalize();
        assert!((q.length() - 1.0).abs() < EPSILON);
        assert!(q.is_normalized());
    }

    #[test]
    fn test_conjugate_properties() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let conj = q.conjugate();
        assert_eq!(conj, Quaternion::new(-1.0, -2.0, -3.0, 4.0));

        // conjugate(conjugate(q)) = q
        let double_conj = conj.conjugate();
        assert_eq!(double_conj, q);
    }

    #[test]
    fn test_inverse_properties() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).with_normalize();
        let inv = q.inverse();
        let product = q * inv;

        // q * q^(-1) should equal identity
        assert!(product.0.x.abs() < EPSILON);
        assert!(product.0.y.abs() < EPSILON);
        assert!(product.0.z.abs() < EPSILON);
        assert!((product.0.w - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotation_z_axis_90_degrees() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec3(v);

        // Rotating X-axis 90° around Z should give Y-axis
        assert!(rotated.x.abs() < EPSILON);
        assert!((rotated.y - 1.0).abs() < EPSILON);
        assert!(rotated.z.abs() < EPSILON);
    }

    #[test]
    fn test_rotation_y_axis_90_degrees() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec3(v);

        // Rotating X-axis 90° around Y should give -Z-axis
        assert!(rotated.x.abs() < EPSILON);
        assert!(rotated.y.abs() < EPSILON);
        assert!((rotated.z + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotation_composition() {
        let q1 = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        let q2 = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);
        let combined = q2 * q1; // Apply q1 first, then q2

        let v = Vec3::new(1.0, 0.0, 0.0);
        let step1 = q1.rotate_vec3(v);
        let step2 = q2.rotate_vec3(step1);
        let direct = combined.rotate_vec3(v);

        assert!((step2.x - direct.x).abs() < EPSILON);
        assert!((step2.y - direct.y).abs() < EPSILON);
        assert!((step2.z - direct.z).abs() < EPSILON);
    }

    #[test]
    fn test_euler_simple_cases() {
        let test_cases = vec![
            Vec3::new(0.0, 0.0, 0.0),       // Zero rotation
            Vec3::new(0.0, FRAC_PI_2, 0.0), // 90° yaw only
            Vec3::new(0.0, 0.0, FRAC_PI_2), // 90° roll only
        ];

        for euler in test_cases {
            let q = Quaternion::from_euler_zxy(euler);
            let recovered = q.to_euler_zxy();
            let q_recovered = Quaternion::from_euler_zxy(recovered);

            let test_vec = Vec3::new(1.0, 2.0, 3.0);
            let rotated1 = q.rotate_vec3(test_vec);
            let rotated2 = q_recovered.rotate_vec3(test_vec);

            assert!((rotated1.x - rotated2.x).abs() < EPSILON_ROUGH);
            assert!((rotated1.y - rotated2.y).abs() < EPSILON_ROUGH);
            assert!((rotated1.z - rotated2.z).abs() < EPSILON_ROUGH);
        }
    }

    #[test]
    fn test_slerp_interpolation() {
        let q_start = Quaternion::identity();
        let q_end = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);

        let q_mid = q_start.slerp(&q_end, 0.5);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q_mid.rotate_vec3(v);

        // At 45°, (1,0,0) should become approximately (0.707, 0.707, 0)
        let expected_x = FRAC_PI_4.cos();
        let expected_y = FRAC_PI_4.sin();
        assert!((rotated.x - expected_x).abs() < EPSILON);
        assert!((rotated.y - expected_y).abs() < EPSILON);
        assert!(rotated.z.abs() < EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::identity();
        assert_eq!(q1.dot(&q2), 1.0);

        let q3 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q4 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(q3.dot(&q4), 0.0);
    }

    #[test]
    fn test_angle_between() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        let angle = q1.angle_between(&q2);
        assert!((0.0..=PI).contains(&angle));
        assert!((angle - FRAC_PI_2).abs() < EPSILON);
    }

    #[test]
    fn test_add_operator() {
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        let result = q1 + q2;
        assert_eq!(result, Quaternion::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn test_sub_operator() {
        let q1 = Quaternion::new(10.0, 12.0, 14.0, 16.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        let result = q1 - q2;
        assert_eq!(result, Quaternion::new(5.0, 6.0, 7.0, 8.0));
    }

    #[test]
    fn test_scalar_mul() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let result1 = q * 2.0;
        let result2 = 2.0 * q;
        assert_eq!(result1, Quaternion::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(result2, result1);
    }

    #[test]
    fn test_scalar_div() {
        let q = Quaternion::new(6.0, 9.0, 12.0, 15.0);
        let result = q / 3.0;
        assert_eq!(result, Quaternion::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_add_assign() {
        let mut q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        q += Quaternion::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q, Quaternion::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_sub_assign() {
        let mut q = Quaternion::new(2.0, 3.0, 4.0, 5.0);
        q -= Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(q, Quaternion::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        q *= 2.0;
        assert_eq!(q, Quaternion::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_div_assign() {
        let mut q = Quaternion::new(6.0, 9.0, 12.0, 15.0);
        q /= 3.0;
        assert_eq!(q, Quaternion::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mul_assign_quaternion() {
        let mut q = Quaternion::identity();
        q *= Quaternion::identity();
        assert_eq!(q, Quaternion::identity());
    }

    #[test]
    fn test_conversion_array() {
        let arr: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let q: Quaternion = arr.into();
        let arr_back: [f32; 4] = q.into();
        assert_eq!(arr, arr_back);
    }

    #[test]
    fn test_conversion_vec4() {
        let vec4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let q: Quaternion = vec4.into();
        let vec4_back: Vec4 = q.into();
        assert_eq!(vec4, vec4_back);
    }

    #[test]
    fn test_chainable_methods() {
        let result = Quaternion::new(1.0, 2.0, 3.0, 4.0)
            .with_normalize()
            .with_conjugate()
            .with_inverse();

        assert!(result.is_normalized());
    }

    #[test]
    fn test_rotation_180_degrees() {
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vec3(v);

        assert!((rotated.x + 1.0).abs() < EPSILON);
        assert!(rotated.y.abs() < EPSILON);
        assert!(rotated.z.abs() < EPSILON);
    }

    #[test]
    fn test_reference_operators() {
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);

        let add1 = q1 + q2;
        let add2 = q1 + q2;
        let add3 = q1 + q2;
        assert_eq!(add1, add2);
        assert_eq!(add2, add3);
    }

    #[test]
    fn test_length_and_length_squared() {
        let q = Quaternion::new(1.0, 2.0, 2.0, 4.0);
        assert_eq!(q.length_squared(), 25.0);
        assert_eq!(q.length(), 5.0);
    }

    #[test]
    fn test_deref() {
        let q = Quaternion::identity();
        let v: &Vec4 = &q;
        assert_eq!(*v, Vec4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_with_conjugate() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).with_conjugate();
        assert_eq!(q, Quaternion::new(-1.0, -2.0, -3.0, 4.0));
    }

    #[test]
    fn test_with_inverse() {
        let q = Quaternion::identity().with_inverse();
        assert_eq!(q, Quaternion::identity());
    }

    #[test]
    fn test_with_slerp() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::identity();
        let result = q1.with_slerp(&q2, 0.5);
        // Should be same as identity when interpolating between identical quaternions
        assert!(result.is_normalized());
    }

    #[test]
    fn test_from_axis_angle_normalization() {
        // Test that from_axis_angle normalizes the input axis
        let axis = Vec3::new(1.0, 1.0, 1.0); // Not normalized
        let q = Quaternion::from_axis_angle(axis, FRAC_PI_2);
        assert!(q.is_normalized());
    }

    #[test]
    fn test_hamilton_product() {
        // Test that quaternion multiplication follows Hamilton product rules
        let q1 = Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_4);
        let q2 = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_4);

        // q1 * q2 should not equal q2 * q1 (non-commutative)
        let result1 = q1 * q2;
        let result2 = q2 * q1;

        // They should be different
        assert!(
            (result1.0.x - result2.0.x).abs() > EPSILON
                || (result1.0.y - result2.0.y).abs() > EPSILON
                || (result1.0.z - result2.0.z).abs() > EPSILON
                || (result1.0.w - result2.0.w).abs() > EPSILON
        );
    }

    #[test]
    fn test_to_matrix4_identity() {
        let q = Quaternion::identity();
        let m = q.to_matrix4();
        let identity = Matrix4::identity();

        // Check that identity quaternion produces identity matrix
        assert!((m.row_x.x - identity.row_x.x).abs() < EPSILON);
        assert!((m.row_y.y - identity.row_y.y).abs() < EPSILON);
        assert!((m.row_z.z - identity.row_z.z).abs() < EPSILON);
        assert!((m.row_w.w - identity.row_w.w).abs() < EPSILON);
    }

    #[test]
    fn test_to_matrix4_rotation_y_90() {
        // 90-degree rotation around Y-axis
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);
        let m = q.to_matrix4();

        // Transform point (1, 0, 0) should give approximately (0, 0, -1)
        let point = Vec3::new(1.0, 0.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(transformed.x.abs() < EPSILON);
        assert!(transformed.y.abs() < EPSILON);
        assert!((transformed.z + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_to_matrix4_rotation_z_90() {
        // 90-degree rotation around Z-axis
        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        let m = q.to_matrix4();

        // Transform point (1, 0, 0) should give approximately (0, 1, 0)
        let point = Vec3::new(1.0, 0.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(transformed.x.abs() < EPSILON);
        assert!((transformed.y - 1.0).abs() < EPSILON);
        assert!(transformed.z.abs() < EPSILON);
    }

    #[test]
    fn test_to_matrix4_orthonormality() {
        // Any rotation should produce an orthonormal matrix
        let q = Quaternion::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).with_normalize(), 0.7);
        let m = q.to_matrix4();

        // Extract rotation part (3x3)
        let col0 = Vec3::new(m.row_x.x, m.row_y.x, m.row_z.x);
        let col1 = Vec3::new(m.row_x.y, m.row_y.y, m.row_z.y);
        let col2 = Vec3::new(m.row_x.z, m.row_y.z, m.row_z.z);

        // Columns should be unit length
        assert!((col0.length() - 1.0).abs() < EPSILON);
        assert!((col1.length() - 1.0).abs() < EPSILON);
        assert!((col2.length() - 1.0).abs() < EPSILON);

        // Columns should be orthogonal (dot product = 0)
        assert!(col0.dot(&col1).abs() < EPSILON);
        assert!(col1.dot(&col2).abs() < EPSILON);
        assert!(col2.dot(&col0).abs() < EPSILON);
    }

    #[test]
    fn test_to_matrix4_preserves_rotation() {
        // Compare quaternion rotation with matrix rotation
        let axis = Vec3::new(1.0, 2.0, 3.0).with_normalize();
        let angle = 1.2;
        let q = Quaternion::from_axis_angle(axis, angle);
        let m = q.to_matrix4();

        let test_points = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0),
        ];

        for point in test_points {
            let rotated_quat = q.rotate_vec3(point);
            let rotated_matrix = m.transform_point(point);

            assert!((rotated_quat.x - rotated_matrix.x).abs() < EPSILON);
            assert!((rotated_quat.y - rotated_matrix.y).abs() < EPSILON);
            assert!((rotated_quat.z - rotated_matrix.z).abs() < EPSILON);
        }
    }

    #[test]
    fn test_row_major_sanity_check() {
        use super::super::vec::Vec3;
        const EPSILON: f32 = 1e-6;

        // Test 1: Verify Vec4 * Matrix4 multiplication is row-vector multiplication
        // For row-major with row-vector multiplication: result = v * M
        // This means: result[i] = sum(v[j] * M[j][i])

        // Create a simple translation matrix
        let translation = Matrix4::from_translation(Vec3::new(5.0, 10.0, 15.0));
        let point = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let result = point * translation;

        // Manual calculation: v * M where M has translation in row_w
        // result.x = 1*1 + 2*0 + 3*0 + 1*5 = 6
        // result.y = 1*0 + 2*1 + 3*0 + 1*10 = 12
        // result.z = 1*0 + 2*0 + 3*1 + 1*15 = 18
        // result.w = 1*0 + 2*0 + 3*0 + 1*1 = 1
        assert!((result.x - 6.0).abs() < EPSILON);
        assert!((result.y - 12.0).abs() < EPSILON);
        assert!((result.z - 18.0).abs() < EPSILON);
        assert!((result.w - 1.0).abs() < EPSILON);

        // Test 2: Verify quaternion-to-matrix conversion matches expected rotation
        // 90-degree rotation around Y-axis should:
        // - Transform (1,0,0) -> (0,0,-1)
        // - Transform (0,1,0) -> (0,1,0) [unchanged]
        // - Transform (0,0,1) -> (1,0,0)

        let q = Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
        let m = q.to_matrix4();

        // Test X-axis vector
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let rotated = m.transform_point(x_axis);
        assert!((rotated.x - 0.0).abs() < EPSILON);
        assert!((rotated.y - 0.0).abs() < EPSILON);
        assert!((rotated.z - (-1.0)).abs() < EPSILON);

        // Test Z-axis vector
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let rotated = m.transform_point(z_axis);
        assert!((rotated.x - 1.0).abs() < EPSILON);
        assert!((rotated.y - 0.0).abs() < EPSILON);
        assert!((rotated.z - 0.0).abs() < EPSILON);

        // Test 3: Verify matrix storage convention
        // Despite "row_*" naming, verify the storage matches our multiplication
        let rotation_z = Matrix4::from_rotation_z(std::f32::consts::FRAC_PI_2);

        // For 90° Z-rotation, the matrix should rotate (1,0,0) -> (0,1,0)
        let x = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotation_z.transform_point(x);
        assert!((rotated.x - 0.0).abs() < EPSILON);
        assert!((rotated.y - 1.0).abs() < EPSILON);
        assert!((rotated.z - 0.0).abs() < EPSILON);

        // Test 4: Verify quaternion and matrix produce identical results
        // for arbitrary rotation
        let axis = Vec3::new(1.0, 2.0, 3.0).with_normalize();
        let angle = 1.234;

        let q = Quaternion::from_axis_angle(axis, angle);
        let m = q.to_matrix4();

        let test_point = Vec3::new(4.0, -2.0, 7.0);
        let rotated_q = q.rotate_vec3(test_point);
        let rotated_m = m.transform_point(test_point);

        assert!((rotated_q.x - rotated_m.x).abs() < EPSILON);
        assert!((rotated_q.y - rotated_m.y).abs() < EPSILON);
        assert!((rotated_q.z - rotated_m.z).abs() < EPSILON);

        // Test 5: Verify matrix composition order
        // Rotation then translation
        let rotation =
            Quaternion::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
        let m_rotation = rotation.to_matrix4();
        let m_translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));

        // Apply rotation first, then translation
        let combined = m_rotation * m_translation;

        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = combined.transform_point(point);

        // First rotate (1,0,0) -> (0,0,-1), then translate by (10,0,0) -> (10,0,-1)
        assert!((result.x - 10.0).abs() < EPSILON);
        assert!((result.y - 0.0).abs() < EPSILON);
        assert!((result.z - (-1.0)).abs() < EPSILON);
    }
}
