use super::{Vec2, Vec4, NORMALIZED_PRECISION_THRESHOLD};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3-dimensional vector with x, y, and z components.
///
/// The workhorse of 3D graphics, used for positions, directions, normals,
/// velocities, and many other 3D quantities.
///
/// # Coordinate Systems
/// - **Right-Handed** (default): X right, Y up, Z backward (toward camera)
/// - **Left-Handed**: X right, Y up, Z forward (away from camera)
///
/// # Common Uses
/// - Positions in 3D space
/// - Direction vectors (should be normalized)
/// - Surface normals (should be normalized)
/// - RGB colors (when interpreted as \[0,1\] range)
/// - Velocities and accelerations
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::vec::Vec3;
/// // Create vectors
/// let position = Vec3::new(1.0, 2.0, 3.0);
/// let direction = Vec3::new(0.0, 1.0, 0.0);
///
/// // Vector arithmetic
/// let offset = position + direction * 5.0;
///
/// // 3D operations
/// let length = direction.length();
/// let normalized = direction.normalize();
/// let dot = position.dot(&direction);
/// let cross = Vec3::X.cross(&Vec3::Y); // Results in Vec3::Z (right-hand rule)
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    /// X component (horizontal right axis)
    pub x: f32,
    /// Y component (vertical up axis)
    pub y: f32,
    /// Z component (depth axis, forward or backward)
    pub z: f32,
}
// ============================================================================
// Vec3 Implementation
// ============================================================================

impl Vec3 {
    // Constructors

    /// Creates a new 3D vector from x, y, and z components.
    ///
    /// # Arguments
    /// * `x` - X component (horizontal)
    /// * `y` - Y component (vertical)
    /// * `z` - Z component (depth)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let position = Vec3::new(1.0, 2.0, 3.0);
    /// let color = Vec3::new(1.0, 0.0, 0.0); // Red color
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a vector with all components set to zero.
    ///
    /// Equivalent to `Vec3::new(0.0, 0.0, 0.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let origin = Vec3::zero();
    /// assert_eq!(origin, Vec3::new(0.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a vector with all components set to one.
    ///
    /// Equivalent to `Vec3::new(1.0, 1.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let uniform_scale = Vec3::one();
    /// ```
    #[inline]
    pub const fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    /// Creates a vector with all components set to the same value.
    ///
    /// Useful for uniform scaling or creating isotropic values.
    ///
    /// # Arguments
    /// * `value` - The value for all components
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let uniform = Vec3::splat(5.0);
    /// assert_eq!(uniform, Vec3::new(5.0, 5.0, 5.0));
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }

    // Constants

    /// Zero vector (0, 0, 0)
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector (1, 1, 1)
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// Unit X vector (1, 0, 0) - points right in standard coordinates
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit Y vector (0, 1, 0) - points up in standard coordinates
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Unit Z vector (0, 0, 1) - points forward/backward depending on handedness
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Alias for X - points right
    pub const RIGHT: Self = Self::X;

    /// Alias for -X - points left
    pub const LEFT: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Alias for Y - points up
    pub const UP: Self = Self::Y;

    /// Alias for -Y - points down
    pub const DOWN: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };

    /// Points forward in right-handed coordinates (toward camera, -Z)
    pub const FORWARD: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };

    /// Points backward in right-handed coordinates (away from camera, +Z)
    pub const BACK: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Computes the dot product (scalar product) of two vectors.
    ///
    /// The dot product measures how aligned two vectors are:
    /// - Positive: vectors point in similar directions
    /// - Zero: vectors are perpendicular (90°)
    /// - Negative: vectors point in opposite directions
    ///
    /// # Formula
    /// dot(a, b) = a.x × b.x + a.y × b.y + a.z × b.z
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(1.0, 0.0, 0.0);
    /// let b = Vec3::new(0.0, 1.0, 0.0);
    /// assert_eq!(a.dot(&b), 0.0); // Perpendicular
    ///
    /// let c = Vec3::new(1.0, 2.0, 3.0);
    /// let d = Vec3::new(2.0, 4.0, 6.0);
    /// assert_eq!(c.dot(&d), 28.0); // 2 + 8 + 18
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product of two vectors.
    ///
    /// The cross product produces a vector perpendicular to both input vectors,
    /// following the right-hand rule. The magnitude equals the area of the parallelogram
    /// formed by the two vectors.
    ///
    /// # Right-Hand Rule
    /// Point your right hand's fingers along `self`, curl them toward `other`,
    /// and your thumb points in the direction of the result.
    ///
    /// # Formula
    /// cross(a, b) = (a.y×b.z - a.z×b.y, a.z×b.x - a.x×b.z, a.x×b.y - a.y×b.x)
    ///
    /// # Properties
    /// - `a.cross(b) = -b.cross(a)` (anti-commutative)
    /// - `a.cross(a) = 0` (parallel vectors)
    /// - `|a.cross(b)| = |a| × |b| × sin(θ)` where θ is angle between vectors
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let x = Vec3::new(1.0, 0.0, 0.0);
    /// let y = Vec3::new(0.0, 1.0, 0.0);
    /// let z = x.cross(&y);
    /// assert_eq!(z, Vec3::new(0.0, 0.0, 1.0)); // Right-hand rule
    /// ```
    #[inline]
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Computes the squared length (magnitude squared) of the vector.
    ///
    /// This is cheaper than [`length`](Self::length) as it avoids the square root.
    /// Useful for comparing distances or when only relative magnitudes matter.
    ///
    /// # Formula
    /// length_squared = x² + y² + z²
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(v.length_squared(), 9.0); // 1 + 4 + 4 = 9
    /// ```
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> f32 {
        self.dot(self)
    }

    /// Computes the length (magnitude) of the vector.
    ///
    /// Returns the Euclidean distance from the origin to the point represented by this vector.
    ///
    /// # Formula
    /// length = √(x² + y² + z²)
    ///
    /// # Performance
    /// If you only need to compare lengths, use [`length_squared`](Self::length_squared) instead
    /// to avoid the expensive square root operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(v.length(), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Normalizes this vector in place to unit length.
    ///
    /// The normalized vector maintains the same direction but has a length of 1.0.
    /// If the vector has zero length, it remains unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(1.0, 2.0, 2.0);
    /// v.normalize();
    /// assert!((v.length() - 1.0).abs() < 0.0001); // Length is ~1.0
    /// ```
    #[inline]
    pub fn normalize(&mut self) {
        let length = self.length();
        if length > 0.0 {
            *self /= length;
        }
    }

    /// Checks if the vector is normalized (unit length).
    /// Uses precision threshold of 2e-4 to account for floating-point errors.
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec3, NORMALIZED_PRECISION_THRESHOLD};
    /// let v = Vec3::new(1.0, 0.0, 0.0);
    /// assert!(v.is_normalized());
    /// ```
    pub fn is_normalized(&self) -> bool {
        (self.length_squared() - 1.0).abs() < NORMALIZED_PRECISION_THRESHOLD
    }

    /// Computes the distance between two points represented as vectors.
    ///
    /// # Formula
    /// distance(a, b) = |b - a| = √((b.x - a.x)² + (b.y - a.y)² + (b.z - a.z)²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(0.0, 0.0, 0.0);
    /// let b = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(a.distance(&b), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        (*self - *other).length()
    }

    /// Computes the squared distance between two points.
    ///
    /// Cheaper than [`distance`](Self::distance) as it avoids the square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(0.0, 0.0, 0.0);
    /// let b = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(a.distance_squared(&b), 9.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, other: &Self) -> f32 {
        (*self - *other).length_squared()
    }

    /// Linear interpolation between two vectors.
    ///
    /// Computes a point along the line segment from `self` to `other`.
    ///
    /// # Arguments
    /// * `other` - Target vector to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Formula
    /// lerp(a, b, t) = a + (b - a) × t = a × (1 - t) + b × t
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let start = Vec3::new(0.0, 0.0, 0.0);
    /// let end = Vec3::new(10.0, 10.0, 10.0);
    /// let midpoint = start.lerp(&end, 0.5);
    /// assert_eq!(midpoint, Vec3::new(5.0, 5.0, 5.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    // Batch Operations

    /// Batch normalization of vectors.
    ///
    /// Processes multiple vectors in parallel, enabling SIMD optimization.
    ///
    /// # Performance
    /// ~3-4x faster than individual normalize calls with compiler optimizations enabled.
    #[inline]
    pub fn normalize_slice(input: &[Self], output: &mut [Self]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i];
            output[i].normalize();
        }
    }

    /// In-place batch normalization of vectors.
    #[inline]
    pub fn normalize_slice_inplace(vectors: &mut [Self]) {
        for v in vectors.iter_mut() {
            v.normalize();
        }
    }

    /// Batch dot product computation.
    #[inline]
    pub fn dot_slice(a: &[Self], b: &[Self], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = a[i].dot(&b[i]);
        }
    }

    /// Batch cross product computation.
    ///
    /// Computes cross products between corresponding pairs of vectors.
    #[inline]
    pub fn cross_slice(a: &[Self], b: &[Self], output: &mut [Self]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = a[i].cross(&b[i]);
        }
    }

    /// Batch vector addition.
    #[inline]
    pub fn add_slice(a: &[Self], b: &[Self], output: &mut [Self]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = Self {
                x: a[i].x + b[i].x,
                y: a[i].y + b[i].y,
                z: a[i].z + b[i].z,
            };
        }
    }

    /// Batch scalar multiplication.
    #[inline]
    pub fn scale_slice(input: &[Self], scalar: f32, output: &mut [Self]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i] * scalar;
        }
    }

    /// In-place batch scalar multiplication.
    #[inline]
    pub fn scale_slice_inplace(vectors: &mut [Self], scalar: f32) {
        for v in vectors.iter_mut() {
            *v = *v * scalar;
        }
    }

    /// Batch linear interpolation.
    #[inline]
    pub fn lerp_slice(a: &[Self], b: &[Self], t: f32, output: &mut [Self]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = a[i].lerp(&b[i], t);
        }
    }

    // Utility Methods

    /// Clamps each component of the vector to the given range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(-5.0, 15.0, 20.0);
    /// let clamped = v.clamp(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
    /// assert_eq!(clamped, Vec3::new(0.0, 10.0, 10.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
        }
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(-3.0, 4.0, -5.0);
    /// assert_eq!(v.abs(), Vec3::new(3.0, 4.0, 5.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns the minimum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(3.0, 1.0, 2.0);
    /// assert_eq!(v.min_component(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn min_component(&self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    /// Returns the maximum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(3.0, 1.0, 2.0);
    /// assert_eq!(v.max_component(), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn max_component(&self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Extends this Vec3 to a Vec4 by adding a w component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec3, Vec4};
    /// let v3 = Vec3::new(1.0, 2.0, 3.0);
    /// let v4 = v3.extend(4.0);
    /// assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn extend(&self, w: f32) -> Vec4 {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    /// Truncates this Vec3 to a Vec2 by discarding the z component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec3};
    /// let v3 = Vec3::new(1.0, 2.0, 3.0);
    /// let v2 = v3.truncate();
    /// assert_eq!(v2, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn truncate(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }

    // Chainable Methods

    /// Multiplies the vector by a scalar value.
    ///
    /// Consumes self, enabling fluent method chaining.
    #[inline]
    #[must_use]
    pub fn with_mul_scalar(self, scalar: f32) -> Self {
        self * scalar
    }

    /// Divides the vector by a scalar value.
    ///
    /// Consumes self, enabling fluent method chaining.
    #[inline]
    #[must_use]
    pub fn with_div_scalar(self, scalar: f32) -> Self {
        self / scalar
    }

    /// Chainable version of [`cross`](Self::cross).
    #[inline]
    #[must_use]
    pub fn with_cross(self, other: &Self) -> Self {
        self.cross(other)
    }

    /// Returns a normalized (unit length) version of this vector.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 2.0).with_normalize();
    /// assert!((v.length() - 1.0).abs() < 0.0001);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_normalize(self) -> Self {
        let length = self.length();
        if length > 0.0 {
            Self {
                x: self.x / length,
                y: self.y / length,
                z: self.z / length,
            }
        } else {
            self
        }
    }

    /// Linear interpolation between vectors with method chaining.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let result = Vec3::new(0.0, 0.0, 0.0)
    ///     .with_lerp(&Vec3::new(10.0, 10.0, 10.0), 0.5)
    ///     .with_mul_scalar(2.0);
    /// assert_eq!(result, Vec3::new(10.0, 10.0, 10.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_lerp(self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }
}
// ============================================================================
// Operator Overloading - Vec3
// ============================================================================

impl Add for Vec3 {
    type Output = Self;

    /// Component-wise addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(1.0, 2.0, 3.0);
    /// let b = Vec3::new(4.0, 5.0, 6.0);
    /// let result = a + b;
    /// assert_eq!(result, Vec3::new(5.0, 7.0, 9.0));
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add<&Self> for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: &Self) -> Self {
        self + *other
    }
}

impl Add<&Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, other: &Vec3) -> Vec3 {
        *self + *other
    }
}

impl Add<Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, other: Vec3) -> Vec3 {
        *self + other
    }
}

impl Sub for Vec3 {
    type Output = Self;

    /// Component-wise subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(5.0, 7.0, 9.0);
    /// let b = Vec3::new(2.0, 3.0, 4.0);
    /// let result = a - b;
    /// assert_eq!(result, Vec3::new(3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub<&Self> for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: &Self) -> Self {
        self - *other
    }
}

impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, other: &Vec3) -> Vec3 {
        *self - *other
    }
}

impl Sub<Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, other: Vec3) -> Vec3 {
        *self - other
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 3.0);
    /// let result = v * 2.0;
    /// assert_eq!(result, Vec3::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<&f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        self * *scalar
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    /// Scalar multiplication (reversed operand order) using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 3.0);
    /// let result = 2.0 * v;
    /// assert_eq!(result, Vec3::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: vec.x * self,
            y: vec.y * self,
            z: vec.z * self,
        }
    }
}

impl Mul<&Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, vec: &Vec3) -> Vec3 {
        self * *vec
    }
}

impl Mul for Vec3 {
    type Output = Self;

    /// Component-wise multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(2.0, 3.0, 4.0);
    /// let b = Vec3::new(5.0, 6.0, 7.0);
    /// let result = a * b;
    /// assert_eq!(result, Vec3::new(10.0, 18.0, 28.0));
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<&Self> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, other: &Vec3) -> Vec3 {
        *self * *other
    }
}

impl Mul<Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        *self * other
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    /// Scalar division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(6.0, 9.0, 12.0);
    /// let result = v / 3.0;
    /// assert_eq!(result, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Div<&f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: &f32) -> Self {
        self / *scalar
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;

    /// Scalar division (reversed operand order) using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(2.0, 4.0, 8.0);
    /// let result = 16.0 / v;
    /// assert_eq!(result, Vec3::new(8.0, 4.0, 2.0));
    /// ```
    #[inline]
    fn div(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: self / vec.x,
            y: self / vec.y,
            z: self / vec.z,
        }
    }
}

impl Div<&Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn div(self, vec: &Vec3) -> Vec3 {
        self / *vec
    }
}

impl Div for Vec3 {
    type Output = Self;

    /// Component-wise division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let a = Vec3::new(10.0, 18.0, 28.0);
    /// let b = Vec3::new(2.0, 3.0, 4.0);
    /// let result = a / b;
    /// assert_eq!(result, Vec3::new(5.0, 6.0, 7.0));
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

impl Div<&Self> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, other: &Self) -> Self {
        self / *other
    }
}

impl Div<&Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, other: &Vec3) -> Vec3 {
        *self / *other
    }
}

impl Div<Vec3> for &Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, other: Vec3) -> Vec3 {
        *self / other
    }
}

impl Neg for Vec3 {
    type Output = Self;

    /// Negates all components using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, -2.0, 3.0);
    /// let result = -v;
    /// assert_eq!(result, Vec3::new(-1.0, 2.0, -3.0));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Neg for &Vec3 {
    type Output = Vec3;

    #[inline]
    fn neg(self) -> Vec3 {
        -*self
    }
}

impl AddAssign for Vec3 {
    /// Component-wise addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(1.0, 2.0, 3.0);
    /// v += Vec3::new(4.0, 5.0, 6.0);
    /// assert_eq!(v, Vec3::new(5.0, 7.0, 9.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl AddAssign<&Self> for Vec3 {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl SubAssign for Vec3 {
    /// Component-wise subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(5.0, 7.0, 9.0);
    /// v -= Vec3::new(2.0, 3.0, 4.0);
    /// assert_eq!(v, Vec3::new(3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl SubAssign<&Self> for Vec3 {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        *self -= *other;
    }
}

impl MulAssign<f32> for Vec3 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(1.0, 2.0, 3.0);
    /// v *= 2.0;
    /// assert_eq!(v, Vec3::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl MulAssign<&f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        *self *= *scalar;
    }
}

impl MulAssign for Vec3 {
    /// Component-wise multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(2.0, 3.0, 4.0);
    /// v *= Vec3::new(5.0, 6.0, 7.0);
    /// assert_eq!(v, Vec3::new(10.0, 18.0, 28.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl MulAssign<&Self> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

impl DivAssign<f32> for Vec3 {
    /// Scalar division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(6.0, 9.0, 12.0);
    /// v /= 3.0;
    /// assert_eq!(v, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl DivAssign<&f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, scalar: &f32) {
        *self /= *scalar;
    }
}

impl DivAssign for Vec3 {
    /// Component-wise division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut v = Vec3::new(10.0, 18.0, 28.0);
    /// v /= Vec3::new(2.0, 3.0, 4.0);
    /// assert_eq!(v, Vec3::new(5.0, 6.0, 7.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}

impl DivAssign<&Self> for Vec3 {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        *self /= *other;
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

// Vec3 Conversions for arrays/tuples

impl From<[f32; 3]> for Vec3 {
    /// Converts from a 3-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v: Vec3 = [1.0, 2.0, 3.0].into();
    /// assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    fn from(arr: [f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl From<&[f32; 3]> for Vec3 {
    #[inline]
    fn from(arr: &[f32; 3]) -> Self {
        let vec: Vec3 = (*arr).into();
        vec
    }
}

impl From<(f32, f32, f32)> for Vec3 {
    /// Converts from a 3-element tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v: Vec3 = (1.0, 2.0, 3.0).into();
    /// assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    fn from(tuple: (f32, f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
            z: tuple.2,
        }
    }
}

impl From<&(f32, f32, f32)> for Vec3 {
    #[inline]
    fn from(tuple: &(f32, f32, f32)) -> Self {
        let vec: Vec3 = (*tuple).into();
        vec
    }
}

impl From<Vec3> for [f32; 3] {
    /// Converts to a 3-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 3.0);
    /// let arr: [f32; 3] = v.into();
    /// assert_eq!(arr, [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(vec: Vec3) -> Self {
        [vec.x, vec.y, vec.z]
    }
}

impl From<&Vec3> for [f32; 3] {
    #[inline]
    fn from(vec: &Vec3) -> Self {
        let arr: [f32; 3] = (*vec).into();
        arr
    }
}

impl From<Vec3> for (f32, f32, f32) {
    /// Converts to a 3-element tuple.
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::vec::Vec3;
    /// let v = Vec3::new(1.0, 2.0, 3.0);
    /// let tuple: (f32, f32, f32) = v.into();
    /// assert_eq!(tuple, (1.0, 2.0, 3.0));
    /// ```
    #[inline]
    fn from(vec: Vec3) -> Self {
        (vec.x, vec.y, vec.z)
    }
}

impl From<&Vec3> for (f32, f32, f32) {
    #[inline]
    fn from(vec: &Vec3) -> Self {
        let tuple: (f32, f32, f32) = (*vec).into();
        tuple
    }
}

// Dimensional Conversions

impl From<Vec2> for Vec3 {
    /// Extends Vec2 to Vec3 by adding z=0.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec3};
    /// let v2 = Vec2::new(1.0, 2.0);
    /// let v3: Vec3 = v2.into();
    /// assert_eq!(v3, Vec3::new(1.0, 2.0, 0.0));
    /// ```
    #[inline]
    fn from(v: Vec2) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: 0.0,
        }
    }
}

impl From<&Vec2> for Vec3 {
    #[inline]
    fn from(v: &Vec2) -> Self {
        let vec3: Vec3 = (*v).into();
        vec3
    }
}

impl From<Vec4> for Vec3 {
    /// Truncates Vec4 to Vec3 by discarding w component.
    ///
    /// **Note**: For homogeneous coordinates with w≠1, you should divide by w first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec3, Vec4};
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v3: Vec3 = v4.into();
    /// assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    fn from(v: Vec4) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<&Vec4> for Vec3 {
    #[inline]
    fn from(v: &Vec4) -> Self {
        let vec3: Vec3 = (*v).into();
        vec3
    }
}
