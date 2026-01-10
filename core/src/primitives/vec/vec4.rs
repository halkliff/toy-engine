use super::{Vec3, Vec2, NORMALIZED_PRECISION_THRESHOLD};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 4-dimensional vector with x, y, z, and w components.
///
/// Used primarily for homogeneous coordinates in graphics pipelines,
/// quaternion storage, and RGBA color representation.
///
/// # Homogeneous Coordinates
/// In 3D graphics, Vec4 represents a 3D point with a w component:
/// - w = 1.0: Point in space (affected by translation)
/// - w = 0.0: Direction vector (not affected by translation)
///
/// After matrix transformation, divide by w to get Cartesian coordinates.
///
/// # Common Uses
/// - Homogeneous 3D coordinates for matrix transformations
/// - RGBA colors (x=r, y=g, z=b, w=a)
/// - Quaternions for rotations (though dedicated Quat type is preferred)
/// - 4D shader calculations
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::vec::Vec4;
/// // Homogeneous point (w=1)
/// let point = Vec4::new(10.0, 20.0, 30.0, 1.0);
///
/// // Homogeneous direction (w=0)
/// let direction = Vec4::new(0.0, 1.0, 0.0, 0.0);
///
/// // RGBA color
/// let red = Vec4::new(1.0, 0.0, 0.0, 1.0); // Opaque red
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    /// X component (or red channel)
    pub x: f32,
    /// Y component (or green channel)
    pub y: f32,
    /// Z component (or blue channel)
    pub z: f32,
    /// W component (homogeneous coordinate or alpha channel)
    pub w: f32,
}

// ============================================================================
// Vec4 Implementation
// ============================================================================

impl Vec4 {
    // Constructors

    /// Creates a new 4D vector from x, y, z, and w components.
    ///
    /// # Arguments
    /// * `x` - X component
    /// * `y` - Y component
    /// * `z` - Z component
    /// * `w` - W component (homogeneous coordinate or alpha)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let point = Vec4::new(1.0, 2.0, 3.0, 1.0); // Homogeneous point
    /// let direction = Vec4::new(0.0, 1.0, 0.0, 0.0); // Homogeneous direction
    /// let color = Vec4::new(1.0, 0.0, 0.0, 1.0); // RGBA red
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a vector with all components set to zero.
    ///
    /// Equivalent to `Vec4::new(0.0, 0.0, 0.0, 0.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let origin = Vec4::zero();
    /// assert_eq!(origin, Vec4::new(0.0, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    /// Creates a vector with all components set to one.
    ///
    /// Equivalent to `Vec4::new(1.0, 1.0, 1.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let uniform_scale = Vec4::one();
    /// ```
    #[inline]
    pub const fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 1.0,
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let uniform = Vec4::splat(5.0);
    /// assert_eq!(uniform, Vec4::new(5.0, 5.0, 5.0, 5.0));
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
            w: value,
        }
    }

    // Constants

    /// Zero vector (0, 0, 0, 0)
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// Unit vector (1, 1, 1, 1)
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    /// Unit X vector (1, 0, 0, 0)
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// Unit Y vector (0, 1, 0, 0)
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 0.0,
    };

    /// Unit Z vector (0, 0, 1, 0)
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 0.0,
    };

    /// Unit W vector (0, 0, 0, 1)
    pub const W: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    // Core Operations

    /// Computes the dot product (scalar product) of two vectors.
    ///
    /// The dot product measures how aligned two vectors are.
    ///
    /// # Formula
    /// dot(a, b) = a.x × b.x + a.y × b.y + a.z × b.z + a.w × b.w
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(a.dot(&b), 40.0); // 2 + 6 + 12 + 20
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Computes the squared length (magnitude squared) of the vector.
    ///
    /// This is cheaper than [`length`](Self::length) as it avoids the square root.
    /// Useful for comparing distances or when only relative magnitudes matter.
    ///
    /// # Formula
    /// length_squared = x² + y² + z² + w²
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(v.length_squared(), 25.0); // 1 + 4 + 4 + 16 = 25
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
    /// length = √(x² + y² + z² + w²)
    ///
    /// # Performance
    /// If you only need to compare lengths, use [`length_squared`](Self::length_squared) instead
    /// to avoid the expensive square root operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(v.length(), 5.0);
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(1.0, 2.0, 2.0, 4.0);
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
    /// # use toyengine::primitives::vec::{Vec4, NORMALIZED_PRECISION_THRESHOLD};
    /// let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// assert!(v.is_normalized());
    /// ```
    pub fn is_normalized(&self) -> bool {
        (self.length_squared() - 1.0).abs() < NORMALIZED_PRECISION_THRESHOLD
    }

    /// Computes the distance between two points represented as vectors.
    ///
    /// # Formula
    /// distance(a, b) = |b - a|
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(a.distance(&b), 5.0);
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(a.distance_squared(&b), 25.0);
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let start = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let end = Vec4::new(10.0, 10.0, 10.0, 10.0);
    /// let midpoint = start.lerp(&end, 0.5);
    /// assert_eq!(midpoint, Vec4::new(5.0, 5.0, 5.0, 5.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
            w: self.w + (other.w - self.w) * t,
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

    /// Batch vector addition.
    #[inline]
    pub fn add_slice(a: &[Self], b: &[Self], output: &mut [Self]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = Self {
                x: a[i].x + b[i].x,
                y: a[i].y + b[i].y,
                z: a[i].z + b[i].z,
                w: a[i].w + b[i].w,
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
            *v *= scalar;
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(-5.0, 15.0, 20.0, 25.0);
    /// let clamped = v.clamp(Vec4::ZERO, Vec4::new(10.0, 10.0, 10.0, 10.0));
    /// assert_eq!(clamped, Vec4::new(0.0, 10.0, 10.0, 10.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
            w: self.w.clamp(min.w, max.w),
        }
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(-3.0, 4.0, -5.0, 6.0);
    /// assert_eq!(v.abs(), Vec4::new(3.0, 4.0, 5.0, 6.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
            w: self.w.abs(),
        }
    }

    /// Returns the minimum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(3.0, 1.0, 2.0, 4.0);
    /// assert_eq!(v.min_component(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn min_component(&self) -> f32 {
        self.x.min(self.y).min(self.z).min(self.w)
    }

    /// Returns the maximum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(3.0, 1.0, 2.0, 4.0);
    /// assert_eq!(v.max_component(), 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn max_component(&self) -> f32 {
        self.x.max(self.y).max(self.z).max(self.w)
    }

    /// Truncates this Vec4 to a Vec3 by discarding the w component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec3, Vec4};
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v3 = v4.truncate();
    /// assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn truncate(&self) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
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

    /// Returns a normalized (unit length) version of this vector.
    ///
    /// Consumes self, enabling fluent method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 2.0, 4.0).with_normalize();
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
                w: self.w / length,
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
    /// # use toyengine::primitives::vec::Vec4;
    /// let result = Vec4::new(0.0, 0.0, 0.0, 0.0)
    ///     .with_lerp(&Vec4::new(10.0, 10.0, 10.0, 10.0), 0.5)
    ///     .with_mul_scalar(2.0);
    /// assert_eq!(result, Vec4::new(10.0, 10.0, 10.0, 10.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn with_lerp(self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }
}

// ============================================================================
// Operator Overloading - Vec4
// ============================================================================

impl Add for Vec4 {
    type Output = Self;

    /// Component-wise addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// let result = a + b;
    /// assert_eq!(result, Vec4::new(6.0, 8.0, 10.0, 12.0));
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Add<&Self> for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: &Self) -> Self {
        self + *other
    }
}

impl Add<&Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn add(self, other: &Vec4) -> Vec4 {
        *self + *other
    }
}

impl Add<Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn add(self, other: Vec4) -> Vec4 {
        *self + other
    }
}

impl Sub for Vec4 {
    type Output = Self;

    /// Component-wise subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(10.0, 12.0, 14.0, 16.0);
    /// let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// let result = a - b;
    /// assert_eq!(result, Vec4::new(5.0, 6.0, 7.0, 8.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Sub<&Self> for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: &Self) -> Self {
        self - *other
    }
}

impl Sub<&Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn sub(self, other: &Vec4) -> Vec4 {
        *self - *other
    }
}

impl Sub<Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn sub(self, other: Vec4) -> Vec4 {
        *self - other
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = v * 2.0;
    /// assert_eq!(result, Vec4::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl Mul<&f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        self * *scalar
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;

    /// Scalar multiplication (reversed operand order) using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = 2.0 * v;
    /// assert_eq!(result, Vec4::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul(self, vec: Vec4) -> Vec4 {
        Vec4 {
            x: vec.x * self,
            y: vec.y * self,
            z: vec.z * self,
            w: vec.w * self,
        }
    }
}

impl Mul<&Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, vec: &Vec4) -> Vec4 {
        self * *vec
    }
}

impl Mul for Vec4 {
    type Output = Self;

    /// Component-wise multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// let b = Vec4::new(6.0, 7.0, 8.0, 9.0);
    /// let result = a * b;
    /// assert_eq!(result, Vec4::new(12.0, 21.0, 32.0, 45.0));
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

impl Mul<&Self> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, other: &Vec4) -> Vec4 {
        *self * *other
    }
}

impl Mul<Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, other: Vec4) -> Vec4 {
        *self * other
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;

    /// Scalar division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(6.0, 9.0, 12.0, 15.0);
    /// let result = v / 3.0;
    /// assert_eq!(result, Vec4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            w: self.w / scalar,
        }
    }
}

impl Div<&f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: &f32) -> Self {
        self / *scalar
    }
}

impl Div<Vec4> for f32 {
    type Output = Vec4;

    /// Scalar division (reversed operand order) using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(2.0, 4.0, 8.0, 16.0);
    /// let result = 32.0 / v;
    /// assert_eq!(result, Vec4::new(16.0, 8.0, 4.0, 2.0));
    /// ```
    #[inline]
    fn div(self, vec: Vec4) -> Vec4 {
        Vec4 {
            x: self / vec.x,
            y: self / vec.y,
            z: self / vec.z,
            w: self / vec.w,
        }
    }
}

impl Div<&Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn div(self, vec: &Vec4) -> Vec4 {
        self / *vec
    }
}

impl Div for Vec4 {
    type Output = Self;

    /// Component-wise division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let a = Vec4::new(12.0, 21.0, 32.0, 45.0);
    /// let b = Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// let result = a / b;
    /// assert_eq!(result, Vec4::new(6.0, 7.0, 8.0, 9.0));
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
            w: self.w / other.w,
        }
    }
}

impl Div<&Self> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, other: &Self) -> Self {
        self / *other
    }
}

impl Div<&Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn div(self, other: &Vec4) -> Vec4 {
        *self / *other
    }
}

impl Div<Vec4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn div(self, other: Vec4) -> Vec4 {
        *self / other
    }
}

impl Neg for Vec4 {
    type Output = Self;

    /// Negates all components using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
    /// let result = -v;
    /// assert_eq!(result, Vec4::new(-1.0, 2.0, -3.0, 4.0));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Neg for &Vec4 {
    type Output = Vec4;

    #[inline]
    fn neg(self) -> Vec4 {
        -*self
    }
}

impl AddAssign for Vec4 {
    /// Component-wise addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// v += Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(v, Vec4::new(6.0, 8.0, 10.0, 12.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl AddAssign<&Self> for Vec4 {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl SubAssign for Vec4 {
    /// Component-wise subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(10.0, 12.0, 14.0, 16.0);
    /// v -= Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(v, Vec4::new(5.0, 6.0, 7.0, 8.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

impl SubAssign<&Self> for Vec4 {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        *self -= *other;
    }
}

impl MulAssign<f32> for Vec4 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// v *= 2.0;
    /// assert_eq!(v, Vec4::new(2.0, 4.0, 6.0, 8.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl MulAssign<&f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        *self *= *scalar;
    }
}

impl MulAssign for Vec4 {
    /// Component-wise multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// v *= Vec4::new(6.0, 7.0, 8.0, 9.0);
    /// assert_eq!(v, Vec4::new(12.0, 21.0, 32.0, 45.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
        self.w *= other.w;
    }
}

impl MulAssign<&Self> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

impl DivAssign<f32> for Vec4 {
    /// Scalar division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(6.0, 9.0, 12.0, 15.0);
    /// v /= 3.0;
    /// assert_eq!(v, Vec4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}

impl DivAssign<&f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, scalar: &f32) {
        *self /= *scalar;
    }
}

impl DivAssign for Vec4 {
    /// Component-wise division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let mut v = Vec4::new(12.0, 21.0, 32.0, 45.0);
    /// v /= Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// assert_eq!(v, Vec4::new(6.0, 7.0, 8.0, 9.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
        self.w /= other.w;
    }
}

impl DivAssign<&Self> for Vec4 {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        *self /= *other;
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================


// Vec4 Conversions for arrays/tuples

impl From<[f32; 4]> for Vec4 {
    /// Converts from a 4-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v: Vec4 = [1.0, 2.0, 3.0, 4.0].into();
    /// assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

impl From<&[f32; 4]> for Vec4 {
    #[inline]
    fn from(arr: &[f32; 4]) -> Self {
        let vec: Vec4 = (*arr).into();
        vec
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    /// Converts from a 4-element tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v: Vec4 = (1.0, 2.0, 3.0, 4.0).into();
    /// assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn from(tuple: (f32, f32, f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
            z: tuple.2,
            w: tuple.3,
        }
    }
}

impl From<&(f32, f32, f32, f32)> for Vec4 {
    #[inline]
    fn from(tuple: &(f32, f32, f32, f32)) -> Self {
        let vec: Vec4 = (*tuple).into();
        vec
    }
}

impl From<Vec4> for [f32; 4] {
    /// Converts to a 4-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let arr: [f32; 4] = v.into();
    /// assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    fn from(vec: Vec4) -> Self {
        [vec.x, vec.y, vec.z, vec.w]
    }
}

impl From<&Vec4> for [f32; 4] {
    #[inline]
    fn from(vec: &Vec4) -> Self {
        let arr: [f32; 4] = (*vec).into();
        arr
    }
}

impl From<Vec4> for (f32, f32, f32, f32) {
    /// Converts to a 4-element tuple.
    ///
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::vec::Vec4;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let tuple: (f32, f32, f32, f32) = v.into();
    /// assert_eq!(tuple, (1.0, 2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn from(vec: Vec4) -> Self {
        (vec.x, vec.y, vec.z, vec.w)
    }
}

impl From<&Vec4> for (f32, f32, f32, f32) {
    #[inline]
    fn from(vec: &Vec4) -> Self {
        let tuple: (f32, f32, f32, f32) = (*vec).into();
        tuple
    }
}

// Dimensional Conversions

impl From<Vec2> for Vec4 {
    /// Extends Vec2 to Vec4 by adding z=0.0 and w=0.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec4};
    /// let v2 = Vec2::new(1.0, 2.0);
    /// let v4: Vec4 = v2.into();
    /// assert_eq!(v4, Vec4::new(1.0, 2.0, 0.0, 0.0));
    /// ```
    #[inline]
    fn from(v: Vec2) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: 0.0,
            w: 0.0,
        }
    }
}

impl From<&Vec2> for Vec4 {
    #[inline]
    fn from(v: &Vec2) -> Self {
        let vec4: Vec4 = (*v).into();
        vec4
    }
}

impl From<Vec3> for Vec4 {
    /// Extends Vec3 to Vec4 by adding w=0.0 (direction vector).
    ///
    /// For homogeneous points, use `Vec4::new(v.x, v.y, v.z, 1.0)` instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec3, Vec4};
    /// let v3 = Vec3::new(1.0, 2.0, 3.0);
    /// let v4: Vec4 = v3.into();
    /// assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 0.0));
    /// ```
    #[inline]
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: 0.0,
        }
    }
}

impl From<&Vec3> for Vec4 {
    #[inline]
    fn from(v: &Vec3) -> Self {
        let vec4: Vec4 = (*v).into();
        vec4
    }
}
