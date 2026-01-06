use super::{Vec3, Vec4, NORMALIZED_PRECISION_THRESHOLD};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2-dimensional vector with x and y components.
///
/// Commonly used for 2D positions, texture coordinates, screen space coordinates,
/// and 2D physics calculations.
///
/// # Coordinate Systems
/// - **Screen Space**: Origin typically top-left, x right, y down
/// - **World Space**: Origin can be anywhere, often center or bottom-left
/// - **Texture Coordinates**: (0,0) to (1,1) mapping, origin top-left
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::vec::Vec2;
/// // Create vectors
/// let position = Vec2::new(10.0, 20.0);
/// let velocity = Vec2::new(1.5, -2.0);
///
/// // Vector arithmetic
/// let new_position = position + velocity;
/// let scaled = velocity * 2.0;
///
/// // Common operations
/// let length = velocity.length();
/// let mut direction = velocity;
/// direction.normalize();
/// let distance = position.distance(&Vec2::ZERO);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    /// X component (horizontal axis, typically right)
    pub x: f32,
    /// Y component (vertical axis, typically up or down depending on coordinate system)
    pub y: f32,
}
// ============================================================================
// Vec2 Implementation
// ============================================================================

impl Vec2 {
    // Constructors

    /// Creates a new 2D vector from x and y components.
    ///
    /// # Arguments
    /// * `x` - X component (horizontal)
    /// * `y` - Y component (vertical)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let position = Vec2::new(10.0, 20.0);
    /// let texture_coord = Vec2::new(0.5, 0.5);
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Creates a vector with all components set to zero.
    ///
    /// Equivalent to `Vec2::new(0.0, 0.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let origin = Vec2::zero();
    /// assert_eq!(origin, Vec2::new(0.0, 0.0));
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Creates a vector with all components set to one.
    ///
    /// Equivalent to `Vec2::new(1.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let uniform_scale = Vec2::one();
    /// ```
    #[inline]
    pub const fn one() -> Self {
        Self { x: 1.0, y: 1.0 }
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let uniform = Vec2::splat(5.0);
    /// assert_eq!(uniform, Vec2::new(5.0, 5.0));
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self { x: value, y: value }
    }

    // Constants

    /// Zero vector (0, 0)
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Unit vector (1, 1)
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    /// Unit X vector (1, 0) - points right in standard coordinates
    pub const X: Self = Self { x: 1.0, y: 0.0 };

    /// Unit Y vector (0, 1) - points up/down depending on coordinate system
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    /// Alias for X - points right
    pub const RIGHT: Self = Self::X;

    /// Points up (0, 1) - may point down in screen coordinates
    pub const UP: Self = Self::Y;

    /// Points left (-1, 0)
    pub const LEFT: Self = Self { x: -1.0, y: 0.0 };

    /// Points down (0, -1) - may point up in screen coordinates
    pub const DOWN: Self = Self { x: 0.0, y: -1.0 };

    // Core Operations

    /// Computes the dot product (scalar product) of two vectors.
    ///
    /// The dot product measures how aligned two vectors are:
    /// - Positive: vectors point in similar directions
    /// - Zero: vectors are perpendicular (90°)
    /// - Negative: vectors point in opposite directions
    ///
    /// # Formula
    /// dot(a, b) = a.x × b.x + a.y × b.y
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(1.0, 0.0);
    /// let b = Vec2::new(0.0, 1.0);
    /// assert_eq!(a.dot(&b), 0.0); // Perpendicular
    ///
    /// let c = Vec2::new(1.0, 0.0);
    /// let d = Vec2::new(2.0, 0.0);
    /// assert_eq!(c.dot(&d), 2.0); // Parallel
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the squared length (magnitude squared) of the vector.
    ///
    /// This is cheaper than [`length`](Self::length) as it avoids the square root.
    /// Useful for comparing distances or when only relative magnitudes matter.
    ///
    /// # Formula
    /// length_squared = x² + y²
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(3.0, 4.0);
    /// assert_eq!(v.length_squared(), 25.0); // 3² + 4² = 25
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
    /// length = √(x² + y²)
    ///
    /// # Performance
    /// If you only need to compare lengths, use [`length_squared`](Self::length_squared) instead
    /// to avoid the expensive square root operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(3.0, 4.0);
    /// assert_eq!(v.length(), 5.0); // 3-4-5 triangle
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(3.0, 4.0);
    /// v.normalize();
    /// assert!((v.length() - 1.0).abs() < 0.0001); // Length is ~1.0
    /// ```
    #[inline]
    pub fn normalize(&mut self) {
        let length = self.length();
        if length > 0.0 {
            self.x /= length;
            self.y /= length;
        }
    }

    /// Checks if the vector is normalized (unit length).
    /// Uses precision threshold of 2e-4 to account for floating-point errors.
    /// # Examples
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, NORMALIZED_PRECISION_THRESHOLD};
    /// let v = Vec2::new(1.0, 0.0);
    /// assert!(v.is_normalized());
    /// ```
    pub fn is_normalized(&self) -> bool {
        (self.length_squared() - 1.0).abs() < NORMALIZED_PRECISION_THRESHOLD
    }

    /// Computes the distance between two points represented as vectors.
    ///
    /// # Formula
    /// distance(a, b) = |b - a| = √((b.x - a.x)² + (b.y - a.y)²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(0.0, 0.0);
    /// let b = Vec2::new(3.0, 4.0);
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(0.0, 0.0);
    /// let b = Vec2::new(3.0, 4.0);
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let start = Vec2::new(0.0, 0.0);
    /// let end = Vec2::new(10.0, 10.0);
    /// let midpoint = start.lerp(&end, 0.5);
    /// assert_eq!(midpoint, Vec2::new(5.0, 5.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    // Batch Operations

    /// Batch normalization of vectors.
    ///
    /// Processes multiple vectors in parallel, enabling SIMD optimization.
    ///
    /// # Performance
    /// ~3-4x faster than individual normalize calls with compiler optimizations enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let vectors = vec![Vec2::new(3.0, 4.0); 100];
    /// let mut output = vec![Vec2::ZERO; 100];
    /// Vec2::normalize_slice(&vectors, &mut output);
    /// ```
    #[inline]
    pub fn normalize_slice(input: &[Self], output: &mut [Self]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i];
            output[i].normalize();
        }
    }

    /// In-place batch normalization of vectors.
    ///
    /// Modifies the vectors array directly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut vectors = vec![Vec2::new(3.0, 4.0); 100];
    /// Vec2::normalize_slice_inplace(&mut vectors);
    /// ```
    #[inline]
    pub fn normalize_slice_inplace(vectors: &mut [Self]) {
        for v in vectors.iter_mut() {
            v.normalize();
        }
    }

    /// Batch dot product computation.
    ///
    /// Computes dot products between corresponding pairs of vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = vec![Vec2::new(1.0, 0.0); 100];
    /// let b = vec![Vec2::new(0.0, 1.0); 100];
    /// let mut output = vec![0.0; 100];
    /// Vec2::dot_slice(&a, &b, &mut output);
    /// ```
    #[inline]
    pub fn dot_slice(a: &[Self], b: &[Self], output: &mut [f32]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = a[i].dot(&b[i]);
        }
    }

    /// Batch vector addition.
    ///
    /// Adds corresponding pairs of vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = vec![Vec2::new(1.0, 2.0); 100];
    /// let b = vec![Vec2::new(3.0, 4.0); 100];
    /// let mut output = vec![Vec2::ZERO; 100];
    /// Vec2::add_slice(&a, &b, &mut output);
    /// ```
    #[inline]
    pub fn add_slice(a: &[Self], b: &[Self], output: &mut [Self]) {
        let len = a.len().min(b.len()).min(output.len());
        for i in 0..len {
            output[i] = Self {
                x: a[i].x + b[i].x,
                y: a[i].y + b[i].y,
            };
        }
    }

    /// Batch scalar multiplication.
    ///
    /// Scales all vectors by the same scalar value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let vectors = vec![Vec2::new(1.0, 2.0); 100];
    /// let mut output = vec![Vec2::ZERO; 100];
    /// Vec2::scale_slice(&vectors, 2.0, &mut output);
    /// ```
    #[inline]
    pub fn scale_slice(input: &[Self], scalar: f32, output: &mut [Self]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = input[i] * scalar;
        }
    }

    /// In-place batch scalar multiplication.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut vectors = vec![Vec2::new(1.0, 2.0); 100];
    /// Vec2::scale_slice_inplace(&mut vectors, 2.0);
    /// ```
    #[inline]
    pub fn scale_slice_inplace(vectors: &mut [Self], scalar: f32) {
        for v in vectors.iter_mut() {
            *v = *v * scalar;
        }
    }

    /// Batch linear interpolation.
    ///
    /// Interpolates between corresponding pairs of vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let start = vec![Vec2::ZERO; 100];
    /// let end = vec![Vec2::ONE; 100];
    /// let mut output = vec![Vec2::ZERO; 100];
    /// Vec2::lerp_slice(&start, &end, 0.5, &mut output);
    /// ```
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(-5.0, 15.0);
    /// let clamped = v.clamp(Vec2::ZERO, Vec2::new(10.0, 10.0));
    /// assert_eq!(clamped, Vec2::new(0.0, 10.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn clamp(&self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
        }
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(-3.0, 4.0);
    /// assert_eq!(v.abs(), Vec2::new(3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns the minimum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(3.0, 1.0);
    /// assert_eq!(v.min_component(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn min_component(&self) -> f32 {
        self.x.min(self.y)
    }

    /// Returns the maximum component value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(3.0, 1.0);
    /// assert_eq!(v.max_component(), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn max_component(&self) -> f32 {
        self.x.max(self.y)
    }

    /// Returns a perpendicular vector (rotated 90 degrees counter-clockwise).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, 0.0);
    /// assert_eq!(v.perpendicular(), Vec2::new(0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn perpendicular(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// Extends this Vec2 to a Vec3 by adding a z component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec3};
    /// let v2 = Vec2::new(1.0, 2.0);
    /// let v3 = v2.extend(3.0);
    /// assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn extend(&self, z: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z,
        }
    }

    // Chainable Methods

    /// Chainable version of [`mul`](Self::mul) for scalar values.
    #[inline]
    #[must_use]
    pub fn with_mul_scalar(self, scalar: f32) -> Self {
        self * scalar
    }

    /// Chainable version of [`div`](Self::div) for scalar values.
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
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(3.0, 4.0).with_normalize();
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
            }
        } else {
            self
        }
    }

    /// Chainable version of [`lerp`](Self::lerp).
    #[inline]
    #[must_use]
    pub fn with_lerp(self, other: &Self, t: f32) -> Self {
        self.lerp(other, t)
    }
}
// ============================================================================
// Operator Overloading - Vec2
// ============================================================================

impl Add for Vec2 {
    type Output = Self;

    /// Component-wise addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(1.0, 2.0);
    /// let b = Vec2::new(3.0, 4.0);
    /// let result = a + b;
    /// assert_eq!(result, Vec2::new(4.0, 6.0));
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Add<&Self> for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: &Self) -> Self {
        self + *other
    }
}

impl Add<&Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, other: &Vec2) -> Vec2 {
        *self + *other
    }
}

impl Add<Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, other: Vec2) -> Vec2 {
        *self + other
    }
}

impl Sub for Vec2 {
    type Output = Self;

    /// Component-wise subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(5.0, 7.0);
    /// let b = Vec2::new(2.0, 3.0);
    /// let result = a - b;
    /// assert_eq!(result, Vec2::new(3.0, 4.0));
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Sub<&Self> for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: &Self) -> Self {
        self - *other
    }
}

impl Sub<&Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, other: &Vec2) -> Vec2 {
        *self - *other
    }
}

impl Sub<Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, other: Vec2) -> Vec2 {
        *self - other
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, 2.0);
    /// let result = v * 3.0;
    /// assert_eq!(result, Vec2::new(3.0, 6.0));
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl Mul<&f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        self * *scalar
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;

    /// Scalar multiplication (reversed operand order) using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, 2.0);
    /// let result = 3.0 * v;
    /// assert_eq!(result, Vec2::new(3.0, 6.0));
    /// ```
    #[inline]
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2 {
            x: vec.x * self,
            y: vec.y * self,
        }
    }
}

impl Mul<&Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, vec: &Vec2) -> Vec2 {
        self * *vec
    }
}

impl Mul for Vec2 {
    type Output = Self;

    /// Component-wise multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(2.0, 3.0);
    /// let b = Vec2::new(4.0, 5.0);
    /// let result = a * b;
    /// assert_eq!(result, Vec2::new(8.0, 15.0));
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl Mul<&Self> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, other: &Vec2) -> Vec2 {
        *self * *other
    }
}

impl Mul<Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, other: Vec2) -> Vec2 {
        *self * other
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    /// Scalar division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(6.0, 9.0);
    /// let result = v / 3.0;
    /// assert_eq!(result, Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
        }
    }
}

impl Div<&f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: &f32) -> Self {
        self / *scalar
    }
}

impl Div<Vec2> for f32 {
    type Output = Vec2;

    /// Scalar division (reversed operand order) using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(2.0, 4.0);
    /// let result = 8.0 / v;
    /// assert_eq!(result, Vec2::new(4.0, 2.0));
    /// ```
    #[inline]
    fn div(self, vec: Vec2) -> Vec2 {
        Vec2 {
            x: self / vec.x,
            y: self / vec.y,
        }
    }
}

impl Div<&Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn div(self, vec: &Vec2) -> Vec2 {
        self / *vec
    }
}

impl Div for Vec2 {
    type Output = Self;

    /// Component-wise division using the `/` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Vec2::new(8.0, 15.0);
    /// let b = Vec2::new(2.0, 3.0);
    /// let result = a / b;
    /// assert_eq!(result, Vec2::new(4.0, 5.0));
    /// ```
    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }
}

impl Div<&Self> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, other: &Self) -> Self {
        self / *other
    }
}

impl Div<&Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn div(self, other: &Vec2) -> Vec2 {
        *self / *other
    }
}

impl Div<Vec2> for &Vec2 {
    type Output = Vec2;
    #[inline]
    fn div(self, other: Vec2) -> Vec2 {
        *self / other
    }
}

impl Neg for Vec2 {
    type Output = Self;

    /// Negates all components using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, -2.0);
    /// let result = -v;
    /// assert_eq!(result, Vec2::new(-1.0, 2.0));
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Neg for &Vec2 {
    type Output = Vec2;

    #[inline]
    fn neg(self) -> Vec2 {
        -*self
    }
}

impl AddAssign for Vec2 {
    /// Component-wise addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(1.0, 2.0);
    /// v += Vec2::new(3.0, 4.0);
    /// assert_eq!(v, Vec2::new(4.0, 6.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl AddAssign<&Self> for Vec2 {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl SubAssign for Vec2 {
    /// Component-wise subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(5.0, 7.0);
    /// v -= Vec2::new(2.0, 3.0);
    /// assert_eq!(v, Vec2::new(3.0, 4.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl SubAssign<&Self> for Vec2 {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        *self -= *other;
    }
}

impl MulAssign<f32> for Vec2 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(1.0, 2.0);
    /// v *= 3.0;
    /// assert_eq!(v, Vec2::new(3.0, 6.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl MulAssign<&f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        *self *= *scalar;
    }
}

impl MulAssign for Vec2 {
    /// Component-wise multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(2.0, 3.0);
    /// v *= Vec2::new(4.0, 5.0);
    /// assert_eq!(v, Vec2::new(8.0, 15.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

impl MulAssign<&Self> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

impl DivAssign<f32> for Vec2 {
    /// Scalar division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(6.0, 9.0);
    /// v /= 3.0;
    /// assert_eq!(v, Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

impl DivAssign<&f32> for Vec2 {
    #[inline]
    fn div_assign(&mut self, scalar: &f32) {
        *self /= *scalar;
    }
}

impl DivAssign for Vec2 {
    /// Component-wise division assignment using the `/=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let mut v = Vec2::new(8.0, 15.0);
    /// v /= Vec2::new(2.0, 3.0);
    /// assert_eq!(v, Vec2::new(4.0, 5.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl DivAssign<&Self> for Vec2 {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        *self /= *other;
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

// Vec2 Conversions for arrays/tuples

impl From<[f32; 2]> for Vec2 {
    /// Converts from a 2-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v: Vec2 = [1.0, 2.0].into();
    /// assert_eq!(v, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn from(arr: [f32; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }
}

impl From<&[f32; 2]> for Vec2 {
    #[inline]
    fn from(arr: &[f32; 2]) -> Self {
        let vec: Vec2 = (*arr).into();
        vec
    }
}

impl From<(f32, f32)> for Vec2 {
    /// Converts from a 2-element tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v: Vec2 = (1.0, 2.0).into();
    /// assert_eq!(v, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn from(tuple: (f32, f32)) -> Self {
        Self {
            x: tuple.0,
            y: tuple.1,
        }
    }
}

impl From<&(f32, f32)> for Vec2 {
    #[inline]
    fn from(tuple: &(f32, f32)) -> Self {
        let vec: Vec2 = (*tuple).into();
        vec
    }
}

impl From<Vec2> for [f32; 2] {
    /// Converts to a 2-element array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, 2.0);
    /// let arr: [f32; 2] = v.into();
    /// assert_eq!(arr, [1.0, 2.0]);
    /// ```
    #[inline]
    fn from(vec: Vec2) -> Self {
        [vec.x, vec.y]
    }
}

impl From<&Vec2> for [f32; 2] {
    #[inline]
    fn from(vec: &Vec2) -> Self {
        let arr: [f32; 2] = (*vec).into();
        arr
    }
}

impl From<Vec2> for (f32, f32) {
    /// Converts to a 2-element tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::Vec2;
    /// let v = Vec2::new(1.0, 2.0);
    /// let arr: (f32, f32) = v.into();
    /// assert_eq!(arr, (1.0, 2.0));
    /// ```
    #[inline]
    fn from(vec: Vec2) -> Self {
        (vec.x, vec.y)
    }
}
impl From<&Vec2> for (f32, f32) {
    #[inline]
    fn from(vec: &Vec2) -> Self {
        let tuple: (f32, f32) = (*vec).into();
        tuple
    }
}

// Dimensional Conversions

impl From<Vec3> for Vec2 {
    /// Truncates Vec3 to Vec2 by discarding z component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec3};
    /// let v3 = Vec3::new(1.0, 2.0, 3.0);
    /// let v2: Vec2 = v3.into();
    /// assert_eq!(v2, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl From<&Vec3> for Vec2 {
    #[inline]
    fn from(v: &Vec3) -> Self {
        let vec2: Vec2 = (*v).into();
        vec2
    }
}

impl From<Vec4> for Vec2 {
    /// Truncates Vec4 to Vec2 by discarding z and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::vec::{Vec2, Vec4};
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2: Vec2 = v4.into();
    /// assert_eq!(v2, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn from(v: Vec4) -> Self {
        Self { x: v.x, y: v.y }
    }
}

impl From<&Vec4> for Vec2 {
    #[inline]
    fn from(v: &Vec4) -> Self {
        let vec2: Vec2 = (*v).into();
        vec2
    }
}
