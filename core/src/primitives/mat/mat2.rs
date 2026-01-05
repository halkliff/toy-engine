use crate::primitives::vec::Vec2;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A 2x2 matrix with row-major order.
///
/// Commonly used for 2D transformations, rotations, scaling, and general linear algebra
/// operations in 2D space. This matrix type is efficient for 2D graphics operations and
/// can represent any linear transformation in 2D.
///
/// # Matrix Layout (Row-Major)
/// ```text
/// [ row_x.x  row_x.y ]
/// [ row_y.x  row_y.y ]
/// ```
///
/// # Common Use Cases
/// - 2D rotations and scaling
/// - Texture transformations
/// - 2D physics simulations
/// - Linear interpolation of transformations
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::mat::Matrix2;
/// # use toyengine::primitives::vec::Vec2;
/// // Create a rotation matrix
/// let angle = std::f32::consts::PI / 4.0; // 45 degrees
/// let rotation = Matrix2::from_angle(angle);
///
/// // Create a scale matrix
/// let scale = Matrix2::from_scale(Vec2::new(2.0, 2.0));
///
/// // Matrix multiplication
/// let combined = rotation * scale;
///
/// // Transform a vector
/// let v = Vec2::new(1.0, 0.0);
/// let transformed = v * combined;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix2 {
    /// First row of the matrix (x-axis in transformation context)
    pub row_x: Vec2,
    /// Second row of the matrix (y-axis in transformation context)
    pub row_y: Vec2,
}

// ============================================================================
// Matrix2 Implementation
// ============================================================================

impl Matrix2 {
    // Constructors

    /// Creates a new Matrix2 from two row vectors.
    ///
    /// The matrix is constructed in row-major order where `row_x` represents
    /// the first row and `row_y` represents the second row.
    ///
    /// # Arguments
    ///
    /// * `row_x` - The first row vector (top row)
    /// * `row_y` - The second row vector (bottom row)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(1.0, 2.0),
    ///     Vec2::new(3.0, 4.0)
    /// );
    /// ```
    #[inline]
    pub const fn new(row_x: Vec2, row_y: Vec2) -> Self {
        Self { row_x, row_y }
    }

    /// Creates an identity matrix.
    ///
    /// The identity matrix leaves vectors unchanged when multiplied.
    ///
    /// # Matrix
    /// ```text
    /// [ 1  0 ]
    /// [ 0  1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let identity = Matrix2::identity();
    /// let v = Vec2::new(3.0, 4.0);
    /// let result = v * identity;
    /// assert_eq!(result, v); // Vector unchanged
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Self {
            row_x: Vec2 { x: 1.0, y: 0.0 },
            row_y: Vec2 { x: 0.0, y: 1.0 },
        }
    }

    /// Creates a zero matrix with all components set to zero.
    ///
    /// # Matrix
    /// ```text
    /// [ 0  0 ]
    /// [ 0  0 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let zero = Matrix2::zero();
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self {
            row_x: Vec2 { x: 0.0, y: 0.0 },
            row_y: Vec2 { x: 0.0, y: 0.0 },
        }
    }

    /// Creates a matrix with all components set to the same value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value for all matrix components
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let uniform = Matrix2::splat(5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            row_x: Vec2 { x: value, y: value },
            row_y: Vec2 { x: value, y: value },
        }
    }

    /// Creates a rotation matrix from an angle in radians.
    ///
    /// Rotates vectors counter-clockwise by the given angle.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Matrix
    /// ```text
    /// [ cos(θ)  -sin(θ) ]
    /// [ sin(θ)   cos(θ) ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let angle = std::f32::consts::PI / 2.0; // 90 degrees
    /// let rotation = Matrix2::from_angle(angle);
    /// let v = Vec2::new(1.0, 0.0);
    /// let rotated = v * rotation;
    /// // rotated is approximately (0, 1)
    /// ```
    #[inline]
    #[must_use]
    pub fn from_angle(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec2 { x: cos, y: -sin },
            row_y: Vec2 { x: sin, y: cos },
        }
    }

    /// Creates a non-uniform scale matrix.
    ///
    /// Scales vectors by different factors along x and y axes.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors for x and y axes
    ///
    /// # Matrix
    /// ```text
    /// [ scale.x    0     ]
    /// [   0      scale.y ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
    /// let v = Vec2::new(1.0, 1.0);
    /// let scaled = v * scale;
    /// assert_eq!(scaled, Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vec2) -> Self {
        Self {
            row_x: Vec2 { x: scale.x, y: 0.0 },
            row_y: Vec2 { x: 0.0, y: scale.y },
        }
    }

    /// Creates a uniform scale matrix.
    ///
    /// Scales vectors equally in all directions.
    ///
    /// # Arguments
    ///
    /// * `scale` - Uniform scale factor
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let scale = Matrix2::from_scale_uniform(2.0);
    /// let v = Vec2::new(1.0, 1.0);
    /// let scaled = v * scale;
    /// assert_eq!(scaled, Vec2::new(2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale_uniform(scale: f32) -> Self {
        Self {
            row_x: Vec2 { x: scale, y: 0.0 },
            row_y: Vec2 { x: 0.0, y: scale },
        }
    }

    /// Creates a matrix from column vectors.
    ///
    /// Useful when working with column-major representations or OpenGL-style matrices.
    ///
    /// # Arguments
    ///
    /// * `col_x` - First column vector
    /// * `col_y` - Second column vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::from_cols(
    ///     Vec2::new(1.0, 2.0),
    ///     Vec2::new(3.0, 4.0)
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(col_x: Vec2, col_y: Vec2) -> Self {
        Self {
            row_x: Vec2 {
                x: col_x.x,
                y: col_y.x,
            },
            row_y: Vec2 {
                x: col_x.y,
                y: col_y.y,
            },
        }
    }

    // Constants

    /// Identity matrix (multiplicative identity)
    /// ```text
    /// [ 1  0 ]
    /// [ 0  1 ]
    /// ```
    pub const IDENTITY: Self = Self {
        row_x: Vec2 { x: 1.0, y: 0.0 },
        row_y: Vec2 { x: 0.0, y: 1.0 },
    };

    /// Zero matrix (additive identity)
    /// ```text
    /// [ 0  0 ]
    /// [ 0  0 ]
    /// ```
    pub const ZERO: Self = Self {
        row_x: Vec2 { x: 0.0, y: 0.0 },
        row_y: Vec2 { x: 0.0, y: 0.0 },
    };

    // Core Operations

    /// Computes the transpose of this matrix.
    ///
    /// The transpose swaps rows and columns. For a rotation matrix,
    /// the transpose equals the inverse.
    ///
    /// # Formula
    /// ```text
    /// [ a  b ]ᵀ   [ a  c ]
    /// [ c  d ]  = [ b  d ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(1.0, 2.0),
    ///     Vec2::new(3.0, 4.0)
    /// );
    /// let transposed = m.transpose();
    /// // transposed rows: (1, 3), (2, 4)
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            row_x: Vec2 {
                x: self.row_x.x,
                y: self.row_y.x,
            },
            row_y: Vec2 {
                x: self.row_x.y,
                y: self.row_y.y,
            },
        }
    }

    /// Computes the determinant of this matrix.
    ///
    /// The determinant indicates:
    /// - Zero: Matrix is singular (non-invertible)
    /// - Positive: Preserves orientation
    /// - Negative: Reverses orientation
    /// - Magnitude: Scale factor of area transformation
    ///
    /// # Formula
    /// det(A) = a·d - b·c
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(3.0, 1.0),
    ///     Vec2::new(2.0, 4.0)
    /// );
    /// let det = m.determinant();
    /// assert_eq!(det, 10.0); // 3*4 - 1*2 = 10
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        self.row_x.x * self.row_y.y - self.row_x.y * self.row_y.x
    }

    /// Computes the inverse of this matrix.
    ///
    /// Returns `None` if the matrix is singular (determinant near zero).
    /// The inverse matrix satisfies: M × M⁻¹ = I (identity)
    ///
    /// # Formula
    /// ```text
    /// [ a  b ]⁻¹        1    [  d  -b ]
    /// [ c  d ]    = --------- [ -c   a ]
    ///               (ad - bc)
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(4.0, 7.0),
    ///     Vec2::new(2.0, 6.0)
    /// );
    /// if let Some(inv) = m.inverse() {
    ///     let identity = m * inv;
    ///     // identity ≈ Matrix2::IDENTITY
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            row_x: Vec2 {
                x: self.row_y.y * inv_det,
                y: -self.row_x.y * inv_det,
            },
            row_y: Vec2 {
                x: -self.row_y.x * inv_det,
                y: self.row_x.x * inv_det,
            },
        })
    }

    /// Computes the Frobenius inner product (dot product) of two matrices.
    ///
    /// This is the sum of products of corresponding elements.
    ///
    /// # Formula
    /// dot(A, B) = Σ aᵢⱼ · bᵢⱼ
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let a = Matrix2::IDENTITY;
    /// let b = Matrix2::IDENTITY;
    /// let dot = a.dot(&b);
    /// assert_eq!(dot, 2.0); // Sum of diagonal elements
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_dot(&self, other: &Self) -> f32 {
        self.row_x.x * other.row_x.x
            + self.row_x.y * other.row_x.y
            + self.row_y.x * other.row_y.x
            + self.row_y.y * other.row_y.y
    }

    /// Computes the trace of this matrix.
    ///
    /// The trace is the sum of diagonal elements. For rotation matrices,
    /// the trace relates to the rotation angle: trace = 2·cos(θ) for 2D rotations.
    ///
    /// # Formula
    /// trace(A) = a₁₁ + a₂₂
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(3.0, 1.0),
    ///     Vec2::new(2.0, 4.0)
    /// );
    /// assert_eq!(m.trace(), 7.0); // 3 + 4
    /// ```
    #[inline]
    #[must_use]
    pub fn trace(&self) -> f32 {
        self.row_x.x + self.row_y.y
    }

    /// Computes the Frobenius norm of this matrix.
    ///
    /// The Frobenius norm is the square root of the sum of squared elements.
    /// It's analogous to vector length for matrices.
    ///
    /// # Formula
    /// ‖A‖_F = √(Σ aᵢⱼ²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// let norm = m.frobenius_norm();
    /// assert!((norm - 1.414).abs() < 0.01); // √2
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm(&self) -> f32 {
        self.frobenius_dot(self).sqrt()
    }

    /// Computes the squared Frobenius norm of this matrix.
    ///
    /// Cheaper than [`frobenius_norm`](Self::frobenius_norm) as it avoids the square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// assert_eq!(m.frobenius_norm_squared(), 2.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm_squared(&self) -> f32 {
        self.frobenius_dot(self)
    }

    /// Transforms a vector by this matrix.
    ///
    /// Equivalent to `vec * matrix` but can be more readable in some contexts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::from_scale(Vec2::new(2.0, 3.0));
    /// let v = Vec2::new(1.0, 1.0);
    /// let result = m.transform_vector(v);
    /// assert_eq!(result, Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vec: Vec2) -> Vec2 {
        vec * (*self)
    }

    /// Extracts the column vectors from this matrix.
    ///
    /// Returns a tuple of (column_x, column_y).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(1.0, 2.0),
    ///     Vec2::new(3.0, 4.0)
    /// );
    /// let (col_x, col_y) = m.cols();
    /// assert_eq!(col_x, Vec2::new(1.0, 3.0));
    /// assert_eq!(col_y, Vec2::new(2.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn cols(&self) -> (Vec2, Vec2) {
        (
            Vec2 {
                x: self.row_x.x,
                y: self.row_y.x,
            },
            Vec2 {
                x: self.row_x.y,
                y: self.row_y.y,
            },
        )
    }

    /// Returns the row vectors as a tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::new(
    ///     Vec2::new(1.0, 2.0),
    ///     Vec2::new(3.0, 4.0)
    /// );
    /// let (row_x, row_y) = m.rows();
    /// assert_eq!(row_x, Vec2::new(1.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn rows(&self) -> (Vec2, Vec2) {
        (self.row_x, self.row_y)
    }

    // Utility Methods

    /// Linear interpolation between two matrices.
    ///
    /// Interpolates component-wise between `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Target matrix to interpolate towards
    /// * `t` - Interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let a = Matrix2::ZERO;
    /// let b = Matrix2::IDENTITY;
    /// let mid = a.lerp(&b, 0.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            row_x: self.row_x.lerp(&other.row_x, t),
            row_y: self.row_y.lerp(&other.row_y, t),
        }
    }

    /// Checks if this matrix is approximately equal to another.
    ///
    /// Uses epsilon comparison for floating-point tolerance.
    ///
    /// # Arguments
    ///
    /// * `other` - Matrix to compare against
    /// * `epsilon` - Maximum difference for equality (typically f32::EPSILON)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let a = Matrix2::IDENTITY;
    /// let b = Matrix2::IDENTITY;
    /// assert!(a.approx_eq(&b, f32::EPSILON));
    /// ```
    #[inline]
    #[must_use]
    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        (self.row_x.x - other.row_x.x).abs() < epsilon
            && (self.row_x.y - other.row_x.y).abs() < epsilon
            && (self.row_y.x - other.row_y.x).abs() < epsilon
            && (self.row_y.y - other.row_y.y).abs() < epsilon
    }

    /// Checks if this matrix is the identity matrix (within epsilon).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// assert!(m.is_identity());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.approx_eq(&Self::IDENTITY, f32::EPSILON)
    }

    /// Checks if this matrix is invertible (determinant != 0).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// assert!(m.is_invertible());
    /// let zero = Matrix2::ZERO;
    /// assert!(!zero.is_invertible());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > f32::EPSILON
    }
}

// ============================================================================
// Default Trait
// ============================================================================

impl Default for Matrix2 {
    /// Returns the identity matrix as the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::default();
    /// assert!(m.is_identity());
    /// ```
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Operator Overloading - Matrix2
// ============================================================================

impl Add for Matrix2 {
    type Output = Self;

    /// Component-wise matrix addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let a = Matrix2::IDENTITY;
    /// let b = Matrix2::IDENTITY;
    /// let result = a + b;
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
        }
    }
}

impl Add<&Matrix2> for Matrix2 {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        Self {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
        }
    }
}

impl Add<&Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn add(self, other: &Matrix2) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
        }
    }
}

impl Add<Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn add(self, other: Matrix2) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
        }
    }
}

impl AddAssign for Matrix2 {
    /// Component-wise matrix addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let mut m = Matrix2::IDENTITY;
    /// m += Matrix2::IDENTITY;
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.row_x += other.row_x;
        self.row_y += other.row_y;
    }
}

impl AddAssign<&Matrix2> for Matrix2 {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.row_x += other.row_x;
        self.row_y += other.row_y;
    }
}

impl Sub for Matrix2 {
    type Output = Self;

    /// Component-wise matrix subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let a = Matrix2::IDENTITY;
    /// let b = Matrix2::IDENTITY;
    /// let result = a - b;
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
        }
    }
}

impl Sub<&Matrix2> for Matrix2 {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        Self {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
        }
    }
}

impl Sub<&Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn sub(self, other: &Matrix2) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
        }
    }
}

impl Sub<Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn sub(self, other: Matrix2) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
        }
    }
}

impl SubAssign for Matrix2 {
    /// Component-wise matrix subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let mut m = Matrix2::IDENTITY;
    /// m -= Matrix2::IDENTITY;
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.row_x -= other.row_x;
        self.row_y -= other.row_y;
    }
}

impl SubAssign<&Matrix2> for Matrix2 {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.row_x -= other.row_x;
        self.row_y -= other.row_y;
    }
}

impl Mul<f32> for Matrix2 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// Multiplies all matrix components by a scalar value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// let result = m * 2.0;
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            row_x: self.row_x * scalar,
            row_y: self.row_y * scalar,
        }
    }
}

impl Mul<f32> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn mul(self, scalar: f32) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x * scalar,
            row_y: self.row_y * scalar,
        }
    }
}

impl Mul<&f32> for Matrix2 {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        Self {
            row_x: self.row_x * *scalar,
            row_y: self.row_y * *scalar,
        }
    }
}

impl Mul<&f32> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn mul(self, scalar: &f32) -> Matrix2 {
        Matrix2 {
            row_x: self.row_x * *scalar,
            row_y: self.row_y * *scalar,
        }
    }
}

impl Mul for Matrix2 {
    type Output = Self;

    /// Matrix multiplication using the `*` operator.
    ///
    /// Performs standard matrix multiplication (not component-wise).
    /// The result represents the composition of two transformations.
    ///
    /// # Formula
    /// (A × B)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let rotation = Matrix2::from_angle(std::f32::consts::PI / 4.0);
    /// let scale = Matrix2::from_scale(Vec2::new(2.0, 2.0));
    /// let combined = rotation * scale; // Apply scale, then rotation
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            row_x: Vec2 {
                x: self.row_x.x * other.row_x.x + self.row_x.y * other.row_y.x,
                y: self.row_x.x * other.row_x.y + self.row_x.y * other.row_y.y,
            },
            row_y: Vec2 {
                x: self.row_y.x * other.row_x.x + self.row_y.y * other.row_y.x,
                y: self.row_y.x * other.row_x.y + self.row_y.y * other.row_y.y,
            },
        }
    }
}

impl Mul<&Matrix2> for Matrix2 {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn mul(self, other: &Matrix2) -> Matrix2 {
        (*self) * (*other)
    }
}

impl Mul<Matrix2> for &Matrix2 {
    type Output = Matrix2;

    #[inline]
    fn mul(self, other: Matrix2) -> Matrix2 {
        (*self) * other
    }
}

impl MulAssign<f32> for Matrix2 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let mut m = Matrix2::IDENTITY;
    /// m *= 2.0;
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.row_x *= scalar;
        self.row_y *= scalar;
    }
}

impl MulAssign<&f32> for Matrix2 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        self.row_x *= *scalar;
        self.row_y *= *scalar;
    }
}

impl MulAssign for Matrix2 {
    /// Matrix multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let mut m = Matrix2::IDENTITY;
    /// m *= Matrix2::IDENTITY;
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<&Matrix2> for Matrix2 {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

// ============================================================================
// Vector-Matrix Multiplication
// ============================================================================

impl Mul<Matrix2> for Vec2 {
    type Output = Vec2;

    /// Vector-matrix multiplication using the `*` operator.
    ///
    /// Transforms a vector by multiplying it with a matrix (row vector × matrix).
    /// This is the standard way to apply transformations to points/vectors.
    ///
    /// # Formula
    /// v' = v × M = [v.x, v.y] × [[m00, m01], [m10, m11]]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
    /// let v = Vec2::new(1.0, 1.0);
    /// let result = v * scale;
    /// assert_eq!(result, Vec2::new(2.0, 3.0));
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix2) -> Self::Output {
        Vec2 {
            x: self.x * rhs.row_x.x + self.y * rhs.row_y.x,
            y: self.x * rhs.row_x.y + self.y * rhs.row_y.y,
        }
    }
}

impl Mul<&Matrix2> for Vec2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: &Matrix2) -> Self::Output {
        self * (*rhs)
    }
}
impl Mul<Matrix2> for &Vec2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: Matrix2) -> Self::Output {
        (*self) * rhs
    }
}
impl Mul<&Matrix2> for &Vec2 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: &Matrix2) -> Self::Output {
        (*self) * (*rhs)
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<[[f32; 2]; 2]> for Matrix2 {
    /// Converts from a 2D array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m: Matrix2 = [[1.0, 2.0], [3.0, 4.0]].into();
    /// ```
    #[inline]
    fn from(arr: [[f32; 2]; 2]) -> Self {
        Self {
            row_x: Vec2 {
                x: arr[0][0],
                y: arr[0][1],
            },
            row_y: Vec2 {
                x: arr[1][0],
                y: arr[1][1],
            },
        }
    }
}

impl From<Matrix2> for [[f32; 2]; 2] {
    /// Converts to a 2D array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// let arr: [[f32; 2]; 2] = m.into();
    /// ```
    #[inline]
    fn from(mat: Matrix2) -> Self {
        [[mat.row_x.x, mat.row_x.y], [mat.row_y.x, mat.row_y.y]]
    }
}

impl From<[f32; 4]> for Matrix2 {
    /// Converts from a flat array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m: Matrix2 = [1.0, 2.0, 3.0, 4.0].into();
    /// ```
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self {
            row_x: Vec2 {
                x: arr[0],
                y: arr[1],
            },
            row_y: Vec2 {
                x: arr[2],
                y: arr[3],
            },
        }
    }
}

impl From<Matrix2> for [f32; 4] {
    /// Converts to a flat array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// let m = Matrix2::IDENTITY;
    /// let arr: [f32; 4] = m.into();
    /// assert_eq!(arr, [1.0, 0.0, 0.0, 1.0]);
    /// ```
    #[inline]
    fn from(mat: Matrix2) -> Self {
        [mat.row_x.x, mat.row_x.y, mat.row_y.x, mat.row_y.y]
    }
}

// Matrix2 Conversions for tuples

impl From<(Vec2, Vec2)> for Matrix2 {
    /// Converts from a tuple of two Vec2 row vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m: Matrix2 = (Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0)).into();
    /// ```
    #[inline]
    fn from(tuple: (Vec2, Vec2)) -> Self {
        Self {
            row_x: tuple.0,
            row_y: tuple.1,
        }
    }
}

impl From<&(Vec2, Vec2)> for Matrix2 {
    #[inline]
    fn from(tuple: &(Vec2, Vec2)) -> Self {
        let mat: Matrix2 = (*tuple).into();
        mat
    }
}

impl From<Matrix2> for (Vec2, Vec2) {
    /// Converts to a tuple of two Vec2 row vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix2;
    /// # use toyengine::primitives::vec::Vec2;
    /// let m = Matrix2::IDENTITY;
    /// let (row_x, row_y): (Vec2, Vec2) = m.into();
    /// ```
    #[inline]
    fn from(mat: Matrix2) -> Self {
        (mat.row_x, mat.row_y)
    }
}

impl From<&Matrix2> for (Vec2, Vec2) {
    #[inline]
    fn from(mat: &Matrix2) -> Self {
        let tuple: (Vec2, Vec2) = (*mat).into();
        tuple
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_constructors() {
        let identity = Matrix2::identity();
        let zero = Matrix2::zero();

        assert_eq!(identity.row_x, Vec2::new(1.0, 0.0));
        assert_eq!(identity.row_y, Vec2::new(0.0, 1.0));

        assert_eq!(zero.row_x, Vec2::new(0.0, 0.0));
        assert_eq!(zero.row_y, Vec2::new(0.0, 0.0));
    }

    #[test]
    fn test_constants() {
        assert_eq!(Matrix2::IDENTITY, Matrix2::identity());
        assert_eq!(Matrix2::ZERO, Matrix2::zero());
    }

    #[test]
    fn test_splat() {
        let m = Matrix2::splat(5.0);
        assert_eq!(m.row_x, Vec2::new(5.0, 5.0));
        assert_eq!(m.row_y, Vec2::new(5.0, 5.0));
    }

    #[test]
    fn test_rotation() {
        let rotation = Matrix2::from_angle(std::f32::consts::PI / 2.0);
        let vec = Vec2::new(1.0, 0.0);
        let rotated = vec * rotation;

        // Should rotate to approximately (0, -1) for row-vector convention
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - -1.0).abs() < 0.001);
    }

    #[test]
    fn test_scale() {
        let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
        let vec = Vec2::new(1.0, 1.0);
        let scaled = vec * scale;
        assert_eq!(scaled, Vec2::new(2.0, 3.0));

        let uniform = Matrix2::from_scale_uniform(2.0);
        let scaled_uniform = Vec2::new(1.0, 1.0) * uniform;
        assert_eq!(scaled_uniform, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_from_cols() {
        let col_x = Vec2::new(1.0, 0.0);
        let col_y = Vec2::new(0.0, 1.0);

        let m = Matrix2::from_cols(col_x, col_y);
        assert_eq!(m, Matrix2::IDENTITY);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix2::new(
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 4.0),
        );
        let transposed = m.transpose();

        assert_eq!(transposed.row_x, Vec2::new(1.0, 3.0));
        assert_eq!(transposed.row_y, Vec2::new(2.0, 4.0));

        // Double transpose should give original
        assert_eq!(transposed.transpose(), m);
    }

    #[test]
    fn test_determinant() {
        let identity = Matrix2::IDENTITY;
        assert_eq!(identity.determinant(), 1.0);

        let m = Matrix2::new(
            Vec2::new(3.0, 1.0),
            Vec2::new(2.0, 4.0),
        );
        assert_eq!(m.determinant(), 10.0);

        let zero = Matrix2::ZERO;
        assert_eq!(zero.determinant(), 0.0);
    }

    #[test]
    fn test_inverse() {
        let identity = Matrix2::IDENTITY;
        let inv_identity = identity.inverse().expect("Identity should be invertible");
        assert!(identity.approx_eq(&inv_identity, 0.001));

        let m = Matrix2::new(
            Vec2::new(4.0, 7.0),
            Vec2::new(2.0, 6.0),
        );
        let inv = m.inverse().expect("Matrix should be invertible");
        let product = m * inv;
        assert!(product.approx_eq(&Matrix2::IDENTITY, 0.001));

        let zero = Matrix2::ZERO;
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_trace() {
        let identity = Matrix2::IDENTITY;
        assert_eq!(identity.trace(), 2.0);

        let m = Matrix2::new(
            Vec2::new(3.0, 1.0),
            Vec2::new(2.0, 4.0),
        );
        assert_eq!(m.trace(), 7.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let identity = Matrix2::IDENTITY;
        let norm = identity.frobenius_norm();
        assert!((norm - 1.414).abs() < 0.01);

        let norm_squared = identity.frobenius_norm_squared();
        assert_eq!(norm_squared, 2.0);
    }

    #[test]
    fn test_addition() {
        let a = Matrix2::IDENTITY;
        let b = Matrix2::IDENTITY;
        let result = a + b;

        let expected = Matrix2::new(
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));

        // Test borrowed
        let result_borrowed = &a + &b;
        assert!(result_borrowed.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_subtraction() {
        let a = Matrix2::from_scale_uniform(3.0);
        let b = Matrix2::from_scale_uniform(1.0);
        let result = a - b;

        let expected = Matrix2::new(
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = Matrix2::IDENTITY;
        let result = m * 2.0;

        let expected = Matrix2::new(
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_matrix_multiplication() {
        let scale = Matrix2::from_scale_uniform(2.0);
        let rotation = Matrix2::from_angle(std::f32::consts::PI / 2.0);

        let combined = scale * rotation;
        let vec = Vec2::new(1.0, 0.0);
        let result = vec * combined;

        // Scale by 2, then rotate (row-vector: clockwise)
        assert!((result.x - 0.0).abs() < 0.001);
        assert!((result.y - -2.0).abs() < 0.001);
    }

    #[test]
    fn test_assignment_operators() {
        let mut m = Matrix2::ZERO;
        m += Matrix2::IDENTITY;
        assert_eq!(m, Matrix2::IDENTITY);

        m *= 2.0;
        let expected = Matrix2::new(
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        );
        assert!(m.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_vector_matrix_multiplication() {
        let v = Vec2::new(3.0, 4.0);
        let m = Matrix2::IDENTITY;
        let result = v * m;
        assert_eq!(result, v);

        // Test with scale
        let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
        let scaled = v * scale;
        assert_eq!(scaled, Vec2::new(6.0, 12.0));
    }

    #[test]
    fn test_array_conversions() {
        let identity = Matrix2::IDENTITY;

        let array: [[f32; 2]; 2] = identity.into();
        let back: Matrix2 = array.into();
        assert_eq!(back, identity);

        let flat: [f32; 4] = identity.into();
        let back_flat: Matrix2 = flat.into();
        assert_eq!(back_flat, identity);
    }

    #[test]
    fn test_transform_vector() {
        let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
        let vec = Vec2::new(1.0, 1.0);
        let result = scale.transform_vector(vec);
        assert_eq!(result, Vec2::new(2.0, 3.0));
    }

    #[test]
    fn test_cols_and_rows() {
        let m = Matrix2::new(
            Vec2::new(1.0, 2.0),
            Vec2::new(3.0, 4.0),
        );

        let (row_x, row_y) = m.rows();
        assert_eq!(row_x, m.row_x);
        assert_eq!(row_y, m.row_y);

        let (col_x, col_y) = m.cols();
        assert_eq!(col_x, Vec2::new(1.0, 3.0));
        assert_eq!(col_y, Vec2::new(2.0, 4.0));
    }

    #[test]
    fn test_lerp() {
        let a = Matrix2::ZERO;
        let b = Matrix2::from_scale_uniform(2.0);
        let mid = a.lerp(&b, 0.5);

        let expected = Matrix2::new(
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        );
        assert!(mid.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_approx_eq() {
        let a = Matrix2::IDENTITY;
        let b = Matrix2::IDENTITY;
        assert!(a.approx_eq(&b, f32::EPSILON));

        let c = Matrix2::from_scale_uniform(1.001);
        assert!(!a.approx_eq(&c, 0.0001));
    }

    #[test]
    fn test_is_identity() {
        let identity = Matrix2::IDENTITY;
        assert!(identity.is_identity());

        let scale = Matrix2::from_scale_uniform(2.0);
        assert!(!scale.is_identity());
    }

    #[test]
    fn test_is_invertible() {
        let identity = Matrix2::IDENTITY;
        assert!(identity.is_invertible());

        let zero = Matrix2::ZERO;
        assert!(!zero.is_invertible());
    }

    #[test]
    fn test_default_trait() {
        let default_mat: Matrix2 = Default::default();
        assert_eq!(default_mat, Matrix2::IDENTITY);
        assert!(default_mat.is_identity());
    }

    #[test]
    fn test_frobenius_dot() {
        let a = Matrix2::IDENTITY;
        let b = Matrix2::IDENTITY;
        let dot = a.frobenius_dot(&b);
        assert_eq!(dot, 2.0);
    }

    #[test]
    fn test_combined_transformations() {
        // Test scale then rotate
        let scale = Matrix2::from_scale(Vec2::new(2.0, 3.0));
        let rotation = Matrix2::from_angle(std::f32::consts::PI / 4.0);

        let combined = scale * rotation;
        let vec = Vec2::new(1.0, 0.0);
        let result = vec * combined;

        // (2, 0) rotated 45 degrees (clockwise for row-vector)
        let angle = -std::f32::consts::PI / 4.0;  // Negative for row-vector convention
        let expected_x = 2.0 * angle.cos();
        let expected_y = 2.0 * angle.sin();

        assert!((result.x - expected_x).abs() < 0.001);
        assert!((result.y - expected_y).abs() < 0.001);
    }
}
