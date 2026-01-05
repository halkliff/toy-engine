use crate::primitives::vec::Vec3;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A 3x3 matrix with row-major order.
///
/// Commonly used for 3D rotations, scaling, shearing, and general linear transformations
/// that don't require translation. This matrix type is the workhorse of 3D graphics for
/// transforming directions, normals, and representing orientation.
///
/// # Matrix Layout (Row-Major)
/// ```text
/// [ row_x.x  row_x.y  row_x.z ]
/// [ row_y.x  row_y.y  row_y.z ]
/// [ row_z.x  row_z.y  row_z.z ]
/// ```
///
/// # Common Use Cases
/// - 3D rotations (around X, Y, Z axes or arbitrary axes)
/// - Non-uniform scaling in 3D
/// - Coordinate system transformations
/// - Normal vector transformations (use inverse transpose)
/// - Texture coordinate transformations
/// - Representing object orientation (rotation part of transforms)
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::mat::Matrix3;
/// # use toyengine::primitives::vec::Vec3;
/// // Create a rotation matrix around Y-axis
/// let angle = std::f32::consts::PI / 4.0; // 45 degrees
/// let rotation = Matrix3::from_rotation_y(angle);
///
/// // Create a scale matrix
/// let scale = Matrix3::from_scale(Vec3::new(2.0, 2.0, 2.0));
///
/// // Combine transformations
/// let combined = rotation * scale;
///
/// // Transform a vector
/// let v = Vec3::new(1.0, 0.0, 0.0);
/// let transformed = v * combined;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix3 {
    /// First row of the matrix (x-axis in transformation context)
    pub row_x: Vec3,
    /// Second row of the matrix (y-axis in transformation context)
    pub row_y: Vec3,
    /// Third row of the matrix (z-axis in transformation context)
    pub row_z: Vec3,
}

// ============================================================================
// Matrix3 Implementation
// ============================================================================

impl Matrix3 {
    // Constructors

    /// Creates a new Matrix3 from three row vectors.
    ///
    /// The matrix is constructed in row-major order where `row_x`, `row_y`, and `row_z`
    /// represent the first, second, and third rows respectively.
    ///
    /// # Arguments
    ///
    /// * `row_x` - The first row vector (top row)
    /// * `row_y` - The second row vector (middle row)
    /// * `row_z` - The third row vector (bottom row)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::new(
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// );
    /// ```
    #[inline]
    pub const fn new(row_x: Vec3, row_y: Vec3, row_z: Vec3) -> Self {
        Self { row_x, row_y, row_z }
    }

    /// Creates an identity matrix.
    ///
    /// The identity matrix leaves vectors unchanged when multiplied.
    ///
    /// # Matrix
    /// ```text
    /// [ 1  0  0 ]
    /// [ 0  1  0 ]
    /// [ 0  0  1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let identity = Matrix3::identity();
    /// let v = Vec3::new(3.0, 4.0, 5.0);
    /// let result = v * identity;
    /// assert_eq!(result, v); // Vector unchanged
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Self {
            row_x: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            row_y: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
            row_z: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
        }
    }

    /// Creates a zero matrix with all components set to zero.
    ///
    /// # Matrix
    /// ```text
    /// [ 0  0  0 ]
    /// [ 0  0  0 ]
    /// [ 0  0  0 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let zero = Matrix3::zero();
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self {
            row_x: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            row_y: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            row_z: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let uniform = Matrix3::splat(5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            row_x: Vec3 { x: value, y: value, z: value },
            row_y: Vec3 { x: value, y: value, z: value },
            row_z: Vec3 { x: value, y: value, z: value },
        }
    }

    /// Creates a rotation matrix around the X-axis.
    ///
    /// Rotates vectors counter-clockwise around the X-axis (right-hand rule).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Matrix
    /// ```text
    /// [ 1    0        0     ]
    /// [ 0  cos(θ)  -sin(θ) ]
    /// [ 0  sin(θ)   cos(θ) ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let angle = std::f32::consts::PI / 2.0; // 90 degrees
    /// let rotation = Matrix3::from_rotation_x(angle);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            row_y: Vec3 { x: 0.0, y: cos, z: -sin },
            row_z: Vec3 { x: 0.0, y: sin, z: cos },
        }
    }

    /// Creates a rotation matrix around the Y-axis.
    ///
    /// Rotates vectors counter-clockwise around the Y-axis (right-hand rule).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Matrix
    /// ```text
    /// [  cos(θ)  0  sin(θ) ]
    /// [    0     1    0    ]
    /// [ -sin(θ)  0  cos(θ) ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let rotation = Matrix3::from_rotation_y(std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec3 { x: cos, y: 0.0, z: sin },
            row_y: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
            row_z: Vec3 { x: -sin, y: 0.0, z: cos },
        }
    }

    /// Creates a rotation matrix around the Z-axis.
    ///
    /// Rotates vectors counter-clockwise around the Z-axis (right-hand rule).
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians
    ///
    /// # Matrix
    /// ```text
    /// [ cos(θ)  -sin(θ)  0 ]
    /// [ sin(θ)   cos(θ)  0 ]
    /// [   0        0     1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let rotation = Matrix3::from_rotation_z(std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec3 { x: cos, y: -sin, z: 0.0 },
            row_y: Vec3 { x: sin, y: cos, z: 0.0 },
            row_z: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
        }
    }

    /// Creates a rotation matrix around an arbitrary axis.
    ///
    /// Uses Rodrigues' rotation formula. The axis should be normalized.
    ///
    /// # Arguments
    ///
    /// * `axis` - Rotation axis (should be normalized)
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let axis = Vec3::new(1.0, 1.0, 0.0).with_normalize();
    /// let rotation = Matrix3::from_axis_angle(axis, std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            row_x: Vec3 {
                x: cos + x * x * one_minus_cos,
                y: x * y * one_minus_cos - z * sin,
                z: x * z * one_minus_cos + y * sin,
            },
            row_y: Vec3 {
                x: y * x * one_minus_cos + z * sin,
                y: cos + y * y * one_minus_cos,
                z: y * z * one_minus_cos - x * sin,
            },
            row_z: Vec3 {
                x: z * x * one_minus_cos - y * sin,
                y: z * y * one_minus_cos + x * sin,
                z: cos + z * z * one_minus_cos,
            },
        }
    }

    /// Creates a non-uniform scale matrix.
    ///
    /// Scales vectors by different factors along x, y, and z axes.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factors for x, y, and z axes
    ///
    /// # Matrix
    /// ```text
    /// [ scale.x    0        0     ]
    /// [   0      scale.y    0     ]
    /// [   0        0      scale.z ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
    /// let v = Vec3::new(1.0, 1.0, 1.0);
    /// let scaled = v * scale;
    /// assert_eq!(scaled, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            row_x: Vec3 { x: scale.x, y: 0.0, z: 0.0 },
            row_y: Vec3 { x: 0.0, y: scale.y, z: 0.0 },
            row_z: Vec3 { x: 0.0, y: 0.0, z: scale.z },
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let scale = Matrix3::from_scale_uniform(2.0);
    /// let v = Vec3::new(1.0, 1.0, 1.0);
    /// let scaled = v * scale;
    /// assert_eq!(scaled, Vec3::new(2.0, 2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale_uniform(scale: f32) -> Self {
        Self {
            row_x: Vec3 { x: scale, y: 0.0, z: 0.0 },
            row_y: Vec3 { x: 0.0, y: scale, z: 0.0 },
            row_z: Vec3 { x: 0.0, y: 0.0, z: scale },
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
    /// * `col_z` - Third column vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::from_cols(
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(col_x: Vec3, col_y: Vec3, col_z: Vec3) -> Self {
        Self {
            row_x: Vec3 { x: col_x.x, y: col_y.x, z: col_z.x },
            row_y: Vec3 { x: col_x.y, y: col_y.y, z: col_z.y },
            row_z: Vec3 { x: col_x.z, y: col_y.z, z: col_z.z },
        }
    }

    // Constants

    /// Identity matrix (multiplicative identity)
    /// ```text
    /// [ 1  0  0 ]
    /// [ 0  1  0 ]
    /// [ 0  0  1 ]
    /// ```
    pub const IDENTITY: Self = Self {
        row_x: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
        row_y: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
        row_z: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
    };

    /// Zero matrix (additive identity)
    /// ```text
    /// [ 0  0  0 ]
    /// [ 0  0  0 ]
    /// [ 0  0  0 ]
    /// ```
    pub const ZERO: Self = Self {
        row_x: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        row_y: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        row_z: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
    };

    // Core Operations

    /// Computes the transpose of this matrix.
    ///
    /// The transpose swaps rows and columns. For a rotation matrix,
    /// the transpose equals the inverse.
    ///
    /// # Formula
    /// ```text
    /// [ a  b  c ]ᵀ   [ a  d  g ]
    /// [ d  e  f ]  = [ b  e  h ]
    /// [ g  h  i ]    [ c  f  i ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::new(
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// );
    /// let transposed = m.transpose();
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            row_x: Vec3 {
                x: self.row_x.x,
                y: self.row_y.x,
                z: self.row_z.x,
            },
            row_y: Vec3 {
                x: self.row_x.y,
                y: self.row_y.y,
                z: self.row_z.y,
            },
            row_z: Vec3 {
                x: self.row_x.z,
                y: self.row_y.z,
                z: self.row_z.z,
            },
        }
    }

    /// Computes the determinant of this matrix.
    ///
    /// The determinant indicates:
    /// - Zero: Matrix is singular (non-invertible)
    /// - Positive: Preserves orientation
    /// - Negative: Reverses orientation
    /// - Magnitude: Scale factor of volume transformation
    ///
    /// # Formula
    /// det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// assert_eq!(m.determinant(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let a = self.row_x.x;
        let b = self.row_x.y;
        let c = self.row_x.z;
        let d = self.row_y.x;
        let e = self.row_y.y;
        let f = self.row_y.z;
        let g = self.row_z.x;
        let h = self.row_z.y;
        let i = self.row_z.z;

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// Computes the inverse of this matrix.
    ///
    /// Returns `None` if the matrix is singular (determinant near zero).
    /// The inverse matrix satisfies: M × M⁻¹ = I (identity)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// if let Some(inv) = m.inverse() {
    ///     let identity = m * inv;
    ///     // identity ≈ Matrix3::IDENTITY
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

        let a = self.row_x.x;
        let b = self.row_x.y;
        let c = self.row_x.z;
        let d = self.row_y.x;
        let e = self.row_y.y;
        let f = self.row_y.z;
        let g = self.row_z.x;
        let h = self.row_z.y;
        let i = self.row_z.z;

        Some(Self {
            row_x: Vec3 {
                x: (e * i - f * h) * inv_det,
                y: (c * h - b * i) * inv_det,
                z: (b * f - c * e) * inv_det,
            },
            row_y: Vec3 {
                x: (f * g - d * i) * inv_det,
                y: (a * i - c * g) * inv_det,
                z: (c * d - a * f) * inv_det,
            },
            row_z: Vec3 {
                x: (d * h - e * g) * inv_det,
                y: (b * g - a * h) * inv_det,
                z: (a * e - b * d) * inv_det,
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let a = Matrix3::IDENTITY;
    /// let b = Matrix3::IDENTITY;
    /// let dot = a.frobenius_dot(&b);
    /// assert_eq!(dot, 3.0); // Sum of diagonal elements
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_dot(&self, other: &Self) -> f32 {
        self.row_x.x * other.row_x.x
            + self.row_x.y * other.row_x.y
            + self.row_x.z * other.row_x.z
            + self.row_y.x * other.row_y.x
            + self.row_y.y * other.row_y.y
            + self.row_y.z * other.row_y.z
            + self.row_z.x * other.row_z.x
            + self.row_z.y * other.row_z.y
            + self.row_z.z * other.row_z.z
    }

    /// Computes the trace of this matrix.
    ///
    /// The trace is the sum of diagonal elements. For rotation matrices,
    /// the trace relates to the rotation angle.
    ///
    /// # Formula
    /// trace(A) = a₁₁ + a₂₂ + a₃₃
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// assert_eq!(m.trace(), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn trace(&self) -> f32 {
        self.row_x.x + self.row_y.y + self.row_z.z
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// let norm = m.frobenius_norm();
    /// assert!((norm - 1.732).abs() < 0.01); // √3
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// assert_eq!(m.frobenius_norm_squared(), 3.0);
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
    /// let v = Vec3::new(1.0, 1.0, 1.0);
    /// let result = m.transform_vector(v);
    /// assert_eq!(result, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vec: Vec3) -> Vec3 {
        vec * (*self)
    }

    /// Extracts the column vectors from this matrix.
    ///
    /// Returns a tuple of (column_x, column_y, column_z).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::new(
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// );
    /// let (col_x, col_y, col_z) = m.cols();
    /// assert_eq!(col_x, Vec3::new(1.0, 4.0, 7.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn cols(&self) -> (Vec3, Vec3, Vec3) {
        (
            Vec3 { x: self.row_x.x, y: self.row_y.x, z: self.row_z.x },
            Vec3 { x: self.row_x.y, y: self.row_y.y, z: self.row_z.y },
            Vec3 { x: self.row_x.z, y: self.row_y.z, z: self.row_z.z },
        )
    }

    /// Returns the row vectors as a tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::new(
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// );
    /// let (row_x, row_y, row_z) = m.rows();
    /// assert_eq!(row_x, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn rows(&self) -> (Vec3, Vec3, Vec3) {
        (self.row_x, self.row_y, self.row_z)
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let a = Matrix3::ZERO;
    /// let b = Matrix3::IDENTITY;
    /// let mid = a.lerp(&b, 0.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            row_x: self.row_x.lerp(&other.row_x, t),
            row_y: self.row_y.lerp(&other.row_y, t),
            row_z: self.row_z.lerp(&other.row_z, t),
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let a = Matrix3::IDENTITY;
    /// let b = Matrix3::IDENTITY;
    /// assert!(a.approx_eq(&b, f32::EPSILON));
    /// ```
    #[inline]
    #[must_use]
    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        (self.row_x.x - other.row_x.x).abs() < epsilon
            && (self.row_x.y - other.row_x.y).abs() < epsilon
            && (self.row_x.z - other.row_x.z).abs() < epsilon
            && (self.row_y.x - other.row_y.x).abs() < epsilon
            && (self.row_y.y - other.row_y.y).abs() < epsilon
            && (self.row_y.z - other.row_y.z).abs() < epsilon
            && (self.row_z.x - other.row_z.x).abs() < epsilon
            && (self.row_z.y - other.row_z.y).abs() < epsilon
            && (self.row_z.z - other.row_z.z).abs() < epsilon
    }

    /// Checks if this matrix is the identity matrix (within epsilon).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// assert!(m.is_invertible());
    /// let zero = Matrix3::ZERO;
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

impl Default for Matrix3 {
    /// Returns the identity matrix as the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::default();
    /// assert!(m.is_identity());
    /// ```
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Operator Overloading - Matrix3
// ============================================================================

impl Add for Matrix3 {
    type Output = Self;

    /// Component-wise matrix addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let a = Matrix3::IDENTITY;
    /// let b = Matrix3::IDENTITY;
    /// let result = a + b;
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
            row_z: self.row_z + other.row_z,
        }
    }
}

impl Add<&Matrix3> for Matrix3 {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        Self {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
            row_z: self.row_z + other.row_z,
        }
    }
}

impl Add<&Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn add(self, other: &Matrix3) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
            row_z: self.row_z + other.row_z,
        }
    }
}

impl Add<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn add(self, other: Matrix3) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
            row_z: self.row_z + other.row_z,
        }
    }
}

impl AddAssign for Matrix3 {
    /// Component-wise matrix addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let mut m = Matrix3::IDENTITY;
    /// m += Matrix3::IDENTITY;
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.row_x += other.row_x;
        self.row_y += other.row_y;
        self.row_z += other.row_z;
    }
}

impl AddAssign<&Matrix3> for Matrix3 {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.row_x += other.row_x;
        self.row_y += other.row_y;
        self.row_z += other.row_z;
    }
}

impl Sub for Matrix3 {
    type Output = Self;

    /// Component-wise matrix subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let a = Matrix3::IDENTITY;
    /// let b = Matrix3::IDENTITY;
    /// let result = a - b;
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
            row_z: self.row_z - other.row_z,
        }
    }
}

impl Sub<&Matrix3> for Matrix3 {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        Self {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
            row_z: self.row_z - other.row_z,
        }
    }
}

impl Sub<&Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn sub(self, other: &Matrix3) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
            row_z: self.row_z - other.row_z,
        }
    }
}

impl Sub<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn sub(self, other: Matrix3) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
            row_z: self.row_z - other.row_z,
        }
    }
}

impl SubAssign for Matrix3 {
    /// Component-wise matrix subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let mut m = Matrix3::IDENTITY;
    /// m -= Matrix3::IDENTITY;
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.row_x -= other.row_x;
        self.row_y -= other.row_y;
        self.row_z -= other.row_z;
    }
}

impl SubAssign<&Matrix3> for Matrix3 {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.row_x -= other.row_x;
        self.row_y -= other.row_y;
        self.row_z -= other.row_z;
    }
}

impl Mul<f32> for Matrix3 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// Multiplies all matrix components by a scalar value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// let result = m * 2.0;
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            row_x: self.row_x * scalar,
            row_y: self.row_y * scalar,
            row_z: self.row_z * scalar,
        }
    }
}

impl Mul<f32> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn mul(self, scalar: f32) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x * scalar,
            row_y: self.row_y * scalar,
            row_z: self.row_z * scalar,
        }
    }
}

impl Mul<&f32> for Matrix3 {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: &f32) -> Self {
        Self {
            row_x: self.row_x * *scalar,
            row_y: self.row_y * *scalar,
            row_z: self.row_z * *scalar,
        }
    }
}

impl Mul<&f32> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn mul(self, scalar: &f32) -> Matrix3 {
        Matrix3 {
            row_x: self.row_x * *scalar,
            row_y: self.row_y * *scalar,
            row_z: self.row_z * *scalar,
        }
    }
}

impl Mul for Matrix3 {
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
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let rotation = Matrix3::from_rotation_y(std::f32::consts::PI / 4.0);
    /// let scale = Matrix3::from_scale(Vec3::new(2.0, 2.0, 2.0));
    /// let combined = rotation * scale; // Apply scale, then rotation
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            row_x: Vec3 {
                x: self.row_x.x * other.row_x.x + self.row_x.y * other.row_y.x + self.row_x.z * other.row_z.x,
                y: self.row_x.x * other.row_x.y + self.row_x.y * other.row_y.y + self.row_x.z * other.row_z.y,
                z: self.row_x.x * other.row_x.z + self.row_x.y * other.row_y.z + self.row_x.z * other.row_z.z,
            },
            row_y: Vec3 {
                x: self.row_y.x * other.row_x.x + self.row_y.y * other.row_y.x + self.row_y.z * other.row_z.x,
                y: self.row_y.x * other.row_x.y + self.row_y.y * other.row_y.y + self.row_y.z * other.row_z.y,
                z: self.row_y.x * other.row_x.z + self.row_y.y * other.row_y.z + self.row_y.z * other.row_z.z,
            },
            row_z: Vec3 {
                x: self.row_z.x * other.row_x.x + self.row_z.y * other.row_y.x + self.row_z.z * other.row_z.x,
                y: self.row_z.x * other.row_x.y + self.row_z.y * other.row_y.y + self.row_z.z * other.row_z.y,
                z: self.row_z.x * other.row_x.z + self.row_z.y * other.row_y.z + self.row_z.z * other.row_z.z,
            },
        }
    }
}

impl Mul<&Matrix3> for Matrix3 {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        self * *other
    }
}

impl Mul<&Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn mul(self, other: &Matrix3) -> Matrix3 {
        (*self) * (*other)
    }
}

impl Mul<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    #[inline]
    fn mul(self, other: Matrix3) -> Matrix3 {
        (*self) * other
    }
}

impl MulAssign<f32> for Matrix3 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let mut m = Matrix3::IDENTITY;
    /// m *= 2.0;
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.row_x *= scalar;
        self.row_y *= scalar;
        self.row_z *= scalar;
    }
}

impl MulAssign<&f32> for Matrix3 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        self.row_x *= *scalar;
        self.row_y *= *scalar;
        self.row_z *= *scalar;
    }
}

impl MulAssign for Matrix3 {
    /// Matrix multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let mut m = Matrix3::IDENTITY;
    /// m *= Matrix3::IDENTITY;
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<&Matrix3> for Matrix3 {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        *self *= *other;
    }
}

// ============================================================================
// Vector-Matrix Multiplication
// ============================================================================

impl Mul<Matrix3> for Vec3 {
    type Output = Vec3;

    /// Vector-matrix multiplication using the `*` operator.
    ///
    /// Transforms a vector by multiplying it with a matrix (row vector × matrix).
    /// This is the standard way to apply transformations to points/vectors.
    ///
    /// # Formula
    /// v' = v × M
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
    /// let v = Vec3::new(1.0, 1.0, 1.0);
    /// let result = v * scale;
    /// assert_eq!(result, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    fn mul(self, rhs: Matrix3) -> Self::Output {
        Vec3 {
            x: self.x * rhs.row_x.x + self.y * rhs.row_y.x + self.z * rhs.row_z.x,
            y: self.x * rhs.row_x.y + self.y * rhs.row_y.y + self.z * rhs.row_z.y,
            z: self.x * rhs.row_x.z + self.y * rhs.row_y.z + self.z * rhs.row_z.z,
        }
    }
}

impl Mul<&Matrix3> for Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: &Matrix3) -> Self::Output {
        self * (*rhs)
    }
}

impl Mul<Matrix3> for &Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Matrix3) -> Self::Output {
        (*self) * rhs
    }
}

impl Mul<&Matrix3> for &Vec3 {
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: &Matrix3) -> Self::Output {
        (*self) * (*rhs)
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<[[f32; 3]; 3]> for Matrix3 {
    /// Converts from a 2D array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m: Matrix3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();
    /// ```
    #[inline]
    fn from(arr: [[f32; 3]; 3]) -> Self {
        Self {
            row_x: Vec3 {
                x: arr[0][0],
                y: arr[0][1],
                z: arr[0][2],
            },
            row_y: Vec3 {
                x: arr[1][0],
                y: arr[1][1],
                z: arr[1][2],
            },
            row_z: Vec3 {
                x: arr[2][0],
                y: arr[2][1],
                z: arr[2][2],
            },
        }
    }
}

impl From<Matrix3> for [[f32; 3]; 3] {
    /// Converts to a 2D array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// let arr: [[f32; 3]; 3] = m.into();
    /// ```
    #[inline]
    fn from(mat: Matrix3) -> Self {
        [
            [mat.row_x.x, mat.row_x.y, mat.row_x.z],
            [mat.row_y.x, mat.row_y.y, mat.row_y.z],
            [mat.row_z.x, mat.row_z.y, mat.row_z.z],
        ]
    }
}

impl From<[f32; 9]> for Matrix3 {
    /// Converts from a flat array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m: Matrix3 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0].into();
    /// ```
    #[inline]
    fn from(arr: [f32; 9]) -> Self {
        Self {
            row_x: Vec3 {
                x: arr[0],
                y: arr[1],
                z: arr[2],
            },
            row_y: Vec3 {
                x: arr[3],
                y: arr[4],
                z: arr[5],
            },
            row_z: Vec3 {
                x: arr[6],
                y: arr[7],
                z: arr[8],
            },
        }
    }
}

impl From<Matrix3> for [f32; 9] {
    /// Converts to a flat array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// let m = Matrix3::IDENTITY;
    /// let arr: [f32; 9] = m.into();
    /// assert_eq!(arr, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// ```
    #[inline]
    fn from(mat: Matrix3) -> Self {
        [
            mat.row_x.x, mat.row_x.y, mat.row_x.z,
            mat.row_y.x, mat.row_y.y, mat.row_y.z,
            mat.row_z.x, mat.row_z.y, mat.row_z.z,
        ]
    }
}

// Matrix3 Conversions for tuples

impl From<(Vec3, Vec3, Vec3)> for Matrix3 {
    /// Converts from a tuple of three Vec3 row vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m: Matrix3 = (
    ///     Vec3::new(1.0, 2.0, 3.0),
    ///     Vec3::new(4.0, 5.0, 6.0),
    ///     Vec3::new(7.0, 8.0, 9.0)
    /// ).into();
    /// ```
    #[inline]
    fn from(tuple: (Vec3, Vec3, Vec3)) -> Self {
        Self {
            row_x: tuple.0,
            row_y: tuple.1,
            row_z: tuple.2,
        }
    }
}

impl From<&(Vec3, Vec3, Vec3)> for Matrix3 {
    #[inline]
    fn from(tuple: &(Vec3, Vec3, Vec3)) -> Self {
        let mat: Matrix3 = (*tuple).into();
        mat
    }
}

impl From<Matrix3> for (Vec3, Vec3, Vec3) {
    /// Converts to a tuple of three Vec3 row vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix3;
    /// # use toyengine::primitives::vec::Vec3;
    /// let m = Matrix3::IDENTITY;
    /// let (row_x, row_y, row_z): (Vec3, Vec3, Vec3) = m.into();
    /// ```
    #[inline]
    fn from(mat: Matrix3) -> Self {
        (mat.row_x, mat.row_y, mat.row_z)
    }
}

impl From<&Matrix3> for (Vec3, Vec3, Vec3) {
    #[inline]
    fn from(mat: &Matrix3) -> Self {
        let tuple: (Vec3, Vec3, Vec3) = (*mat).into();
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
        let identity = Matrix3::identity();
        let zero = Matrix3::zero();

        assert_eq!(identity.row_x, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(identity.row_y, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(identity.row_z, Vec3::new(0.0, 0.0, 1.0));

        assert_eq!(zero.row_x, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(zero.row_y, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(zero.row_z, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_constants() {
        assert_eq!(Matrix3::IDENTITY, Matrix3::identity());
        assert_eq!(Matrix3::ZERO, Matrix3::zero());
    }

    #[test]
    fn test_splat() {
        let m = Matrix3::splat(5.0);
        assert_eq!(m.row_x, Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(m.row_y, Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(m.row_z, Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_rotation_x() {
        let rotation = Matrix3::from_rotation_x(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(0.0, 1.0, 0.0);
        let rotated = vec * rotation;

        // Y should rotate to -Z for row-vector convention
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - -1.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation_y() {
        let rotation = Matrix3::from_rotation_y(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = vec * rotation;

        // X should rotate to Z for row-vector convention
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation_z() {
        let rotation = Matrix3::from_rotation_z(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = vec * rotation;

        // X should rotate to -Y for row-vector convention
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - -1.0).abs() < 0.001);
        assert!((rotated.z - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_axis_angle_rotation() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = std::f32::consts::PI / 2.0;
        let rotation = Matrix3::from_axis_angle(axis, angle);

        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = vec * rotation;

        // Should rotate around Z-axis
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - -1.0).abs() < 0.001);
        assert!((rotated.z - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_scale() {
        let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let vec = Vec3::new(1.0, 1.0, 1.0);
        let scaled = vec * scale;
        assert_eq!(scaled, Vec3::new(2.0, 3.0, 4.0));

        let uniform = Matrix3::from_scale_uniform(2.0);
        let scaled_uniform = Vec3::new(1.0, 1.0, 1.0) * uniform;
        assert_eq!(scaled_uniform, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_from_cols() {
        let col_x = Vec3::new(1.0, 0.0, 0.0);
        let col_y = Vec3::new(0.0, 1.0, 0.0);
        let col_z = Vec3::new(0.0, 0.0, 1.0);

        let m = Matrix3::from_cols(col_x, col_y, col_z);
        assert_eq!(m, Matrix3::IDENTITY);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let transposed = m.transpose();

        assert_eq!(transposed.row_x, Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(transposed.row_y, Vec3::new(2.0, 5.0, 8.0));
        assert_eq!(transposed.row_z, Vec3::new(3.0, 6.0, 9.0));

        // Double transpose should give original
        assert_eq!(transposed.transpose(), m);
    }

    #[test]
    fn test_determinant() {
        let identity = Matrix3::IDENTITY;
        assert_eq!(identity.determinant(), 1.0);

        let scale = Matrix3::from_scale_uniform(2.0);
        assert!((scale.determinant() - 8.0).abs() < 0.001);

        let zero = Matrix3::ZERO;
        assert_eq!(zero.determinant(), 0.0);
    }

    #[test]
    fn test_inverse() {
        let identity = Matrix3::IDENTITY;
        let inv_identity = identity.inverse().expect("Identity should be invertible");
        assert!(identity.approx_eq(&inv_identity, 0.001));

        let scale = Matrix3::from_scale_uniform(2.0);
        let inv_scale = scale.inverse().expect("Scale should be invertible");
        let product = scale * inv_scale;
        assert!(product.approx_eq(&Matrix3::IDENTITY, 0.001));

        let zero = Matrix3::ZERO;
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_trace() {
        let identity = Matrix3::IDENTITY;
        assert_eq!(identity.trace(), 3.0);

        let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(scale.trace(), 9.0);
    }

    #[test]
    fn test_frobenius_norm() {
        let identity = Matrix3::IDENTITY;
        let norm = identity.frobenius_norm();
        assert!((norm - 1.732).abs() < 0.01);

        let norm_squared = identity.frobenius_norm_squared();
        assert_eq!(norm_squared, 3.0);
    }

    #[test]
    fn test_addition() {
        let a = Matrix3::IDENTITY;
        let b = Matrix3::IDENTITY;
        let result = a + b;

        let expected = Matrix3::new(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));

        // Test borrowed
        let result_borrowed = &a + &b;
        assert!(result_borrowed.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_subtraction() {
        let a = Matrix3::from_scale_uniform(3.0);
        let b = Matrix3::from_scale_uniform(1.0);
        let result = a - b;

        let expected = Matrix3::new(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = Matrix3::IDENTITY;
        let result = m * 2.0;

        let expected = Matrix3::new(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_matrix_multiplication() {
        let scale = Matrix3::from_scale_uniform(2.0);
        let rotation = Matrix3::from_rotation_z(std::f32::consts::PI / 2.0);

        let combined = scale * rotation;
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let result = vec * combined;

        // Scale then rotate
        assert!((result.x - 0.0).abs() < 0.001);
        assert!((result.y - -2.0).abs() < 0.001);
    }

    #[test]
    fn test_assignment_operators() {
        let mut m = Matrix3::ZERO;
        m += Matrix3::IDENTITY;
        assert_eq!(m, Matrix3::IDENTITY);

        m *= 2.0;
        let expected = Matrix3::new(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        );
        assert!(m.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_vector_matrix_multiplication() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let m = Matrix3::IDENTITY;
        let result = v * m;
        assert_eq!(result, v);

        // Test with scale
        let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let scaled = v * scale;
        assert_eq!(scaled, Vec3::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_array_conversions() {
        let identity = Matrix3::IDENTITY;

        let array: [[f32; 3]; 3] = identity.into();
        let back: Matrix3 = array.into();
        assert_eq!(back, identity);

        let flat: [f32; 9] = identity.into();
        let back_flat: Matrix3 = flat.into();
        assert_eq!(back_flat, identity);
    }

    #[test]
    fn test_transform_vector() {
        let scale = Matrix3::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let vec = Vec3::new(1.0, 1.0, 1.0);
        let result = scale.transform_vector(vec);
        assert_eq!(result, Vec3::new(2.0, 3.0, 4.0));
    }

    #[test]
    fn test_cols_and_rows() {
        let m = Matrix3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );

        let (row_x, row_y, row_z) = m.rows();
        assert_eq!(row_x, m.row_x);
        assert_eq!(row_y, m.row_y);
        assert_eq!(row_z, m.row_z);

        let (col_x, col_y, col_z) = m.cols();
        assert_eq!(col_x, Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(col_y, Vec3::new(2.0, 5.0, 8.0));
        assert_eq!(col_z, Vec3::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn test_lerp() {
        let a = Matrix3::ZERO;
        let b = Matrix3::from_scale_uniform(2.0);
        let mid = a.lerp(&b, 0.5);

        let expected = Matrix3::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(mid.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_approx_eq() {
        let a = Matrix3::IDENTITY;
        let b = Matrix3::IDENTITY;
        assert!(a.approx_eq(&b, f32::EPSILON));

        let c = Matrix3::from_scale_uniform(1.001);
        assert!(!a.approx_eq(&c, 0.0001));
    }

    #[test]
    fn test_default_trait() {
        let default_mat: Matrix3 = Default::default();
        assert_eq!(default_mat, Matrix3::IDENTITY);
    }

    #[test]
    fn test_frobenius_dot() {
        let a = Matrix3::IDENTITY;
        let b = Matrix3::IDENTITY;
        let dot = a.frobenius_dot(&b);
        assert_eq!(dot, 3.0);
    }
}