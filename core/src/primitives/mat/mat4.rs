use crate::primitives::vec::{Vec3, Vec4};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A 4x4 matrix with row-major order.
///
/// The fundamental matrix type for 3D graphics, capable of representing any affine
/// transformation including translation, rotation, scaling, shearing, and perspective
/// projection. This is the workhorse of modern 3D game engines.
///
/// # Matrix Layout (Row-Major)
/// ```text
/// [ row_x.x  row_x.y  row_x.z  row_x.w ]
/// [ row_y.x  row_y.y  row_y.z  row_y.w ]
/// [ row_z.x  row_z.y  row_z.z  row_z.w ]
/// [ row_w.x  row_w.y  row_w.z  row_w.w ]
/// ```
///
/// # Homogeneous Coordinates
/// Matrix4 uses homogeneous coordinates (4D) to represent 3D transformations:
/// - The upper-left 3x3 submatrix handles rotation and scale
/// - The first three elements of row_w handle translation
/// - row_w.w is typically 1.0 for affine transformations
///
/// # Common Use Cases
/// - Model transformations (translate, rotate, scale objects)
/// - View matrix (camera positioning and orientation)
/// - Projection matrix (perspective or orthographic)
/// - Model-View-Projection (MVP) matrix for rendering
/// - Skeletal animation and bone transformations
/// - Shadow mapping transformations
///
/// # Matrix Multiplication Order
/// This implementation uses row-major storage with row-vector multiplication:
/// ```text
/// result = vector * matrix
/// combined = matrix1 * matrix2
/// ```
///
/// # Examples
///
/// ```rust
/// # use toyengine::primitives::mat::Matrix4;
/// # use toyengine::primitives::vec::{Vec3, Vec4};
/// // Create a transformation matrix
/// let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));
/// let rotation = Matrix4::from_rotation_y(std::f32::consts::PI / 4.0);
/// let scale = Matrix4::from_scale(Vec3::new(2.0, 2.0, 2.0));
///
/// // Combine transformations (order matters!)
/// let model = translation * rotation * scale;
///
/// // Create a camera view matrix
/// let view = Matrix4::look_at(
///     Vec3::new(0.0, 0.0, 10.0), // eye position
///     Vec3::new(0.0, 0.0, 0.0),  // look at target
///     Vec3::new(0.0, 1.0, 0.0)   // up direction
/// );
///
/// // Create a perspective projection
/// let projection = Matrix4::perspective(
///     std::f32::consts::PI / 4.0, // 45 degree FOV
///     16.0 / 9.0,                 // aspect ratio
///     0.1,                        // near plane
///     100.0                       // far plane
/// );
///
/// // Full MVP matrix
/// let mvp = model * view * projection;
///
/// // Transform a point
/// let point = Vec3::new(1.0, 2.0, 3.0);
/// let transformed = mvp.transform_point(point);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Matrix4 {
    /// First row of the matrix (x-axis in transformation context)
    pub row_x: Vec4,
    /// Second row of the matrix (y-axis in transformation context)
    pub row_y: Vec4,
    /// Third row of the matrix (z-axis in transformation context)
    pub row_z: Vec4,
    /// Fourth row of the matrix (translation and homogeneous coordinate)
    pub row_w: Vec4,
}

// ============================================================================
// Matrix4 Implementation
// ============================================================================

impl Matrix4 {
    // Constructors

    /// Creates a new Matrix4 from four row vectors.
    ///
    /// The matrix is constructed in row-major order where `row_x`, `row_y`, `row_z`,
    /// and `row_w` represent the first, second, third, and fourth rows respectively.
    ///
    /// # Arguments
    ///
    /// * `row_x` - The first row vector (top row)
    /// * `row_y` - The second row vector (second row)
    /// * `row_z` - The third row vector (third row)
    /// * `row_w` - The fourth row vector (bottom row, typically translation)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec4;
    /// let m = Matrix4::new(
    ///     Vec4::new(1.0, 0.0, 0.0, 0.0),
    ///     Vec4::new(0.0, 1.0, 0.0, 0.0),
    ///     Vec4::new(0.0, 0.0, 1.0, 0.0),
    ///     Vec4::new(0.0, 0.0, 0.0, 1.0)
    /// );
    /// ```
    #[inline]
    pub const fn new(row_x: Vec4, row_y: Vec4, row_z: Vec4, row_w: Vec4) -> Self {
        Self {
            row_x,
            row_y,
            row_z,
            row_w,
        }
    }

    /// Creates an identity matrix.
    ///
    /// The identity matrix leaves vectors and points unchanged when multiplied.
    /// This is the multiplicative identity for matrix operations.
    ///
    /// # Matrix
    /// ```text
    /// [ 1  0  0  0 ]
    /// [ 0  1  0  0 ]
    /// [ 0  0  1  0 ]
    /// [ 0  0  0  1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let identity = Matrix4::identity();
    /// let point = Vec3::new(3.0, 4.0, 5.0);
    /// let result = identity.transform_point(point);
    /// assert_eq!(result, point); // Point unchanged
    /// ```
    #[inline]
    pub const fn identity() -> Self {
        Self {
            row_x: Vec4::new(1.0, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, 1.0, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, 1.0, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Creates a zero matrix with all components set to zero.
    ///
    /// # Matrix
    /// ```text
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let zero = Matrix4::zero();
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self {
            row_x: Vec4::ZERO,
            row_y: Vec4::ZERO,
            row_z: Vec4::ZERO,
            row_w: Vec4::ZERO,
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
    /// # use toyengine::primitives::mat::Matrix4;
    /// let uniform = Matrix4::splat(5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            row_x: Vec4::splat(value),
            row_y: Vec4::splat(value),
            row_z: Vec4::splat(value),
            row_w: Vec4::splat(value),
        }
    }

    // Constants

    /// Identity matrix (multiplicative identity)
    /// ```text
    /// [ 1  0  0  0 ]
    /// [ 0  1  0  0 ]
    /// [ 0  0  1  0 ]
    /// [ 0  0  0  1 ]
    /// ```
    pub const IDENTITY: Self = Self {
        row_x: Vec4::new(1.0, 0.0, 0.0, 0.0),
        row_y: Vec4::new(0.0, 1.0, 0.0, 0.0),
        row_z: Vec4::new(0.0, 0.0, 1.0, 0.0),
        row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
    };

    /// Zero matrix (additive identity)
    /// ```text
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// [ 0  0  0  0 ]
    /// ```
    pub const ZERO: Self = Self {
        row_x: Vec4::ZERO,
        row_y: Vec4::ZERO,
        row_z: Vec4::ZERO,
        row_w: Vec4::ZERO,
    };

    // Transformation Constructors

    /// Creates a translation matrix.
    ///
    /// Translates (moves) points in 3D space by the given offset.
    ///
    /// # Arguments
    ///
    /// * `translation` - Translation offset in x, y, and z
    ///
    /// # Matrix
    /// ```text
    /// [ 1  0  0  0 ]
    /// [ 0  1  0  0 ]
    /// [ 0  0  1  0 ]
    /// [ tx ty tz 1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let translation = Matrix4::from_translation(Vec3::new(10.0, 20.0, 30.0));
    /// let point = Vec3::new(1.0, 2.0, 3.0);
    /// let moved = translation.transform_point(point);
    /// assert_eq!(moved, Vec3::new(11.0, 22.0, 33.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            row_x: Vec4::new(1.0, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, 1.0, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, 1.0, 0.0),
            row_w: Vec4::new(translation.x, translation.y, translation.z, 1.0),
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
    /// [ 1    0        0     0 ]
    /// [ 0  cos(θ)  -sin(θ) 0 ]
    /// [ 0  sin(θ)   cos(θ) 0 ]
    /// [ 0    0        0     1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let angle = std::f32::consts::PI / 2.0; // 90 degrees
    /// let rotation = Matrix4::from_rotation_x(angle);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec4::new(1.0, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, cos, sin, 0.0),
            row_z: Vec4::new(0.0, -sin, cos, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    /// [  cos(θ)  0  sin(θ) 0 ]
    /// [    0     1    0    0 ]
    /// [ -sin(θ)  0  cos(θ) 0 ]
    /// [    0     0    0    1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let rotation = Matrix4::from_rotation_y(std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec4::new(cos, 0.0, -sin, 0.0),
            row_y: Vec4::new(0.0, 1.0, 0.0, 0.0),
            row_z: Vec4::new(sin, 0.0, cos, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    /// [ cos(θ)  -sin(θ)  0  0 ]
    /// [ sin(θ)   cos(θ)  0  0 ]
    /// [   0        0     1  0 ]
    /// [   0        0     0  1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let rotation = Matrix4::from_rotation_z(std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            row_x: Vec4::new(cos, sin, 0.0, 0.0),
            row_y: Vec4::new(-sin, cos, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, 1.0, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Creates a rotation matrix around an arbitrary axis.
    ///
    /// Uses Rodrigues' rotation formula to compute the rotation matrix.
    /// The axis should be normalized for correct results.
    ///
    /// # Arguments
    ///
    /// * `axis` - Rotation axis (should be normalized)
    /// * `angle` - Rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let axis = Vec3::new(1.0, 1.0, 0.0).with_normalize();
    /// let rotation = Matrix4::from_axis_angle(axis, std::f32::consts::PI / 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        let xx = x * x * one_minus_cos;
        let yy = y * y * one_minus_cos;
        let zz = z * z * one_minus_cos;
        let xy = x * y * one_minus_cos;
        let xz = x * z * one_minus_cos;
        let yz = y * z * one_minus_cos;

        Self {
            row_x: Vec4::new(xx + cos, xy - z * sin, xz + y * sin, 0.0),
            row_y: Vec4::new(xy + z * sin, yy + cos, yz - x * sin, 0.0),
            row_z: Vec4::new(xz - y * sin, yz + x * sin, zz + cos, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    /// [ scale.x    0        0     0 ]
    /// [   0      scale.y    0     0 ]
    /// [   0        0      scale.z 0 ]
    /// [   0        0        0     1 ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let scale = Matrix4::from_scale(Vec3::new(2.0, 3.0, 4.0));
    /// let point = Vec3::new(1.0, 1.0, 1.0);
    /// let scaled = scale.transform_point(point);
    /// assert_eq!(scaled, Vec3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            row_x: Vec4::new(scale.x, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, scale.y, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, scale.z, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let scale = Matrix4::from_scale_uniform(2.0);
    /// let point = Vec3::new(1.0, 1.0, 1.0);
    /// let scaled = scale.transform_point(point);
    /// assert_eq!(scaled, Vec3::new(2.0, 2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale_uniform(scale: f32) -> Self {
        Self {
            row_x: Vec4::new(scale, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, scale, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, scale, 0.0),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
    /// * `col_w` - Fourth column vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec4;
    /// let m = Matrix4::from_cols(
    ///     Vec4::new(1.0, 2.0, 3.0, 4.0),
    ///     Vec4::new(5.0, 6.0, 7.0, 8.0),
    ///     Vec4::new(9.0, 10.0, 11.0, 12.0),
    ///     Vec4::new(13.0, 14.0, 15.0, 16.0)
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(col_x: Vec4, col_y: Vec4, col_z: Vec4, col_w: Vec4) -> Self {
        Self {
            row_x: Vec4::new(col_x.x, col_y.x, col_z.x, col_w.x),
            row_y: Vec4::new(col_x.y, col_y.y, col_z.y, col_w.y),
            row_z: Vec4::new(col_x.z, col_y.z, col_z.z, col_w.z),
            row_w: Vec4::new(col_x.w, col_y.w, col_z.w, col_w.w),
        }
    }
    // Camera and Projection Constructors

    /// Creates a perspective projection matrix.
    ///
    /// This creates a frustum projection matrix commonly used for realistic 3D rendering.
    /// Objects further from the camera appear smaller (perspective foreshortening).
    ///
    /// # Arguments
    ///
    /// * `fov_y` - Vertical field of view in radians
    /// * `aspect_ratio` - Aspect ratio (width / height)
    /// * `near` - Near clipping plane distance (must be > 0)
    /// * `far` - Far clipping plane distance (must be > near)
    ///
    /// # Coordinate System
    /// - Assumes right-handed coordinate system
    /// - Maps to NDC (Normalized Device Coordinates) in range [-1, 1]
    ///
    /// # Examples
    ///
    /// ``rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let projection = Matrix4::perspective(
    ///     std::f32::consts::PI / 4.0, // 45 degree FOV
    ///     16.0 / 9.0,                 // 16:9 aspect ratio
    ///     0.1,                        // near plane at 0.1
    ///     1000.0                      // far plane at 1000.0
    /// );
    /// ``
    #[inline]
    #[must_use]
    pub fn perspective(fov_y: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let range_inv = 1.0 / (near - far);

        Self {
            row_x: Vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, f, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, (near + far) * range_inv, -1.0),
            row_w: Vec4::new(0.0, 0.0, 2.0 * near * far * range_inv, 0.0),
        }
    }

    /// Creates an orthographic projection matrix.
    ///
    /// This creates a parallel projection without perspective foreshortening.
    /// Objects maintain their size regardless of distance from the camera.
    /// Commonly used for 2D rendering, UI, and CAD applications.
    ///
    /// # Arguments
    ///
    /// * `left` - Left edge of the view volume
    /// * `right` - Right edge of the view volume
    /// * `bottom` - Bottom edge of the view volume
    /// * `top` - Top edge of the view volume
    /// * `near` - Near clipping plane distance
    /// * `far` - Far clipping plane distance
    ///
    /// # Examples
    ///
    /// ``rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// // Create an orthographic projection for a 800x600 viewport
    /// let projection = Matrix4::orthographic(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);
    /// ``
    #[inline]
    #[must_use]
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let width_inv = 1.0 / (right - left);
        let height_inv = 1.0 / (top - bottom);
        let depth_inv = 1.0 / (far - near);

        Self {
            row_x: Vec4::new(2.0 * width_inv, 0.0, 0.0, 0.0),
            row_y: Vec4::new(0.0, 2.0 * height_inv, 0.0, 0.0),
            row_z: Vec4::new(0.0, 0.0, -2.0 * depth_inv, 0.0),
            row_w: Vec4::new(
                -(right + left) * width_inv,
                -(top + bottom) * height_inv,
                -(far + near) * depth_inv,
                1.0,
            ),
        }
    }

    /// Creates a view matrix looking at a target from an eye position.
    ///
    /// This creates a view transformation matrix that positions and orients a camera.
    /// The resulting matrix transforms world coordinates to view (camera) space.
    ///
    /// # Arguments
    ///
    /// * `eye` - Camera position in world space
    /// * `target` - Point the camera is looking at
    /// * `up` - Up direction vector (typically Vec3::new(0.0, 1.0, 0.0))
    ///
    /// # Coordinate System
    /// - Assumes right-handed coordinate system
    /// - Camera looks down the negative Z-axis in view space
    ///
    /// # Examples
    ///
    /// ``rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let view = Matrix4::look_at(
    ///     Vec3::new(0.0, 5.0, 10.0), // Camera at (0, 5, 10)
    ///     Vec3::new(0.0, 0.0, 0.0),  // Looking at origin
    ///     Vec3::new(0.0, 1.0, 0.0)   // Y is up
    /// );
    /// ``
    #[inline]
    #[must_use]
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).with_normalize(); // Forward (toward target)
        let s = f.cross(&up).with_normalize();   // Right
        let u = s.cross(&f);                     // Up

        Self {
            row_x: Vec4::new(s.x, s.y, s.z, -s.dot(&eye)),
            row_y: Vec4::new(u.x, u.y, u.z, -u.dot(&eye)),
            row_z: Vec4::new(-f.x, -f.y, -f.z, f.dot(&eye)),
            row_w: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    // Core Operations

    /// Computes the transpose of this matrix.
    ///
    /// The transpose swaps rows and columns. For orthogonal matrices (pure rotations),
    /// the transpose equals the inverse.
    ///
    /// # Formula
    /// ```text
    /// [ a  b  c  d ]ᵀ   [ a  e  i  m ]
    /// [ e  f  g  h ]  = [ b  f  j  n ]
    /// [ i  j  k  l ]    [ c  g  k  o ]
    /// [ m  n  o  p ]    [ d  h  l  p ]
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let transposed = m.transpose();
    /// assert_eq!(transposed, m); // Identity is symmetric
    /// ```
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            row_x: Vec4::new(self.row_x.x, self.row_y.x, self.row_z.x, self.row_w.x),
            row_y: Vec4::new(self.row_x.y, self.row_y.y, self.row_z.y, self.row_w.y),
            row_z: Vec4::new(self.row_x.z, self.row_y.z, self.row_z.z, self.row_w.z),
            row_w: Vec4::new(self.row_x.w, self.row_y.w, self.row_z.w, self.row_w.w),
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
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// assert_eq!(m.determinant(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let a = self.row_x;
        let b = self.row_y;
        let c = self.row_z;
        let d = self.row_w;

        let det_a = a.x * (
            b.y * (c.z * d.w - c.w * d.z) -
            b.z * (c.y * d.w - c.w * d.y) +
            b.w * (c.y * d.z - c.z * d.y)
        );

        let det_b = a.y * (
            b.x * (c.z * d.w - c.w * d.z) -
            b.z * (c.x * d.w - c.w * d.x) +
            b.w * (c.x * d.z - c.z * d.x)
        );

        let det_c = a.z * (
            b.x * (c.y * d.w - c.w * d.y) -
            b.y * (c.x * d.w - c.w * d.x) +
            b.w * (c.x * d.y - c.y * d.x)
        );

        let det_d = a.w * (
            b.x * (c.y * d.z - c.z * d.y) -
            b.y * (c.x * d.z - c.z * d.x) +
            b.z * (c.x * d.y - c.y * d.x)
        );

        det_a - det_b + det_c - det_d
    }

    /// Computes the inverse of this matrix.
    ///
    /// Returns `None` if the matrix is singular (determinant near zero).
    /// The inverse matrix satisfies: M × M⁻¹ = I (identity)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::from_scale_uniform(2.0);
    /// if let Some(inv) = m.inverse() {
    ///     let identity = m * inv;
    ///     // identity ≈ Matrix4::IDENTITY
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

        // Calculate cofactor matrix and transpose (adjugate)
        let a = self.row_x;
        let b = self.row_y;
        let c = self.row_z;
        let d = self.row_w;

        Some(Self {
            row_x: Vec4::new(
                inv_det * (b.y * (c.z * d.w - c.w * d.z) - b.z * (c.y * d.w - c.w * d.y) + b.w * (c.y * d.z - c.z * d.y)),
                inv_det * -(a.y * (c.z * d.w - c.w * d.z) - a.z * (c.y * d.w - c.w * d.y) + a.w * (c.y * d.z - c.z * d.y)),
                inv_det * (a.y * (b.z * d.w - b.w * d.z) - a.z * (b.y * d.w - b.w * d.y) + a.w * (b.y * d.z - b.z * d.y)),
                inv_det * -(a.y * (b.z * c.w - b.w * c.z) - a.z * (b.y * c.w - b.w * c.y) + a.w * (b.y * c.z - b.z * c.y)),
            ),
            row_y: Vec4::new(
                inv_det * -(b.x * (c.z * d.w - c.w * d.z) - b.z * (c.x * d.w - c.w * d.x) + b.w * (c.x * d.z - c.z * d.x)),
                inv_det * (a.x * (c.z * d.w - c.w * d.z) - a.z * (c.x * d.w - c.w * d.x) + a.w * (c.x * d.z - c.z * d.x)),
                inv_det * -(a.x * (b.z * d.w - b.w * d.z) - a.z * (b.x * d.w - b.w * d.x) + a.w * (b.x * d.z - b.z * d.x)),
                inv_det * (a.x * (b.z * c.w - b.w * c.z) - a.z * (b.x * c.w - b.w * c.x) + a.w * (b.x * c.z - b.z * c.x)),
            ),
            row_z: Vec4::new(
                inv_det * (b.x * (c.y * d.w - c.w * d.y) - b.y * (c.x * d.w - c.w * d.x) + b.w * (c.x * d.y - c.y * d.x)),
                inv_det * -(a.x * (c.y * d.w - c.w * d.y) - a.y * (c.x * d.w - c.w * d.x) + a.w * (c.x * d.y - c.y * d.x)),
                inv_det * (a.x * (b.y * d.w - b.w * d.y) - a.y * (b.x * d.w - b.w * d.x) + a.w * (b.x * d.y - b.y * d.x)),
                inv_det * -(a.x * (b.y * c.w - b.w * c.y) - a.y * (b.x * c.w - b.w * c.x) + a.w * (b.x * c.y - b.y * c.x)),
            ),
            row_w: Vec4::new(
                inv_det * -(b.x * (c.y * d.z - c.z * d.y) - b.y * (c.x * d.z - c.z * d.x) + b.z * (c.x * d.y - c.y * d.x)),
                inv_det * (a.x * (c.y * d.z - c.z * d.y) - a.y * (c.x * d.z - c.z * d.x) + a.z * (c.x * d.y - c.y * d.x)),
                inv_det * -(a.x * (b.y * d.z - b.z * d.y) - a.y * (b.x * d.z - b.z * d.x) + a.z * (b.x * d.y - b.y * d.x)),
                inv_det * (a.x * (b.y * c.z - b.z * c.y) - a.y * (b.x * c.z - b.z * c.x) + a.z * (b.x * c.y - b.y * c.x)),
            ),
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
    /// # use toyengine::primitives::mat::Matrix4;
    /// let a = Matrix4::IDENTITY;
    /// let b = Matrix4::IDENTITY;
    /// let dot = a.frobenius_dot(&b);
    /// assert_eq!(dot, 4.0); // Sum of diagonal elements
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_dot(&self, other: &Self) -> f32 {
        self.row_x.dot(&other.row_x) +
        self.row_y.dot(&other.row_y) +
        self.row_z.dot(&other.row_z) +
        self.row_w.dot(&other.row_w)
    }

    /// Computes the trace of this matrix.
    ///
    /// The trace is the sum of diagonal elements.
    ///
    /// # Formula
    /// trace(A) = a₁₁ + a₂₂ + a₃₃ + a₄₄
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// assert_eq!(m.trace(), 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn trace(&self) -> f32 {
        self.row_x.x + self.row_y.y + self.row_z.z + self.row_w.w
    }

    /// Computes the Frobenius norm of this matrix.
    ///
    /// The Frobenius norm is the square root of the sum of squared elements.
    ///
    /// # Formula
    /// ‖A‖_F = √(Σ aᵢⱼ²)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let norm = m.frobenius_norm();
    /// assert!((norm - 2.0).abs() < 0.01); // √4 = 2
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm(&self) -> f32 {
        self.frobenius_norm_squared().sqrt()
    }

    /// Computes the squared Frobenius norm of this matrix.
    ///
    /// Cheaper than [`frobenius_norm`](Self::frobenius_norm) as it avoids the square root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// assert_eq!(m.frobenius_norm_squared(), 4.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn frobenius_norm_squared(&self) -> f32 {
        self.frobenius_dot(self)
    }

    // Utility Methods

    /// Transforms a 4D vector by this matrix.
    ///
    /// Equivalent to `vec * matrix` but can be more readable in some contexts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec4;
    /// let m = Matrix4::IDENTITY;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// let result = m.transform_vector(v);
    /// assert_eq!(result, v);
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vec: Vec4) -> Vec4 {
        vec * (*self)
    }

    /// Transforms a 3D point by this matrix.
    ///
    /// Treats the point as a homogeneous coordinate with w=1.0 (affected by translation).
    /// After transformation, divides by w to get the resulting 3D point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    /// let point = Vec3::new(1.0, 2.0, 3.0);
    /// let result = translation.transform_point(point);
    /// assert_eq!(result, Vec3::new(11.0, 2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        let v = Vec4::new(point.x, point.y, point.z, 1.0);
        let transformed = v * (*self);

        // Perspective divide
        if transformed.w != 0.0 && transformed.w != 1.0 {
            Vec3::new(
                transformed.x / transformed.w,
                transformed.y / transformed.w,
                transformed.z / transformed.w,
            )
        } else {
            Vec3::new(transformed.x, transformed.y, transformed.z)
        }
    }

    /// Transforms a 3D direction vector by this matrix.
    ///
    /// Treats the vector as a homogeneous coordinate with w=0.0 (not affected by translation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    /// let direction = Vec3::new(1.0, 0.0, 0.0);
    /// let result = translation.transform_direction(direction);
    /// assert_eq!(result, direction); // Direction unaffected by translation
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_direction(&self, direction: Vec3) -> Vec3 {
        let v = Vec4::new(direction.x, direction.y, direction.z, 0.0);
        let transformed = v * (*self);
        Vec3::new(transformed.x, transformed.y, transformed.z)
    }

    /// Extracts the column vectors from this matrix.
    ///
    /// Returns a tuple of (column_x, column_y, column_z, column_w).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let (col_x, col_y, col_z, col_w) = m.cols();
    /// ```
    #[inline]
    #[must_use]
    pub fn cols(&self) -> (Vec4, Vec4, Vec4, Vec4) {
        (
            Vec4::new(self.row_x.x, self.row_y.x, self.row_z.x, self.row_w.x),
            Vec4::new(self.row_x.y, self.row_y.y, self.row_z.y, self.row_w.y),
            Vec4::new(self.row_x.z, self.row_y.z, self.row_z.z, self.row_w.z),
            Vec4::new(self.row_x.w, self.row_y.w, self.row_z.w, self.row_w.w),
        )
    }

    /// Returns the row vectors as a tuple.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let (row_x, row_y, row_z, row_w) = m.rows();
    /// ```
    #[inline]
    #[must_use]
    pub fn rows(&self) -> (Vec4, Vec4, Vec4, Vec4) {
        (self.row_x, self.row_y, self.row_z, self.row_w)
    }

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
    /// # use toyengine::primitives::mat::Matrix4;
    /// let a = Matrix4::ZERO;
    /// let b = Matrix4::IDENTITY;
    /// let mid = a.lerp(&b, 0.5);
    /// ```
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            row_x: self.row_x.lerp(&other.row_x, t),
            row_y: self.row_y.lerp(&other.row_y, t),
            row_z: self.row_z.lerp(&other.row_z, t),
            row_w: self.row_w.lerp(&other.row_w, t),
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
    /// # use toyengine::primitives::mat::Matrix4;
    /// let a = Matrix4::IDENTITY;
    /// let b = Matrix4::IDENTITY;
    /// assert!(a.approx_eq(&b, f32::EPSILON));
    /// ```
    #[inline]
    #[must_use]
    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        (self.row_x.x - other.row_x.x).abs() < epsilon &&
        (self.row_x.y - other.row_x.y).abs() < epsilon &&
        (self.row_x.z - other.row_x.z).abs() < epsilon &&
        (self.row_x.w - other.row_x.w).abs() < epsilon &&
        (self.row_y.x - other.row_y.x).abs() < epsilon &&
        (self.row_y.y - other.row_y.y).abs() < epsilon &&
        (self.row_y.z - other.row_y.z).abs() < epsilon &&
        (self.row_y.w - other.row_y.w).abs() < epsilon &&
        (self.row_z.x - other.row_z.x).abs() < epsilon &&
        (self.row_z.y - other.row_z.y).abs() < epsilon &&
        (self.row_z.z - other.row_z.z).abs() < epsilon &&
        (self.row_z.w - other.row_z.w).abs() < epsilon &&
        (self.row_w.x - other.row_w.x).abs() < epsilon &&
        (self.row_w.y - other.row_w.y).abs() < epsilon &&
        (self.row_w.z - other.row_w.z).abs() < epsilon &&
        (self.row_w.w - other.row_w.w).abs() < epsilon
    }
}

// ============================================================================
// Default Trait
// ============================================================================

impl Default for Matrix4 {
    /// Returns the identity matrix as the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m: Matrix4 = Default::default();
    /// assert_eq!(m, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ============================================================================
// Operator Overloading - Matrix4
// ============================================================================

impl Add for Matrix4 {
    type Output = Self;

    /// Component-wise matrix addition using the `+` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let a = Matrix4::IDENTITY;
    /// let b = Matrix4::IDENTITY;
    /// let result = a + b;
    /// ```
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            row_x: self.row_x + other.row_x,
            row_y: self.row_y + other.row_y,
            row_z: self.row_z + other.row_z,
            row_w: self.row_w + other.row_w,
        }
    }
}

impl Add<&Matrix4> for Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn add(self, other: &Matrix4) -> Matrix4 {
        self + *other
    }
}

impl Add<&Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn add(self, other: &Matrix4) -> Matrix4 {
        *self + *other
    }
}

impl Add<Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn add(self, other: Matrix4) -> Matrix4 {
        *self + other
    }
}

impl AddAssign for Matrix4 {
    /// Component-wise matrix addition assignment using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let mut m = Matrix4::ZERO;
    /// m += Matrix4::IDENTITY;
    /// assert_eq!(m, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.row_x += other.row_x;
        self.row_y += other.row_y;
        self.row_z += other.row_z;
        self.row_w += other.row_w;
    }
}

impl AddAssign<&Matrix4> for Matrix4 {
    #[inline]
    fn add_assign(&mut self, other: &Matrix4) {
        *self += *other;
    }
}

impl Sub for Matrix4 {
    type Output = Self;

    /// Component-wise matrix subtraction using the `-` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let a = Matrix4::IDENTITY;
    /// let b = Matrix4::ZERO;
    /// let result = a - b;
    /// assert_eq!(result, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            row_x: self.row_x - other.row_x,
            row_y: self.row_y - other.row_y,
            row_z: self.row_z - other.row_z,
            row_w: self.row_w - other.row_w,
        }
    }
}

impl Sub<&Matrix4> for Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn sub(self, other: &Matrix4) -> Matrix4 {
        self - *other
    }
}

impl Sub<&Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn sub(self, other: &Matrix4) -> Matrix4 {
        *self - *other
    }
}

impl Sub<Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn sub(self, other: Matrix4) -> Matrix4 {
        *self - other
    }
}

impl SubAssign for Matrix4 {
    /// Component-wise matrix subtraction assignment using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let mut m = Matrix4::IDENTITY;
    /// m -= Matrix4::ZERO;
    /// assert_eq!(m, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.row_x -= other.row_x;
        self.row_y -= other.row_y;
        self.row_z -= other.row_z;
        self.row_w -= other.row_w;
    }
}

impl SubAssign<&Matrix4> for Matrix4 {
    #[inline]
    fn sub_assign(&mut self, other: &Matrix4) {
        *self -= *other;
    }
}

impl Mul<f32> for Matrix4 {
    type Output = Self;

    /// Scalar multiplication using the `*` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let result = m * 2.0;
    /// ```
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            row_x: self.row_x * scalar,
            row_y: self.row_y * scalar,
            row_z: self.row_z * scalar,
            row_w: self.row_w * scalar,
        }
    }
}

impl Mul<f32> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, scalar: f32) -> Matrix4 {
        *self * scalar
    }
}

impl Mul<&f32> for Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, scalar: &f32) -> Matrix4 {
        self * *scalar
    }
}

impl Mul<&f32> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, scalar: &f32) -> Matrix4 {
        *self * *scalar
    }
}

impl Mul for Matrix4 {
    type Output = Self;

    /// Matrix multiplication using the `*` operator.
    ///
    /// Computes the matrix product of self × other.
    /// Order matters: A × B ≠ B × A
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));
    /// let rotation = Matrix4::from_rotation_y(std::f32::consts::PI / 4.0);
    /// let combined = translation * rotation; // Order matters!
    /// ```
    #[inline]
    fn mul(self, other: Self) -> Self {
        let a = self;
        let b = other;

        Self {
            row_x: Vec4::new(
                a.row_x.x * b.row_x.x + a.row_x.y * b.row_y.x + a.row_x.z * b.row_z.x + a.row_x.w * b.row_w.x,
                a.row_x.x * b.row_x.y + a.row_x.y * b.row_y.y + a.row_x.z * b.row_z.y + a.row_x.w * b.row_w.y,
                a.row_x.x * b.row_x.z + a.row_x.y * b.row_y.z + a.row_x.z * b.row_z.z + a.row_x.w * b.row_w.z,
                a.row_x.x * b.row_x.w + a.row_x.y * b.row_y.w + a.row_x.z * b.row_z.w + a.row_x.w * b.row_w.w,
            ),
            row_y: Vec4::new(
                a.row_y.x * b.row_x.x + a.row_y.y * b.row_y.x + a.row_y.z * b.row_z.x + a.row_y.w * b.row_w.x,
                a.row_y.x * b.row_x.y + a.row_y.y * b.row_y.y + a.row_y.z * b.row_z.y + a.row_y.w * b.row_w.y,
                a.row_y.x * b.row_x.z + a.row_y.y * b.row_y.z + a.row_y.z * b.row_z.z + a.row_y.w * b.row_w.z,
                a.row_y.x * b.row_x.w + a.row_y.y * b.row_y.w + a.row_y.z * b.row_z.w + a.row_y.w * b.row_w.w,
            ),
            row_z: Vec4::new(
                a.row_z.x * b.row_x.x + a.row_z.y * b.row_y.x + a.row_z.z * b.row_z.x + a.row_z.w * b.row_w.x,
                a.row_z.x * b.row_x.y + a.row_z.y * b.row_y.y + a.row_z.z * b.row_z.y + a.row_z.w * b.row_w.y,
                a.row_z.x * b.row_x.z + a.row_z.y * b.row_y.z + a.row_z.z * b.row_z.z + a.row_z.w * b.row_w.z,
                a.row_z.x * b.row_x.w + a.row_z.y * b.row_y.w + a.row_z.z * b.row_z.w + a.row_z.w * b.row_w.w,
            ),
            row_w: Vec4::new(
                a.row_w.x * b.row_x.x + a.row_w.y * b.row_y.x + a.row_w.z * b.row_z.x + a.row_w.w * b.row_w.x,
                a.row_w.x * b.row_x.y + a.row_w.y * b.row_y.y + a.row_w.z * b.row_z.y + a.row_w.w * b.row_w.y,
                a.row_w.x * b.row_x.z + a.row_w.y * b.row_y.z + a.row_w.z * b.row_z.z + a.row_w.w * b.row_w.z,
                a.row_w.x * b.row_x.w + a.row_w.y * b.row_y.w + a.row_w.z * b.row_z.w + a.row_w.w * b.row_w.w,
            ),
        }
    }
}

impl Mul<&Matrix4> for Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, other: &Matrix4) -> Matrix4 {
        self * *other
    }
}

impl Mul<&Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, other: &Matrix4) -> Matrix4 {
        *self * *other
    }
}

impl Mul<Matrix4> for &Matrix4 {
    type Output = Matrix4;
    #[inline]
    fn mul(self, other: Matrix4) -> Matrix4 {
        *self * other
    }
}

impl MulAssign<f32> for Matrix4 {
    /// Scalar multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let mut m = Matrix4::IDENTITY;
    /// m *= 2.0;
    /// ```
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.row_x *= scalar;
        self.row_y *= scalar;
        self.row_z *= scalar;
        self.row_w *= scalar;
    }
}

impl MulAssign<&f32> for Matrix4 {
    #[inline]
    fn mul_assign(&mut self, scalar: &f32) {
        *self *= *scalar;
    }
}

impl MulAssign for Matrix4 {
    /// Matrix multiplication assignment using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec3;
    /// let mut m = Matrix4::IDENTITY;
    /// m *= Matrix4::from_scale_uniform(2.0);
    /// ```
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<&Matrix4> for Matrix4 {
    #[inline]
    fn mul_assign(&mut self, other: &Matrix4) {
        *self *= *other;
    }
}

// ============================================================================
// Vector-Matrix Multiplication
// ============================================================================

impl Mul<Matrix4> for Vec4 {
    type Output = Vec4;

    /// Row-vector matrix multiplication using the `*` operator.
    ///
    /// Computes: result = vector × matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec4;
    /// let m = Matrix4::IDENTITY;
    /// let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// let result = v * m;
    /// assert_eq!(result, v);
    /// ```
    #[inline]
    fn mul(self, matrix: Matrix4) -> Vec4 {
        Vec4::new(
            self.x * matrix.row_x.x + self.y * matrix.row_y.x + self.z * matrix.row_z.x + self.w * matrix.row_w.x,
            self.x * matrix.row_x.y + self.y * matrix.row_y.y + self.z * matrix.row_z.y + self.w * matrix.row_w.y,
            self.x * matrix.row_x.z + self.y * matrix.row_y.z + self.z * matrix.row_z.z + self.w * matrix.row_w.z,
            self.x * matrix.row_x.w + self.y * matrix.row_y.w + self.z * matrix.row_z.w + self.w * matrix.row_w.w,
        )
    }
}

impl Mul<&Matrix4> for Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, matrix: &Matrix4) -> Vec4 {
        self * *matrix
    }
}

impl Mul<Matrix4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, matrix: Matrix4) -> Vec4 {
        *self * matrix
    }
}

impl Mul<&Matrix4> for &Vec4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, matrix: &Matrix4) -> Vec4 {
        *self * *matrix
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

// Matrix4 Conversions for 2D arrays

impl From<[[f32; 4]; 4]> for Matrix4 {
    /// Converts a 2D array to a Matrix4.
    ///
    /// The array is interpreted in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let array = [
    ///     [1.0, 0.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ];
    /// let m: Matrix4 = array.into();
    /// assert_eq!(m, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn from(array: [[f32; 4]; 4]) -> Self {
        Self {
            row_x: Vec4::from(array[0]),
            row_y: Vec4::from(array[1]),
            row_z: Vec4::from(array[2]),
            row_w: Vec4::from(array[3]),
        }
    }
}

impl From<&[[f32; 4]; 4]> for Matrix4 {
    #[inline]
    fn from(array: &[[f32; 4]; 4]) -> Self {
        Self::from(*array)
    }
}

impl From<Matrix4> for [[f32; 4]; 4] {
    /// Converts a Matrix4 to a 2D array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let array: [[f32; 4]; 4] = m.into();
    /// ```
    #[inline]
    fn from(matrix: Matrix4) -> Self {
        [
            matrix.row_x.into(),
            matrix.row_y.into(),
            matrix.row_z.into(),
            matrix.row_w.into(),
        ]
    }
}

impl From<&Matrix4> for [[f32; 4]; 4] {
    #[inline]
    fn from(matrix: &Matrix4) -> Self {
        (*matrix).into()
    }
}

// Matrix4 Conversions for flat arrays

impl From<[f32; 16]> for Matrix4 {
    /// Converts a flat array to a Matrix4 in row-major order.
    ///
    /// Elements are interpreted as: [row0, row1, row2, row3]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let array = [
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ];
    /// let m: Matrix4 = array.into();
    /// assert_eq!(m, Matrix4::IDENTITY);
    /// ```
    #[inline]
    fn from(array: [f32; 16]) -> Self {
        Self {
            row_x: Vec4::new(array[0], array[1], array[2], array[3]),
            row_y: Vec4::new(array[4], array[5], array[6], array[7]),
            row_z: Vec4::new(array[8], array[9], array[10], array[11]),
            row_w: Vec4::new(array[12], array[13], array[14], array[15]),
        }
    }
}

impl From<&[f32; 16]> for Matrix4 {
    #[inline]
    fn from(array: &[f32; 16]) -> Self {
        Self::from(*array)
    }
}

impl From<Matrix4> for [f32; 16] {
    /// Converts a Matrix4 to a flat array in row-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let array: [f32; 16] = m.into();
    /// ```
    #[inline]
    fn from(matrix: Matrix4) -> Self {
        [
            matrix.row_x.x, matrix.row_x.y, matrix.row_x.z, matrix.row_x.w,
            matrix.row_y.x, matrix.row_y.y, matrix.row_y.z, matrix.row_y.w,
            matrix.row_z.x, matrix.row_z.y, matrix.row_z.z, matrix.row_z.w,
            matrix.row_w.x, matrix.row_w.y, matrix.row_w.z, matrix.row_w.w,
        ]
    }
}

impl From<&Matrix4> for [f32; 16] {
    #[inline]
    fn from(matrix: &Matrix4) -> Self {
        (*matrix).into()
    }
}

// Matrix4 Conversions for tuples (Vec4)

impl From<(Vec4, Vec4, Vec4, Vec4)> for Matrix4 {
    /// Converts a tuple of four Vec4 to a Matrix4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// # use toyengine::primitives::vec::Vec4;
    /// let tuple = (Vec4::X, Vec4::Y, Vec4::Z, Vec4::W);
    /// let m: Matrix4 = tuple.into();
    /// ```
    #[inline]
    fn from(tuple: (Vec4, Vec4, Vec4, Vec4)) -> Self {
        Self {
            row_x: tuple.0,
            row_y: tuple.1,
            row_z: tuple.2,
            row_w: tuple.3,
        }
    }
}

impl From<&(Vec4, Vec4, Vec4, Vec4)> for Matrix4 {
    #[inline]
    fn from(tuple: &(Vec4, Vec4, Vec4, Vec4)) -> Self {
        Self::from(*tuple)
    }
}

impl From<Matrix4> for (Vec4, Vec4, Vec4, Vec4) {
    /// Converts a Matrix4 to a tuple of four Vec4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use toyengine::primitives::mat::Matrix4;
    /// let m = Matrix4::IDENTITY;
    /// let tuple: (_, _, _, _) = m.into();
    /// ```
    #[inline]
    fn from(matrix: Matrix4) -> Self {
        (matrix.row_x, matrix.row_y, matrix.row_z, matrix.row_w)
    }
}

impl From<&Matrix4> for (Vec4, Vec4, Vec4, Vec4) {
    #[inline]
    fn from(matrix: &Matrix4) -> Self {
        (*matrix).into()
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
        let identity = Matrix4::identity();
        let zero = Matrix4::zero();

        // Verify identity matrix structure
        assert_eq!(identity.row_x, Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(identity.row_y, Vec4::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(identity.row_z, Vec4::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(identity.row_w, Vec4::new(0.0, 0.0, 0.0, 1.0));

        // Verify zero matrix
        assert_eq!(zero.row_x, Vec4::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(zero.row_y, Vec4::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(zero.row_z, Vec4::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(zero.row_w, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_constants() {
        assert_eq!(Matrix4::IDENTITY, Matrix4::identity());
        assert_eq!(Matrix4::ZERO, Matrix4::zero());
    }

    #[test]
    fn test_translation() {
        let translation = Matrix4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let transformed = translation.transform_point(point);
        assert_eq!(transformed, Vec3::new(11.0, 22.0, 33.0));
    }

    #[test]
    fn test_scale() {
        let scale = Matrix4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        let point = Vec3::new(1.0, 1.0, 1.0);
        let scaled = scale.transform_point(point);
        assert_eq!(scaled, Vec3::new(2.0, 3.0, 4.0));

        // Test uniform scale
        let uniform_scale = Matrix4::from_scale_uniform(2.0);
        let scaled_uniform = uniform_scale.transform_point(Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(scaled_uniform, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_rotation_z() {
        let rotation_90 = Matrix4::from_rotation_z(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotation_90.transform_direction(vec);

        // Should be approximately (0, 1, 0) after 90-degree rotation around Z
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 1.0).abs() < 0.001);
        assert!((rotated.z - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation_x() {
        let rotation_90 = Matrix4::from_rotation_x(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(0.0, 1.0, 0.0);
        let rotated = rotation_90.transform_direction(vec);

        // Y should rotate to Z
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation_y() {
        let rotation_90 = Matrix4::from_rotation_y(std::f32::consts::PI / 2.0);
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotation_90.transform_direction(vec);

        // X should rotate to -Z
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - -1.0).abs() < 0.001);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix4::from_scale_uniform(2.0);
        let m2 = Matrix4::from_translation(Vec3::new(5.0, 0.0, 0.0));

        // For row-vector multiplication: v * M1 * M2
        // Combined matrix applies M2 first (translation), then M1 (scale)
        let combined = m2 * m1;
        let point = Vec3::new(1.0, 1.0, 1.0);
        let result = combined.transform_point(point);
        assert_eq!(result, Vec3::new(12.0, 2.0, 2.0)); // (1+5)*2 = 12, 1*2 = 2
    }

    #[test]
    fn test_transpose() {
        let original = Matrix4::new(
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(5.0, 6.0, 7.0, 8.0),
            Vec4::new(9.0, 10.0, 11.0, 12.0),
            Vec4::new(13.0, 14.0, 15.0, 16.0),
        );
        let transposed = original.transpose();

        assert_eq!(transposed.row_x, Vec4::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(transposed.row_y, Vec4::new(2.0, 6.0, 10.0, 14.0));
        assert_eq!(transposed.row_z, Vec4::new(3.0, 7.0, 11.0, 15.0));
        assert_eq!(transposed.row_w, Vec4::new(4.0, 8.0, 12.0, 16.0));

        // Transpose twice should give original
        assert_eq!(transposed.transpose(), original);
    }

    #[test]
    fn test_determinant() {
        let identity = Matrix4::IDENTITY;
        let det_identity = identity.determinant();
        assert!((det_identity - 1.0).abs() < 0.001);

        let scale2x = Matrix4::from_scale_uniform(2.0);
        let det_scale = scale2x.determinant();
        assert!(det_scale > 0.0, "Determinant should be positive for scale");

        // Zero matrix has zero determinant
        let zero = Matrix4::ZERO;
        assert_eq!(zero.determinant(), 0.0);
    }

    #[test]
    fn test_inverse() {
        // Test identity inverse
        let identity = Matrix4::IDENTITY;
        let inv_identity = identity.inverse().expect("Identity should be invertible");
        assert!(identity.approx_eq(&inv_identity, 0.001));

        // Test scale inverse
        let scale = Matrix4::from_scale_uniform(2.0);
        let inv_scale = scale.inverse().expect("Scale should be invertible");
        let product = scale * inv_scale;
        assert!(product.approx_eq(&Matrix4::IDENTITY, 0.001));

        // Zero matrix should not be invertible
        let zero = Matrix4::ZERO;
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_vector_matrix_multiplication() {
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let m = Matrix4::IDENTITY;
        let result = v * m;
        assert_eq!(result, v);

        // Test with translation
        let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let point = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let translated = point * translation;
        assert_eq!(translated, Vec4::new(11.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_perspective_projection() {
        let perspective = Matrix4::perspective(
            std::f32::consts::PI / 4.0,
            16.0 / 9.0,
            0.1,
            1000.0,
        );

        // Projection matrix should be invertible
        assert!(perspective.determinant() != 0.0);
        assert!(perspective.inverse().is_some());
    }

    #[test]
    fn test_orthographic_projection() {
        let ortho = Matrix4::orthographic(0.0, 800.0, 0.0, 600.0, -1.0, 1.0);

        // Orthographic matrix should be invertible
        assert!(ortho.determinant() != 0.0);
        assert!(ortho.inverse().is_some());
    }

    #[test]
    fn test_look_at() {
        let view = Matrix4::look_at(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        // View matrix should be invertible
        assert!(view.determinant() != 0.0);
        assert!(view.inverse().is_some());
    }

    #[test]
    fn test_array_conversions() {
        let identity = Matrix4::IDENTITY;

        // Test 2D array conversion
        let array: [[f32; 4]; 4] = identity.into();
        let back: Matrix4 = array.into();
        assert_eq!(back, identity);

        // Test flat array conversion
        let flat: [f32; 16] = identity.into();
        let back_flat: Matrix4 = flat.into();
        assert_eq!(back_flat, identity);
    }

    #[test]
    fn test_addition_operators() {
        let a = Matrix4::IDENTITY;
        let b = Matrix4::IDENTITY;

        // Test owned
        let result = a + b;
        let expected = Matrix4::new(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));

        // Test borrowed
        let result_borrowed = &a + &b;
        assert!(result_borrowed.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_subtraction_operators() {
        let a = Matrix4::from_scale_uniform(3.0);
        let b = Matrix4::from_scale_uniform(1.0);
        let result = a - b;
        // Subtraction is component-wise, not geometric scale subtraction
        let expected = Matrix4::new(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),  // 1.0 - 1.0 = 0.0
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_scalar_multiplication() {
        let m = Matrix4::IDENTITY;
        let result = m * 2.0;

        // Scalar multiplication multiplies ALL elements
        let expected = Matrix4::new(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );
        assert!(result.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_assignment_operators() {
        let mut m = Matrix4::ZERO;
        m += Matrix4::IDENTITY;
        assert_eq!(m, Matrix4::IDENTITY);

        m *= 2.0;
        let expected = Matrix4::new(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 2.0),
        );
        assert!(m.approx_eq(&expected, 0.001));

        m -= Matrix4::IDENTITY;
        let expected_after_sub = Matrix4::from_scale_uniform(1.0);
        assert!(m.approx_eq(&expected_after_sub, 0.001));
    }

    #[test]
    fn test_default_trait() {
        let default_mat: Matrix4 = Default::default();
        assert_eq!(default_mat, Matrix4::IDENTITY);
    }

    #[test]
    fn test_transform_point() {
        let translation = Matrix4::from_translation(Vec3::new(5.0, 10.0, 15.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let result = translation.transform_point(point);
        assert_eq!(result, Vec3::new(6.0, 12.0, 18.0));
    }

    #[test]
    fn test_transform_direction() {
        // Directions should not be affected by translation
        let translation = Matrix4::from_translation(Vec3::new(100.0, 100.0, 100.0));
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let result = translation.transform_direction(direction);
        assert_eq!(result, direction);

        // But should be affected by rotation
        let rotation = Matrix4::from_rotation_z(std::f32::consts::PI / 2.0);
        let rotated = rotation.transform_direction(direction);
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_trace() {
        let identity = Matrix4::IDENTITY;
        assert_eq!(identity.trace(), 4.0);

        let scale = Matrix4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(scale.trace(), 10.0); // 2 + 3 + 4 + 1
    }

    #[test]
    fn test_frobenius_norm() {
        let identity = Matrix4::IDENTITY;
        let norm = identity.frobenius_norm();
        assert!((norm - 2.0).abs() < 0.01); // √4 = 2

        let norm_squared = identity.frobenius_norm_squared();
        assert_eq!(norm_squared, 4.0);
    }

    #[test]
    fn test_lerp() {
        let a = Matrix4::ZERO;
        let b = Matrix4::from_scale_uniform(2.0);
        let mid = a.lerp(&b, 0.5);
        // Lerp is component-wise: 0.5 * (0 + 2) for each component
        let expected = Matrix4::new(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.5),  // 0.5 * (0 + 1) = 0.5
        );
        assert!(mid.approx_eq(&expected, 0.001));
    }

    #[test]
    fn test_cols_and_rows() {
        let m = Matrix4::new(
            Vec4::new(1.0, 2.0, 3.0, 4.0),
            Vec4::new(5.0, 6.0, 7.0, 8.0),
            Vec4::new(9.0, 10.0, 11.0, 12.0),
            Vec4::new(13.0, 14.0, 15.0, 16.0),
        );

        let (row_x, row_y, row_z, row_w) = m.rows();
        assert_eq!(row_x, m.row_x);
        assert_eq!(row_y, m.row_y);
        assert_eq!(row_z, m.row_z);
        assert_eq!(row_w, m.row_w);

        let (col_x, col_y, col_z, col_w) = m.cols();
        assert_eq!(col_x, Vec4::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(col_y, Vec4::new(2.0, 6.0, 10.0, 14.0));
        assert_eq!(col_z, Vec4::new(3.0, 7.0, 11.0, 15.0));
        assert_eq!(col_w, Vec4::new(4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn test_from_cols() {
        let col_x = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let col_y = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let col_z = Vec4::new(0.0, 0.0, 1.0, 0.0);
        let col_w = Vec4::new(0.0, 0.0, 0.0, 1.0);

        let m = Matrix4::from_cols(col_x, col_y, col_z, col_w);
        assert_eq!(m, Matrix4::IDENTITY);
    }

    #[test]
    fn test_axis_angle_rotation() {
        // Test that axis-angle rotation works correctly
        let axis = Vec3::new(0.0, 0.0, 1.0);  // Z-axis
        let angle = std::f32::consts::PI / 2.0;
        let axis_angle = Matrix4::from_axis_angle(axis, angle);

        // Rotate a vector and verify result
        let vec = Vec3::new(1.0, 0.0, 0.0);
        let rotated = axis_angle.transform_direction(vec);

        // For row-vector multiplication, rotation is clockwise (result is (0, -1, 0))
        assert!((rotated.x - 0.0).abs() < 0.001, "x: expected 0.0, got {}", rotated.x);
        assert!((rotated.y - -1.0).abs() < 0.001, "y: expected -1.0, got {}", rotated.y);
        assert!((rotated.z - 0.0).abs() < 0.001, "z: expected 0.0, got {}", rotated.z);
    }

    #[test]
    fn test_combined_transformations() {
        // Test scale, rotate, translate order (common SRT pattern)
        let scale = Matrix4::from_scale_uniform(2.0);
        let rotation = Matrix4::from_rotation_z(std::f32::consts::PI / 2.0);
        let translation = Matrix4::from_translation(Vec3::new(10.0, 0.0, 0.0));

        // For row-vector multiplication: v * S * R * T
        // This applies scale first, then rotation, then translation
        let srt = scale * rotation * translation;

        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = srt.transform_point(point);

        // Scale by 2: (2, 0, 0)
        // Rotate 90° around Z: (0, 2, 0)
        // Translate: (10, 2, 0)
        assert!((result.x - 10.0).abs() < 0.001);
        assert!((result.y - 2.0).abs() < 0.001);
        assert!((result.z - 0.0).abs() < 0.001);
    }
}
