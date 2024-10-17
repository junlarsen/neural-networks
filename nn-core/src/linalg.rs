//! Linear algebra operations and data types.
//!
//! This module provides basic linear algebra operations such as matrix multiplication and vector
//! addition, dot product, etc. It also has vector and matrix data types that can be used to
//! represent and manipulate data in neural networks.

use rand::distributions::{Distribution, Uniform};

/// A vector of f32 elements
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Vector {
    inner: Vec<f32>,
}

impl Vector {
    /// Create a new vector from an inner `Vec<f32>`.
    pub fn new<T: Into<Vec<f32>>>(data: T) -> Self {
        Self { inner: data.into() }
    }

    /// Create a new vector of a given size, populated with an uniform distribution.
    pub fn from_uniform_distribution(size: usize, min: f32, max: f32) -> Self {
        let sampler = Uniform::new(min, max);
        let mut rng = rand::thread_rng();
        Self {
            inner: sampler.sample_iter(&mut rng).take(size).collect(),
        }
    }

    /// Compute the inner product of two vectors.
    pub fn inner_product(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for (a, b) in self.inner.iter().zip(other.inner.iter()) {
            sum += a * b;
        }
        sum
    }

    /// Add another vector to the current vector.
    pub fn add(&mut self, other: &Self) {
        let tuples = self.inner.iter_mut().zip(other.inner.iter());
        for (lhs, rhs) in tuples {
            *lhs += rhs;
        }
    }

    /// Multiply the current vector by a scalar value.
    pub fn multiply(&mut self, scalar: f32) {
        for elem in self.inner.iter_mut() {
            *elem *= scalar;
        }
    }

    /// Get the value at the given index in the vector.
    pub fn get(&self, index: usize) -> f32 {
        self.inner[index]
    }

    /// Set the value at the given index in the vector.
    pub fn set(&mut self, index: usize, value: f32) {
        self.inner[index] = value;
    }

    /// Get the size of the vector.
    pub fn size(&self) -> usize {
        self.inner.len()
    }
}

/// A 2-dimensional matrix of f32 elements.
pub struct Matrix2D {
    inner: Vector,
    rows: usize,
    cols: usize,
}

impl Matrix2D {
    /// Create a new matrix with the given number of rows and columns.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            inner: Vector::new(vec![0.0; rows * cols]),
            rows,
            cols,
        }
    }

    /// Create a vector with a single row from a vector.
    pub fn from_vector(vector: Vector) -> Self {
        let size = vector.size();
        Self {
            inner: vector,
            rows: size,
            cols: 1,
        }
    }

    /// Perform the dot product of two matrices.
    ///
    /// The dot product of two matrices is a new matrix where each element is the dot product of
    /// the corresponding elements in the two input matrices.
    ///
    /// In other words, the dot product of two matrices A and B is a matrix C where C[i][j] is
    /// the dot product of A[i][k] and B[k][j] for all i, j, and k.
    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "cannot perform dot product on matrices with different dimensions"
        );
        let mut result = Matrix2D::new(self.rows, other.cols);
        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0.0;
                for i in 0..self.cols {
                    sum += self.get(row, i) * other.get(i, col);
                }
                result.set(row, col, sum);
            }
        }
        result
    }

    /// Read the value at the given row and column in the 2-dimensional matrix.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.inner.inner[row * self.cols + col]
    }

    /// Set the value at the given row and column in the 2-dimensional matrix.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.inner.inner[row * self.cols + col] = value;
    }

    /// Transpose the matrix, turning rows into columns and vice versa.
    pub fn transpose(&self) -> Self {
        let mut transposed = Matrix2D::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                transposed.set(col, row, self.get(row, col));
            }
        }
        transposed
    }

    /// Get the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::{Matrix2D, Vector};

    #[test]
    fn test_vector_addition() {
        let mut v = Vector::new([1.0, 2.0]);
        let w = Vector::new([3.0, 4.0]);
        v.add(&w);
        assert_eq!(v.inner, [4.0, 6.0]);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let mut v = Vector::new([1.0, 2.0]);
        v.multiply(2.0);
        assert_eq!(v.inner, [2.0, 4.0]);
    }

    #[test]
    fn test_vector_size() {
        let v = Vector::new([1.0, 2.0]);
        assert_eq!(v.size(), 2);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix2D::new(1, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        let t = m.transpose();
        assert_eq!(t.cols(), 1);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix2D::from_vector(Vector::new([1.0, 2.0]));
        let b = Matrix2D::from_vector(Vector::new([3.0, 4.0])).transpose();
        let c = a.dot(&b);
        assert_eq!(c.get(0, 0), 3.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 6.0);
        assert_eq!(c.get(1, 1), 8.0);
    }
}
