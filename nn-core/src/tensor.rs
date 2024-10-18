//! Linear algebra operations and data types.
//!
//! This module provides the tensor data type, which is a multi-dimensional array of f32 elements,
//! as well as standard linear algebra operations such as matrix multiplication and vector

use rand::distributions::{Distribution, Uniform};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("tensor shape mismatch")]
    ShapeMismatch,
    #[error("tensor shape is not valid, only 1-d and 2-d tensors are supported")]
    InvalidShape,
}

/// The shape of a tensor, represented by a vector of dimensions.
///
/// The tensor implementation is currently limited to 2-dimensional tensors.
///
/// A tensor shape of [2, 3] represents a 2-d matrix with 2 rows and 3 columns.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new tensor shape from a vector of dimensions
    pub fn new<T: Into<Vec<usize>>>(dims: T) -> Result<Self, TensorError> {
        let dims = dims.into();
        if dims.len() > 2 {
            return Err(TensorError::InvalidShape);
        }
        Ok(Self { dims })
    }

    /// Shorthand for creating a 1-dimensional vector
    pub fn vector(len: usize) -> Self {
        TensorShape { dims: [len].into() }
    }

    /// Shorthand for creating a 2-dimensional matrix
    pub fn matrix(rows: usize, cols: usize) -> Self {
        TensorShape {
            dims: [rows, cols].into(),
        }
    }

    /// Get the number of dimensions in the tensor shape
    pub fn dims(&self) -> usize {
        self.dims.len()
    }

    /// Get the size of the tensor
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A tensor of f32 elements.
///
/// A tensor is a multidimensional array of f32 elements. We use it to represent vectors and
/// matrices in neural networks.
///
/// At the moment, we only support 1-tensors and 2-tensors.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Tensor {
    inner: Vec<f32>,
    shape: TensorShape,
}

impl Tensor {
    /// Create a new tensor of the given shape, with all elements set to zero.
    pub fn zeros(size: usize) -> Tensor {
        let shape = TensorShape::vector(size);
        Tensor {
            inner: vec![0.0; size],
            shape,
        }
    }

    /// Create a tensor of the given shape, with all elements being uniformly distributed across
    /// the given range.
    pub fn uniform(size: usize, min: f32, max: f32) -> Tensor {
        let shape = TensorShape::vector(size);
        let sampler = Uniform::new(min, max);
        let mut rng = rand::thread_rng();
        let mut data = vec![0.0; shape.size()];
        for elem in data.iter_mut() {
            *elem = sampler.sample(&mut rng);
        }
        Tensor { inner: data, shape }
    }

    /// Create a new tensor from a raw buffer and a shape
    pub fn new(data: Vec<f32>, shape: TensorShape) -> Result<Self, TensorError> {
        if data.len() != shape.size() {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Self { inner: data, shape })
    }

    /// Create a new tensor from a vector, inheriting the dimensions from the vector
    pub fn vector(data: Vec<f32>) -> Self {
        let shape = TensorShape::vector(data.len());
        Self { inner: data, shape }
    }

    /// Get the value at the given index in the buffer
    pub fn get(&self, index: usize) -> f32 {
        self.inner[index]
    }

    /// Set the value at the given index in the buffer
    pub fn set(&mut self, index: usize, value: f32) {
        self.inner[index] = value;
    }

    /// Get the size of the tensor
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }
}

/// Vector-like operations on Tensor values
///
/// The purpose of this module is to provide operations for Tensors, as if they were
/// single-dimensional vectors.
pub mod vector {
    use crate::tensor::{Tensor, TensorShape};
    use std::cmp::max;

    /// Add another vector to the current vector.
    ///
    /// This operation updates each element in the current vector by adding the corresponding
    /// element in the other vector.
    pub fn add(lhs: &mut Tensor, rhs: &Tensor) {
        lhs.inner
            .iter_mut()
            .zip(rhs.inner.iter())
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });
    }

    /// Get the inner product value of the vector.
    ///
    /// This value is the sum of the products of each element pair in the two vectors.
    pub fn inner(lhs: &Tensor, rhs: &Tensor) -> f32 {
        lhs.inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Multiply the current tensor by a scalar value.
    ///
    /// This operation updates each of the elements in the tensor by multiplying them by the
    /// given scalar value.
    ///
    /// This operation is also often called scaling a vector.
    pub fn multiply(lhs: &mut Tensor, scalar: f32) {
        lhs.inner.iter_mut().for_each(|elem| *elem *= scalar);
    }

    /// Transpose the vector into a (1, n) 2-dimensional matrix.
    ///
    /// This operation essentially flips the "columns" in the vector to become rows in a matrix.
    /// The resulting tensor is no longer single-dimensional.
    pub fn transpose(lhs: &mut Tensor) {
        lhs.shape = TensorShape::matrix(1, lhs.size());
    }

    /// Get the buffer index value for a given index into the vector.
    pub fn index(shape: &TensorShape, index: usize) -> usize {
        max(index, shape.size())
    }
}

/// Matrix-like operations on Tensor values
///
/// The purpose of this module is to provide operations for Tensors, as if they were
/// two-dimensional matrices.
pub mod matrix {
    use crate::tensor::{Tensor, TensorShape};

    /// Perform the dot product (matrix multiplication) of two tensors.
    ///
    /// The dot product of two tensors is a new tensor where each element is the dot product of
    /// the corresponding elements in the two input tensors.
    ///
    /// This implementation only supports two-dimensional matrices.
    pub fn dot(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        assert_eq!(
            rows(lhs.shape()),
            cols(rhs.shape()),
            "cannot perform dot product on tensors with different dimensions"
        );
        let output_size = rows(lhs.shape()) * cols(rhs.shape());
        let output_shape = TensorShape::matrix(rows(lhs.shape()), cols(rhs.shape()));
        let mut buffer = vec![0.0; output_size];
        for row in 0..rows(lhs.shape()) {
            for col in 0..cols(rhs.shape()) {
                let mut sum = 0.0;
                for i in 0..cols(lhs.shape()) {
                    sum +=
                        lhs.get(index(lhs.shape(), row, i)) * rhs.get(index(rhs.shape(), i, col));
                }
                buffer[index(&output_shape, row, col)] = sum;
            }
        }
        Tensor::new(buffer, output_shape).expect("failed to create dot product")
    }

    /// Transpose the matrix, flipping rows and columns.
    ///
    /// This operation also re-positions the tensor elements. It implies a new allocation of the
    /// same size as the original tensor.
    pub fn transpose(tensor: &Tensor) -> Tensor {
        let mut buffer = vec![0.0; tensor.size()];
        let output_shape = TensorShape::matrix(cols(tensor.shape()), rows(tensor.shape()));
        for row in 0..rows(tensor.shape()) {
            for col in 0..cols(tensor.shape()) {
                buffer[index(&output_shape, col, row)] =
                    tensor.get(index(tensor.shape(), row, col));
            }
        }
        Tensor::new(buffer, output_shape).expect("failed to create transpose")
    }

    /// Get the buffer index value for a given row and column in the matrix.
    pub fn index(shape: &TensorShape, row: usize, col: usize) -> usize {
        assert!(row < shape.dims[0], "row index out of bounds");
        assert!(col < shape.dims[1], "column index out of bounds");

        row * shape.dims[1] + col
    }

    /// Get the row count of the matrix
    pub fn rows(shape: &TensorShape) -> usize {
        shape.dims[0]
    }

    /// Get the column count of the matrix
    pub fn cols(shape: &TensorShape) -> usize {
        shape.dims[1]
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{matrix, vector, Tensor, TensorError, TensorShape};

    #[test]
    fn test_vector_addition() {
        let mut v = Tensor::vector(vec![1.0, 2.0]);
        let w = Tensor::vector(vec![3.0, 4.0]);

        vector::add(&mut v, &w);
        assert_eq!(v.inner, [4.0, 6.0]);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let mut v = Tensor::vector(vec![1.0, 2.0]);

        vector::multiply(&mut v, 2.0);
        assert_eq!(v.inner, [2.0, 4.0]);
    }

    #[test]
    fn test_vector_size() {
        let v = Tensor::vector(vec![1.0, 2.0]);
        assert_eq!(v.size(), 2);
    }

    #[test]
    fn test_matrix_transpose() -> Result<(), TensorError> {
        let shape = TensorShape::matrix(1, 2);
        let m = Tensor::new(vec![1.0, 2.0], shape)?;
        let t = matrix::transpose(&m);
        assert_eq!(t.get(matrix::index(t.shape(), 0, 0)), 1.0);
        assert_eq!(t.get(matrix::index(t.shape(), 1, 0)), 2.0);
        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<(), TensorError> {
        let shape = TensorShape::matrix(2, 1);
        let a = Tensor::new(vec![1.0, 2.0], shape.clone())?;
        let b = Tensor::new(vec![3.0, 4.0], shape)?;
        // Transpose the second matrix to align its rows as columns for the dot product
        let b = matrix::transpose(&b);
        let c = matrix::dot(&a, &b);

        println!("{:?}", c);

        assert_eq!(c.get(matrix::index(c.shape(), 0, 0)), 3.0);
        assert_eq!(c.get(matrix::index(c.shape(), 0, 1)), 4.0);
        assert_eq!(c.get(matrix::index(c.shape(), 1, 0)), 6.0);
        assert_eq!(c.get(matrix::index(c.shape(), 1, 1)), 8.0);
        Ok(())
    }
}
