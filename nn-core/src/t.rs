//! Tensor data type and linear algebra operations.

use rand::distributions::{Distribution, Uniform};
use std::ops::Index;
use crate::tensor::matrix::rows;

/// A rank-1 tensor, effectively a vector.
pub type Tensor1D = Tensor<1>;

/// A rank-2 tensor, effectively a matrix.
pub type Tensor2D = Tensor<2>;

/// A multidimensional array of f32 elements.
///
/// The `N` argument specifies the rank (number of dimensions) of the tensor.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Tensor<const N: usize> {
    dims: [usize; N],
    data: Vec<f32>,
}

impl<const N: usize> Tensor<N> {
    /// Create a new tensor of the given shape, with all elements set to zero.
    pub fn zeros(dims: [usize; N]) -> Self {
        let size = dims.iter().product();
        let data = vec![0.0; size];
        Self { dims, data }
    }

    /// Create a tensor of the given shape, with all elements being uniformly distributed across
    /// the given range.
    pub fn uniform(dims: [usize; N], min: f32, max: f32) -> Self {
        let sampler = Uniform::new(min, max);
        let mut rng = rand::thread_rng();
        let mut data = vec![0.0; dims.iter().product()];
        for elem in data.iter_mut() {
            *elem = sampler.sample(&mut rng);
        }
        Self { dims, data }
    }

    /// Create a new tensor from a raw buffer and a shape
    pub fn raw<T: Into<Vec<f32>>>(vec: T, dims: [usize; N]) -> Self {
        Self {
            dims,
            data: vec.into(),
        }
    }

    /// Get the rank of the tensor
    pub fn rank(&self) -> usize {
        N
    }
}

impl Tensor1D {
    /// Add another vector to the vector.
    ///
    /// This operation updates each element in the current vector by adding the corresponding
    /// element in the other vector.
    pub fn add(&mut self, rhs: &Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });
    }

    /// Get the inner product value of the vector.
    ///
    /// This value is the sum of the products of each element pair in the two vectors.
    pub fn inner_product(&self, rhs: &Self) -> f32 {
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Scale the tensor by a scalar value.
    ///
    /// This operation updates each of the elements in the tensor by multiplying them by the
    /// given scalar value.
    pub fn scale(&mut self, scalar: f32) {
        self.data.iter_mut().for_each(|elem| *elem *= scalar);
    }

    /// Get the length of the vector
    pub fn length(&self) -> usize {
        self.dims[0]
    }

    /// Transpose the vector into a (1, n) 2-dimensional matrix.
    pub fn transpose(self) -> Tensor2D {
        Tensor2D::raw(self.data, [1, self.dims[0]])
    }

    /// Get the element count
    pub fn size(&self) -> usize {
        self.length()
    }
}

impl Index<usize> for Tensor1D {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        self.data.index(index)
    }
}

impl Tensor2D {
    /// Get the number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.dims[0]
    }

    /// Get the number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.dims[1]
    }

    pub fn dot(&self, tensor: &Tensor2D) -> Tensor2D {
        unimplemented!()
    }

    pub fn transpose(&self) -> Tensor2D {
        let mut buf = vec![0.0; self.size()];
        let dimensions = [self.cols(), self.rows()];

        for row in 0..self.rows() {
            for col in 0..self.cols() {

            }
        }

        Tensor2D::raw(buf, dimensions)
    }

    /// Get the element count
    pub fn size(&self) -> usize {
        return self.dims[0] * self.dims[1];
    }
}

impl Index<(usize, usize)> for Tensor2D {
    type Output = f32;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        let idx = row * self.dims[1] + col;
        self.data.index(idx)
    }
}
