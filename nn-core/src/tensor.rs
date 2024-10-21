//! Tensor data type and linear algebra operations.

use rand::distributions::{Distribution, Uniform};
use std::ops::{Index, IndexMut};

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
    pub fn from_vec<T: Into<Vec<f32>>>(vec: T) -> Self {
        let v = vec.into();
        let len = v.len();
        Self::raw(v, [len])
    }

    /// Add another vector to the vector.
    pub fn pairwise_summation(&mut self, rhs: &Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(lhs, rhs)| {
                *lhs += rhs;
            });
    }

    /// Subtract another vector from the vector.
    pub fn pairwise_subtraction(&mut self, rhs: &Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(lhs, rhs)| {
                *lhs -= rhs;
            });
    }

    /// Multiply each element in the vector by the corresponding element in the other vector.
    pub fn pairwise_multiplication(&mut self, rhs: &Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(lhs, rhs)| {
                *lhs *= rhs;
            });
    }

    /// Divide each element in the vector by the corresponding element in the other vector.
    pub fn pairwise_division(&mut self, rhs: &Self) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(lhs, rhs)| {
                *lhs /= rhs;
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
        let len = self.length();
        Tensor2D::raw(self.data, [1, len])
    }

    /// Get the element count
    pub fn size(&self) -> usize {
        self.length()
    }
}

impl Index<usize> for Tensor1D {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.length(), "index out of bounds");
        self.data.index(index)
    }
}

impl IndexMut<usize> for Tensor1D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.length(), "index out of bounds");
        self.data.index_mut(index)
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

    /// Perform the dot product (matrix multiplication) of two tensors.
    ///
    /// The dot product of two tensors is a new tensor where each element is the dot product of
    /// the corresponding elements in the two input tensors.
    ///
    /// This implementation only supports two-dimensional matrices.
    pub fn dot(&self, tensor: &Tensor2D) -> Tensor2D {
        assert_eq!(
            self.rows(),
            tensor.cols(),
            "cannot perform dot product on tensors with different dimensions"
        );
        let mut output = Tensor2D::zeros([self.rows(), tensor.cols()]);
        for row in 0..self.rows() {
            for col in 0..tensor.cols() {
                let mut sum = 0.0;
                for i in 0..self.cols() {
                    sum += self.index((row, i)) * tensor.index((i, col));
                }
                *output.index_mut((row, col)) = sum;
            }
        }
        output
    }

    /// Transpose the matrix, flipping rows and columns.
    ///
    /// This operation also re-positions the tensor elements. It implies a new allocation of the
    /// same size as the original tensor.
    pub fn transpose(&self) -> Tensor2D {
        let mut output = Tensor2D::zeros([self.cols(), self.rows()]);
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                *output.index_mut((col, row)) = *self.index((row, col));
            }
        }
        output
    }

    /// Get the element count
    pub fn size(&self) -> usize {
        self.dims[0] * self.dims[1]
    }
}

impl Index<(usize, usize)> for Tensor2D {
    type Output = f32;
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows(), "row index out of bounds");
        assert!(col < self.cols(), "column index out of bounds");
        let idx = row * self.cols() + col;
        self.data.index(idx)
    }
}

impl IndexMut<(usize, usize)> for Tensor2D {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows(), "row index out of bounds");
        assert!(col < self.cols(), "column index out of bounds");
        let idx = row * self.cols() + col;
        self.data.index_mut(idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor1D, Tensor2D};

    #[test]
    fn test_tensor1d_summation() {
        let mut v = Tensor1D::raw([1.0, 2.0], [2]);
        let w = Tensor1D::raw([3.0, 4.0], [2]);

        v.pairwise_summation(&w);
        assert_eq!(v.data, [4.0, 6.0]);
    }

    #[test]
    fn test_tensor1d_subtraction() {
        let mut v = Tensor1D::raw([1.0, 2.0], [2]);
        let w = Tensor1D::raw([3.0, 4.0], [2]);

        v.pairwise_subtraction(&w);
        assert_eq!(v.data, [-2.0, -2.0]);
    }

    #[test]
    fn test_tensor1d_multiplication() {
        let mut v = Tensor1D::raw([1.0, 2.0], [2]);
        let w = Tensor1D::raw([3.0, 4.0], [2]);

        v.pairwise_multiplication(&w);
        assert_eq!(v.data, [3.0, 8.0]);
    }

    #[test]
    fn test_tensor1d_division() {
        let mut v = Tensor1D::raw([1.0, 2.0], [2]);
        let w = Tensor1D::raw([3.0, 4.0], [2]);

        v.pairwise_division(&w);
        assert_eq!(v.data, [1.0 / 3.0, 0.5]);
    }

    #[test]
    fn test_tensor1d_scale() {
        let mut v = Tensor1D::raw([1.0, 2.0], [2]);

        v.scale(2.0);
        assert_eq!(v.data, [2.0, 4.0]);
    }

    #[test]
    fn test_tensor1d_length() {
        let v = Tensor1D::raw([1.0, 2.0], [2]);
        assert_eq!(v.length(), 2);
    }

    #[test]
    fn test_tensor1d_transpose() {
        let v = Tensor1D::raw([1.0, 2.0], [2]);
        let t = v.transpose();
        assert_eq!(t.data, [1.0, 2.0]);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 2.0);
        assert_eq!(t.dims, [1, 2]);
    }

    #[test]
    fn test_tensor2d_dot() {
        let a = Tensor2D::raw([1.0, 2.0], [2, 1]);
        let b = Tensor2D::raw([3.0, 4.0], [2, 1]).transpose();
        let c = a.dot(&b);
        assert_eq!(c.data, [3.0, 4.0, 6.0, 8.0]);
    }
}
