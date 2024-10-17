//! Linear algebra operations and data types.
//!
//! This module provides basic linear algebra operations such as matrix multiplication and vector
//! addition, dot product, etc. It also has vector and matrix data types that can be used to
//! represent and manipulate data in neural networks.

/// A vector of f32 elements
pub struct Vector {
    v: Vec<f32>,
}

impl Vector {
    /// Create a new vector from an inner `Vec<f32>`.
    pub fn new<T: Into<Vec<f32>>>(data: T) -> Self {
        Self { v: data.into() }
    }

    /// Add another vector to the current vector.
    ///
    /// This implementation requires that the vectors have the same size.
    pub fn add(mut self, other: &Self) -> Self {
        let tuples = self.v.iter_mut().zip(other.v.iter());
        for (lhs, rhs) in tuples {
            *lhs += rhs;
        }
        self
    }

    /// Multiply the current vector by a scalar value.
    pub fn multiply_scalar(mut self, scalar: f32) -> Self {
        for elem in self.v.iter_mut() {
            *elem *= scalar;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::Vector;

    #[test]
    fn test_vector_addition() {
        let v = Vector::new([1.0, 2.0]);
        let w = Vector::new([3.0, 4.0]);
        let x = v.add(&w);
        assert_eq!(x.v, [4.0, 6.0]);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector::new([1.0, 2.0]);
        let x = v.multiply_scalar(2.0);
        assert_eq!(x.v, [2.0, 4.0]);
    }
}
