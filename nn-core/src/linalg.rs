//! Linear algebra operations and data types.
//!
//! This module provides basic linear algebra operations such as matrix multiplication and vector
//! addition, dot product, etc. It also has vector and matrix data types that can be used to
//! represent and manipulate data in neural networks.

/// A vector f32 vector with a fixed size.
pub struct Vector<const N: usize> {
    data: [f32; N],
}

pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;

impl<const N: usize> Vector<N> {
    pub fn new(data: [f32; N]) -> Self {
        Self { data }
    }

    /// Add another vector to the current vector.
    ///
    /// This implementation requires that the vectors have the same size.
    pub fn add(mut self, other: &Self) -> Self {
        let tuples = self.data.iter_mut().zip(other.data.iter());
        for (lhs, rhs) in tuples {
            *lhs += rhs;
        }
        self
    }

    /// Multiply the current vector by a scalar value.
    pub fn multiply_scalar(mut self, scalar: f32) -> Self {
        for elem in self.data.iter_mut() {
            *elem *= scalar;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::Vector2;

    #[test]
    fn test_vector_addition() {
        let v = Vector2::new([1.0, 2.0]);
        let w = Vector2::new([3.0, 4.0]);
        let x = v.add(&w);
        assert_eq!(x.data, [4.0, 6.0]);
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector2::new([1.0, 2.0]);
        let x = v.multiply_scalar(2.0);
        assert_eq!(x.data, [2.0, 4.0]);
    }
}
