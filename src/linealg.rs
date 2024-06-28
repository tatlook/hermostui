use std::{fmt::Display, ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign}};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    pub fn dot(&self, rhs: &Self) -> f32 {
        assert_eq!(self.size(), rhs.size());
        let mut sum = 0.0;
        for i in 0..self.0.len() {
            sum += self.0[i] * rhs.0[i];
        }
        sum
    }

    pub fn len_sq(&self) -> f32 {
        self.dot(self)
    }

    pub fn normalize(&self) -> Self {
        if self.size() == 0 {
            return self.clone();
        }
        let len = (self.len_sq()).sqrt();
        self.clone() * (1.0 / len)
    }

    /// Not vector length, but dimension
    pub fn size(&self) -> usize {
        self.0.len()
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size(), rhs.size());
        let mut v = self.0.clone();
        for i in 0..v.len() {
            v[i] += rhs.0[i];
        }
        Vector(v)
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.size(), rhs.size());
        for i in 0..self.0.len() {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Sub for Vector {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size(), rhs.size());
        let mut v = self.0.clone();
        for i in 0..v.len() {
            v[i] -= rhs.0[i];
        }
        Vector(v)
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.size(), rhs.size());
        for i in 0..self.0.len() {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Mul<f32> for Vector {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut v = self.0;
        for i in 0..v.len() {
            v[i] *= rhs;
        }
        Vector(v)
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i != self.0.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self(shape)
    }

    pub fn size(&self) -> usize {
        self.0.iter().product()
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i != self.0.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    S(f32),
    V(Vector),
}

fn check_shape(t1: &Tensor, t2: &Tensor) {
    let s1 = t1.shape();
    let s2 = t2.shape();
    if s1 != s2 {
        panic!("Dimension mismatch: {} vs {}", s1, s2);
    }
}

impl Tensor {

    pub fn as_scalar(&self) -> Result<f32, ()> {
        match self {
            Tensor::S(s) => Ok(*s),
            _ => Err(()),
        }
    }

    pub fn as_vector(&self) -> Result<Vector, ()> {
        match self {
            Tensor::S(s) => Ok(Vector(vec![*s])),
            Tensor::V(v) => Ok(v.clone()),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            Tensor::S(_) => Shape(vec![]),
            Tensor::V(v) => Shape(vec![v.size()]),
        }
    }

    pub fn set(&mut self, value: Tensor) {
        check_shape(self, &value);
        *self = value
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Self;
    fn mul(self, rhs: Tensor) -> Self::Output {
        match (self, rhs) {
            (Tensor::S(s1), Tensor::S(s2)) => Tensor::S(s1 * s2),
            (Tensor::S(s1), Tensor::V(v1)) => Tensor::V(v1 * s1),
            (Tensor::V(v1), Tensor::S(s1)) => Tensor::V(v1 * s1),
            (Tensor::V(v1), Tensor::V(v2)) => Tensor::S(v1.dot(&v2)),
        }
    }
}

impl MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        *self = self.clone() * rhs
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Self;

    fn sub(self, rhs: Tensor) -> Self::Output {
        check_shape(&self, &rhs);
        match (self, rhs) {
            (Tensor::S(s1), Tensor::S(s2)) => Tensor::S(s1 - s2),
            (Tensor::V(v1), Tensor::V(v2)) => Tensor::V(v1 - v2),
            _ => unreachable!()
        }
    }
}

impl SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: Tensor) {
        *self = self.clone() - rhs
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::S(s) => writeln!(f, "Tensor {}", s),
            Tensor::V(v) => writeln!(f, "Tensor {}", v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_subtraction() {
        let v1 = Vector(vec![1.0, 2.0, 3.0]);
        let v2 = Vector(vec![4.0, 5.0, 6.0]);
        let expected = Vector(vec![-3.0, -3.0, -3.0]);
        assert_eq!(v1 - v2, expected);
    }

    #[test]
    fn vector_subtraction_assign() {
        let mut v1 = Vector(vec![1.0, 2.0, 3.0]);
        let v2 = Vector(vec![4.0, 5.0, 6.0]);
        v1 -= v2;
        let expected = Vector(vec![-3.0, -3.0, -3.0]);
        assert_eq!(v1, expected);
    }

    #[test]
    fn vector_dot_product() {
        let v1 = Vector(vec![1.0, 2.0, 3.0]);
        let v2 = Vector(vec![4.0, 5.0, 6.0]);
        let expected = 32.0;
        assert_eq!(v1.dot(&v2), expected);
    }

    #[test]
    fn vector_multiply_scalar() {
        let v = Vector(vec![1.0, 2.0, 3.0]);
        let expected = Vector(vec![2.0, 4.0, 6.0]);
        assert_eq!(v * 2.0, expected);
    }

    #[test]
    fn vector_length_squared() {
        let v = Vector(vec![3.0, 4.0, 0.0]);
        let expected = 25.0;
        assert_eq!(v.len_sq(), expected);
    }

    #[test]
    fn vector_size() {
        let v = Vector(vec![1.0, 2.0, 3.0, 4.0]);
        let expected = 4;
        assert_eq!(v.size(), expected);
    }

    #[test]
    fn vector_display() {
        let v = Vector(vec![1.0, 2.0, 3.0]);
        let expected = "[1, 2, 3]".to_string();
        assert_eq!(format!("{}", v), expected);
    }
}
