use std::{
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    pub fn hadamard(&self, rhs: &Self) -> Self {
        assert_eq!(self.size(), rhs.size());
        let mut v = self.0.clone();
        for i in 0..v.len() {
            v[i] *= rhs.0[i];
        }
        Vector(v)
    }

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

/// Contains list of basis vectors
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix(Vec<Vector>);

impl Matrix {
    pub fn new(data: Vec<Vector>) -> Self {
        Self(data)
    }

    /// Applies the linear transformation to the vector
    pub fn apply(&self, rhs: &Vector) -> Vector {
        assert_eq!(self.0.len(), rhs.size());
        let mut vector = Vector(vec![0.0; self.0[0].size()]);
        for i in 0..rhs.size() {
            vector += self.0[i].clone() * rhs.0[i];
        }
        vector
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![Vector(vec![0.0; self.0.len()]); self.0[0].size()];
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                data[j].0[i] = self.0[i].0[j];
            }
        }
        Self(data)
    }

    pub fn shape(&self) -> Shape {
        Shape::M(self.0[0].size(), self.0.len()) // TODO: self.0.len() can be zero
    }

    pub fn mul_matrix(&self, rhs: &Self) -> Self {
        assert_eq!(self.0[0].size(), rhs.0.len());
        let mut data = vec![Vector(vec![0.0; rhs.0[0].size()]); self.0.len()];
        for i in 0..self.0.len() {
            for j in 0..rhs.0[0].size() {
                for k in 0..self.0[i].size() {
                    data[i].0[j] += self.0[i].0[k] * rhs.0[k].0[j];
                }
            }
        }
        Self(data)
    }

    pub fn mul_scalar(&self, rhs: f32) -> Self {
        let mut data = vec![Vector(vec![0.0; self.0[0].size()]); self.0.len()];
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                data[i].0[j] = self.0[i].0[j] * rhs;
            }
        }
        Self(data)
    }
}

impl Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape());
        let mut data = vec![Vector(vec![0.0; self.0[0].size()]); self.0.len()];
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                data[i].0[j] = self.0[i].0[j] - rhs.0[i].0[j];
            }
        }
        Self(data)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.0.len() {
            write!(f, "{}", self.0[i])?;
            if i != self.0.len() - 1 {
                write!(f, "\n")?;
            }
        }
        write!(f, "]")
    }
}

/// For matrix, this is (#rows, #columns)
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    N, S, V(usize), M(usize, usize),
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shape::N => write!(f, "N"),
            Shape::S => write!(f, "S"),
            Shape::V(v) => write!(f, "V({})", v),
            Shape::M(r, c) => write!(f, "M({}, {})", r, c),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    /// Empty
    N, 
    /// Scalar
    S(f32), 
    V(Vector),
    M(Matrix),
}

fn check_shape(t1: &Tensor, t2: &Tensor) {
    let s1 = t1.shape();
    let s2 = t2.shape();
    if s1 != s2 {
        panic!("Dimension mismatch: {} vs {}", s1, s2);
    }
}

impl Tensor {
    pub fn as_scalar(&self) -> f32 {
        match self {
            Tensor::S(s) => *s,
            _ => panic!("Can't cast {self} to scalar"),
        }
    }

    pub fn as_vector(&self) -> Vector {
        match self {
            Tensor::S(s) => Vector(vec![*s]),
            Tensor::V(v) => v.clone(),
            _ => panic!("Can't cast {self} to vector"),
        }
    }

    pub fn as_matrix(&self) -> Matrix {
        match self {
            Tensor::N => panic!("Can't cast {self} to matrix"),
            Tensor::S(s) => Matrix(vec![Vector(vec![*s])]),
            Tensor::V(v) => Matrix(vec![v.clone()]),
            Tensor::M(m) => m.clone(),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            Tensor::N => Shape::N,
            Tensor::S(_) => Shape::S,
            Tensor::V(v) => Shape::V(v.size()),
            Tensor::M(m) => m.shape(),
        }
    }

    pub fn zeroes(shape: Shape) -> Self {
        match shape {
            Shape::N => Tensor::N,
            Shape::S => Tensor::S(0.0),
            Shape::V(l) => Tensor::V(Vector(vec![0.0; l])),
            Shape::M(r, c) => Tensor::M(Matrix(vec![Vector(vec![0.0; r]); c])),
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Self;
    fn mul(self, rhs: Tensor) -> Self::Output {
        match (self, rhs) {
            (Tensor::N, _) => Tensor::N,
            (_, Tensor::N) => Tensor::N,
            (Tensor::S(s1), Tensor::S(s2)) => Tensor::S(s1 * s2),
            (Tensor::S(s1), Tensor::V(v1)) => Tensor::V(v1 * s1),
            (Tensor::V(v1), Tensor::S(s1)) => Tensor::V(v1 * s1),
            (Tensor::V(v1), Tensor::V(v2)) => Tensor::S(v1.dot(&v2)),
            (Tensor::M(m1), Tensor::M(m2)) => Tensor::M(m1.mul_matrix(&m2)),
            (Tensor::S(s1), Tensor::M(m1)) => Tensor::M(m1.mul_scalar(s1)),
            (Tensor::M(m1), Tensor::V(v1)) => Tensor::V(m1.apply(&v1)),
            (Tensor::V(_), Tensor::M(_)) => panic!("First matrix then vector, please"),
            (Tensor::M(_), Tensor::S(_)) => panic!("First scalar then matrix, please"),
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
            (Tensor::N, Tensor::N) => Tensor::N,
            (Tensor::S(s1), Tensor::S(s2)) => Tensor::S(s1 - s2),
            (Tensor::V(v1), Tensor::V(v2)) => Tensor::V(v1 - v2),
            (Tensor::M(m1), Tensor::M(m2)) => Tensor::M(m1 - m2),
            _ => unreachable!(),
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
            Tensor::N => writeln!(f, "Tensor Empty"),
            Tensor::S(s) => writeln!(f, "Tensor {}", s),
            Tensor::V(v) => writeln!(f, "Tensor {}", v),
            Tensor::M(m) => writeln!(f, "Tensor {}", m),
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

    #[test]
    fn matrix_apply() {
        let m = Matrix(vec![
            Vector(vec![1.0, 2.0, 3.0]),
            Vector(vec![4.0, 5.0, 6.0]),
        ]);
        let v = Vector(vec![1.0, 2.0]);
        let expected = Vector(vec![9.0, 12.0, 15.0]);
        assert_eq!(m.apply(&v), expected);
    }

    #[test]
    fn matrix_transpose() {
        let m = Matrix(vec![
            Vector(vec![1.0, 2.0, 3.0]),
            Vector(vec![4.0, 5.0, 6.0]),
        ]);
        let expected = Matrix(vec![
            Vector(vec![1.0, 4.0]),
            Vector(vec![2.0, 5.0]),
            Vector(vec![3.0, 6.0]),
        ]);
        assert_eq!(m.transpose(), expected);
    }

    #[test]
    fn zero_shape() {
        let m = Tensor::zeroes(Shape::M(3, 2));
        assert_eq!(m.shape(), Shape::M(3, 2));
        let v = Tensor::zeroes(Shape::V(2));
        assert_eq!(v.shape(), Shape::V(2));
        let v = m * v;
        assert_eq!(v.shape(), Shape::V(3));
    }

    #[test]
    #[should_panic]
    fn matrix_wrong_shape1() {
        let m = Tensor::zeroes(Shape::M(3, 2));
        let v = Tensor::zeroes(Shape::V(3));
        let _ = m * v;
    }

    #[test]
    #[should_panic]
    fn matrix_wrong_shape2() {
        let m = Tensor::zeroes(Shape::M(2, 3));
        let v = Tensor::zeroes(Shape::V(2));
        let _ = m * v;
    }
}
