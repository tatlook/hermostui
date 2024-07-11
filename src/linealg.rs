use std::{
    fmt::Display, iter::zip, ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign}
};

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector(pub Vec<f32>);

impl Vector {
    pub fn hadamard(mut self, rhs: &Self) -> Self {
        assert_eq!(self.size(), rhs.size());
        for i in 0..self.size() {
            self.0[i] *= rhs.0[i];
        }
        self
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

impl AddAssign<&Self> for Vector {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.size(), rhs.size());
        for i in 0..self.0.len() {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Add<&Self> for Vector {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl SubAssign<&Self> for Vector {
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.size(), rhs.size());
        for i in 0..self.0.len() {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Sub<&Self> for Vector {
    type Output = Vector;
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..self.0.len() {
            self.0[i] *= rhs;
        }
    }
}

impl Mul<f32> for Vector {
    type Output = Self;
    fn mul(mut self, rhs: f32) -> Self::Output {
        self *= rhs;
        self
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix(Vec<Vector>);

impl Matrix {
    pub fn new(data: Vec<Vector>) -> Self {
        Self(data)
    }

    pub fn data_mut(&mut self) -> &mut Vec<Vector> {
        &mut self.0
    }

    /// Applies the linear transformation to the vector
    pub fn apply(self, rhs: &Vector) -> Vector {
        assert_eq!(self.0.len(), rhs.size());
        let mut vector = Vector(vec![0.0; self.0[0].size()]);
        for (i, v) in self.0.into_iter().enumerate() {
            vector += &(v * rhs.0[i]);
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

    pub fn mul_assign_scalar(&mut self, rhs: f32) {
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                self.0[i].0[j] *= rhs;
            }
        }
    }

    pub fn mul_scalar(mut self, rhs: f32) -> Self {
        self.mul_assign_scalar(rhs);
        self
    }
}

impl AddAssign<&Self> for Matrix {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape(), rhs.shape());
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                self.0[i].0[j] += rhs.0[i].0[j];
            }
        }
    }
}

impl Add<&Self> for Matrix {
    type Output = Self;
    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl SubAssign<&Self> for Matrix {
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape(), rhs.shape());
        for i in 0..self.0.len() {
            for j in 0..self.0[i].size() {
                self.0[i].0[j] -= rhs.0[i].0[j];
            }
        }
    }
}

impl Sub<&Self> for Matrix {
    type Output = Self;
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
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
    N, S, V(usize), M(usize, usize), L(Vec<Shape>),
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Shape::N => write!(f, "N"),
            Shape::S => write!(f, "S"),
            Shape::V(v) => write!(f, "V({})", v),
            Shape::M(r, c) => write!(f, "M({}, {})", r, c),
            Shape::L(l) => {
                write!(f, "L(")?;
                for i in 0..l.len() {
                    write!(f, "{}", l[i])?;
                    if i != l.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Tensor {
    /// Empty
    N, 
    /// Scalar
    S(f32), 
    V(Vector),
    M(Matrix),
    /// A list of tensors
    L(Vec<Tensor>),
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

    pub fn as_vector(self) -> Vector {
        match self {
            Tensor::V(v) => v,
            _ => panic!("Can't cast {self} to vector"),
        }
    }

    pub fn as_matrix(self) -> Matrix {
        match self {
            Tensor::M(m) => m,
            _ => panic!("Can't cast {self} to matrix"),
        }
    }

    pub fn as_tensor_list_ref(&self) -> &Vec<Tensor> {
        match self {
            Tensor::L(l) => l,
            _ => panic!("Can't cast {self} to tensor list"),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            Tensor::N => Shape::N,
            Tensor::S(_) => Shape::S,
            Tensor::V(v) => Shape::V(v.size()),
            Tensor::M(m) => m.shape(),
            Tensor::L(l) => Shape::L(l.iter().map(|t| t.shape()).collect()),
        }
    }

    pub fn zeroes(shape: Shape) -> Self {
        match shape {
            Shape::N => Tensor::N,
            Shape::S => Tensor::S(0.0),
            Shape::V(l) => Tensor::V(Vector(vec![0.0; l])),
            Shape::M(r, c) => Tensor::M(Matrix(vec![Vector(vec![0.0; r]); c])),
            Shape::L(l) => Tensor::L(l.iter().map(|s| Tensor::zeroes(s.clone())).collect()),
        }
    }

    pub fn rand(shape: Shape) -> Self {
        let mut rng = rand::thread_rng();
        let range = -1.0..1.0;
        match shape {
            Shape::N => Tensor::N,
            Shape::S => Tensor::S(rng.gen_range(range)),
            Shape::V(l) => {
                let mut v = vec![];
                for _ in 0..l {
                    v.push(rng.gen_range(range.clone()));
                }
                Tensor::V(Vector(v))
            },
            Shape::M(r, c) => {
                let mut li = vec![];
                for _ in 0..c {
                    let mut v = vec![];
                    for _ in 0..r {
                        v.push(rng.gen_range(range.clone()));
                    }
                    li.push(Vector(v));
                }
                Tensor::M(Matrix(li))
            },
            Shape::L(l) => Tensor::L(l.iter().map(|s| Tensor::rand(s.clone())).collect()),
        }
    }
}

impl AddAssign<&Self> for Tensor {
    fn add_assign(&mut self, rhs: &Self) {
        check_shape(&self, &rhs);
        match (self, rhs) {
            (Tensor::N, Tensor::N) => return,
            (Tensor::S(s1), Tensor::S(s2)) => *s1 += s2,
            (Tensor::V(v1), Tensor::V(v2)) => *v1 += v2,
            (Tensor::M(m1), Tensor::M(m2)) => *m1 += m2,
            (Tensor::L(l1), Tensor::L(l2)) => {
                for (t1, t2) in zip(l1, l2) {
                    *t1 += t2;
                }
            },
            _ => unreachable!(),
        }
    }
}

impl Add<&Self> for Tensor {
    type Output = Self;
    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl SubAssign<&Self> for Tensor {
    fn sub_assign(&mut self, rhs: &Self) {
        check_shape(&self, &rhs);
        match (self, rhs) {
            (Tensor::N, Tensor::N) => return,
            (Tensor::S(s1), Tensor::S(s2)) => *s1 -= s2,
            (Tensor::V(v1), Tensor::V(v2)) => *v1 -= v2,
            (Tensor::M(m1), Tensor::M(m2)) => *m1 -= m2,
            (Tensor::L(l1), Tensor::L(l2)) => {
                for (t1, t2) in zip(l1, l2) {
                    *t1 -= t2;
                }
            },
            _ => unreachable!(),
        }
    }
}

impl Sub<&Self> for Tensor {
    type Output = Self;
    
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, rhs: f32) {
        match self {
            Tensor::N => return,
            Tensor::S(s) => *s *= rhs,
            Tensor::V(v) => *v *= rhs,
            Tensor::M(m) => m.mul_assign_scalar(rhs),
            Tensor::L(l) => {
                for t in l {
                    *t *= rhs;
                }
            }
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(mut self, rhs: f32) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::N => writeln!(f, "Tensor Empty"),
            Tensor::S(s) => writeln!(f, "Tensor {}", s),
            Tensor::V(v) => writeln!(f, "Tensor {}", v),
            Tensor::M(m) => writeln!(f, "Tensor {}", m),
            Tensor::L(l) => {
                writeln!(f, "Tensor [")?;
                for t in l {
                    writeln!(f, "Tensor {}", t)?;
                }
                write!(f, "]")
            },
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
        assert_eq!(v1 - &v2, expected);
    }

    #[test]
    fn vector_subtraction_assign() {
        let mut v1 = Vector(vec![1.0, 2.0, 3.0]);
        let v2 = Vector(vec![4.0, 5.0, 6.0]);
        v1 -= &v2;
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
    }

    #[test]
    #[should_panic]
    fn matrix_wrong_shape1() {
        if let (Tensor::M(m), Tensor::V(v)) = (Tensor::zeroes(Shape::M(3, 2)), Tensor::zeroes(Shape::V(3))) {
            let _ = m.apply(&v);
        } else {
            println!("WTF");
        }
    }
}
