//! Utilities to approximate equality of floating point values.

// likely will be replaced VERY soon

/// The max epsilon accepted on `f32`s.
pub const F32_MAX_ERROR: f32 = 1e-3;

/// The expected minimum epsilon accepted on `f32`s.
pub const F32_AVG_ERROR: f32 = 1e-5;

/// The best expected epsilon accepted on `f32`s.
pub const F32_MIN_ERROR: f32 = 1e-6;

/// The max epsilon accepted on `f64`s.
pub const F64_MAX_ERROR: f64 = 1e-3;

/// The expected minimum epsilon accepted on `f64`s.
pub const F64_AVG_ERROR: f64 = 1e-6;

/// The best expected epsilon accepted on `f64`s.
pub const F64_MIN_ERROR: f64 = 1e-13;

/// Checks the relative distance based off epsilon.
pub trait RelativeEq<Rhs: ?Sized> {
    /// Enumerates the equality of `self`
    fn approx_eq(&self, rhs: &Rhs) -> ApproxEquality;
}

impl RelativeEq<Self> for f32 {
    fn approx_eq(&self, rhs: &Self) -> ApproxEquality
    where
        Self: Sized,
    {
        let dif = (self - rhs).abs();

        if dif < F32_MIN_ERROR {
            ApproxEquality::Precise
        } else if dif < F32_AVG_ERROR {
            ApproxEquality::Partial
        } else if dif < F32_MAX_ERROR {
            ApproxEquality::Relative
        } else {
            ApproxEquality::Scarce
        }
    }
}

impl RelativeEq<Self> for f64 {
    fn approx_eq(&self, rhs: &Self) -> ApproxEquality
    where
        Self: Sized,
    {
        let dif = (self - rhs).abs();

        if dif < F64_MIN_ERROR {
            ApproxEquality::Precise
        } else if dif < F64_AVG_ERROR {
            ApproxEquality::Partial
        } else if dif < F64_MAX_ERROR {
            ApproxEquality::Relative
        } else {
            ApproxEquality::Scarce
        }
    }
}

impl<const N: usize, T: RelativeEq<U>, U> RelativeEq<[U; N]> for [T; N] {
    fn approx_eq(&self, rhs: &[U; N]) -> ApproxEquality {
        let mut eq = ApproxEquality::Precise;
        for (t_val, u_val) in self.iter().zip(rhs.iter()) {
            let eq_rating = t_val.approx_eq(u_val);
            match eq_rating {
                ApproxEquality::Precise => {
                    // already the best, can't change equality for the worse; leave it as-is
                }
                ApproxEquality::Partial => {
                    if eq != ApproxEquality::Relative {
                        eq = eq_rating;
                    }
                }
                ApproxEquality::Relative => {
                    eq = eq_rating;
                }
                ApproxEquality::Scarce => {
                    break; // can't improve from here; not equal
                }
            }
        }
        eq
    }
}

impl<T: RelativeEq<U> + Copy, U: Copy> RelativeEq<[U]> for [T] {
    fn approx_eq(&self, rhs: &[U]) -> ApproxEquality {
        let mut eq = ApproxEquality::Precise;
        for (t_val, u_val) in self.iter().zip(rhs.iter()) {
            let eq_rating = t_val.approx_eq(u_val);
            match eq_rating {
                ApproxEquality::Precise => {
                    // already the best, can't change equality for the worse; leave it as-is
                }
                ApproxEquality::Partial => {
                    if eq != ApproxEquality::Relative {
                        eq = eq_rating;
                    }
                }
                ApproxEquality::Relative => {
                    eq = eq_rating;
                }
                ApproxEquality::Scarce => {
                    break; // can't improve from here; not equal
                }
            }
        }
        eq
    }
}

/// The approximated equality enumerated.
#[repr(u8)]
#[derive(Debug, PartialEq, Eq)]
pub enum ApproxEquality {
    /// Very strong epsilon.
    Precise = 0,

    /// Good epsilon.
    Partial = 1,

    /// Acceptable epsilon
    Relative = 2,

    /// No relative equality.
    Scarce = 3,
}

/// Approximates equality based off the relative difference.
pub fn approx_eq<A: RelativeEq<B> + ?Sized, B: ?Sized>(a: &A, b: &B) -> bool {
    let eq = a.approx_eq(b);
    eq == ApproxEquality::Precise
}
