//! Provides the necessary means of abstraction which make advanced cases much simpler.
//!
//! Note: UNSTABLE!!! This API might recieve major breaking changes with each update.
//! `v0.4.0` will likely have the final API, `v0.3.x` will not change the API though.

mod data;
pub use data::Dataset;

pub mod internal;
pub use internal::Relu;

pub mod layers;
