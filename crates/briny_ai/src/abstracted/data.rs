pub struct Dataset<'a, X, Y> {
    inputs: &'a [X],
    labels: &'a [Y],
    shape: [usize; 2], // (num_samples, sample_len)
}

impl<'a, X, Y> Dataset<'a, X, Y> {
    /// # Panics
    ///
    /// When the length of the given data isn't equal to
    /// `num_samples` and `sample_len` as provided, this method
    /// will panic.
    pub const fn new(
        inputs: &'a [X],
        labels: &'a [Y],
        num_samples: usize,
        sample_len: usize,
    ) -> Self {
        Self::construct(inputs, labels, num_samples, sample_len)
            .expect("data length and shape specified size don't match")
    }

    pub const fn construct(
        inputs: &'a [X],
        labels: &'a [Y],
        num_samples: usize,
        sample_len: usize,
    ) -> Option<Self> {
        if inputs.len() != num_samples * sample_len {
            return None;
        }
        if labels.len() != num_samples {
            return None;
        }
        Some(Self {
            inputs,
            labels,
            shape: [num_samples, sample_len],
        })
    }

    #[must_use]
    pub const fn num_samples(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub const fn sample_len(&self) -> usize {
        self.shape[1]
    }

    #[must_use]
    pub fn get(&self, i: usize) -> (&[X], &Y) {
        let len = self.sample_len();
        let start = i * len;
        let x = &self.inputs[start..start + len];
        let y = &self.labels[i];
        (x, y)
    }

    /// Return a batch of samples as slices over the specified range.
    ///
    /// Panics if the range is out of bounds.
    #[must_use]
    pub fn get_batch(&self, range: core::ops::Range<usize>) -> (&'a [X], &'a [Y]) {
        let sample_len = self.sample_len();
        let start_idx = range.start * sample_len;
        let end_idx = range.end * sample_len;

        let x_batch = &self.inputs[start_idx..end_idx];
        let y_batch = &self.labels[range];

        (x_batch, y_batch)
    }
}
