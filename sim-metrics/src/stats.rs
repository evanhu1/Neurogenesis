//! Small numeric helpers shared across the metric layers. All ignore
//! non-finite inputs so a single `NaN`/`inf` can't poison an aggregate.

/// Mean of the finite, present values in an iterator of optionals. `None` when
/// nothing finite was seen.
pub fn mean_option(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_u64;
    for value in values.flatten() {
        if value.is_finite() {
            sum += value;
            count = count.saturating_add(1);
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Mean of the finite values in an iterator, rounded to the nearest `u32`.
pub fn mean_round_u32(values: impl Iterator<Item = u32>) -> u32 {
    mean_f64(values.map(|value| value as f64)).round() as u32
}

/// Mean of the finite values in an iterator. `0.0` when nothing finite was seen.
pub fn mean_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_u64;
    for value in values {
        if value.is_finite() {
            sum += value;
            count = count.saturating_add(1);
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
