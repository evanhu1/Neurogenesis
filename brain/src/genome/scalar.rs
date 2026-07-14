use super::*;

pub fn inter_alpha_from_log_time_constant(log_time_constant: f32) -> f32 {
    let clamped_log_time_constant =
        log_time_constant.clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX);
    let time_constant = clamped_log_time_constant
        .exp()
        .clamp(INTER_TIME_CONSTANT_MIN, INTER_TIME_CONSTANT_MAX);
    1.0 - (-1.0 / time_constant).exp()
}
