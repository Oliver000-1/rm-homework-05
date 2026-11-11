#include "armor/tracking.hpp"

#include <cmath>
#include <stdexcept>

namespace armor {
namespace {
bool positiveFinite(double value) {
    return std::isfinite(value) && value > 0;
}
} // namespace

const char* trackingStatusName(TrackingStatus status) {
    switch (status) {
        case TrackingStatus::Waiting: return "waiting";
        case TrackingStatus::Initialized: return "initialized";
        case TrackingStatus::Corrected: return "corrected";
        case TrackingStatus::PredictionOnly: return "prediction_only";
        case TrackingStatus::Lost: return "lost";
    }
    throw std::logic_error("Unknown tracking status");
}

PositionTracker::PositionTracker(const MotionOptions& options) : options_(options), filter_(6, 3, 0, CV_64F) {
    if (!positiveFinite(options_.acceleration_std_mps2) ||
        !positiveFinite(options_.prediction_seconds) || !positiveFinite(options_.max_gap_seconds))
        throw std::runtime_error("Invalid motion time/noise parameters");
    for (int axis = 0; axis < 3; ++axis) {
        const double sigma = options_.measurement_std_m[axis];
        if (!positiveFinite(sigma)) throw std::runtime_error("Measurement standard deviations must be positive");
        filter_.measurementMatrix.at<double>(axis, axis) = 1;
        filter_.measurementNoiseCov.at<double>(axis, axis) = sigma * sigma;
    }
}

void PositionTracker::initialize(const cv::Vec3d& position) {
    filter_.statePost.setTo(0);
    filter_.errorCovPost.setTo(0);
    for (int axis = 0; axis < 3; ++axis) {
        filter_.statePost.at<double>(axis) = position[axis];
        filter_.errorCovPost.at<double>(axis, axis) = filter_.measurementNoiseCov.at<double>(axis, axis);
        filter_.errorCovPost.at<double>(axis + 3, axis + 3) = 1;  // 初始速度标准差 1 m/s。
    }
}

void PositionTracker::predict(double dt) {
    // 三轴独立，单帧内随机加速度分别以 0.5*dt^2 和 dt 进入位置与速度。
    const double variance = options_.acceleration_std_mps2 * options_.acceleration_std_mps2;
    const double pos_coeff = 0.5 * dt * dt;
    for (int axis = 0; axis < 3; ++axis) {
        const int speed = axis + 3;
        filter_.transitionMatrix.at<double>(axis, speed) = dt;
        filter_.processNoiseCov.at<double>(axis, axis) = pos_coeff * pos_coeff * variance;
        filter_.processNoiseCov.at<double>(axis, speed) = pos_coeff * dt * variance;
        filter_.processNoiseCov.at<double>(speed, axis) = pos_coeff * dt * variance;
        filter_.processNoiseCov.at<double>(speed, speed) = dt * dt * variance;
    }
    filter_.predict();
}

Motion PositionTracker::motion() const {
    const auto& state = filter_.statePost;
    const cv::Vec3d position(state.at<double>(0), state.at<double>(1), state.at<double>(2));
    const cv::Vec3d velocity(state.at<double>(3), state.at<double>(4), state.at<double>(5));
    // 未来预测只用于展示，滤波器状态仍对应当前这一帧。
    return {position, velocity, position + velocity * options_.prediction_seconds};
}

TrackingUpdate PositionTracker::step(double time_seconds, const std::optional<cv::Vec3d>& position) {
    if (!std::isfinite(time_seconds) || (frame_time_ && time_seconds <= *frame_time_))
        throw std::runtime_error("Frame times must be finite and increasing");
    if (position && !cv::checkRange(cv::Mat(*position)))
        throw std::runtime_error("Non-finite position reached the filter");
    const double dt = frame_time_ ? time_seconds - *frame_time_ : 0;
    frame_time_ = time_seconds;
    const bool expired = measurement_time_ && time_seconds - *measurement_time_ >= options_.max_gap_seconds;
    if (expired) measurement_time_.reset();

    if (!measurement_time_) {
        if (!position) return {expired ? TrackingStatus::Lost : TrackingStatus::Waiting, std::nullopt};
        initialize(*position);
        measurement_time_ = time_seconds;
        return {TrackingStatus::Initialized, motion()};
    }

    predict(dt);
    if (!position) return {TrackingStatus::PredictionOnly, motion()};
    filter_.correct(cv::Mat(*position));
    measurement_time_ = time_seconds;
    return {TrackingStatus::Corrected, motion()};
}

} // namespace armor
