#pragma once

#include <opencv2/video/tracking.hpp>
#include <optional>

namespace armor {

struct MotionOptions {
    cv::Vec3d measurement_std_m{0.02, 0.02, 0.08};  // 现场可用静止段的标准差替换。
    double acceleration_std_mps2 = 2.0;
    double prediction_seconds = 0.2;
    double max_gap_seconds = 0.5;
};

enum class TrackingStatus { Waiting, Initialized, Corrected, PredictionOnly, Lost };
const char* trackingStatusName(TrackingStatus status);

struct Motion {
    cv::Vec3d position_m;
    cv::Vec3d velocity_mps;
    cv::Vec3d future_m;
};

struct TrackingUpdate {
    TrackingStatus status = TrackingStatus::Waiting;
    std::optional<Motion> motion;
};

// 位置、速度都在固定相机坐标系中。每一帧调用一次，缺测用 nullopt。
class PositionTracker {
    MotionOptions options_;
    cv::KalmanFilter filter_;
    std::optional<double> frame_time_;
    std::optional<double> measurement_time_;
    void initialize(const cv::Vec3d& position);
    void predict(double dt);
    Motion motion() const;
public:
    explicit PositionTracker(const MotionOptions& options);
    TrackingUpdate step(double time_seconds, const std::optional<cv::Vec3d>& position);
};

} // namespace armor
