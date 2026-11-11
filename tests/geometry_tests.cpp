#include "armor/geometry.hpp"
#include "armor/tracking.hpp"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool condition, const char* name) {
    if (!condition) throw std::runtime_error(name);
}

void testPose() {
    const armor::Calibration calibration{
        {{1280, 720}, "test-camera", 4, 6},
        (cv::Mat_<double>(3, 3) << 900, 0, 640, 0, 905, 360, 0, 0, 1),
        (cv::Mat_<double>(1, 5) << -0.08, 0.025, 0.001, -0.002, 0.0)
    };
    const auto object = armor::armorCorners(.14, .056);
    const cv::Vec3d rotation(.12, .35, .08), position(.17, -.05, 2.2);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object, rotation, position, calibration.camera_matrix, calibration.distortion, projected);
    std::array<cv::Point2f, 4> corners;
    std::copy(projected.begin(), projected.end(), corners.begin());
    const auto pose = armor::solveArmorPose(object, corners, calibration, 1);
    expect(pose && cv::norm(pose->position_m - position) < .002, "recover metric XYZ with distortion");
    expect(cv::norm(pose->rotation - rotation) < .015 && pose->rms_px < .01, "recover the rotation vector");
    std::swap(corners[1], corners[3]);
    expect(!armor::solveArmorPose(object, corners, calibration, 1), "reject mirrored index order");
    std::swap(corners[1], corners[3]);
    corners[0].x = -1;
    expect(!armor::solveArmorPose(object, corners, calibration, 1), "reject out-of-image model point");
    expect(!armor::solveArmorPose(object, {{{10, 10}, {10, 11}, {11, 11}, {11, 10}}}, calibration, 1),
           "reject degenerate tiny quad");

    armor::requireSameGeometry(calibration.image, calibration.image);
    for (const auto& image : {armor::ImageGeometry{{640, 360}, "test-camera", 4, 6},
                              armor::ImageGeometry{{1280, 720}, "other-camera", 4, 6},
                              armor::ImageGeometry{{1280, 720}, "test-camera", 0, 6}}) {
        bool rejected = false;
        try { armor::requireSameGeometry(calibration.image, image); }
        catch (const std::runtime_error&) { rejected = true; }
        expect(rejected, "reject calibration size, serial, or ROI mismatch");
    }
}

void testTracker() {
    armor::PositionTracker tracker({});
    using Status = armor::TrackingStatus;
    expect(tracker.step(0, std::nullopt).status == Status::Waiting, "waiting before first position");
    const auto initial = tracker.step(.1, cv::Vec3d(0, 0, 2));
    expect(initial.status == Status::Initialized && cv::norm(initial.motion->velocity_mps) == 0, "initialize at measurement");
    double time = .1;
    armor::TrackingUpdate last;
    for (int i = 0; i < 16; ++i) {
        time += (i % 2 == 0 ? .08 : .12);
        last = tracker.step(time, cv::Vec3d(time - .1, 0, 2));
        expect(last.status == Status::Corrected, "correct at nonuniform sample times");
    }
    expect(std::abs(last.motion->velocity_mps[0] - 1) < .05, "recover velocity");
    expect(cv::norm(last.motion->future_m - (last.motion->position_m + .2 * last.motion->velocity_mps)) < 1e-12,
           "forecast is a separate future position");
    const auto missed = tracker.step(time + .2, std::nullopt);
    expect(missed.status == Status::PredictionOnly &&
           cv::norm(missed.motion->position_m - last.motion->future_m) < 1e-12, "short occlusion advances from current state");
    const auto lost = tracker.step(time + .6, std::nullopt);
    expect(lost.status == Status::Lost && !lost.motion, "long occlusion invalidates motion");
    const auto recovered = tracker.step(time + .7, cv::Vec3d(5, 0, 3));
    expect(recovered.status == Status::Initialized && recovered.motion->position_m[0] == 5 &&
           cv::norm(recovered.motion->velocity_mps) == 0, "reacquisition resets velocity and position");
    expect(tracker.step(time + 1.3, cv::Vec3d(9, 0, 3)).status == Status::Initialized,
           "gap without frames also resets before correcting");
    bool rejected = false;
    try { tracker.step(time + 1.3, std::nullopt); }
    catch (const std::runtime_error&) { rejected = true; }
    expect(rejected, "reject repeated timestamp");

    armor::PositionTracker stationary({});
    double raw_error = 0, filtered_error = 0;
    for (int i = 0; i < 70; ++i) {
        const double noise = (i % 2 == 0 ? .06 : -.06);
        const auto filtered = stationary.step(i * .1, cv::Vec3d(0, 0, 2 + noise));
        const double error = filtered.motion->position_m[2] - 2;
        if (i >= 10) {
            raw_error += noise * noise;
            filtered_error += error * error;
        }
    }
    expect(filtered_error < raw_error, "reduce a noisy stationary sequence's error");
}
} // namespace

int main() {
    try {
        testPose();
        testTracker();
        std::cout << "PASS metric pose, calibration matching, and position tracking\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
