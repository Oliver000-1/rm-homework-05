#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace armor {

struct ImageGeometry {
    cv::Size size;
    std::string camera_serial;  // 空串表示照片来源未与现场相机绑定。
    int offset_x = 0;
    int offset_y = 0;
};

struct Calibration {
    ImageGeometry image;
    cv::Mat camera_matrix;
    cv::Mat distortion;
};

cv::FileStorage readYaml(const std::filesystem::path& path);
void writeNewYaml(const std::filesystem::path& path, cv::FileStorage& yaml);
ImageGeometry readImageGeometry(const cv::FileStorage& yaml);
void writeImageGeometry(cv::FileStorage& yaml, const ImageGeometry& image);
void requireSameGeometry(const ImageGeometry& expected, const ImageGeometry& actual);
Calibration loadCalibration(const std::filesystem::path& path);

struct Pose {
    cv::Vec3d rotation;
    cv::Vec3d position_m;
    double rms_px;
};

// 物理顺序统一为左上、左下、右下、右上，原点取装甲板中心。
std::vector<cv::Point3f> armorCorners(double width_m, double height_m);
double reprojectionRms(const std::vector<cv::Point3f>& object_points,
                       const std::vector<cv::Point2f>& observed,
                       const cv::Vec3d& rotation, const cv::Vec3d& position,
                       const Calibration& calibration);
std::optional<Pose> solveArmorPose(const std::vector<cv::Point3f>& object_points,
                                  const std::array<cv::Point2f, 4>& image_points,
                                  const Calibration& calibration, double max_rms_px);

} // namespace armor
