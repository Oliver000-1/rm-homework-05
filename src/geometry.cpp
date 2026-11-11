#include "armor/geometry.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace armor {
namespace fs = std::filesystem;

cv::FileStorage readYaml(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.u8string());
    std::string contents((std::istreambuf_iterator<char>(input)), {});
    if (contents.empty()) throw std::runtime_error("Empty YAML: " + path.u8string());
    cv::FileStorage yaml(contents, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if (!yaml.isOpened()) throw std::runtime_error("Invalid YAML: " + path.u8string());
    return yaml;
}

void writeNewYaml(const fs::path& path, cv::FileStorage& yaml) {
    const auto contents = yaml.releaseAndGetString();
    auto temporary = path;
    temporary += ".tmp";
    const auto staged = temporary / "profile.yml";
    if (!fs::create_directory(temporary))
        throw std::runtime_error("Output staging directory exists: " + temporary.u8string());
    try {
        if (fs::exists(path)) throw std::runtime_error("Choose a new output file: " + path.u8string());
        std::ofstream output(staged, std::ios::binary);
        output << contents;
        output.close();
        if (!output) throw std::runtime_error("Cannot write " + staged.u8string());
        fs::rename(staged, path);
    } catch (...) {
        std::error_code error;
        fs::remove(staged, error);
        fs::remove(temporary, error);
        throw;
    }
    std::error_code error;
    fs::remove(temporary, error);
}

ImageGeometry readImageGeometry(const cv::FileStorage& yaml) {
    const auto integer = [&](const char* key, int min_value) {
        const auto field = yaml[key];
        if (!field.isInt() || static_cast<int>(field) < min_value)
            throw std::runtime_error(std::string("Missing or invalid YAML field: ") + key);
        return static_cast<int>(field);
    };
    ImageGeometry image;
    image.size = {integer("image_width", 1), integer("image_height", 1)};
    const auto serial = yaml["camera_serial"];
    if (!serial.empty()) {
        if (!serial.isString()) throw std::runtime_error("camera_serial must be a string");
        serial >> image.camera_serial;
    }
    if (!image.camera_serial.empty()) {
        image.offset_x = integer("roi_offset_x", 0);
        image.offset_y = integer("roi_offset_y", 0);
    }
    return image;
}

void writeImageGeometry(cv::FileStorage& yaml, const ImageGeometry& image) {
    yaml << "image_width" << image.size.width << "image_height" << image.size.height;
    if (!image.camera_serial.empty()) {
        yaml << "camera_serial" << image.camera_serial
             << "roi_offset_x" << image.offset_x << "roi_offset_y" << image.offset_y;
    }
}

void requireSameGeometry(const ImageGeometry& expected, const ImageGeometry& actual) {
    if (expected.size != actual.size)
        throw std::runtime_error("Resolution differs from the calibration/capture record");
    if (expected.camera_serial.empty() || expected.camera_serial != actual.camera_serial)
        throw std::runtime_error("Calibration needs a capture record from this MVS camera");
    if (expected.offset_x != actual.offset_x || expected.offset_y != actual.offset_y)
        throw std::runtime_error("ROI offsets differ from the calibration/capture record");
}

Calibration loadCalibration(const fs::path& path) {
    const auto yaml = readYaml(path);
    Calibration calibration;
    calibration.image = readImageGeometry(yaml);
    yaml["camera_matrix"] >> calibration.camera_matrix;
    yaml["distortion_coefficients"] >> calibration.distortion;
    auto& k = calibration.camera_matrix;
    auto& d = calibration.distortion;
    if (k.rows != 3 || k.cols != 3 || k.channels() != 1 ||
        d.total() != 5 || (d.rows != 1 && d.cols != 1) || d.channels() != 1 ||
        !cv::checkRange(k) || !cv::checkRange(d)) {
        throw std::runtime_error("Expected finite 3x3 camera matrix and five distortion coefficients");
    }
    k.convertTo(k, CV_64F);
    d.convertTo(d, CV_64F);
    if (k.at<double>(0, 0) <= 0 || k.at<double>(1, 1) <= 0 ||
        k.at<double>(0, 1) != 0 || k.at<double>(1, 0) != 0 || k.at<double>(2, 0) != 0 ||
        k.at<double>(2, 1) != 0 || k.at<double>(2, 2) != 1) {
        throw std::runtime_error("Invalid pinhole camera matrix");
    }
    return calibration;
}

std::vector<cv::Point3f> armorCorners(double width_m, double height_m) {
    if (!(width_m > 0) || !(height_m > 0) || !std::isfinite(width_m) || !std::isfinite(height_m))
        throw std::runtime_error("Measure the armor keypoint width and height in millimeters");
    const auto x = static_cast<float>(width_m / 2);
    const auto y = static_cast<float>(height_m / 2);
    return {{-x, -y, 0}, {-x, y, 0}, {x, y, 0}, {x, -y, 0}};
}

double reprojectionRms(const std::vector<cv::Point3f>& object_points,
                       const std::vector<cv::Point2f>& observed,
                       const cv::Vec3d& rotation, const cv::Vec3d& position,
                       const Calibration& calibration) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rotation, position, calibration.camera_matrix,
                      calibration.distortion, projected);
    return cv::norm(observed, projected, cv::NORM_L2) / std::sqrt(static_cast<double>(observed.size()));
}

std::optional<Pose> solveArmorPose(const std::vector<cv::Point3f>& object_points,
                                  const std::array<cv::Point2f, 4>& image_points,
                                  const Calibration& calibration, double max_rms_px) {
    const auto size = calibration.image.size;
    const auto in_image = [&](const cv::Point2f& p) {
        return p.x >= 0 && p.y >= 0 && p.x < size.width && p.y < size.height;
    };
    if (!std::all_of(image_points.begin(), image_points.end(), in_image)) return std::nullopt;
    const std::vector<cv::Point2f> observed(image_points.begin(), image_points.end());
    // 前面朝向相机时，四点按约定绕行。反向、自交及过小的四边形都不参与解算。
    if (!cv::isContourConvex(observed) || cv::contourArea(observed, true) >= -16) return std::nullopt;
    Pose pose{};
    if (!cv::solvePnP(object_points, observed, calibration.camera_matrix, calibration.distortion,
                      pose.rotation, pose.position_m, false, cv::SOLVEPNP_IPPE)) return std::nullopt;
    if (!cv::checkRange(cv::Mat(pose.rotation)) || !cv::checkRange(cv::Mat(pose.position_m)))
        return std::nullopt;
    cv::Matx33d rotation;
    cv::Rodrigues(pose.rotation, rotation);
    for (const auto& p : object_points) {
        const auto camera_point = rotation * cv::Vec3d(p.x, p.y, p.z) + pose.position_m;
        if (!(camera_point[2] > 0)) return std::nullopt;
    }
    pose.rms_px = reprojectionRms(object_points, observed, pose.rotation, pose.position_m, calibration);
    if (!std::isfinite(pose.rms_px) || pose.rms_px > max_rms_px) return std::nullopt;
    return pose;
}

} // namespace armor
