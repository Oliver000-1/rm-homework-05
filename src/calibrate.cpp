#include "armor/geometry.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {

struct Options {
    fs::path images;
    fs::path output;
    int columns = 0;
    int rows = 0;
    double square_mm = 0;
    bool check_only = false;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto value = [&]() {
            if (++i >= argc) throw std::runtime_error("Missing value for " + key);
            return std::string(argv[i]);
        };
        const auto number = [&]() {
            const auto text = value();
            std::size_t end = 0;
            const double result = std::stod(text, &end);
            if (end != text.size() || !std::isfinite(result)) throw std::runtime_error("Invalid number for " + key);
            return result;
        };
        if (key == "--help" || key == "-h") {
            std::cout << "armor_calibrate --images DIR --cols N --rows N --square-mm MM --output camera.yml\n"
                      << "  --check-only: detect chessboard corners; square size and output are optional\n"
                      << "  --cols and --rows count inner corners, not squares\n";
            std::exit(0);
        } else if (key == "--images") options.images = fs::u8path(value());
        else if (key == "--output") options.output = fs::u8path(value());
        else if (key == "--square-mm") options.square_mm = number();
        else if (key == "--cols" || key == "--rows") {
            const auto count = number();
            if (count != std::floor(count) || count < 3 || count > 100)
                throw std::runtime_error("Chessboard inner corner counts must be integers in 3..100");
            (key == "--cols" ? options.columns : options.rows) = static_cast<int>(count);
        } else if (key == "--check-only") options.check_only = true;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (!fs::is_directory(options.images) || !options.columns || !options.rows)
        throw std::runtime_error("Supply --images DIR --cols N --rows N");
    if (!options.check_only && (!(options.square_mm > 0) || options.square_mm > 1000 || options.output.empty()))
        throw std::runtime_error("Supply a measured --square-mm and a new --output path");
    if (!options.output.empty() && fs::exists(options.output))
        throw std::runtime_error("Output already exists: " + options.output.u8string());
    return options;
}

std::vector<fs::path> imageFiles(const fs::path& directory) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        auto extension = entry.path().extension().u8string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" ||
            extension == ".jpeg" || extension == ".bmp" || extension == ".tif" || extension == ".tiff"))
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

cv::Mat readGray(const fs::path& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) return {};
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.empty()) return {};
    return cv::imdecode(bytes, cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION);
}

struct View {
    std::string name;
    std::vector<cv::Point2f> pixels;
};

std::vector<View> detectViews(const Options& options, armor::ImageGeometry& image) {
    std::vector<View> views;
    for (const auto& file : imageFiles(options.images)) {
        const auto gray = readGray(file);
        const auto name = file.filename().u8string();
        if (gray.empty()) {
            std::cout << "unreadable " << name << '\n';
            continue;
        }
        if (image.size.empty()) image.size = gray.size();
        if (image.size != gray.size()) throw std::runtime_error("Mixed image resolutions at " + name);
        std::vector<cv::Point2f> pixels;
        const cv::Size pattern(options.columns, options.rows);
        if (cv::findChessboardCornersSB(gray, pattern, pixels, cv::CALIB_CB_NORMALIZE_IMAGE)) {
            std::cout << "ok " << name << " (" << pixels.size() << " inner corners)\n";
            views.push_back({name, std::move(pixels)});
        } else {
            std::cout << "no full chessboard " << name << '\n';
        }
    }
    return views;
}

void calibrate(const Options& options, const std::vector<View>& views, armor::Calibration& calibration) {
    if (views.size() < 12) throw std::runtime_error("Need at least 12 usable views for fit and validation; aim for 20+ diverse views");
    std::vector<cv::Point3f> board;
    const auto square_m = static_cast<float>(options.square_mm / 1000.0);
    for (int row = 0; row < options.rows; ++row)
        for (int col = 0; col < options.columns; ++col)
            board.emplace_back(col * square_m, row * square_m, 0);

    std::vector<std::vector<cv::Point2f>> fit_pixels;
    std::vector<std::string> fit_names;
    std::vector<const View*> validation;
    for (std::size_t i = 0; i < views.size(); ++i) {
        if ((i + 1) % 5 == 0) validation.push_back(&views[i]);
        else {
            fit_names.push_back(views[i].name);
            fit_pixels.push_back(views[i].pixels);
        }
    }
    const std::vector<std::vector<cv::Point3f>> objects(fit_pixels.size(), board);
    std::vector<cv::Mat> rotations, translations;
    std::vector<double> fit_errors;
    const double rms = cv::calibrateCamera(objects, fit_pixels, calibration.image.size,
                                           calibration.camera_matrix, calibration.distortion,
                                           rotations, translations, cv::noArray(), cv::noArray(), fit_errors);
    if (!std::isfinite(rms) || !cv::checkRange(calibration.camera_matrix) ||
        !cv::checkRange(calibration.distortion))
        throw std::runtime_error("Calibration is not finite; collect sharper and more varied views");

    std::vector<double> validation_errors;
    double validation_squares = 0;
    for (const auto* view : validation) {
        cv::Vec3d rotation, position;
        if (!cv::solvePnP(board, view->pixels, calibration.camera_matrix, calibration.distortion, rotation, position))
            throw std::runtime_error("Cannot estimate chessboard pose in validation image " + view->name);
        const double error = armor::reprojectionRms(board, view->pixels, rotation, position, calibration);
        if (!std::isfinite(error)) throw std::runtime_error("Invalid validation reprojection error");
        validation_errors.push_back(error);
        validation_squares += error * error;
    }
    const double validation_rms = std::sqrt(validation_squares / validation.size());
    cv::FileStorage yaml(".yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    armor::writeImageGeometry(yaml, calibration.image);
    yaml << "camera_matrix" << calibration.camera_matrix << "distortion_coefficients" << calibration.distortion
         << "inner_columns" << options.columns << "inner_rows" << options.rows << "square_mm" << options.square_mm
         << "fit_rms_px" << rms << "validation_rms_px" << validation_rms << "fit_images" << "[";
    for (std::size_t i = 0; i < fit_errors.size(); ++i)
        yaml << "{" << "file" << fit_names[i] << "rms_px" << fit_errors[i] << "}";
    yaml << "]" << "validation_images" << "[";
    for (std::size_t i = 0; i < validation.size(); ++i)
        yaml << "{" << "file" << validation[i]->name << "rms_px" << validation_errors[i] << "}";
    yaml << "]";
    armor::writeNewYaml(options.output, yaml);
    std::cout << std::fixed << std::setprecision(3)
              << "Fit: " << fit_pixels.size() << " views, " << rms << " px RMS\n"
              << "Held out: " << validation.size() << " views, " << validation_rms << " px RMS\n"
              << "Saved " << options.output.u8string() << '\n';
    if (calibration.image.camera_serial.empty())
        std::cout << "Photo source has no MVS capture record. This profile is for offline checks.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        cv::setNumThreads(1);
        armor::Calibration calibration;
        const auto capture = options.images / "capture.yml";
        if (fs::exists(capture)) calibration.image = armor::readImageGeometry(armor::readYaml(capture));
        const auto views = detectViews(options, calibration.image);
        std::cout << views.size() << " usable views at " << calibration.image.size.width << 'x'
                  << calibration.image.size.height << '\n';
        if (options.check_only) return views.empty() ? 1 : 0;
        calibrate(options, views, calibration);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Calibration error: " << error.what() << '\n';
        return 1;
    }
}
