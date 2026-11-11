#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace armor {
struct CameraOptions {
    std::string serial,format="BayerRG8",exercise_error; // 【可调】序列号、像素格式；故障注入仅供演练
    double exposure_us=1000,gain_db=-1; // 【可调】曝光微秒、增益dB；-1表示使用相机支持的最小增益
};
struct FrameInfo {
    std::uint64_t number{},device_ticks{};
    double received_seconds{},convert_ms{};
};
class UsbCamera {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    UsbCamera();
    ~UsbCamera();
    UsbCamera(const UsbCamera&)=delete;
    static int list();
    void open(const CameraOptions& options);
    bool read(cv::Mat& owned_bgr,FrameInfo& info);
    void close() noexcept;
    std::map<std::string,std::string> metadata() const;
    bool cleanup_ok() const;
};
}
