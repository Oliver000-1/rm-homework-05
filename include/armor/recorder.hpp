#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace armor {
// 录像线程只保留一张最新的独立图像；不创建无界队列。
class Recorder {
    std::string path_,error_;
    cv::VideoWriter writer_;
    cv::Mat latest_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
    int frames_=0;
    double wall_=0;
    void run() noexcept;
public:
    Recorder(const std::string& path,const cv::Mat& image);
    ~Recorder();
    void update(const cv::Mat& image);
    void stop();
    std::string error();
    std::string summary() const;
};
}
