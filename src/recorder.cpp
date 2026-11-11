#include "armor/recorder.hpp"
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace armor {
Recorder::Recorder(const std::string& path,const cv::Mat& image):path_(path),latest_(image.clone()) {
    if(!writer_.open(path,cv::VideoWriter::fourcc('M','J','P','G'),10,image.size(),true))
        throw std::runtime_error("MJPG recording unavailable");
    worker_=std::thread(&Recorder::run,this);
}
Recorder::~Recorder(){stop();}
void Recorder::update(const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(image.size()!=latest_.size())throw std::runtime_error("Recording image size changed");
    latest_=image.clone();
}
void Recorder::run() noexcept {
    using Clock=std::chrono::steady_clock;
    auto started=Clock::now();
    try {
        while(!stop_.load()) {
            auto due=started+std::chrono::milliseconds(frames_*100);
            cv::Mat image;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if(condition_.wait_until(lock,due,[&]{return stop_.load();}))break;
                image=latest_; // 独立图像的引用计数延长本帧寿命；update只替换，不写旧像素。
            }
            if(Clock::now()-due>std::chrono::milliseconds(500))
                throw std::runtime_error("Recording writer >500ms behind; stopped without accumulating frames");
            if(frames_%10==0) {
                auto path=std::filesystem::u8path(path_);
                if(std::filesystem::space(path.parent_path()).available<1024ull*1024*1024)
                    throw std::runtime_error("Recording stopped: less than 1 GiB disk free");
                if(std::filesystem::exists(path) && std::filesystem::file_size(path)>128ull*1024*1024)
                    throw std::runtime_error("Recording reached 128 MiB; start a new segment");
            }
            writer_.write(image);++frames_;
        }
    } catch(const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);error_=error.what();
    }
    writer_.release();
    wall_=std::chrono::duration<double>(Clock::now()-started).count();
}
void Recorder::stop() {
    stop_.store(true);condition_.notify_all();
    if(worker_.joinable())worker_.join();
}
std::string Recorder::error(){std::lock_guard<std::mutex> lock(mutex_);return error_;}
std::string Recorder::summary() const {
    // 只在stop/join之后读取，避免与录像线程竞争。
    std::ostringstream out;
    out<<path_<<"; fps=10; frames="<<frames_<<"; video_seconds="<<frames_/10.0
       <<"; wall_seconds="<<wall_<<"; error="<<error_;
    return out.str();
}
}
