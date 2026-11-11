#pragma once
// 不依赖相机/OpenCV的模型输出解释；可独立编译测试。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace armor {
constexpr int kInput = 640;
constexpr int kClasses = 6;
constexpr int kChannels = 4 + kClasses + 4 * 2;
inline constexpr const char* kNames[] = {"B1", "B2", "B3", "R1", "R2", "R3"};

struct Point { float x{}, y{}; };
struct Box { float x1{}, y1{}, x2{}, y2{}; };
struct Detection { Box box; float score{}; int label{}; std::array<Point, 4> points{}; };
struct Letterbox {
    int source_width{}, source_height{}, resized_width{}, resized_height{}, left{}, top{};
    float ratio{};
};

inline Letterbox letterbox(int width, int height) {
    if (width <= 0 || height <= 0) throw std::runtime_error("Invalid source dimensions");
    Letterbox p;
    p.source_width = width; p.source_height = height;
    p.ratio = std::min(float(kInput) / width, float(kInput) / height);
    p.resized_width = static_cast<int>(std::lround(width * p.ratio));
    p.resized_height = static_cast<int>(std::lround(height * p.ratio));
    p.left = static_cast<int>(std::round((kInput - p.resized_width) / 2.0 - 0.1));
    p.top = static_cast<int>(std::round((kInput - p.resized_height) / 2.0 - 0.1));
    return p;
}

inline float overlap(const Box& a, const Box& b) {
    float area_a = std::max(0.f, a.x2-a.x1) * std::max(0.f, a.y2-a.y1);
    float area_b = std::max(0.f, b.x2-b.x1) * std::max(0.f, b.y2-b.y1);
    float inter = std::max(0.f, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) *
                  std::max(0.f, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    return inter / (area_a + area_b - inter + 1e-7f);
}

inline std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, std::size_t maximum=300) {
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const auto& next : candidates) {
        bool suppress = false;
        for (const auto& previous : kept) {
            if (previous.label == next.label && overlap(previous.box, next.box) > threshold) {
                suppress = true; break;
            }
        }
        if (!suppress) kept.push_back(next);
        if (kept.size() >= maximum) break;
    }
    return kept;
}

inline std::vector<Detection> decode(const float* data, std::size_t count, const Letterbox& p,
                                      float confidence=0.25f, float iou=0.7f) {
    if (!data || !count || p.ratio <= 0) throw std::runtime_error("Invalid model output");
    std::vector<Detection> candidates;
    const auto at = [&](int channel, std::size_t index) { return data[std::size_t(channel)*count + index]; };
    for (std::size_t i=0; i<count; ++i) {
        int label=0;
        float score=at(4,i);
        for (int c=1; c<kClasses; ++c) if (at(4+c,i) > score) { label=c; score=at(4+c,i); }
        if (!std::isfinite(score) || !(score > confidence)) continue;
        float x=at(0,i), y=at(1,i), w=at(2,i), h=at(3,i);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h) || w<=0 || h<=0) continue;
        Detection d;
        d.score=score; d.label=label;
        d.box={(x-w/2-p.left)/p.ratio, (y-h/2-p.top)/p.ratio,
               (x+w/2-p.left)/p.ratio, (y+h/2-p.top)/p.ratio};
        for (int k=0; k<4; ++k) {
            d.points[k]={(at(10+2*k,i)-p.left)/p.ratio, (at(11+2*k,i)-p.top)/p.ratio};
        }
        candidates.push_back(d);
    }
    auto result=nms(std::move(candidates), iou);
    // 框按原图边界裁切。关键点保留原始预测坐标，PnP 需要拒绝出界点。
    for (auto& d:result) {
        d.box.x1=std::clamp(d.box.x1,0.f,float(p.source_width));
        d.box.x2=std::clamp(d.box.x2,0.f,float(p.source_width));
        d.box.y1=std::clamp(d.box.y1,0.f,float(p.source_height));
        d.box.y2=std::clamp(d.box.y2,0.f,float(p.source_height));
    }
    return result;
}

inline std::size_t checked_bgr_bytes(int width,int height) {
    if(width<=0 || height<=0 || width>16384 || height>16384)
        throw std::runtime_error("Invalid image size");
    auto bytes=std::size_t(width)*std::size_t(height)*3;
    if(bytes>64u*1024u*1024u) throw std::runtime_error("BGR frame exceeds 64 MiB limit");
    return bytes;
}

// 一次成功借帧，一次归还。即使处理中抛异常，析构也执行归还。
template<class Release> class FrameLease {
    Release release_;
    bool active_=true;
public:
    explicit FrameLease(Release release):release_(release){}
    FrameLease(const FrameLease&)=delete;
    ~FrameLease() { if(active_) release_(); }
    void release() { if(active_) { active_=false; release_(); } }
};
} // namespace armor
