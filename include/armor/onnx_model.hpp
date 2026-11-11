#pragma once
#include "core.hpp"
#include <memory>
#include <string>

namespace armor {
// 实现放在.cpp中：调用官方ONNX Runtime C++接口，无Python子进程。
class OnnxModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit OnnxModel(const std::string& path);
    ~OnnxModel();
    OnnxModel(const OnnxModel&)=delete;
    std::vector<float> raw(const std::vector<float>& nchw);
    std::vector<Detection> predict(const std::vector<float>& nchw,const Letterbox& transform,
                                   float confidence,float iou);
    std::string version() const;
};
}
