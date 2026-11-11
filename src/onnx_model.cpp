// 【模型推理】用 ONNX Runtime 的 C++ 接口加载已有模型，不启动 Python。
// 输入形状固定 [1,3,640,640]；输出按这份模型的 [1,18,8400] 结构解释。
#include "armor/onnx_model.hpp"
#include <onnxruntime_cxx_api.h>
#include <filesystem>

namespace armor {
struct OnnxModel::Impl {
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"armor-cpp"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    std::string input_name,output_name;
    Impl(const std::string& path) {
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.AddConfigEntry("session.intra_op.allow_spinning","0");
        auto native=std::filesystem::u8path(path);
        session=Ort::Session(environment,native.c_str(),options);
        if(session.GetInputCount()!=1 || session.GetOutputCount()!=1)
            throw std::runtime_error("Expected one ONNX input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name=session.GetInputNameAllocated(0,allocator).get();
        output_name=session.GetOutputNameAllocated(0,allocator).get();
        const auto input_type=session.GetInputTypeInfo(0);
        const auto output_type=session.GetOutputTypeInfo(0);
        const auto input=input_type.GetTensorTypeAndShapeInfo();
        const auto output=output_type.GetTensorTypeAndShapeInfo();
        if(input.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           input.GetShape()!=std::vector<int64_t>{1,3,kInput,kInput} ||
           output.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           output.GetShape()!=std::vector<int64_t>{1,kChannels,8400})
            throw std::runtime_error("Use the supplied FP32 best.onnx: input[1,3,640,640], output[1,18,8400]");
    }
};
OnnxModel::OnnxModel(const std::string& path):impl_(std::make_unique<Impl>(path)){}
OnnxModel::~OnnxModel()=default;
std::string OnnxModel::version() const { return OrtGetApiBase()->GetVersionString(); }
std::vector<float> OnnxModel::raw(const std::vector<float>& nchw) {
    // 将预处理后的浮点数组包装为模型输入，运行一次，返回原始输出。
    if(nchw.size()!=std::size_t(3*kInput*kInput)) throw std::runtime_error("Invalid input tensor size");
    std::array<int64_t,4> dimensions{1,3,kInput,kInput};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(memory,const_cast<float*>(nchw.data()),nchw.size(),
                                             dimensions.data(),dimensions.size());
    const char* inputs[]={impl_->input_name.c_str()};
    const char* outputs[]={impl_->output_name.c_str()};
    auto result=impl_->session.Run(Ort::RunOptions{nullptr},inputs,&input,1,outputs,1);
    auto info=result[0].GetTensorTypeAndShapeInfo();
    if(info.GetShape()!=std::vector<int64_t>{1,kChannels,8400}) throw std::runtime_error("Output shape changed");
    auto pointer=result[0].GetTensorData<float>();
    return {pointer,pointer+info.GetElementCount()};
}
std::vector<Detection> OnnxModel::predict(const std::vector<float>& nchw,const Letterbox& p,float conf,float iou) {
    // decode 负责按置信度筛选、去掉重复框，并把坐标还原到原图。
    auto output=raw(nchw); // 每帧约0.6MB，作用域结束就释放，不积攒所有预测。
    return decode(output.data(),8400,p,conf,iou);
}
}
