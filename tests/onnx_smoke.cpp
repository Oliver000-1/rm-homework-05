#include "armor/onnx_model.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc,char** argv) {
    try {
        if(argc!=5 && argc!=6)throw std::runtime_error("Usage: ort_smoke MODEL INPUT_FLOAT32.bin RAW_OUTPUT.bin DETECTIONS.csv [REFERENCE.csv]");
        std::vector<float> input(3*armor::kInput*armor::kInput);
        std::ifstream stream(argv[2],std::ios::binary);
        stream.read(reinterpret_cast<char*>(input.data()),static_cast<std::streamsize>(input.size()*sizeof(float)));
        if(stream.gcount()!=static_cast<std::streamsize>(input.size()*sizeof(float)))throw std::runtime_error("Bad input fixture");
        armor::OnnxModel model(argv[1]);
        auto output=model.raw(input);
        std::ofstream binary(argv[3],std::ios::binary);
        binary.write(reinterpret_cast<const char*>(output.data()),static_cast<std::streamsize>(output.size()*sizeof(float)));
        auto detections=armor::decode(output.data(),8400,armor::letterbox(1280,720));
        std::ofstream csv(argv[4]);csv<<std::setprecision(10);
        for(const auto& d:detections) {
            csv<<d.label<<','<<d.score<<','<<d.box.x1<<','<<d.box.y1<<','<<d.box.x2<<','<<d.box.y2;
            for(const auto& point:d.points)csv<<','<<point.x<<','<<point.y;
            csv<<'\n';
        }
        if(!binary || !csv)throw std::runtime_error("Fixture output write failed");
        if(argc==6) {
            std::ifstream reference(argv[5]);
            if(!reference)throw std::runtime_error("Missing reference detections");
            std::vector<std::vector<double>> expected;std::string line;
            while(std::getline(reference,line)) {
                if(line.empty())continue;
                std::istringstream row(line);std::vector<double> values;std::string field;
                while(std::getline(row,field,','))values.push_back(std::stod(field));
                if(values.size()!=14)throw std::runtime_error("Invalid reference row");
                expected.push_back(std::move(values));
            }
            if(expected.size()!=detections.size())throw std::runtime_error("Reference target count differs");
            for(std::size_t i=0;i<detections.size();++i) {
                const auto& d=detections[i];
                std::vector<double> values{double(d.label),d.score,d.box.x1,d.box.y1,d.box.x2,d.box.y2};
                for(const auto& point:d.points){values.push_back(point.x);values.push_back(point.y);}
                for(std::size_t k=0;k<values.size();++k)
                    if(!std::isfinite(values[k]) || std::abs(values[k]-expected[i][k])>.002)
                        throw std::runtime_error("Reference box/keypoint mismatch >0.002");
            }
            std::cout<<"PASS real-model reference boxes and four keypoints\n";
        }
        std::cout<<"ONNX Runtime "<<model.version()<<"; detections="<<detections.size()<<'\n';
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
