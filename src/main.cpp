// 第五次作业：取图 → YOLO 四角点 → PnP → 卡尔曼状态和未来位置。
// 标定取图与识别共用同一个 MVS 采集入口。
#include "armor/camera.hpp"
#include "armor/core.hpp"
#include "armor/geometry.hpp"
#include "armor/onnx_model.hpp"
#include "armor/recorder.hpp"
#include "armor/tracking.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <charconv>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>

namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static volatile std::sig_atomic_t quit_requested=0;
static void request_quit(int){quit_requested=1;} // 信号里不调用相机/写盘/分配内存。
static double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}

struct Options {
    // 【可调参数】平时优先用运行命令传参，不必直接修改这些默认值。
    // 相机的格式、曝光和增益默认值在 include/armor/camera.hpp 的 CameraOptions 里。
    armor::CameraOptions camera;
    std::string video,model="models/best.onnx",output="outputs/live",light_note="not recorded";
    bool list=false,preview=false,headless=false,record=false,doctor=false;
    bool capture_calibration=false;
    std::string calibration_path;
    std::array<int,4> corner_order{0,1,2,3};
    int target_label=-1;
    double armor_width_mm=0,armor_height_mm=0,max_rms_px=3;
    armor::MotionOptions motion;
    double seconds=60,grab_fps=10; // 最长运行秒数；程序取图上限（不等于相机硬件帧率）
    float confidence=.25f,iou=.7f; // 置信度门槛；同类重叠框筛选阈值，均不是准确率
    int max_frames=0;
};
static Options parse(int argc,char** argv) {
    // 例如 --seconds 30：key 读到参数名，value()/number() 读其后面的值。
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        auto value=[&](){if(i+1>=argc)throw std::runtime_error("Missing value for "+key);return std::string(argv[++i]);};
        auto number=[&](){auto s=value();std::size_t n=0;double d=std::stod(s,&n);if(n!=s.size()||!std::isfinite(d))throw std::runtime_error("Bad number: "+key);return d;};
        if(key=="--help" || key=="-h") {
            std::cout<<"RM Homework 05 | C++17 MVS USB + OpenCV + ONNX Runtime\n"
                     <<"--list-cameras | --doctor | --preview | --video FILE\n"
                     <<"--serial SERIAL --pixel-format BayerRG8|BGR8|Mono8 --exposure-us 1000 --gain-db 0\n"
                     <<"--model models/best.onnx --conf 0.25 --iou 0.7 --seconds 60 --max-frames 10\n"
                     <<"--grab-fps 10 --headless --record --output outputs/live --light-note TEXT\n"
                     <<"--capture-calibration: S saves an original frame and MVS geometry for chessboard calibration\n"
                     <<"--calibration camera.yml --armor-width-mm W --armor-height-mm H\n"
                     <<"--corner-order 1234 (model indices for TL BL BR TR) --target B1|B2|B3|R1|R2|R3\n"
                     <<"--max-reprojection-px 3 --measurement-std-mm SX SY SZ --accel-std 2\n"
                     <<"--exercise-error format|timeout (simulated, requires --preview and a camera)\n"
                     <<"Q/Esc quit; S snapshot; R toggle 10-FPS recording\n";std::exit(0);
        } else if(key=="--serial")o.camera.serial=value();
        else if(key=="--video")o.video=value();
        else if(key=="--model")o.model=value();
        else if(key=="--output")o.output=value();
        else if(key=="--light-note")o.light_note=value();
        else if(key=="--calibration")o.calibration_path=value();
        else if(key=="--armor-width-mm")o.armor_width_mm=number();
        else if(key=="--armor-height-mm")o.armor_height_mm=number();
        else if(key=="--max-reprojection-px")o.max_rms_px=number();
        else if(key=="--accel-std")o.motion.acceleration_std_mps2=number();
        else if(key=="--measurement-std-mm") {
            for(int axis=0;axis<3;++axis)o.motion.measurement_std_m[axis]=number()/1000.0;
        } else if(key=="--corner-order") {
            auto order=value(),sorted=order;
            std::sort(sorted.begin(),sorted.end());
            if(sorted!="1234")throw std::runtime_error("--corner-order must be a permutation of 1234");
            for(std::size_t j=0;j<o.corner_order.size();++j)o.corner_order[j]=order[j]-'1';
        } else if(key=="--target") {
            const auto label=value();
            for(int c=0;c<armor::kClasses;++c)if(label==armor::kNames[c])o.target_label=c;
            if(o.target_label<0)throw std::runtime_error("Unknown --target class: "+label);
        }
        else if(key=="--pixel-format")o.camera.format=value();
        else if(key=="--exercise-error")o.camera.exercise_error=value();
        else if(key=="--seconds")o.seconds=number();
        else if(key=="--grab-fps")o.grab_fps=number();
        else if(key=="--exposure-us")o.camera.exposure_us=number();
        else if(key=="--gain-db")o.camera.gain_db=number();
        else if(key=="--conf")o.confidence=static_cast<float>(number());
        else if(key=="--iou")o.iou=static_cast<float>(number());
        else if(key=="--max-frames") {auto n=number();if(n!=std::floor(n)||n<1||n>1000000)throw std::runtime_error("Invalid max-frames");o.max_frames=static_cast<int>(n);}
        else if(key=="--list-cameras")o.list=true;
        else if(key=="--preview")o.preview=true;
        else if(key=="--capture-calibration") {o.capture_calibration=true;o.preview=true;}
        else if(key=="--headless")o.headless=true;
        else if(key=="--record")o.record=true;
        else if(key=="--doctor")o.doctor=true;
        else throw std::runtime_error("Unknown argument: "+key);
    }
    if(o.seconds<=0 || o.seconds>600 || o.grab_fps<=0 || o.grab_fps>30 || o.camera.exposure_us<=0 ||
       o.confidence<=0 || o.confidence>1 || o.iou<0 || o.iou>1 || o.camera.gain_db < -1)
        throw std::runtime_error("Invalid limits: seconds(0..600], grab-fps(0..30], positive exposure, conf(0..1], iou[0..1]");
    if(o.camera.format!="BayerRG8"&&o.camera.format!="BGR8"&&o.camera.format!="Mono8")
        throw std::runtime_error("Unsupported --pixel-format");
    if(!o.camera.exercise_error.empty() && (!o.preview || !o.video.empty() ||
       (o.camera.exercise_error!="format"&&o.camera.exercise_error!="timeout")))
        throw std::runtime_error("Fault injection requires camera --preview --exercise-error format|timeout");
    if(o.capture_calibration && (o.headless || !o.video.empty() || !o.camera.exercise_error.empty()))
        throw std::runtime_error("--capture-calibration requires a physical camera and a window");
    if(!o.calibration_path.empty() && (o.preview || o.armor_width_mm<=0 || o.armor_height_mm<=0 || o.max_rms_px<=0))
        throw std::runtime_error("--calibration requires detection and positive measured armor dimensions and RMS limit");
    if(o.calibration_path.empty() && (o.armor_width_mm!=0 || o.armor_height_mm!=0))
        throw std::runtime_error("Use measured armor dimensions with --calibration");
    return o;
}

struct Resources {double rss_mb=0,available_mb=0;};
static Resources resources() {
    Resources r;
#ifdef __linux__
    std::ifstream status("/proc/self/status"),memory("/proc/meminfo");std::string line;
    while(std::getline(status,line))if(line.rfind("VmRSS:",0)==0){std::istringstream in(line.substr(6));in>>r.rss_mb;r.rss_mb/=1024;}
    while(std::getline(memory,line))if(line.rfind("MemAvailable:",0)==0){std::istringstream in(line.substr(13));in>>r.available_mb;r.available_mb/=1024;}
    if(r.rss_mb<=0 || r.available_mb<=0)throw std::runtime_error("Cannot read Linux memory counters");
    if(r.rss_mb>3072 || r.available_mb<1024)throw std::runtime_error("Memory guard: RSS>3072 MiB or available<1024 MiB");
#endif
    return r;
}

static cv::Mat preprocess(const cv::Mat& bgr,std::vector<float>& input,armor::Letterbox& p) {
    // 保持原图比例缩放，再补灰边变成 640×640；p 保存缩放和边距，后面用于还原坐标。
    p=armor::letterbox(bgr.cols,bgr.rows);
    cv::Mat small,padded;
    cv::resize(bgr,small,cv::Size(p.resized_width,p.resized_height),0,0,cv::INTER_LINEAR);
    cv::copyMakeBorder(small,padded,p.top,armor::kInput-p.resized_height-p.top,
                       p.left,armor::kInput-p.resized_width-p.left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    input.resize(3*armor::kInput*armor::kInput);
    constexpr int plane=armor::kInput*armor::kInput;
    // OpenCV输入BGR；模型输入RGB、NCHW、FP32、除255。
    for(int y=0;y<armor::kInput;++y) {
        const auto* row=padded.ptr<cv::Vec3b>(y);
        for(int x=0;x<armor::kInput;++x)for(int c=0;c<3;++c)
            input[c*plane+y*armor::kInput+x]=row[x][2-c]/255.f;
    }
    return padded;
}
static void draw(cv::Mat& image,const std::vector<armor::Detection>& detections) {
    // 每个检测结果包含一个框、类别和模型输出的四个关键点；点不是用框四角代替的。
    // 画面里的编号只代表本帧顺序，不代表跨帧追踪同一个目标。
    constexpr int colors[][3]={{0,220,255},{255,180,0},{80,255,80},{255,100,240}};
    for(std::size_t i=0;i<detections.size();++i) {
        const auto& d=detections[i];const auto& c=colors[i%4];cv::Scalar color(c[0],c[1],c[2]);
        cv::rectangle(image,cv::Point(int(d.box.x1),int(d.box.y1)),cv::Point(int(d.box.x2),int(d.box.y2)),color,2);
        std::ostringstream title;title<<'#'<<i+1<<' '<<armor::kNames[d.label]<<' '<<std::fixed<<std::setprecision(2)<<d.score;
        cv::putText(image,title.str(),{int(d.box.x1),std::max(20,int(d.box.y1)-8)},cv::FONT_HERSHEY_SIMPLEX,.6,color,2);
        for(int k=0;k<4;++k) {
            auto p=d.points[k];if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<0||p.y<0||p.x>=image.cols||p.y>=image.rows)continue;
            cv::Point point(int(std::lround(p.x)),int(std::lround(p.y)));
            cv::circle(image,point,6,cv::Scalar(0,0,0),-1);cv::circle(image,point,4,color,-1);
            cv::putText(image,std::to_string(i+1)+":"+std::to_string(k+1),point+cv::Point(7,-5),cv::FONT_HERSHEY_SIMPLEX,.45,color,1);
        }
    }
}
static cv::Mat display_size(const cv::Mat& image) {
    double ratio=std::min({1.0,960.0/image.cols,1080.0/image.rows});
    cv::Size size(std::max(2,int(image.cols*ratio)/2*2),std::max(2,int(image.rows*ratio)/2*2));
    cv::Mat result;cv::resize(image,result,size,0,0,cv::INTER_AREA);return result;
}
static void save(const fs::path& file,const cv::Mat& image) {
    // 编码后使用C++文件流，兼容宿主Windows上的中文路径；Ubuntu也用同一路径。
    std::vector<unsigned char> bytes;
    if(!cv::imencode(".png",image,bytes))throw std::runtime_error("PNG encoding failed");
    std::ofstream stream(file,std::ios::binary);stream.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    stream.close();
    if(!stream)throw std::runtime_error("PNG write failed");
}
static std::string stamp() {
    auto time=std::chrono::system_clock::now();auto t=std::chrono::system_clock::to_time_t(time);
    std::ostringstream out;out<<std::put_time(std::localtime(&t),"%Y%m%d-%H%M%S")<<'-'
       <<std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count()%1000000;return out.str();
}

static armor::ImageGeometry live_geometry(const armor::UsbCamera& camera, cv::Size size) {
    const auto metadata=camera.metadata();
    const auto number=[&](const char* key) {
        const auto item=metadata.find(key);
        if(item==metadata.end())throw std::runtime_error(std::string("Missing MVS geometry: ")+key);
        const auto& text=item->second;
        int result=0;
        const auto end=text.data()+text.size();
        const auto parsed=std::from_chars(text.data(),end,result);
        if(parsed.ec!=std::errc{} || parsed.ptr!=end || result<0)
            throw std::runtime_error(std::string("Cannot verify MVS geometry: ")+key+"="+text);
        return result;
    };
    if(size!=cv::Size(number("Width"),number("Height")))
        throw std::runtime_error("Frame dimensions differ from reported MVS dimensions");
    return {size,metadata.at("serial"),number("OffsetX"),number("OffsetY")};
}

struct PoseFrame {
    const char* result="no_target";
    const armor::Detection* detection=nullptr;
    std::optional<armor::Pose> pose;
};

static PoseFrame locate_target(const std::vector<armor::Detection>& detections, const Options& options,
                               const std::vector<cv::Point3f>& object, const armor::Calibration& calibration) {
    const armor::Detection* selected=nullptr;
    for(const auto& d:detections) {
        if(options.target_label>=0 && d.label!=options.target_label)continue;
        if(selected)return {"multiple_targets",nullptr,std::nullopt};
        selected=&d;
    }
    if(!selected)return {};
    std::array<cv::Point2f,4> pixels;
    for(std::size_t i=0;i<pixels.size();++i) {
        const auto p=selected->points[options.corner_order[i]];
        pixels[i]={p.x,p.y};
    }
    auto pose=armor::solveArmorPose(object,pixels,calibration,options.max_rms_px);
    return {pose?"valid":"invalid_pose",selected,std::move(pose)};
}

static std::string vector_text(const cv::Vec3d& value) {
    std::ostringstream text;
    text<<std::fixed<<std::setprecision(3)<<value[0]<<", "<<value[1]<<", "<<value[2];
    return text.str();
}

static cv::Mat motion_panel(const cv::Mat& image, const PoseFrame& pose, const armor::TrackingUpdate& track,
                            bool live, double horizon) {
    cv::Mat result;
    // 显示图宽高为偶数，状态栏也保持偶数高度，供录像编码器使用。
    cv::copyMakeBorder(image,result,0,114,0,std::max(0,640-image.cols),cv::BORDER_CONSTANT,cv::Scalar(22,22,22));
    const auto line=[&](int row,const std::string& text,const cv::Scalar& color) {
        cv::putText(result,text,{10,image.rows+19+21*row},cv::FONT_HERSHEY_SIMPLEX,.51,color,1,cv::LINE_AA);
    };
    const auto future="Future +"+cv::format("%.2f",horizon)+" s [m]: ";
    line(0,std::string(live?"LIVE MVS":"REPLAY")+" | "+pose.result+" | "+armor::trackingStatusName(track.status),{225,225,225});
    if(pose.pose) {
        const auto& p=*pose.pose;
        line(1,"Raw XYZ [m]: "+vector_text(p.position_m)+"   Distance: "+
             cv::format("%.3f m",cv::norm(p.position_m))+"   RMS: "+cv::format("%.2f px",p.rms_px),{75,205,255});
    } else line(1,"Raw XYZ [m]: --   Distance: --",{75,205,255});
    line(2,"Filtered XYZ [m]: "+(track.motion?vector_text(track.motion->position_m):"--"),{105,240,145});
    line(3,"Velocity XYZ [m/s]: "+(track.motion?vector_text(track.motion->velocity_mps):"--"),{225,225,225});
    line(4,future+(track.motion?vector_text(track.motion->future_m):"--"),{255,200,95});
    return result;
}

static void write_tracking(std::ostream& csv, std::uint64_t frame, double time, double dt, const PoseFrame& pose,
                           const armor::TrackingUpdate& track) {
    const auto vector=[&](const cv::Vec3d& value) {csv<<','<<value[0]<<','<<value[1]<<','<<value[2];};
    csv<<frame<<','<<time<<','<<dt<<','<<pose.result<<','<<armor::trackingStatusName(track.status)<<',';
    if(pose.detection)csv<<armor::kNames[pose.detection->label];
    if(pose.pose) {
        vector(pose.pose->position_m);
        csv<<','<<cv::norm(pose.pose->position_m);
        vector(pose.pose->rotation);
        csv<<','<<pose.pose->rms_px;
    } else csv<<",,,,,,,,";
    if(track.motion) {
        vector(track.motion->position_m);
        vector(track.motion->velocity_mps);
        vector(track.motion->future_m);
    } else csv<<",,,,,,,,,";
    csv<<'\n';
}

int main(int argc,char** argv) {
    try {
        auto options=parse(argc,argv);
        if(options.list)return armor::UsbCamera::list()>0?0:1;
        if(options.doctor) {
            auto r=resources();std::cout<<"language=C++17\nOpenCV="<<CV_VERSION<<"\nrss_mib="<<r.rss_mb
                                       <<"\navailable_mib="<<r.available_mb<<"\n";
#ifdef ARMOR_WITH_MVS
            std::cout<<"MVS=compiled_with_local_headers_and_library\n";
#else
            std::cout<<"MVS=disabled_for_host_checks\n";
#endif
            return 0;
        }
        std::optional<armor::Calibration> calibration;
        std::optional<armor::PositionTracker> tracker;
        std::vector<cv::Point3f> armor_points;
        if(!options.calibration_path.empty()) {
            calibration=armor::loadCalibration(fs::u8path(options.calibration_path));
            if(options.video.empty() && calibration->image.camera_serial.empty())
                throw std::runtime_error("Live measurement requires a calibration made from this camera's capture session");
            armor_points=armor::armorCorners(options.armor_width_mm/1000.0,options.armor_height_mm/1000.0);
            tracker.emplace(options.motion);
        }
        std::signal(SIGINT,request_quit);std::signal(SIGTERM,request_quit);
        cv::setNumThreads(1);resources();
        fs::path session=fs::u8path(options.output)/stamp();
        fs::create_directories(session);
        if(fs::space(session).available<1024ull*1024*1024)throw std::runtime_error("Need >=1 GiB disk free");
        std::cout<<"Output: "<<session.u8string()<<'\n';
        // 相机尚未打开时先检查图形后端，避免Qt原生错误发生在持有相机缓冲期间。
        bool window=false;
        if(!options.headless) {cv::namedWindow("Armor C++",cv::WINDOW_NORMAL);cv::imshow("Armor C++",cv::Mat::zeros(120,320,CV_8UC3));cv::waitKey(1);window=true;}
        armor::UsbCamera camera;cv::VideoCapture replay;
        std::unique_ptr<armor::OnnxModel> model;
        std::unique_ptr<armor::Recorder> recording;
        std::map<std::string,std::string> report;
        report["language"]="C++17";report["opencv_version"]=CV_VERSION;
        report["source"]=options.video.empty()?"physical MVS USB camera":"recorded video replay (not hardware validation)";
        report["light_note"]=options.light_note;report["model"]=options.preview?"none (preview)":options.model;
        report["exercise_error_is_simulated"]=options.camera.exercise_error.empty()?"false":"true";
        report["grab_fps_limit"]=std::to_string(options.grab_fps);
        report["onnx_threads"]="intra=2,inter=1;CPU;sequential";
        if(calibration) {
            report["calibration_file"]=options.calibration_path;
            report["armor_width_mm"]=std::to_string(options.armor_width_mm);
            report["armor_height_mm"]=std::to_string(options.armor_height_mm);
            for(const auto index:options.corner_order)report["corner_order_TL_BL_BR_TR"]+=char('1'+index);
            report["target"]=options.target_label<0?"single detection":armor::kNames[options.target_label];
            report["max_reprojection_px"]=std::to_string(options.max_rms_px);
            report["prediction_seconds"]=std::to_string(options.motion.prediction_seconds);
            report["max_gap_seconds"]=std::to_string(options.motion.max_gap_seconds);
            report["measurement_std_m"]=vector_text(options.motion.measurement_std_m);
            report["acceleration_std_mps2"]=std::to_string(options.motion.acceleration_std_mps2);
            report["tracking_time_source"]=options.video.empty()?"monotonic host receive time":"video frame index / source FPS";
        }
        int frames=0,detections=0,snapshots=0,segments=0,exit_code=0;
        std::size_t pose_frames=0;
        std::uint64_t number=0,source_gaps=0,previous=0;
        double started=0,convert_sum=0,predict_sum=0,peak_rss=0,last_print=-1,last_valid=0;
        double saturated_sum=0,dark_std_sum=0;int samples=0;
        auto cpu_started=std::clock();
        cv::Mat last_result;
        std::optional<armor::ImageGeometry> capture_geometry;
        const auto calibration_frames=session/"calibration-frames";
        auto stop_record=[&]() {
            if(!recording)return;
            recording->stop();
            report["recording."+std::to_string(segments)]=recording->summary();
            const auto error=recording->error();
            if(!error.empty()) {
                std::cerr<<"Recording error: "<<error<<'\n';
                exit_code=1;
            }
            recording.reset();
        };
        auto toggle_record=[&](const cv::Mat& image) {
            if(recording)stop_record();
            else {if(++segments>10)throw std::runtime_error("Max 10 recording segments per run");
                  recording=std::make_unique<armor::Recorder>((session/("recording-"+std::to_string(segments)+".avi")).u8string(),image);}
        };
        try {
            if(!options.preview) {
                // 模型只创建一次，空输入预热一次；--preview 时不加载模型。
                model=std::make_unique<armor::OnnxModel>(options.model);report["onnxruntime_version"]=model->version();
                std::vector<float> blank(3*armor::kInput*armor::kInput,0);model->raw(blank);resources();
            }
            if(quit_requested)throw std::runtime_error("Stop requested before capture");
            double replay_fps=30;
            // 没指定 --video 就打开工业相机；指定时从文件回放，便于不接相机先检查流程。
            if(options.video.empty())camera.open(options.camera);
            else {if(!replay.open(options.video))throw std::runtime_error("Cannot open replay video");
                  double f=replay.get(cv::CAP_PROP_FPS);if(std::isfinite(f)&&f>0&&f<=1000)replay_fps=f;}
            std::ofstream log(session/"frames.csv");
            std::ofstream predictions(session/"detections.csv");
            std::ofstream trajectory;
            if(calibration) {
                trajectory.open(session/"tracking.csv");
                trajectory<<"frame,time_s,dt_s,pose_status,track_status,label,raw_x_m,raw_y_m,raw_z_m,distance_m,"
                          <<"rvec_x_rad,rvec_y_rad,rvec_z_rad,reprojection_rms_px,filtered_x_m,filtered_y_m,filtered_z_m,"
                          <<"vx_mps,vy_mps,vz_mps,future_x_m,future_y_m,future_z_m\n";
                trajectory<<std::setprecision(12);
                if(!trajectory)throw std::runtime_error("Cannot write tracking.csv");
            }
            log<<"frame,host_received_seconds,device_ticks_unsynchronized,elapsed_seconds,convert_copy_ms,predict_ms,host_received_to_result_ms,detections,rss_mib\n";
            predictions<<"frame,label,confidence,x1,y1,x2,y2,k1x,k1y,k2x,k2y,k3x,k3y,k4x,k4y\n";
            log<<std::setprecision(12);predictions<<std::setprecision(9);
            started=last_valid=now();cpu_started=std::clock();
            double first_source_time=0,previous_source_time=0;
            std::vector<float> input;
            while(!quit_requested && now()-started<options.seconds && (!options.max_frames||frames<options.max_frames)) {
                // 每轮处理一帧：取图 → 预处理 → 推理 → 画结果 → 显示/写记录。
                double tick=now();auto memory=resources();peak_rss=std::max(peak_rss,memory.rss_mb);
                cv::Mat raw;armor::FrameInfo meta;
                if(options.video.empty()) {
                    if(!camera.read(raw,meta)) {
                        if(window) {
                            int key=cv::waitKey(1)&255;
                            if(key=='q'||key=='Q'||key==27)quit_requested=1;
                        }
                        if(now()-last_valid>=5)throw std::runtime_error("Five seconds without a valid frame");
                        continue;
                    }
                } else {
                    // 文件回放按时间跳过旧帧；完整逐帧离线处理仍保留在上次项目。
                    std::uint64_t target=static_cast<std::uint64_t>((now()-started)*replay_fps);
                    while(number<target && replay.grab())++number;
                    if(!replay.read(raw))break;
                    ++number;meta.number=number;meta.received_seconds=now();
                }
                last_valid=now();armor::checked_bgr_bytes(raw.cols,raw.rows);
                if(raw.type()!=CV_8UC3)throw std::runtime_error("Expected owned BGR8 image");
                if(calibration && raw.size()!=calibration->image.size)
                    throw std::runtime_error("Original frame resolution differs from calibration; recalibrate this capture mode");
                if(options.capture_calibration && capture_geometry && raw.size()!=capture_geometry->size)
                    throw std::runtime_error("Resolution changed while capturing calibration frames");
                if(frames==0 && options.video.empty()) {
                    if(calibration)armor::requireSameGeometry(calibration->image,live_geometry(camera,raw.size()));
                    if(options.capture_calibration) {
                        capture_geometry=live_geometry(camera,raw.size());
                        fs::create_directory(calibration_frames);
                        cv::FileStorage yaml(".yml",cv::FileStorage::WRITE|cv::FileStorage::MEMORY);
                        armor::writeImageGeometry(yaml,*capture_geometry);
                        armor::writeNewYaml(calibration_frames/"capture.yml",yaml);
                        std::cout<<"Press S for original chessboard frames: "<<calibration_frames.u8string()<<'\n';
                    }
                }
                if(previous && meta.number>previous)source_gaps+=meta.number-previous-1;
                previous=meta.number;
                if(frames==0)save(session/"first-unannotated-bgr.png",raw);
                std::vector<armor::Detection> result;double predict_ms=0;
                // preview 下 model 为空，仅展示原图；否则转换输入并执行模型。
                if(model) {auto begin=now();armor::Letterbox p;preprocess(raw,input,p);
                           result=model->predict(input,p,options.confidence,options.iou);predict_ms=(now()-begin)*1000;}
                auto annotated=raw.clone();draw(annotated,result);
                PoseFrame pose;
                armor::TrackingUpdate track;
                if(calibration) {
                    pose=locate_target(result,options,armor_points,*calibration);
                    std::optional<cv::Vec3d> measurement;
                    if(pose.pose) {
                        ++pose_frames;
                        const auto& p=*pose.pose;
                        measurement=p.position_m;
                        const float axis_length=static_cast<float>(std::min(options.armor_width_mm,options.armor_height_mm)/2000.0);
                        cv::drawFrameAxes(annotated,calibration->camera_matrix,calibration->distortion,
                                          p.rotation,p.position_m,axis_length,2);
                    }
                    const double source_time=options.video.empty()?meta.received_seconds:(meta.number-1)/replay_fps;
                    if(frames==0)first_source_time=previous_source_time=source_time;
                    track=tracker->step(source_time,measurement);
                    write_tracking(trajectory,meta.number,source_time-first_source_time,source_time-previous_source_time,pose,track);
                    previous_source_time=source_time;
                }
                ++frames;detections+=static_cast<int>(result.size());
                convert_sum+=meta.convert_ms;predict_sum+=predict_ms;
                memory=resources();peak_rss=std::max(peak_rss,memory.rss_mb);
                double elapsed=now()-started,age_ms=(now()-meta.received_seconds)*1000;
                log<<meta.number<<','<<meta.received_seconds<<','<<meta.device_ticks<<','<<elapsed<<','<<meta.convert_ms<<','<<predict_ms<<','<<age_ms<<','<<result.size()<<','<<memory.rss_mb<<'\n';
                for(const auto& d:result) {predictions<<meta.number<<','<<armor::kNames[d.label]<<','<<d.score<<','<<d.box.x1<<','<<d.box.y1<<','<<d.box.x2<<','<<d.box.y2;
                    for(const auto& point:d.points)predictions<<','<<point.x<<','<<point.y;
                    predictions<<'\n';}
                if(elapsed-last_print>=1) {
                    if(fs::space(session).available<1024ull*1024*1024)throw std::runtime_error("Less than 1 GiB disk free");
                    std::uint64_t saturated=0;
                    for(int y=0;y<raw.rows;++y){auto row=raw.ptr<cv::Vec3b>(y);for(int x=0;x<raw.cols;++x)if(row[x][0]==255||row[x][1]==255||row[x][2]==255)++saturated;}
                    cv::Mat gray;cv::cvtColor(raw,gray,cv::COLOR_BGR2GRAY);cv::Scalar mean,stddev;
                    cv::meanStdDev(gray(cv::Rect(0,0,std::max(1,gray.cols/4),std::max(1,gray.rows/4))),mean,stddev);
                    saturated_sum+=double(saturated)/raw.total();dark_std_sum+=stddev[0];++samples;
                    double cpu=100.0*double(std::clock()-cpu_started)/CLOCKS_PER_SEC/std::max(elapsed,1e-6);
                    std::cout<<"frames="<<frames<<" fps="<<frames/std::max(elapsed,1e-6)<<" predict_ms="<<predict_ms
                             <<" convert_ms="<<meta.convert_ms<<" rss_mib="<<memory.rss_mb<<" process_cpu_percent="<<cpu<<'\n';
                    log.flush();predictions.flush();last_print=elapsed;
                    if(calibration) {
                        trajectory.flush();
                        if(!trajectory)throw std::runtime_error("tracking.csv write failed");
                        std::cout<<"  "<<armor::trackingStatusName(track.status)<<" raw_xyz_m="
                                 <<(pose.pose?vector_text(pose.pose->position_m):"--")<<" distance_m="
                                 <<(pose.pose?cv::format("%.3f",cv::norm(pose.pose->position_m)):"--")<<'\n';
                    }
                }
                last_result=display_size(annotated);
                std::ostringstream title;title<<(model?"C++ YOLO":"C++ PREVIEW")<<" FPS "<<std::fixed<<std::setprecision(1)
                                            <<frames/std::max(elapsed,1e-6)<<"  RSS "<<memory.rss_mb<<" MiB";
                cv::putText(last_result,title.str(),{8,24},cv::FONT_HERSHEY_SIMPLEX,.55,{0,240,255},1);
                cv::putText(last_result,options.capture_calibration?"Q/Esc quit | S save original chessboard frame":
                            "Q/Esc quit | S snapshot | R record",{8,46},cv::FONT_HERSHEY_SIMPLEX,.5,{0,240,255},1);
                if(calibration)last_result=motion_panel(last_result,pose,track,options.video.empty(),options.motion.prediction_seconds);
                if(options.record && frames==1)toggle_record(last_result);
                if(recording) {
                    if(recording->error().empty())recording->update(last_result);
                    else stop_record();
                }
                // Q/Esc 退出，S 截图，R 开始或停止录像。
                int key=-1;
                if(window){cv::imshow("Armor C++",last_result);key=cv::waitKey(1)&255;
                    // -1 can mean an unsupported property. Only 0 confirms invisibility.
                    if(cv::getWindowProperty("Armor C++",cv::WND_PROP_VISIBLE)==0)break;}
                if(key=='q'||key=='Q'||key==27)break;
                if((key=='s'||key=='S') && snapshots<100) {
                    ++snapshots;
                    if(options.capture_calibration) {
                        std::ostringstream name;
                        name<<"chessboard-"<<std::setfill('0')<<std::setw(3)<<snapshots<<".png";
                        save(calibration_frames/name.str(),raw);
                        std::cout<<"Saved original "<<name.str()<<'\n';
                    } else save(session/("snapshot-"+std::to_string(snapshots)+".png"),last_result);
                }
                if(key=='r'||key=='R')toggle_record(last_result);
                std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0,1/options.grab_fps-(now()-tick))));
            }
            if(calibration) {
                trajectory.close();
                if(!trajectory)throw std::runtime_error("tracking.csv write failed");
            }
            if(!log || !predictions)throw std::runtime_error("CSV write failed");
        } catch(const std::exception& error){report["error"]=error.what();std::cerr<<error.what()<<'\n';exit_code=1;}
        // 9.先结束录像，再关闭相机。每帧借用在camera.read内部早已归还。
        stop_record();camera.close();replay.release();
        for(const auto& item:camera.metadata())report["camera."+item.first]=item.second;
        if(!camera.cleanup_ok())exit_code=1;
        if(window)cv::destroyAllWindows();
        if(!last_result.empty())save(session/"last-result.png",last_result);
        double elapsed=started?now()-started:0;
        report["processed_frames"]=std::to_string(frames);report["frame_detections"]=std::to_string(detections);
        if(calibration)report["valid_pose_frames"]=std::to_string(pose_frames);
        if(options.capture_calibration)report["calibration_snapshots"]=std::to_string(snapshots);
        report["elapsed_seconds"]=std::to_string(elapsed);report["processing_fps"]=std::to_string(elapsed?frames/elapsed:0);
        report["mean_predict_ms"]=std::to_string(frames?predict_sum/frames:0);
        report["mean_convert_copy_ms"]=std::to_string(frames?convert_sum/frames:0);
        report["peak_sampled_rss_mib"]=std::to_string(peak_rss);
        report["source_frame_number_gaps"]=std::to_string(source_gaps);
        report["sampled_saturated_fraction"]=std::to_string(samples?saturated_sum/samples:0);
        report["sampled_top_left_roi_std"]=std::to_string(samples?dark_std_sum/samples:0);
        report["cpu_percent_one_core_100"]=std::to_string(elapsed?100.0*double(std::clock()-cpu_started)/CLOCKS_PER_SEC/elapsed:0);
        report["stop_requested"]=quit_requested?"true":"false";
        report["exit_code"]=std::to_string(exit_code || frames==0?1:0);
        std::ifstream os_release("/etc/os-release");std::string os_line;
        while(std::getline(os_release,os_line))if(os_line.rfind("PRETTY_NAME=",0)==0)report["ubuntu"]=os_line.substr(12);
        std::ofstream file(session/"report.txt");
        for(auto& item:report){std::replace(item.second.begin(),item.second.end(),'\n',' ');std::replace(item.second.begin(),item.second.end(),'\r',' ');file<<item.first<<'='<<item.second<<'\n';}
        if(!file)throw std::runtime_error("Report write failed");
        std::cout<<"Finished; report: "<<(session/"report.txt").u8string()<<'\n';
        return exit_code || frames==0?1:0;
    } catch(const std::exception& error){std::cerr<<"Startup/cleanup error: "<<error.what()<<'\n';return 1;}
}
