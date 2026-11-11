// 【相机采集】SDK 初始化 → 找设备 → 打开并设置 → 借一帧 → 复制/转换 → 归还 → 关闭。
// SDK 借出的内存不是程序永久拥有的，必须先复制出独立图像，再归还给 SDK。
#include "armor/camera.hpp"
#include "armor/core.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#ifdef ARMOR_WITH_MVS
#include <MvCameraControl.h>
#endif

namespace armor {
#ifdef ARMOR_WITH_MVS
static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::string hexcode(int value) {std::ostringstream s;s<<"0x"<<std::hex<<static_cast<unsigned>(value);return s.str();}
static void check(int result,const char* operation) {
    if(result!=MV_OK) throw std::runtime_error(std::string(operation)+": "+hexcode(result));
}
template<std::size_t N> static std::string text(const unsigned char(&data)[N]) {
    const auto* start=reinterpret_cast<const char*>(data);
    return {start,std::find(start,start+N,'\0')};
}
#endif
struct UsbCamera::Impl {
    void* handle=nullptr;
    bool initialized=false,opened=false,grabbing=false,clean=true;
    std::uint64_t borrowed=0,returned=0;
    CameraOptions options;
    cv::Mat converted;
    std::map<std::string,std::string> data;
    ~Impl(){close();}
#ifdef ARMOR_WITH_MVS
    double set_float(const char* node,double requested,bool minimum=false) {
        // 先询问相机支持范围，再设置并回读实际值；设备不支持的曝光/增益不能强行写入。
        MVCC_FLOATVALUE value{};
        check(MV_CC_GetFloatValue(handle,node,&value),node);
        float target=minimum?value.fMin:static_cast<float>(requested);
        if(!std::isfinite(target) || target<value.fMin || target>value.fMax)
            throw std::runtime_error(std::string(node)+" outside camera range "+std::to_string(value.fMin)+".."+std::to_string(value.fMax));
        check(MV_CC_SetFloatValue(handle,node,target),node);
        check(MV_CC_GetFloatValue(handle,node,&value),node);
        return value.fCurValue;
    }
    void optional_nodes() {
        for(const char* name:{"Width","Height","OffsetX","OffsetY"}) {
            MVCC_INTVALUE_EX value{};
            int ret=MV_CC_GetIntValueEx(handle,name,&value);
            data[name]=ret==MV_OK?std::to_string(value.nCurValue):"unavailable:"+hexcode(ret);
        }
        for(const char* name:{"Gamma","AcquisitionFrameRate"}) {
            MVCC_FLOATVALUE value{};
            int ret=MV_CC_GetFloatValue(handle,name,&value);
            data[name]=ret==MV_OK?std::to_string(value.fCurValue):"unavailable:"+hexcode(ret);
        }
        MVCC_STRINGVALUE value{};
        int ret=MV_CC_GetStringValue(handle,"DeviceVersion",&value);
        const char* version=value.chCurValue;
        data["firmware"]=ret==MV_OK?std::string(version,std::find(version,version+sizeof(value.chCurValue),'\0'))
                                  :"unavailable:"+hexcode(ret);
        MVCC_ENUMVALUE wb{};
        ret=MV_CC_GetEnumValue(handle,"BalanceWhiteAuto",&wb);
        data["BalanceWhiteAuto"]=ret==MV_OK?std::to_string(wb.nCurValue):"unavailable:"+hexcode(ret);
    }
#endif
    void close() noexcept {
#ifdef ARMOR_WITH_MVS
        auto finish=[&](const char* name,int code) {
            data[std::string("cleanup.")+name]=code==MV_OK?"OK":hexcode(code);
            std::cerr<<name<<": "<<data[std::string("cleanup.")+name]<<'\n';
            if(code!=MV_OK) clean=false;
        };
        // read()内的FrameLease已归还借出帧，再执行设备清理。只清理已经取得的资源。
        if(grabbing){finish("StopGrabbing",MV_CC_StopGrabbing(handle));grabbing=false;}
        if(opened){finish("CloseDevice",MV_CC_CloseDevice(handle));opened=false;}
        if(handle){finish("DestroyHandle",MV_CC_DestroyHandle(handle));handle=nullptr;}
        if(initialized){finish("Finalize",MV_CC_Finalize());initialized=false;}
#endif
        converted.release();
        data["buffers.borrowed"]=std::to_string(borrowed);
        data["buffers.returned"]=std::to_string(returned);
        data["buffers.balanced"]=borrowed==returned?"true":"false";
        if(borrowed!=returned) clean=false;
    }
};
UsbCamera::UsbCamera():impl_(std::make_unique<Impl>()){}
UsbCamera::~UsbCamera()=default;
void UsbCamera::close() noexcept {impl_->close();}
bool UsbCamera::cleanup_ok() const{return impl_->clean;}
std::map<std::string,std::string> UsbCamera::metadata() const{return impl_->data;}

int UsbCamera::list() {
#ifdef ARMOR_WITH_MVS
    check(MV_CC_Initialize(),"Initialize");
    int count=0;
    try {
        MV_CC_DEVICE_INFO_LIST devices{};
        check(MV_CC_EnumDevices(MV_USB_DEVICE,&devices),"EnumDevices(USB)");
        for(unsigned i=0;i<devices.nDeviceNum;++i) {
            const auto* d=devices.pDeviceInfo[i];
            if(!d || d->nTLayerType!=MV_USB_DEVICE)continue;
            const auto& usb=d->SpecialInfo.stUsb3VInfo;
            std::cout<<"model="<<text(usb.chModelName)<<" serial="<<text(usb.chSerialNumber)<<'\n';++count;
        }
    } catch(...) {MV_CC_Finalize();throw;}
    check(MV_CC_Finalize(),"Finalize");
    return count;
#else
    throw std::runtime_error("Built without MVS; rebuild in Ubuntu with -DENABLE_MVS=ON");
#endif
}

void UsbCamera::open(const CameraOptions& options) {
#ifdef ARMOR_WITH_MVS
    auto& s=*impl_;
    if(s.initialized)throw std::runtime_error("Camera already open");
    s.options=options;
    try {
        // 1.初始化 → 2.枚举USB → 3.序列号匹配。多相机时禁止盲选第0台。
        check(MV_CC_Initialize(),"Initialize");s.initialized=true;
        s.data["sdk_version"]=hexcode(MV_CC_GetSDKVersion());
        MV_CC_DEVICE_INFO_LIST devices{};
        check(MV_CC_EnumDevices(MV_USB_DEVICE,&devices),"EnumDevices(USB)");
        MV_CC_DEVICE_INFO* chosen=nullptr;unsigned found=0;
        for(unsigned i=0;i<devices.nDeviceNum;++i) {
            auto* d=devices.pDeviceInfo[i];
            if(!d || d->nTLayerType!=MV_USB_DEVICE)continue;
            ++found;
            if(options.serial.empty() || text(d->SpecialInfo.stUsb3VInfo.chSerialNumber)==options.serial)chosen=d;
        }
        if(!chosen || (options.serial.empty() && found!=1))
            throw std::runtime_error("Specify a matching --serial after --list-cameras; check USB and MVS client ownership");
        s.data["model"]=text(chosen->SpecialInfo.stUsb3VInfo.chModelName);
        s.data["serial"]=text(chosen->SpecialInfo.stUsb3VInfo.chSerialNumber);
        // 4.创建句柄 → 5.打开 → 6.配置并回读。
        check(MV_CC_CreateHandle(&s.handle,chosen),"CreateHandle");
        check(MV_CC_OpenDevice(s.handle,MV_ACCESS_Exclusive,0),"OpenDevice");s.opened=true;
        for(auto setting:{std::pair{"AcquisitionMode","Continuous"},{"TriggerMode","Off"},
                          {"ExposureAuto","Off"},{"ExposureMode","Timed"},{"GainAuto","Off"}})
            check(MV_CC_SetEnumValueByString(s.handle,setting.first,setting.second),setting.first);
        unsigned type=PixelType_Gvsp_BayerRG8;
        if(options.format=="BGR8")type=PixelType_Gvsp_BGR8_Packed;
        else if(options.format=="Mono8")type=PixelType_Gvsp_Mono8;
        else if(options.format!="BayerRG8")throw std::runtime_error("Unsupported pixel format argument");
        check(MV_CC_SetEnumValue(s.handle,"PixelFormat",type),"PixelFormat");
        s.data["pixel_format"]=options.format;
        s.data["exposure_us"]=std::to_string(s.set_float("ExposureTime",options.exposure_us));
        s.data["gain_db"]=std::to_string(s.set_float("Gain",options.gain_db,options.gain_db<0));
        s.optional_nodes();
        MVCC_INTVALUE_EX width{},height{};
        check(MV_CC_GetIntValueEx(s.handle,"Width",&width),"Width");
        check(MV_CC_GetIntValueEx(s.handle,"Height",&height),"Height");
        if(width.nCurValue<=0 || height.nCurValue<=0 || width.nCurValue>16384 || height.nCurValue>16384)
            throw std::runtime_error("Invalid configured ROI");
        checked_bgr_bytes(static_cast<int>(width.nCurValue),static_cast<int>(height.nCurValue));
        check(MV_CC_SetImageNodeNum(s.handle,2),"SetImageNodeNum(2)");
        check(MV_CC_SetGrabStrategy(s.handle,MV_GrabStrategy_LatestImagesOnly),"LatestImagesOnly");
        s.data["sdk_buffer_nodes"]="2";s.data["grab_strategy"]="LatestImagesOnly";
        // 7.开始采集；不保存设备永久配置，不扩张usbfs限额。
        check(MV_CC_StartGrabbing(s.handle),"StartGrabbing");s.grabbing=true;
    } catch(...) {s.close();throw;}
#else
    (void)options;throw std::runtime_error("This build has no MVS support");
#endif
}

bool UsbCamera::read(cv::Mat& owned,FrameInfo& meta) {
    // 输出 owned 是程序独立拥有的 BGR 图像；meta 保存帧号、时间和转换耗时。
#ifdef ARMOR_WITH_MVS
    auto& s=*impl_;
    if(!s.grabbing)throw std::runtime_error("Camera is not grabbing");
    if(s.options.exercise_error=="timeout") {std::this_thread::sleep_for(std::chrono::seconds(1));return false;}
    MV_FRAME_OUT frame{};
    int result=MV_CC_GetImageBuffer(s.handle,&frame,1000);
    if(static_cast<unsigned>(result)==MV_E_NODATA)return false;
    check(result,"GetImageBuffer");++s.borrowed;
    // 8.只要Get成功，后面任何异常都会经过FrameLease归还。
    int free_result=MV_OK;
    FrameLease lease([&]() noexcept {
        free_result=MV_CC_FreeImageBuffer(s.handle,&frame);
        if(free_result==MV_OK)++s.returned;
        else {s.clean=false;std::cerr<<"FreeImageBuffer: "<<hexcode(free_result)<<'\n';}
    });
    meta.received_seconds=now_seconds();
    const auto& info=frame.stFrameInfo;
    int w=info.nWidth,h=info.nHeight;
    checked_bgr_bytes(w,h);
    if(!frame.pBufAddr)throw std::runtime_error("Empty SDK image pointer");
    unsigned type=info.enPixelType;
    if(s.options.exercise_error=="format")type=0xffffffff;
    bool bgr=type==PixelType_Gvsp_BGR8_Packed,mono=type==PixelType_Gvsp_Mono8,bayer=type==PixelType_Gvsp_BayerRG8;
    if(!bgr && !mono && !bayer)throw std::runtime_error("Unexpected pixel format; refusing reinterpretation");
    unsigned requested=s.options.format=="BGR8"?PixelType_Gvsp_BGR8_Packed:
                       s.options.format=="Mono8"?PixelType_Gvsp_Mono8:PixelType_Gvsp_BayerRG8;
    if(type!=requested)throw std::runtime_error("Actual pixel format differs from requested format");
    std::size_t row=std::size_t(w)*(bgr?3:1);
    if(row*std::size_t(h)!=info.nFrameLen)
        throw std::runtime_error("Expected tight frame without padding/Chunk/compression");
    double began=now_seconds();
    // BGR 直接深拷贝；灰度扩成三通道；Bayer 格式由 SDK 转成 BGR 再深拷贝。
    if(bgr)owned=cv::Mat(h,w,CV_8UC3,frame.pBufAddr,row).clone();
    else if(mono)cv::cvtColor(cv::Mat(h,w,CV_8UC1,frame.pBufAddr,row),owned,cv::COLOR_GRAY2BGR);
    else {
        s.converted.create(h,w,CV_8UC3);
        if(!s.converted.isContinuous())throw std::runtime_error("Non-contiguous conversion destination");
        MV_CC_PIXEL_CONVERT_PARAM_EX p{};
        p.nWidth=w;p.nHeight=h;
        p.enSrcPixelType=static_cast<MvGvspPixelType>(type);
        p.enDstPixelType=PixelType_Gvsp_BGR8_Packed;
        p.pSrcData=frame.pBufAddr;p.nSrcDataLen=info.nFrameLen;p.pDstBuffer=s.converted.data;
        p.nDstBufferSize=static_cast<unsigned>(s.converted.total()*s.converted.elemSize());
        check(MV_CC_ConvertPixelTypeEx(s.handle,&p),"ConvertPixelTypeEx");
        if(p.nDstLen!=p.nDstBufferSize)throw std::runtime_error("Invalid converted frame length");
        owned=s.converted.clone(); // 独立持有：下一次转换不会覆盖正在识别/显示的图像。
    }
    meta.convert_ms=(now_seconds()-began)*1000;
    meta.number=info.nFrameNum;
    meta.device_ticks=(std::uint64_t(info.nDevTimeStampHigh)<<32)|info.nDevTimeStampLow;
    lease.release();check(free_result,"FreeImageBuffer");
    // 这里之后owned完全属于OpenCV；没有借出的SDK帧再交给YOLO。
    return true;
#else
    (void)owned;(void)meta;throw std::runtime_error("This build has no MVS support");
#endif
}
}
