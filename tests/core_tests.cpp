#include "armor/core.hpp"
#include <iostream>
#include <functional>
#include <limits>

int main() {
    int checks=0;
    const auto expect=[&](bool ok,const char* name) { if(!ok) throw std::runtime_error(name); ++checks; };
    try {
        auto p=armor::letterbox(1280,720);
        expect(p.resized_width==640 && p.resized_height==360 && p.left==0 && p.top==140,"letterbox landscape");
        auto q=armor::letterbox(1440,1080);
        expect(q.resized_height==480 && q.top==80,"letterbox camera");
        auto r=armor::letterbox(720,1280);
        expect(r.left==140 && r.top==0,"letterbox portrait");
        expect(armor::checked_bgr_bytes(1440,1080)==4665600,"bounded frame size");
        bool rejected=false;
        try { armor::checked_bgr_bytes(16384,16384); } catch(...) { rejected=true; }
        expect(rejected,"reject oversized frame");
        int returned=0;
        try { armor::FrameLease guard([&]{++returned;}); throw std::runtime_error("conversion failure"); } catch(...) {}
        expect(returned==1,"exception returns borrowed frame");
        { armor::FrameLease guard([&]{++returned;}); guard.release(); guard.release(); }
        expect(returned==2,"release exactly once");
        std::vector<float> tensor(armor::kChannels*3,0.f);
        const auto set=[&](int c,int i,float v){tensor[c*3+i]=v;};
        for(int i=0;i<3;++i) {
            set(0,i,320);set(1,i,320);set(2,i,100);set(3,i,80);
            set(i==2?7:6,i,i==0?.9f:.8f); // 前两条B3重叠，第三条R1不得被跨类抑制。
            for(int k=0;k<4;++k) {set(10+2*k,i,300+k*5);set(11+2*k,i,300+k*6);}
        }
        auto detections=armor::decode(tensor.data(),3,p);
        expect(detections.size()==2,"class-aware NMS");
        expect(detections[0].label==2,"B3 class mapping");
        expect(std::abs(detections[0].box.x1-540)<.001f,"box restored to source pixels");
        expect(std::abs(detections[0].points[0].x-600)<.001f &&
               std::abs(detections[0].points[0].y-320)<.001f,"four model points restored");
        expect(detections[0].points[0].x!=detections[0].box.x1,"points are not box corners");
        set(6,0,std::numeric_limits<float>::quiet_NaN());
        set(10,2,-10);
        auto finite=armor::decode(tensor.data(),3,p);
        expect(finite.size()==2,"nonfinite score ignored");
        expect(finite[1].points[0].x<0,"off-screen keypoint remains invalid for pose");
        expect(armor::overlap({0,0,10,10},{20,20,30,30})==0,"nonoverlapping IoU");
        std::cout<<"PASS "<<checks<<" C++ checks\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
