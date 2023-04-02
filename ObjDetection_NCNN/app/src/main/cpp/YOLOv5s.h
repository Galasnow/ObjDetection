//
// YOLOv5s v6.1 support
// Original model from https://github.com/ultralytics/yolov5/releases
// Model converted through pnnx (https://github.com/pnnx/pnnx), (https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
// Reference https://github.com/Tencent/ncnn/blob/master/examples/yolov5_pnnx.cpp
// Reference https://zhuanlan.zhihu.com/p/471357671
//

#ifndef YOLOv5s_H
#define YOLOv5s_H

#include "net.h"

namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

typedef struct BoxInfo {
    float x1;
    float y1;
    float w;
    float h;
    float score;
    int label;
} BoxInfo;

class YOLOv5s {
public:
    YOLOv5s(AAssetManager *mgr, const char *param, const char *bin, bool useGPU, int threads_number);

    ~YOLOv5s();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float threshold, float nms_threshold);
//    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//                                    "hair drier", "toothbrush"};
private:
//    static std::vector<BoxInfo>
//    decode_infer(ncnn::Mat &data, int stride, const ncnn::Mat& in_pad, //const yolocv::YoloSize &frame_size,
//                 //int net_size, int num_classes,
//                 const std::vector<yolocv::YoloSize> &anchors, float threshold);

//    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    ncnn::Net *Net;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
//    int input_size = 640;
//    int num_class = 80;
    std::vector<YoloLayerData> layers{
            {"out0",    8,  {{10,  13}, {16,  30},  {33,  23}}},
            {"out1",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"out2",    32, {{116, 90}, {156, 198}, {373, 326}}},

    };

public:
    static YOLOv5s *detector;
    static bool hasGPU;
    static bool toUseGPU;
};


#endif //YOLOv5s_H
