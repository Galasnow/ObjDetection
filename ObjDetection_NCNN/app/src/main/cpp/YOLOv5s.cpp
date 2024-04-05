//
// YOLOv5s v6.1 support
// Original model from https://github.com/ultralytics/yolov5/releases
// Model converted through pnnx (https://github.com/pnnx/pnnx), (https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
// Reference https://github.com/Tencent/ncnn/blob/master/examples/yolov5_pnnx.cpp
// Reference https://zhuanlan.zhihu.com/p/471357671
//

#include "YOLOv5s.h"
#include "cpu.h"

bool YOLOv5s::hasGPU = true;
bool YOLOv5s::toUseGPU = false;
YOLOv5s *YOLOv5s::detector = nullptr;

YOLOv5s::YOLOv5s(AAssetManager *mgr, const char *param, const char *bin, bool useGPU, int threads_number) {

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;

    Net = new ncnn::Net();
    // opt 需要在加载前设置
    if (toUseGPU) {
        // enable vulkan compute
        this->Net->opt.use_vulkan_compute = true;
        // turn on for adreno
        this->Net->opt.use_image_storage = true;
        this->Net->opt.use_tensor_storage = true;
    }
    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    this->Net->opt.use_bf16_storage = true;

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(threads_number);
    if(threads_number)
        this->Net->opt.num_threads = threads_number;
//  this number is automatically set to the number of all big cores (details in NCNN option.h).
//  However, for some SOC with 3 different architectures
//  (e.g. Snapdragon 8 Gen 1, Kryo 1*Cortex-X2 @3.0 GHz + 3*Cortex-A710 @2.5GHz + 4*Cortex-A510 @1.8GHz),
//  and some small models such as NanoDet-Plus,
//  it may be much better to set this number to the number of super large cores,
//  for Snapdragon 8 Gen 1, the best number is 1.

    this->Net->opt.blob_allocator = &blob_pool_allocator;
    this->Net->opt.workspace_allocator = &workspace_pool_allocator;

    if(this->Net->load_param(mgr, param))
        exit(-1);
    if(this->Net->load_model(mgr, bin))
        exit(-1);
}

YOLOv5s::~YOLOv5s() {
    Net->clear();
    delete Net;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + expf(-x)));
}

static inline float intersection_area(const BoxInfo& a, const BoxInfo& b)
{
    if (a.x1 > b.x1 + b.w || a.x1 + a.w < b.x1 || a.y1 > b.y1 + b.h || a.y1 + a.h < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x1 + a.w, b.x1 + b.w) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y1 + a.h, b.y1 + b.h) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<BoxInfo>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j)
    {
        while (faceobjects[i].score > p)
            i++;

        while (faceobjects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<BoxInfo>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<BoxInfo>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const BoxInfo& a = faceobjects[i];

        int keep = 1;
        for (int j : picked)
        {
            const BoxInfo& b = faceobjects[j];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<BoxInfo>& objects)
{
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    BoxInfo obj;
                    obj.x1 = x0;
                    obj.y1 = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.score = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

std::vector<BoxInfo> YOLOv5s::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);

    const int target_size = 640;

    int img_w = img_size.width;
    int img_h = img_size.height;

    // yolov5/models/common.py DetectMultiBackend
    const int max_stride = 64;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = int(h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = int(w * scale);
    }

    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2RGB, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in_net, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_pad.substract_mean_normalize(mean, norm);

    auto ex = Net->create_extractor();

//  this number is automatically set to the number of all big cores (details in NCNN option.h).
//  However, for some SOC with 3 different architectures
//  (e.g. Snapdragon 8 Gen 1, Kryo 1*Cortex-X2 @3.0 GHz + 3*Cortex-A710 @2.5GHz + 4*Cortex-A510 @1.8GHz),
//  and some small models such as NanoDet-Plus,
//  it may be much better to set this number to the number of super large cores,
//  for Snapdragon 8 Gen 1, the best number is 1.

    ex.input("in0", in_pad);

    std::vector<BoxInfo> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat out0;
        ex.extract("out0", out0);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<BoxInfo> objects8;
        generate_proposals(anchors, 8, in_pad, out0, threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out1;
        ex.extract("out1", out1);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<BoxInfo> objects16;
        generate_proposals(anchors, 16, in_pad, out1, threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out2;
        ex.extract("out2", out2);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<BoxInfo> objects32;
        generate_proposals(anchors, 32, in_pad, out2, threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    std::vector<BoxInfo> result;
    // apply nms with nms_threshold
    std::vector<int> picked;

    //ignore the label when NMS
    bool agnostic = false;

    nms_sorted_bboxes(proposals, picked, nms_threshold, agnostic);

    int count = picked.size();

    result.resize(count);
    for (int i = 0; i < count; i++)
    {
        result[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (result[i].x1 - (wpad / 2)) / scale;
        float y0 = (result[i].y1 - (hpad / 2)) / scale;
        float x1 = (result[i].x1 + result[i].w - (wpad / 2)) / scale;
        float y1 = (result[i].y1 + result[i].h - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        result[i].x1 = x0;
        result[i].y1 = y0;
        result[i].w = x1 - x0;
        result[i].h = y1 - y0;
    }

    return result;
}
