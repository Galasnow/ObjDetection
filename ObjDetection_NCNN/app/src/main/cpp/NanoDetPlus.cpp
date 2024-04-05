// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu.h"
#include "YOLOv5s.h"
#include "NanoDetPlus.h"

//#include <cstdlib>
//#include <cfloat>
//#include <cstdio>
//#include <vector>
//#include <omp.h>


bool NanoDetPlus::hasGPU = true;
bool NanoDetPlus::toUseGPU = false;
NanoDetPlus* NanoDetPlus::detector = nullptr;

NanoDetPlus::NanoDetPlus(AAssetManager *mgr, const char *param, const char *bin, bool useGPU, int threads_number) {

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    this->Net = new ncnn::Net();
    hasGPU = ncnn::get_gpu_count() > 0;
    toUseGPU = hasGPU && useGPU;
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

NanoDetPlus::~NanoDetPlus()
{
    delete this->Net;
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
        areas[i] = faceobjects[i].w*faceobjects[i].h;
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

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<BoxInfo>& objects)
{
    const int num_grid = pred.h;

    int num_grid_x = pred.w;
    int num_grid_y = pred.h;

    const int num_class = 80; // number of classes. 80 for COCO
    const int reg_max_1 = (pred.c - num_class) / 4;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float s = pred.channel(k).row(i)[j];
                if (s > score)
                {
                    label = k;
                    score = s;
                }
            }

            score = sigmoid(score);

            if (score >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4);
                for (int k = 0; k < reg_max_1 * 4; k++)
                {
                    bbox_pred[k] = pred.channel(num_class + k).row(i)[j];
                }
                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = j * stride;
                float pb_cy = i * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                BoxInfo obj;
                obj.x1 = x0;
                obj.y1 = y0;
                obj.w = x1 - x0;
                obj.h = y1 - y0;
                obj.label = label;
                obj.score = score;

                objects.push_back(obj);
            }
        }
    }
}

std::vector<BoxInfo> NanoDetPlus::detect(JNIEnv *env, jobject image, float score_threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);

    // pad to multiple of max_stride
    int w = img_size.width;
    int h = img_size.height;

    const int max_stride = 64;

    float scale = 1.f;
    if (w > h) {
        scale = (float) 320 / w;
        w = 320;
        h = h * scale;
    } else {
        scale = (float) 320 / h;
        h = 320;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGBA2BGR, w,h);

    // pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = this->Net->create_extractor();
    ex.input("in0", in_pad);

    std::vector<BoxInfo> proposals;

    // stride 8
    {
        ncnn::Mat pred;
        ex.extract("231", pred);

        std::vector<BoxInfo> objects8;
        generate_proposals(pred, 8, in_pad, score_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat pred;
        ex.extract("228", pred);

        std::vector<BoxInfo> objects16;
        generate_proposals(pred, 16, in_pad, score_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat pred;
        ex.extract("225", pred);

        std::vector<BoxInfo> objects32;
        generate_proposals(pred, 32, in_pad, score_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // stride 64
    {
        ncnn::Mat pred;
        ex.extract("222", pred);

        std::vector<BoxInfo> objects64;
        generate_proposals(pred, 64, in_pad, score_threshold, objects64);

        proposals.insert(proposals.end(), objects64.begin(), objects64.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    std::vector<BoxInfo> results;
    results.resize(count);

    for (int i = 0; i < count; i++) {
        results[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (results[i].x1 - (wpad / 2)) / scale;
        float y0 = (results[i].y1 - (hpad / 2)) / scale;
        float x1 = (results[i].x1 + results[i].w - (wpad / 2)) / scale;
        float y1 = (results[i].y1 + results[i].h - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_size.width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_size.height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_size.width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_size.height - 1)), 0.f);

        results[i].x1 = x0;
        results[i].y1 = y0;
        results[i].w = x1 - x0;
        results[i].h = y1 - y0;
    }

    return results;
}