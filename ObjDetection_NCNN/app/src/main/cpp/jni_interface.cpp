#include <jni.h>
#include <string>
#include <ncnn/gpu.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "NanoDetPlus.h"
#include "YOLOv5s.h"



JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() > 0) {
        NanoDetPlus::hasGPU = true;
        YOLOv5s::hasGPU = true;
    }
//    LOGD("jni onload");
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    ncnn::destroy_gpu_instance();
    delete NanoDetPlus::detector;
    delete YOLOv5s::detector;
//    LOGD("jni onunload");
}

/*********************************************************************************************
                                         NanoDet-Plus
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_objdetection_NanoDetPlus_init(JNIEnv *env, jobject thiz, jobject assetManager, jboolean useGPU, jint threads_number) {
    if (NanoDetPlus::detector != nullptr) {
        delete NanoDetPlus::detector;
        NanoDetPlus::detector = nullptr;
    }
    if (NanoDetPlus::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        NanoDetPlus::detector = new NanoDetPlus(mgr, "NanoDetPlus.param", "NanoDetPlus.bin", useGPU, threads_number);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_objdetection_NanoDetPlus_detect(JNIEnv *env, jobject thiz, jobject image, jfloat threshold,
                                         jfloat nms_threshold) {
    auto result = NanoDetPlus::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/objdetection/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.w, box.h, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}


/*********************************************************************************************
                                         YOLOv5s
 ********************************************************************************************/
extern "C" JNIEXPORT void JNICALL
Java_com_objdetection_YOLOv5s_init(JNIEnv *env, jobject thiz, jobject assetManager, jboolean useGPU, jint threads_number) {
    if (YOLOv5s::detector != nullptr) {
        delete YOLOv5s::detector;
        YOLOv5s::detector = nullptr;
    }
    if (YOLOv5s::detector == nullptr) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        YOLOv5s::detector = new YOLOv5s(mgr, "YOLOv5s.param", "YOLOv5s.bin", useGPU, threads_number);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_objdetection_YOLOv5s_detect(JNIEnv *env, jobject thiz, jobject image, jfloat threshold,
                                     jfloat nms_threshold) {
    auto result = YOLOv5s::detector->detect(env, image, threshold, nms_threshold);

    auto box_cls = env->FindClass("com/objdetection/Box");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.w, box.h, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}
