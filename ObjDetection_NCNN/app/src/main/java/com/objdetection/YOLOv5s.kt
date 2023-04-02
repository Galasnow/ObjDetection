package com.objdetection

import android.content.res.AssetManager
import android.graphics.Bitmap

object YOLOv5s {
    external fun init(manager: AssetManager?, useGPU: Boolean, threadsNumber: Int)
    external fun detect(bitmap: Bitmap?, threshold: Float, nms_threshold: Float): Array<Box>?

    init {
        System.loadLibrary("objdetection")
    }
}
