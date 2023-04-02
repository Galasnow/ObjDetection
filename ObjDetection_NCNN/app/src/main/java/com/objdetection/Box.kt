package com.objdetection

import android.graphics.Color
import android.graphics.RectF
import java.util.*

class Box(
    var x0: Float,
    var y0: Float,
    var w: Float,
    var h: Float,
    private val label: Int,
    private val score: Float
) {
    fun getRect(): RectF {
        return RectF(x0, y0, x0+w, y0+h)
    }

    fun getLabel(): String {
        return labels[label]
    }

    fun getScore(): Float {
        return score
    }

    fun getColor(): Int {
        val random = Random(label.toLong())
        return Color.argb(255, random.nextInt(256), random.nextInt(256), random.nextInt(256))
    }

    companion object {
        private val labels = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane","bus","train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        )
    }
}