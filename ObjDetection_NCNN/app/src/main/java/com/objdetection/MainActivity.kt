package com.objdetection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.preference.PreferenceManager
import com.objdetection.databinding.ActivityMainBinding
import wseemann.media.FFmpegMediaMetadataRetriever
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
//    private val context: Context? = null

    private val NANODET = 1
    private val YOLOV5S = 2

    private var useModel = NANODET
    private var useGPU = false

    var threshold = 0.3f
    var nmsThreshold = 0.7f

    var videoSpeed = 1.0f
    var videoCurFrameLoc: Long = 0
    private val videoSpeedMax = 20 + 1
    private val videoSpeedMin = 1

    private val detectCamera = AtomicBoolean(false)
    private val detectPhoto = AtomicBoolean(false)
    private val detectVideo = AtomicBoolean(false)

    private var errorFlag = 1

    private var targetWidth = 0
    private var targetHeight = 0
    private var width = 0
    private var height = 0

    private var threadsNumber = 0
    private var startTime: Long = 0
    private var endTime: Long = 0
//    private var Time1: Long = 0
//    private var Time2: Long = 0
//    private var Time3: Long = 0

    private var totalFPS = 0.0f
    private var FPSCount = 0

    private var mutableBitmap: Bitmap? = null
//    private var mReusableBitmap: Bitmap? = null
    private var detectService = Executors.newSingleThreadExecutor()
    private lateinit var mCameraProvider: ProcessCameraProvider
    private lateinit var mPreview: Preview
    private lateinit var mImageCapture: ImageCapture
    private lateinit var mImageAnalysis: ImageAnalysis
    private lateinit var mResolutionSelector: ResolutionSelector
    private var cameraExecutor = Executors.newSingleThreadExecutor()
//    private var analyzing = true
//    private var pauseAnalysis = false

    private var mmr: FFmpegMediaMetadataRetriever? = null

//    var reusableBitmaps: MutableSet<SoftReference<Bitmap>>? = null
//    private lateinit var memoryCache: LruCache<String, BitmapDrawable>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // View binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Get the information of using of GPU and model
        val intent = intent
        useGPU = intent.getBooleanExtra("useGPU", false)
        useModel = intent.getIntExtra("useModel", NANODET)

        // Request permissions
        if (Build.VERSION.SDK_INT >= 33) {      // Android 13 support
            val permissionList = arrayOfNulls<String>(3)
            permissionList[0] = Manifest.permission.READ_MEDIA_IMAGES //图片
            permissionList[1] = Manifest.permission.READ_MEDIA_VIDEO //视频
            permissionList[2] = Manifest.permission.CAMERA
            ActivityCompat.requestPermissions(this@MainActivity, permissionList, 0)
        } else {
            val permissionList = arrayOfNulls<String>(2)
            permissionList[0] = Manifest.permission.READ_EXTERNAL_STORAGE
            permissionList[1] = Manifest.permission.CAMERA
            ActivityCompat.requestPermissions(this@MainActivity, permissionList, 0)
        }

        initModel()
        initView()
        initViewListener()

        val prefs: SharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        // Number of threads in CPU Mode
        threadsNumber = prefs.getString("numThreads", "0")?.toInt()!!

        // Replace StartActivityForResult()
        // Select picture
        val photoActivity =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
                if (it.data != null && it.resultCode == Activity.RESULT_OK) {
                    runByPhoto(it.resultCode, it.data)
                } else {
                    Toast.makeText(this, "Error", Toast.LENGTH_SHORT).show()
                }
            }

        binding.btnPhoto.setOnClickListener {
            val permission =
                if (Build.VERSION.SDK_INT >= 33)
                    ActivityCompat.checkSelfPermission(
                    this@MainActivity,
                    Manifest.permission.READ_MEDIA_IMAGES
                )
                else
                    ActivityCompat.checkSelfPermission(
                    this@MainActivity,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                )
            if (permission != PackageManager.PERMISSION_GRANTED) {
                if (Build.VERSION.SDK_INT >= 33)
                    ActivityCompat.requestPermissions(
                        this@MainActivity,
                        arrayOf(Manifest.permission.READ_MEDIA_IMAGES),
                        777
                    )
                else
                    ActivityCompat.requestPermissions(
                        this@MainActivity,
                        arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                        777
                )
            } else {
                val intentPhoto =
                    if (Build.VERSION.SDK_INT >= 33)
                        Intent("android.provider.action.PICK_IMAGES")
                        /**
                        * set the number of selected images (when > 1)
                        * set the max number use " MediaStore.getPickImagesMaxLimit(). "
                        */
                        //intent.putExtra("android.provider.extra.PICK_IMAGES_MAX", MediaStore.getPickImagesMaxLimit());
                    else
                        Intent(Intent.ACTION_PICK)

                intentPhoto.type = "image/*"
                photoActivity.launch(intentPhoto)
            }
        }

        // Select video
        val videoActivity =
            registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
                if (it.data != null && it.resultCode == Activity.RESULT_OK) {
                    runByVideo(it.resultCode, it.data)
                } else {
                    Toast.makeText(this, "Error", Toast.LENGTH_SHORT).show()
                }
            }

        binding.btnVideo.setOnClickListener {
            val permission =
                if (Build.VERSION.SDK_INT >= 33)
                    ActivityCompat.checkSelfPermission(
                        this@MainActivity,
                        Manifest.permission.READ_MEDIA_VIDEO
                    )
                else
                    ActivityCompat.checkSelfPermission(
                        this@MainActivity,
                        Manifest.permission.READ_EXTERNAL_STORAGE
                    )
            if (permission != PackageManager.PERMISSION_GRANTED) {
                if (Build.VERSION.SDK_INT >= 33)
                    ActivityCompat.requestPermissions(
                        this@MainActivity,
                        arrayOf(Manifest.permission.READ_MEDIA_VIDEO),
                        777
                    )
                else
                    ActivityCompat.requestPermissions(
                        this@MainActivity,
                        arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                        777
                    )
            } else {
                val intentVideo =
                    if (Build.VERSION.SDK_INT >= 33)
                        Intent("android.provider.action.PICK_IMAGES")
                    else
                        Intent(Intent.ACTION_PICK)

                intentVideo.type = "video/*"
                videoActivity.launch(intentVideo)
            }
        }
    }

    // Init the model
    private fun initModel() {
        when (useModel) {
            NANODET -> NanoDetPlus.init(assets, useGPU, threadsNumber)
            YOLOV5S -> YOLOv5s.init(assets, useGPU, threadsNumber)
        }
    }
    // Init the interface
    private fun initView() {
        binding.sbVideo.visibility = View.GONE
        binding.sbVideoSpeed.min = videoSpeedMin
        binding.sbVideoSpeed.max = videoSpeedMax
        binding.sbVideoSpeed.visibility = View.GONE
        binding.btnBack.visibility = View.GONE
    }
    // Init the interface response
    private fun initViewListener() {
        binding.toolBar.setNavigationIcon(R.drawable.actionbar_dark_back_icon)
        binding.toolBar.setNavigationOnClickListener { finish() }
        if (useModel == NANODET) {
            threshold = 0.4f
            nmsThreshold = 0.6f
        } else if (useModel == YOLOV5S) {
            threshold = 0.3f
            nmsThreshold = 0.5f
        }
        binding.nmsSeek.progress = (nmsThreshold * 100).toInt()
        binding.thresholdSeek.progress = (threshold * 100).toInt()
        val format = "THR: %.2f, NMS: %.2f"
        binding.valTxtView.text =
            String.format(Locale.ENGLISH, format, threshold, nmsThreshold)

        binding.nmsSeek.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {
                nmsThreshold = (i / 100f)
                binding.valTxtView.text =
                    String.format(Locale.ENGLISH, format, threshold, nmsThreshold)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        binding.thresholdSeek.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {
                threshold = (i / 100f)
                binding.valTxtView.text =
                    String.format(Locale.ENGLISH, format, threshold, nmsThreshold)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        binding.sbVideo.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {
                videoCurFrameLoc = i.toLong()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {
                videoCurFrameLoc = seekBar.progress.toLong()
            }
        })

        binding.sbVideoSpeed.setOnSeekBarChangeListener(object : OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, i: Int, b: Boolean) {
                videoSpeed = i.toFloat()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {
                Toast.makeText(
                    this@MainActivity,
                    "Video Speed:" + seekBar.progress,
                    Toast.LENGTH_SHORT
                ).show()
            }
        })

        if (ActivityCompat.checkSelfPermission(this@MainActivity,Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera(binding.viewFinder)
        }
        else {
            ActivityCompat.requestPermissions(
              this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_PERMISSIONS
            )
        }
    }

    // Companion object
    companion object {
//         Used to load the 'objdetection' library on application startup.
//         init {
//             System.loadLibrary("objdetection")
//          }
        private const val TAG = "ObjDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
    }


    private fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                mCameraProvider = cameraProviderFuture.get()
                bindPreview(mCameraProvider, previewView)
            } catch (e: ExecutionException) {
                e.printStackTrace()
            } catch (e: InterruptedException) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Bind lifecycle object
    private fun bindPreview(
        cameraProvider: ProcessCameraProvider,
        previewView: PreviewView
    ) {
        // Set Texture View for PreviewView mode
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        mPreview = Preview.Builder()
            .build()

        mImageCapture = ImageCapture.Builder()
            .setTargetRotation(previewView.display.rotation)
            .build()

        targetWidth = 640
        targetHeight = 480

        mResolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(ResolutionStrategy(Size(targetWidth, targetHeight), FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build()

        mImageAnalysis = ImageAnalysis.Builder()
            // Enable the following line if RGBA output is needed.
            // .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetRotation(previewView.display.rotation)
            .setResolutionSelector(mResolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        mImageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            // Insert your code here.
            detectOnModel(imageProxy, rotationDegrees)
            // After done, release the ImageProxy object
            imageProxy.close()
        }

        cameraProvider.unbindAll()
        try {
            cameraProvider.bindToLifecycle(
                this, cameraSelector,
                mPreview, mImageCapture,mImageAnalysis
            )
        // Bind the view's surface to preview use case.
        mPreview.setSurfaceProvider(previewView.surfaceProvider)
        } catch (e: java.lang.Exception) {
            Log.e("Camera", "camera provider bind error:", e)
        }
    }

    private fun detectOnModel(image: ImageProxy, rotationDegrees: Int) {
        if (detectCamera.get() || detectPhoto.get() || detectVideo.get()) {
            return
        }
        if (detectService == null) {
            detectCamera.set(false)
            return
        }
        detectCamera.set(true)
        startTime = System.currentTimeMillis()
        val bitmapsrc = imageToBitmap(image) // format conversion

        detectService.execute {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            width = bitmapsrc.width
            height = bitmapsrc.height
            val bitmap = Bitmap.createBitmap(bitmapsrc, 0, 0, width, height, matrix, false)
            detectAndDraw(bitmap)
            if(errorFlag == 1)
                showResultOnUI()
        }

    }

    private fun imageToBitmap(image: ImageProxy): Bitmap {
        val nv21 = imageToNV21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        try {
            out.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun imageToNV21(image: ImageProxy): ByteArray {
        val planes = image.planes
        val y = planes[0]
        val u = planes[1]
        val v = planes[2]
        val yBuffer = y.buffer
        val uBuffer = u.buffer
        val vBuffer = v.buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        // U and V are swapped
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        return nv21
    }

    private fun showResultOnUI() {
        runOnUiThread {
            detectCamera.set(false)
            binding.imageView.setImageBitmap(mutableBitmap)

            endTime = System.currentTimeMillis()
            val dur = endTime - startTime

            if(errorFlag == 1 && dur > 1) {
//                Log.d(TAG, "dur time is: $dur")
                val fps = 1000.0f / dur
                totalFPS += fps
                FPSCount++
                val modelName: String = getModelName()
                binding.tvInfo.text = String.format(
                    Locale.CHINESE,
                    "%s\nSize: %dx%d\nTime: %.3f s\nFPS: %.3f\nAVG_FPS: %.3f",
                    modelName, height, width, dur / 1000.0f, fps, totalFPS / FPSCount
                )
            }
        }
    }

    private fun detectAndDraw(image: Bitmap): Bitmap? {
        var result: Array<Box>? = null
        when (useModel) {
            NANODET -> result = NanoDetPlus.detect(image, threshold, nmsThreshold)
            YOLOV5S -> result = YOLOv5s.detect(image, threshold, nmsThreshold)
        }

        if (result == null) {
            detectCamera.set(false)
            errorFlag = 0
            return image
        }
        else
        {
            mutableBitmap = drawBoxRects(image, result)
        }

        return mutableBitmap
    }

    private fun getModelName(): String {
        var modelName = "NULL"
        when (useModel) {
            NANODET -> modelName = "NanoDet-Plus"
            YOLOV5S -> modelName = "YOLOv5s"
        }
        return if (useGPU) "[ GPU ] $modelName" else "[ CPU ] $modelName"
    }

    private fun drawBoxRects(bitmap: Bitmap, results: Array<Box>?): Bitmap {

        if (results.isNullOrEmpty()) {
            return bitmap
        }
        // Copy，Bitmap cannot be modified directly.
        val bitmapCopy = bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(bitmapCopy)
        val boxPaint = Paint()
        boxPaint.alpha = 200
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 4 * bitmapCopy.width / 800.0f
        boxPaint.textSize = 30 * bitmapCopy.width / 800.0f
        for (box in results) {
            boxPaint.color = box.getColor()
            boxPaint.style = Paint.Style.FILL
            canvas.drawText(
                box.getLabel() + java.lang.String.format(
                    Locale.CHINESE,
                    " %.3f",
                    box.getScore()
                ), box.x0 + 3, box.y0 + 30 * bitmapCopy.width / 1000.0f, boxPaint
            )
            boxPaint.style = Paint.Style.STROKE
            canvas.drawRect(box.getRect(), boxPaint)
        }

        return bitmapCopy
    }

    private fun runByPhoto(resultCode: Int, data: Intent?) {
        if (resultCode != RESULT_OK || data == null) {
            Toast.makeText(this, "Photo error", Toast.LENGTH_SHORT).show()
            return
        }
        if (detectVideo.get()) {
            Toast.makeText(this, "Video is running", Toast.LENGTH_SHORT).show()
            return
        }
        detectPhoto.set(true)
        var image: Bitmap? = getPicture(data.data)
        if (image == null) {
            Toast.makeText(this, "Photo is null", Toast.LENGTH_SHORT).show()
            return
        }

        // Resize too large image
        val MAX_BITMAP_SIZE = 100*1024*1024
        if((image.width * image.height) > MAX_BITMAP_SIZE){
            val ratio = width * height.toFloat() / MAX_BITMAP_SIZE
            val finalWidth = width/ratio.toInt()
            val finalHeight = height/ratio.toInt()
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true)
        }

        mCameraProvider.unbindAll()
        binding.viewFinder.visibility = View.INVISIBLE
        binding.btnBack.visibility = View.VISIBLE
        binding.btnBack.setOnClickListener {
            if (detectVideo.get() || detectPhoto.get()) {
                detectPhoto.set(false)
                detectVideo.set(false)
                binding.sbVideo.visibility = View.GONE
                binding.sbVideoSpeed.visibility = View.GONE
                binding.viewFinder.visibility = View.VISIBLE
                binding.btnBack.visibility = View.GONE
                startCamera(binding.viewFinder)
            }
        }

        val thread = Thread({
            val start = System.currentTimeMillis()
            val imageCopy = image.copy(Bitmap.Config.ARGB_8888, true)
            width = image.width
            height = image.height

            mutableBitmap = detectAndDraw(imageCopy)
            val dur = System.currentTimeMillis() - start
            runOnUiThread {
                val modelName = getModelName()
                binding.imageView.setImageBitmap(mutableBitmap)
                binding.tvInfo.text = String.format(
                    Locale.CHINESE, "%s\nSize: %dx%d\nTime: %.3f s\nFPS: %.3f",
                    modelName, height, width, dur / 1000.0f, 1000.0f / dur
                )
            }
        }, "photo detect")
        thread.start()
    }

    private fun getPicture(selectedImage: Uri?): Bitmap? {
        val filePathColumn = arrayOf(MediaStore.Images.Media.DATA)
        val cursor: Cursor =
            selectedImage?.let { this.contentResolver.query(it, filePathColumn, null, null, null) }
                ?: return null
        cursor.moveToFirst()
        val columnIndex = cursor.getColumnIndex(filePathColumn[0])
        val picturePath = cursor.getString(columnIndex)
        cursor.close()
        val bitmap = BitmapFactory.decodeFile(picturePath) ?: return null
        val rotate: Int = readPictureDegree(picturePath)
        return rotateBitmapByDegree(bitmap, rotate)
    }

    private fun readPictureDegree(path: String?): Int {
        var degree = 0
        try {
            val exifInterface = ExifInterface(path!!)
            val orientation = exifInterface.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> degree = 90
                ExifInterface.ORIENTATION_ROTATE_180 -> degree = 180
                ExifInterface.ORIENTATION_ROTATE_270 -> degree = 270
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return degree
    }

    private fun rotateBitmapByDegree(bm: Bitmap, degree: Int): Bitmap {
        var returnBm: Bitmap? = null
        val matrix = Matrix()
        matrix.postRotate(degree.toFloat())
        try {
            returnBm = Bitmap.createBitmap(
                bm, 0, 0, bm.width,
                bm.height, matrix, true
            )
        } catch (e: OutOfMemoryError) {
            e.printStackTrace()
        }
        if (returnBm == null) {
            returnBm = bm
        }
        if (bm != returnBm) {
            bm.recycle()
        }
        return returnBm
    }


    private fun runByVideo(resultCode: Int, data: Intent?) {
        if (resultCode != Activity.RESULT_OK || data == null) {
            Toast.makeText(this, "Video error", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            val uri = data.data
            val cursor: Cursor? = uri?.let { contentResolver.query(it, null, null, null, null) }
            if (cursor != null) {
                cursor.moveToFirst()
//                String imgNo = cursor.getString(0); // file ID
                val v_path = cursor.getString(1) // file path
//                val v_size = cursor.getString(2) // file size
//                val v_name = cursor.getString(3) // file name
                detectOnVideo(v_path)
            } else {
                Toast.makeText(this, "Video is null", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Video is null", Toast.LENGTH_SHORT).show()
        }
    }

    private fun detectOnVideo(path: String?) {
        if (detectVideo.get()) {
            Toast.makeText(this, "Video is running", Toast.LENGTH_SHORT).show()
            return
        }
        detectVideo.set(true)
        Toast.makeText(this@MainActivity, "FPS is not accurate!", Toast.LENGTH_SHORT).show()
        binding.sbVideo.visibility = View.VISIBLE
        binding.sbVideoSpeed.visibility = View.VISIBLE
        mCameraProvider.unbindAll()
        binding.viewFinder.visibility = View.INVISIBLE
        binding.btnBack.visibility = View.VISIBLE

        binding.btnBack.setOnClickListener {
            if (detectVideo.get() || detectPhoto.get()) {
                detectPhoto.set(false)
                detectVideo.set(false)
                binding.sbVideo.visibility = View.GONE
                binding.sbVideoSpeed.visibility = View.GONE
                binding.viewFinder.visibility = View.VISIBLE
                binding.btnBack.visibility = View.GONE
                startCamera(binding.viewFinder)
            }
        }

        val thread = Thread({
            mmr = FFmpegMediaMetadataRetriever()
            mmr!!.setDataSource(path)
            val dur: String =
                mmr!!.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_DURATION) // ms
            val sfps: String =
                mmr!!.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_FRAMERATE) // fps
//                String sWidth = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);  // w
//                String sHeight = mmr.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);  // h
            val rota: String =
                mmr!!.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION) // rotation
            val duration = dur.toInt()
            val fps = sfps.toFloat()

            val rotate = rota.toFloat()
            binding.sbVideo.max = duration * 1000
            var frameDis = 1.0f / fps * 1000 * 1000 * videoSpeed
            videoCurFrameLoc = 0
            while (detectVideo.get() && videoCurFrameLoc < duration * 1000) {
                videoCurFrameLoc = (videoCurFrameLoc + frameDis).toLong()
                binding.sbVideo.progress = videoCurFrameLoc.toInt()
                val b: Bitmap = mmr!!.getFrameAtTime(
                    videoCurFrameLoc,
                    FFmpegMediaMetadataRetriever.OPTION_CLOSEST
                )
                    ?: continue
                val matrix = Matrix()
                matrix.postRotate(rotate)
                width = b.width
                height = b.height
                val bitmap = Bitmap.createBitmap(b, 0, 0, width, height, matrix, false)
                startTime = System.currentTimeMillis()
                detectAndDraw(bitmap.copy(Bitmap.Config.ARGB_8888, true))
                showResultOnUI()
                frameDis = 1.0f / fps * 1000 * 1000 * videoSpeed
            }
            mmr!!.release()
            if (detectVideo.get()) {
                runOnUiThread {
                    binding.sbVideo.visibility = View.GONE
                    binding.sbVideoSpeed.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "Video end!", Toast.LENGTH_LONG).show()
                }
            }
            detectVideo.set(false)
        }, "video detect")
        thread.start()
    }


    override fun onDestroy() {
        detectCamera.set(false)
        detectVideo.set(false)
        if (detectService != null) {
            detectService.shutdown()
            detectService = null
        }

        mmr?.release()

//        analyzing = false
        mCameraProvider.unbindAll()
        if (cameraExecutor != null) {
            cameraExecutor.shutdown()
            cameraExecutor = null
        }

        super.onDestroy()

    }
}
