package com.objdetection

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.objdetection.databinding.ActivityWelcomeBinding

class WelcomeActivity : AppCompatActivity() {
    private val NANODET = 1
    private val YOLOV5S = 2
    private var useGPU = false
    private lateinit var binding: ActivityWelcomeBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityWelcomeBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.warningButton.setOnClickListener {
            val builder = AlertDialog.Builder(this@WelcomeActivity)
            builder.setTitle("Warning")
            builder.setMessage("Please allow screen rotating.\nApp may crash when changing from landscape to portrait")
            builder.setCancelable(true)
            builder.setPositiveButton("OK", null)
            val dialog = builder.create()
            dialog.show()
        }

        binding.settingButton.setOnClickListener {
            val intent = Intent(this@WelcomeActivity, SettingsActivity::class.java)
            this@WelcomeActivity.startActivity(intent)
        }

        binding.tbUseGpu.setOnCheckedChangeListener{_, isChecked ->
            useGPU = isChecked
            if (useGPU) {
                val builder = AlertDialog.Builder(this@WelcomeActivity)
                builder.setTitle("Warning")
                builder.setMessage("If the GPU is too old, it may not work well in GPU mode.\nIt may take a long time to init.")
                builder.setCancelable(true)
                builder.setPositiveButton("OK", null)
                val dialog = builder.create()
                dialog.show()
            } else {
                Toast.makeText(this@WelcomeActivity, "CPU Mode", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnStartDetect0.setOnClickListener {
            val intent = Intent(this@WelcomeActivity, MainActivity::class.java)
            intent.putExtra("useGPU", useGPU)
            intent.putExtra("useModel", NANODET)
            this@WelcomeActivity.startActivity(intent)
        }

        binding.btnStartDetect1.setOnClickListener {
            val intent = Intent(this@WelcomeActivity, MainActivity::class.java)
            intent.putExtra("useGPU", useGPU)
            intent.putExtra("useModel", YOLOV5S)
            this@WelcomeActivity.startActivity(intent)
        }

    }
}