package com.example.s25nputest

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import android.system.Os
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.s25nputest.ui.theme.S25NPUTestTheme
import java.io.File
import kotlin.math.abs

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "S25NPUTest"

        private const val QNN_EP_NAME =
            "QNNExecutionProvider"

        private const val QNN_PLUGIN_LIBRARY =
            "libonnxruntime_providers_qnn.so"

        private const val MODEL_NAME =
            "tiny_qdq_matmul.onnx"
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var sessionOptions: OrtSession.SessionOptions? = null

    private var qnnPluginRegistered = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val status = runHtpInference()

        setContent {
            S25NPUTestTheme {
                QnnStatusScreen(status)
            }
        }
    }

    private fun runHtpInference(): String {
        return try {

            // ---------------------------------------------------------
            // 1. Android native library directory
            // ---------------------------------------------------------

            val nativeLibDir =
                applicationInfo.nativeLibraryDir

            Log.i(
                TAG,
                "Native library dir = $nativeLibDir"
            )

            // ---------------------------------------------------------
            // 2. QNN / HTP DSP library search path
            //
            // 반드시 OrtEnvironment 생성 전에 설정
            // ---------------------------------------------------------

            Os.setenv(
                "ADSP_LIBRARY_PATH",
                nativeLibDir,
                true
            )

            Log.i(
                TAG,
                "ADSP_LIBRARY_PATH = $nativeLibDir"
            )

            // ---------------------------------------------------------
            // 3. Plugin / HTP libraries
            // ---------------------------------------------------------

            val pluginPath =
                "$nativeLibDir/$QNN_PLUGIN_LIBRARY"

            val htpLibraryPath =
                "$nativeLibDir/libQnnHtp.so"

            if (!File(pluginPath).exists()) {
                throw IllegalStateException(
                    "QNN plugin not found:\n$pluginPath"
                )
            }

            if (!File(htpLibraryPath).exists()) {
                throw IllegalStateException(
                    "QNN HTP library not found:\n$htpLibraryPath"
                )
            }

            Log.i(
                TAG,
                "QNN plugin = $pluginPath"
            )

            Log.i(
                TAG,
                "QNN HTP library = $htpLibraryPath"
            )

            // ---------------------------------------------------------
            // 4. ONNX Runtime
            // ---------------------------------------------------------

            val env =
                OrtEnvironment.getEnvironment()

            ortEnvironment = env

            Log.i(
                TAG,
                "ORT version = ${env.version}"
            )

            // ---------------------------------------------------------
            // 5. Register standalone QNN EP plugin
            // ---------------------------------------------------------

            env.registerExecutionProviderLibrary(
                QNN_EP_NAME,
                pluginPath
            )

            qnnPluginRegistered = true

            // ---------------------------------------------------------
            // 6. Find QNN EP devices
            // ---------------------------------------------------------

            val qnnDevices =
                env.epDevices.filter {
                    it.epName == QNN_EP_NAME
                }

            if (qnnDevices.isEmpty()) {
                throw IllegalStateException(
                    "No QNN EP device found"
                )
            }

            qnnDevices.forEach {
                Log.i(
                    TAG,
                    "QNN device = $it"
                )
            }

            // ---------------------------------------------------------
            // 7. Session options
            // ---------------------------------------------------------

            val options =
                OrtSession.SessionOptions()

            sessionOptions = options

            // 핵심:
            // QNN이 처리하지 못하는 연산을
            // CPU EP로 넘기는 것을 금지.
            options.addConfigEntry(
                "session.disable_cpu_ep_fallback",
                "1"
            )

            // ---------------------------------------------------------
            // 8. Force HTP = Snapdragon NPU
            // ---------------------------------------------------------

            val providerOptions = mapOf(
                "backend_type" to "htp",

                // 모델의 graph I/O Q/DQ도
                // QNN에서 처리하도록 함.
                "offload_graph_io_quantization" to "0"
            )

            options.addExecutionProvider(
                qnnDevices,
                providerOptions
            )

            Log.i(
                TAG,
                "QNN backend requested = HTP"
            )

            Log.i(
                TAG,
                "CPU fallback = DISABLED"
            )

            // ---------------------------------------------------------
            // 9. Read ONNX model from assets
            // ---------------------------------------------------------

            val modelBytes =
                assets.open(MODEL_NAME).use {
                    it.readBytes()
                }

            Log.i(
                TAG,
                "Model size = ${modelBytes.size} bytes"
            )

            // ---------------------------------------------------------
            // 10. Create session
            //
            // 여기서 성공해야 모델 전체가 QNN HTP에서
            // 처리 가능한 상태라는 강한 증거가 됨.
            // ---------------------------------------------------------

            val sessionCreateStart =
                System.nanoTime()

            val session =
                env.createSession(
                    modelBytes,
                    options
                )

            val sessionCreateMs =
                (System.nanoTime() - sessionCreateStart) /
                        1_000_000.0

            ortSession = session

            Log.i(
                TAG,
                "HTP session created in %.3f ms"
                    .format(sessionCreateMs)
            )

            Log.i(
                TAG,
                "Inputs = ${session.inputNames}"
            )

            Log.i(
                TAG,
                "Outputs = ${session.outputNames}"
            )

            // ---------------------------------------------------------
            // 11. Input
            //
            // 모델:
            // [1, 2, 3, 4] × Identity
            //
            // Expected ≈ [1, 2, 3, 4]
            // ---------------------------------------------------------

            val inputData = arrayOf(
                floatArrayOf(
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f
                )
            )

            OnnxTensor.createTensor(
                env,
                inputData
            ).use { inputTensor ->

                // -----------------------------------------------------
                // 12. Actual inference
                // -----------------------------------------------------

                val runStart =
                    System.nanoTime()

                session.run(
                    mapOf(
                        "input" to inputTensor
                    )
                ).use { result ->

                    val inferenceMs =
                        (System.nanoTime() - runStart) /
                                1_000_000.0

                    val outputTensor =
                        result[0] as OnnxTensor

                    @Suppress("UNCHECKED_CAST")
                    val output =
                        outputTensor.value
                                as Array<FloatArray>

                    val values =
                        output[0]

                    Log.i(
                        TAG,
                        "Output = ${values.contentToString()}"
                    )

                    Log.i(
                        TAG,
                        "Inference time = %.3f ms"
                            .format(inferenceMs)
                    )

                    // -------------------------------------------------
                    // 13. Validate output
                    // -------------------------------------------------

                    val expected =
                        floatArrayOf(
                            1.0f,
                            2.0f,
                            3.0f,
                            4.0f
                        )

                    var maxError = 0.0f

                    for (i in expected.indices) {
                        val error =
                            abs(
                                values[i] -
                                        expected[i]
                            )

                        if (error > maxError) {
                            maxError = error
                        }
                    }

                    if (maxError > 0.11f) {
                        throw IllegalStateException(
                            "Output validation failed. " +
                                    "Max error = $maxError"
                        )
                    }

                    // -------------------------------------------------
                    // PASS
                    // -------------------------------------------------

                    """
                    HTP INFERENCE PASS

                    ONNX Runtime:
                    ${env.version}

                    Backend:
                    QNN HTP / NPU

                    CPU fallback:
                    DISABLED

                    Input:
                    [1.0, 2.0, 3.0, 4.0]

                    Output:
                    ${values.contentToString()}

                    Max error:
                    $maxError

                    Session creation:
                    %.3f ms

                    Inference:
                    %.3f ms
                    """.trimIndent()
                        .format(
                            sessionCreateMs,
                            inferenceMs
                        )
                }
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "HTP inference failed",
                e
            )

            """
            HTP INFERENCE FAILED

            ${e.javaClass.simpleName}

            ${e.message}

            Check Logcat:
            tag = S25NPUTest
            """.trimIndent()
        }
    }

    override fun onDestroy() {

        // 반드시 Session -> SessionOptions -> Plugin 순서
        try {
            ortSession?.close()
            ortSession = null
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Failed to close session",
                e
            )
        }

        try {
            sessionOptions?.close()
            sessionOptions = null
        } catch (e: Exception) {
            Log.e(
                TAG,
                "Failed to close SessionOptions",
                e
            )
        }

        if (qnnPluginRegistered) {

            try {

                ortEnvironment
                    ?.unregisterExecutionProviderLibrary(
                        QNN_EP_NAME
                    )

                qnnPluginRegistered = false

                Log.i(
                    TAG,
                    "QNN plugin unregistered"
                )

            } catch (e: Exception) {

                Log.e(
                    TAG,
                    "Failed to unregister QNN plugin",
                    e
                )
            }
        }

        super.onDestroy()
    }
}


@Composable
fun QnnStatusScreen(status: String) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),

        horizontalAlignment =
            Alignment.CenterHorizontally,

        verticalArrangement =
            Arrangement.Center
    ) {

        Text(
            text = "Galaxy S25 NPU Test",
            style =
                MaterialTheme.typography.headlineMedium
        )

        Text(
            text = status,
            modifier =
                Modifier.padding(top = 24.dp)
        )
    }
}