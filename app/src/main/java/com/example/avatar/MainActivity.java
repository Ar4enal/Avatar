package com.example.avatar;

import android.graphics.SurfaceTexture;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.glutil.EglManager;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.LinkedBlockingDeque;

import com.google.protobuf.InvalidProtocolBufferException;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;

import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    
    private static final String BINARY_GRAPH_NAME = "holistic_tracking_aar.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.FRONT;

    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;
    private static final String INPUT_NUM_FACES_SIDE_PACKET_NAME = "num_faces";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_FACE_MESH = "face_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_POS_ROI = "pose_roi";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_RIGHT_HAND = "right_hand_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_LEFT_HAND = "left_hand_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_POSE = "pose_landmarks";
    private static final String FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel";
    // Max number of faces to detect/process.
    private static final int NUM_FACES = 1;
    private static final boolean USE_FRONT_CAMERA = true;
    private boolean haveAddedSidePackets = false;

    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;

    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private ExternalTextureConverter converter;

    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        setupConnectionFactory();
        publishToAMQP();

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);

        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor.getVideoSurfaceOutput().setFlipY(FLIP_FRAMES_VERTICALLY);

        PermissionHelper.checkAndRequestCameraPermissions(this);

        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
        if (Log.isLoggable(TAG, Log.VERBOSE)) {
            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME_FACE_MESH,
                    (packet) -> {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        Log.v(TAG, "Received face mesh landmarks packet.");
                        try {
                            NormalizedLandmarkList multiFaceLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            Log.v(
                                    TAG,
                                    "[TS:"
                                            + packet.getTimestamp()
                                            + "] ");
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String face_landmarks = getHolisticLandmarksDebugString(multiFaceLandmarks, "face");
                            //publishMessage(face_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(multiFaceLandmarks, "face");
                            publishJsonMessage(landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });

            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME_RIGHT_HAND,
                    (packet) -> {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        Log.v(TAG, "Received right hand landmarks packet.");
                        try {
                            NormalizedLandmarkList RightHandLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            Log.v(
                                    TAG,
                                    "[TS:"
                                            + packet.getTimestamp()
                                            + "] ");
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String right_hand_landmarks = getHolisticLandmarksDebugString(RightHandLandmarks, "right_hand");
                            //publishMessage(right_hand_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(RightHandLandmarks, "right_hand");
                            publishJsonMessage(landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });

            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME_LEFT_HAND,
                    (packet) -> {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        Log.v(TAG, "Received left hand landmarks packet.");
                        try {
                            NormalizedLandmarkList LeftHandLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            Log.v(
                                    TAG,
                                    "[TS:"
                                            + packet.getTimestamp()
                                            + "] ");
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String left_hand_landmarks = getHolisticLandmarksDebugString(LeftHandLandmarks, "left_hand");
                            //publishMessage(left_hand_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(LeftHandLandmarks, "left_hand");
                            publishJsonMessage(landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });

            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME_POSE,
                    (packet) -> {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        Log.v(TAG, "Received pose landmarks packet.");
                        try {
                            NormalizedLandmarkList PoseLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            Log.v(
                                    TAG,
                                    "[TS:"
                                            + packet.getTimestamp()
                                            + "] ");
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String pose_landmarks = getHolisticLandmarksDebugString(PoseLandmarks, "pose");
                            //publishMessage(pose_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(PoseLandmarks, "pose");
                            publishJsonMessage(landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });
        }
    }

    private static String getHolisticLandmarksDebugString(NormalizedLandmarkList landmarks, String location) {
        String landmarksString = "";
        if (location == "face"){
            int landmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                landmarksString +=
                        "\t\tFaceLandmark["
                                + landmarkIndex
                                + "]: ("
                                + landmark.getX()
                                + ", "
                                + landmark.getY()
                                + ", "
                                + landmark.getZ()
                                + ")\n";
                ++landmarkIndex;
        }}
        else if(location == "right_hand"){
                int rhlandmarkIndex = 0;
                for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                    landmarksString +=
                            "\t\tRightHandLandmark["
                                    + rhlandmarkIndex
                                    + "]: ("
                                    + landmark.getX()
                                    + ", "
                                    + landmark.getY()
                                    + ", "
                                    + landmark.getZ()
                                    + ")\n";
                    ++rhlandmarkIndex;
        }}
        else if(location == "left_hand"){
                    int lhlandmarkIndex = 0;
                    for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                        landmarksString +=
                                "\t\tLeftHandLandmark["
                                        + lhlandmarkIndex
                                        + "]: ("
                                        + landmark.getX()
                                        + ", "
                                        + landmark.getY()
                                        + ", "
                                        + landmark.getZ()
                                        + ")\n";
                        ++lhlandmarkIndex;
        }}
        else if(location == "pose"){
                        int plandmarkIndex = 0;
                        for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                            landmarksString +=
                                    "\t\tPoseLandmark["
                                            + plandmarkIndex
                                            + "]: ("
                                            + landmark.getX()
                                            + ", "
                                            + landmark.getY()
                                            + ", "
                                            + landmark.getZ()
                                            + ")\n";
                            ++plandmarkIndex;
        }}
        return landmarksString;
    }

    private static JSONObject getLandmarksJsonObject(NormalizedLandmarkList landmarks, String location) throws JSONException {
        JSONObject landmarks_json_object = new JSONObject();
        if (location == "face"){
            int landmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()){
                JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("landmark_index", landmarkIndex);
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());
                String tag = "face_landmark[" + landmarkIndex + "]";
                landmarks_json_object.put(tag, landmarks_json_object_part);
                ++landmarkIndex;
            }
        }
        else if(location == "right_hand"){
            int rlandmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("landmark_index", rlandmarkIndex);
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());
                String tag = "right_hand_landmark[" + rlandmarkIndex + "]";
                landmarks_json_object.put(tag, landmarks_json_object_part);
                ++rlandmarkIndex;
            }
        }
        else if(location == "left_hand"){
                int llandmarkIndex = 0;
                for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                    JSONObject landmarks_json_object_part = new JSONObject();
                    landmarks_json_object_part.put("landmark_index", llandmarkIndex);
                    landmarks_json_object_part.put("X", landmark.getX());
                    landmarks_json_object_part.put("Y", landmark.getY());
                    landmarks_json_object_part.put("Z", landmark.getZ());
                    String tag = "left_hand_landmark[" + llandmarkIndex + "]";
                    landmarks_json_object.put(tag, landmarks_json_object_part);
                    ++llandmarkIndex;
                }
        }
        else if(location == "pose"){
            int plandmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("landmark_index", plandmarkIndex);
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());
                String tag = "pose_landmark[" + plandmarkIndex + "]";
                landmarks_json_object.put(tag, landmarks_json_object_part);
                ++plandmarkIndex;
            }
        }
        return landmarks_json_object;
    }

    Thread subscribeThread;
    Thread publishThread;
    @Override
    protected void onDestroy() {
        super.onDestroy();
        publishThread.interrupt();
        subscribeThread.interrupt();
    }

    private final BlockingDeque<String> queue = new LinkedBlockingDeque();
    private final BlockingDeque<JSONObject> json_queue = new LinkedBlockingDeque();
    void publishMessage(String message) {
        //Adds a message to internal blocking queue
        try {
            //Log.d("","[q] " + message);
            queue.putLast(message);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    void publishJsonMessage(JSONObject message) {
        //Adds a message to internal blocking queue
        try {
            //Log.d("","[q] " + message);
            json_queue.putLast(message);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    ConnectionFactory factory = new ConnectionFactory();
    private void setupConnectionFactory() {
        factory.setHost("gerbil.rmq.cloudamqp.com");
        //factory.setHost("18.162.114.173");
        factory.setPort(5672);
        factory.setUsername("lmtpgusv");
        factory.setPassword("pfZmzsmdTM6ZselTSUXM-dlwSAh07cDN");
        factory.setVirtualHost("lmtpgusv");
/*        try{
            factory.setUri("amqp://guest:guest@192.168.1.108:5672/");
            factory.setAutomaticRecoveryEnabled(false);}
        catch (KeyManagementException | NoSuchAlgorithmException | URISyntaxException e1) {
            e1.printStackTrace();
        }*/
    }

    @Override
    protected void onResume() {
        super.onResume();
        converter = new ExternalTextureConverter(eglManager.getContext());
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        converter.close();
        previewDisplayView.setVisibility(View.GONE);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    // 计算最佳的预览大小
    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // 设置预览大小
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        // 根据是否旋转调整预览图像大小
        boolean isCameraRotated = cameraHelper.isCameraRotated();
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }


    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }

    // 相机启动后事件
    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        // 显示预览
        previewFrameTexture = surfaceTexture;
        previewDisplayView.setVisibility(View.VISIBLE);
        // onCameraStarted gets called each time the activity resumes, but we only want to do this once.
        if (!haveAddedSidePackets) {
            float focalLength = cameraHelper.getFocalLengthPixels();
            if (focalLength != Float.MIN_VALUE) {
                Packet focalLengthSidePacket = processor.getPacketCreator().createFloat32(focalLength);
                Map<String, Packet> inputSidePackets = new HashMap<>();
                inputSidePackets.put(FOCAL_LENGTH_STREAM_NAME, focalLengthSidePacket);
                processor.setInputSidePackets(inputSidePackets);
            }
            haveAddedSidePackets = true;
        }
    }

    // 设置相机大小
    protected Size cameraTargetResolution() {
        return null;
    }

    // 启动相机
    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(this::onCameraStarted);
        CameraHelper.CameraFacing cameraFacing =
                USE_FRONT_CAMERA ? CameraHelper.CameraFacing.FRONT : CameraHelper.CameraFacing.BACK;
        cameraHelper.startCamera(this, cameraFacing, null, cameraTargetResolution());
    }

    public void publishToAMQP()
    {
        publishThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while(true) {
                    try {
                        Connection connection = factory.newConnection();
                        Channel ch = connection.createChannel();
                        ch.confirmSelect();

                        while (true) {
                            //String message = queue.takeFirst();
                            JSONObject message = json_queue.takeFirst();
                            try{
                                ch.exchangeDeclare("oneplus", "topic" ,true);
                                //String queueName = ch.queueDeclare().getQueue();
                                //ch.queueBind(queueName, "oneplus", "chat");
                                //ch.basicPublish("oneplus", "chat", null, message.getBytes());
                                ch.basicPublish("oneplus", "chat", null, message.toString().getBytes());
                                Log.d("", "[s] " + message);
                                ch.waitForConfirmsOrDie();
                            } catch (Exception e){
                                Log.d("","[f] " + message);
                                //queue.putFirst(message);
                                json_queue.putFirst(message);
                                throw e;
                            }
                        }
                    } catch (InterruptedException e) {
                        break;
                    } catch (Exception e) {
                        Log.d("", "Connection broken: " + e.getClass().getName());
                        try {
                            Thread.sleep(5000); //sleep and then try again
                        } catch (InterruptedException e1) {
                            break;
                        }
                    }
                }
            }
        });
        publishThread.start();
    }


}