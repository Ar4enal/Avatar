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
    
    private static final String BINARY_GRAPH_NAME = "holistic_iris.binarypb";
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
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_IRIS = "iris_landmarks";
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
                        //Log.v(TAG, "Received face mesh landmarks packet.");
                        try {
                            NormalizedLandmarkList multiFaceLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String face_landmarks = getHolisticLandmarksDebugString(multiFaceLandmarks, "face");
                            //publishMessage(face_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(multiFaceLandmarks, "face");
                            JSONObject face_landmarks_json_object = getFaceLandmarkJsonObject(landmarks_json_object);
                            publishJsonMessage(face_landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });
/*
            processor.addPacketCallback(
                    OUTPUT_LANDMARKS_STREAM_NAME_RIGHT_HAND,
                    (packet) -> {
                        byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                        //Log.v(TAG, "Received right hand landmarks packet.");
                        try {
                            NormalizedLandmarkList RightHandLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
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
                        //Log.v(TAG, "Received left hand landmarks packet.");
                        try {
                            NormalizedLandmarkList LeftHandLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
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
                        //Log.v(TAG, "Received pose landmarks packet.");
                        try {
                            NormalizedLandmarkList PoseLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                            //+ getMultiFaceLandmarksDebugString(multiFaceLandmarks));
                            //String pose_landmarks = getHolisticLandmarksDebugString(PoseLandmarks, "pose");
                            //publishMessage(pose_landmarks);
                            JSONObject landmarks_json_object = getLandmarksJsonObject(PoseLandmarks, "pose");
                            publishJsonMessage(landmarks_json_object);
                        } catch (InvalidProtocolBufferException | JSONException e) {
                            e.printStackTrace();
                        }
                    });*/
        }
    }

    private JSONObject getFaceLandmarkJsonObject(JSONObject landmarks_json_object) throws JSONException {
        JSONObject face_landmarks_json_object = new JSONObject();
        String mouth = getMouthDistance(landmarks_json_object);
        face_landmarks_json_object.put("open_mouth", mouth); //get mouth distance
        String left_eye = getLeftEyeDistance(landmarks_json_object);
        face_landmarks_json_object.put("blink_left_eye", left_eye); //get left_eye
        String right_eye = getRightEyeDistance(landmarks_json_object);
        face_landmarks_json_object.put("blink_right_eye", right_eye); //get right_eye
        String left_mouth_angle = getLeftMouthAngle(landmarks_json_object);
        face_landmarks_json_object.put("left_mouth_angle", left_mouth_angle); //get left_mouth_angle
        String right_mouth_angle = getRightMouthAngle(landmarks_json_object);
        face_landmarks_json_object.put("right_mouth_angle", right_mouth_angle); //get right_mouth_angle
        return face_landmarks_json_object;
    }

    private String getMouthDistance(JSONObject landmarks_json_object){
        String mouth_distance_rank = "";
        try {
            JSONObject up_Y = landmarks_json_object.getJSONObject("face_landmark[13]");
            double up_mouth_Y = up_Y.getDouble("Y");
            JSONObject down_Y = landmarks_json_object.getJSONObject("face_landmark[14]");
            double down_mouth_Y = down_Y.getDouble("Y");
            double mouth_distance = Double.parseDouble(String.format("%.3f", down_mouth_Y - up_mouth_Y));
            if(mouth_distance <= 0.005){
                mouth_distance_rank = "0";
            }
            else if(mouth_distance >= 0.09){
                mouth_distance_rank = "49";
            }
            else{
                mouth_distance_rank = String.valueOf((int)Math.round(mouth_distance*565-1.825));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return mouth_distance_rank;
    }

    private String getLeftEyeDistance(JSONObject landmarks_json_object){
        String blink_left_eye = "0";
        try {
            JSONObject left_up_Z = landmarks_json_object.getJSONObject("face_landmark[4]");
            double center = left_up_Z.getDouble("Z");
            JSONObject left_up_Y = landmarks_json_object.getJSONObject("face_landmark[386]");
            double left_up_eye_Y = left_up_Y.getDouble("Y");
            JSONObject left_down_Y = landmarks_json_object.getJSONObject("face_landmark[374]");
            double left_down_eye_Y = left_down_Y.getDouble("Y");
            double left_eye_distance = left_down_eye_Y - left_up_eye_Y;
            double center_Z = Double.parseDouble(String.format("%.2f", center));
            //Log.v("left", String.valueOf(left_eye_distance));
            //Log.v("center", String.valueOf(center_Z));
            if (left_eye_distance < center_Z*(-0.1)){
                blink_left_eye = "1";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return blink_left_eye;
    }

    private String getRightEyeDistance(JSONObject landmarks_json_object){
        String blink_right_eye = "0";
        try {
            JSONObject right_up_Z = landmarks_json_object.getJSONObject("face_landmark[4]");
            double center = right_up_Z.getDouble("Z");
            JSONObject right_up_Y = landmarks_json_object.getJSONObject("face_landmark[159]");
            double right_up_eye_Y = right_up_Y.getDouble("Y");
            JSONObject right_down_Y = landmarks_json_object.getJSONObject("face_landmark[145]");
            double right_down_eye_Y = right_down_Y.getDouble("Y");
            double right_eye_distance = right_down_eye_Y - right_up_eye_Y;
            double center_Z = Double.parseDouble(String.format("%.2f", center));
            //Log.v("right", String.valueOf(right_eye_distance));
            //Log.v("center", String.valueOf(center_Z));
            if (right_eye_distance < center_Z*(-0.1)){
                blink_right_eye = "1";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return blink_right_eye;
    }

    private String getRightMouthAngle(JSONObject landmarks_json_object){
        String mouth_angle = "0";
        try {
            JSONObject x1 = landmarks_json_object.getJSONObject("face_landmark[267]");
            double left_up_mouth_x1 = x1.getDouble("X");
            JSONObject x2 = landmarks_json_object.getJSONObject("face_landmark[409]");
            double left_up_mouth_x2 = x2.getDouble("X");
            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[267]");
            double left_up_mouth_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[409]");
            double left_up_mouth_y2 = y2.getDouble("Y");

            JSONObject x3 = landmarks_json_object.getJSONObject("face_landmark[312]");
            double left_down_mouth_x3 = x3.getDouble("X");
            JSONObject x4 = landmarks_json_object.getJSONObject("face_landmark[415]");
            double left_down_mouth_x4 = x4.getDouble("X");
            JSONObject y3 = landmarks_json_object.getJSONObject("face_landmark[312]");
            double left_down_mouth_y3 = y3.getDouble("Y");
            JSONObject y4 = landmarks_json_object.getJSONObject("face_landmark[415]");
            double left_down_mouth_y4 = y4.getDouble("Y");
            double angle1 = Math.atan2(left_up_mouth_y1 - left_up_mouth_y2, left_up_mouth_x1 - left_up_mouth_x2);
            double angle2 = Math.atan2(left_down_mouth_y3 - left_down_mouth_y4, left_down_mouth_x3 - left_down_mouth_x4);
            double angle = Math.abs(angle1) - Math.abs(angle2);
            angle = Double.parseDouble(String.format("%.2f", angle));
            //Log.v("right_angle", String.valueOf(angle));

            if (angle >-0.2 & angle <= -0.1){
                mouth_angle = "1";
            }else if (angle >-0.1 & angle <= 0){
                mouth_angle = "2";
            }else if (angle >0 & angle <= 0.1){
                mouth_angle = "3";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return mouth_angle;
    }

    private String getLeftMouthAngle(JSONObject landmarks_json_object){
        String mouth_angle = "0";
        try {
            JSONObject x1 = landmarks_json_object.getJSONObject("face_landmark[37]");
            double left_up_mouth_x1 = x1.getDouble("X");
            JSONObject x2 = landmarks_json_object.getJSONObject("face_landmark[185]");
            double left_up_mouth_x2 = x2.getDouble("X");
            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[37]");
            double left_up_mouth_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[185]");
            double left_up_mouth_y2 = y2.getDouble("Y");

            JSONObject x3 = landmarks_json_object.getJSONObject("face_landmark[82]");
            double left_down_mouth_x3 = x3.getDouble("X");
            JSONObject x4 = landmarks_json_object.getJSONObject("face_landmark[191]");
            double left_down_mouth_x4 = x4.getDouble("X");
            JSONObject y3 = landmarks_json_object.getJSONObject("face_landmark[82]");
            double left_down_mouth_y3 = y3.getDouble("Y");
            JSONObject y4 = landmarks_json_object.getJSONObject("face_landmark[191]");
            double left_down_mouth_y4 = y4.getDouble("Y");
            double angle1 = Math.atan2(left_up_mouth_y1 - left_up_mouth_y2, left_up_mouth_x1 - left_up_mouth_x2);
            double angle2 = Math.atan2(left_down_mouth_y3 - left_down_mouth_y4, left_down_mouth_x3 - left_down_mouth_x4);
            double angle = Math.abs(angle1) - Math.abs(angle2);
            angle = Double.parseDouble(String.format("%.2f", angle));
            //Log.v("left_angle", String.valueOf(angle));

            if (angle >0.1 & angle <= 0.2){
                mouth_angle = "1";
            }else if (angle >0 & angle <= 0.1){
                mouth_angle = "2";
            }else if (angle >-0.1 & angle <= 0){
                mouth_angle = "3";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return mouth_angle;
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
        else if(location == "iris"){
            int irislandmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                landmarksString +=
                        "\t\tIrisLandmark["
                                + irislandmarkIndex
                                + "]: ("
                                + landmark.getX()
                                + ", "
                                + landmark.getY()
                                + ", "
                                + landmark.getZ()
                                + ")\n";
                ++irislandmarkIndex;
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
                        ch.exchangeDeclare("oneplus", "topic" ,true);
                        while (true) {
                            //String message = queue.takeFirst();
                            JSONObject message = json_queue.takeFirst();
                            try{
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