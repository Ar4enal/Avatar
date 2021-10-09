package com.example.avatar;

import android.content.Context;
import android.content.DialogInterface;
import android.graphics.SurfaceTexture;
import android.os.Bundle;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.os.StrictMode;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;
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
import org.webrtc.Camera1Enumerator;
import org.webrtc.IceCandidate;
import org.webrtc.MediaConstraints;
import org.webrtc.MediaStream;
import org.webrtc.PeerConnection;
import org.webrtc.PeerConnectionFactory;
import org.webrtc.SessionDescription;
import org.webrtc.SurfaceViewRenderer;
import org.webrtc.VideoCapturer;
import org.webrtc.VideoTrack;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, SignalingClient.Callback{
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
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_FACE_MESH = "face_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_POS_ROI = "pose_roi";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_RIGHT_HAND = "right_hand_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_LEFT_HAND = "left_hand_landmarks";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME_POSE = "pose_landmarks";
    private static final String FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel";

    private static final boolean USE_FRONT_CAMERA = true;
    private boolean haveAddedSidePackets = false;
    private static String serveraddress = "";   //192.168.50.3
    private static final Integer port = 5567;
    private Button enter_ip;

    PeerConnectionFactory peerConnectionFactory;
    PeerConnection peerConnection;
    MediaStream mediaStream;
    SurfaceViewRenderer localView;
    SurfaceViewRenderer remoteView;

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
        bindView();
        final String[] serverAddress = {""};

        if (android.os.Build.VERSION.SDK_INT > 9) {
            StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
            StrictMode.setThreadPolicy(policy);
        }
        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        //setupConnectionFactory();
        //publishToAMQP();
/*        confirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                serverAddress[0] = server_ip.getText().toString();
                send_TCP(serverAddress[0], port);
                Log.v("TCP server ip", serverAddress[0]);
            }
        });*/
        //send_TCP(serveraddress, port);


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
        processor.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME_FACE_MESH,
                (packet) -> {
                    byte[] landmarksRaw = PacketGetter.getProtoBytes(packet);
                    //Log.d(TAG, "Received face mesh landmarks packet.");
                    try {
                        NormalizedLandmarkList multiFaceLandmarks = NormalizedLandmarkList.parseFrom(landmarksRaw);
                        JSONObject landmarks_json_object = getLandmarksJsonObject(multiFaceLandmarks, "face");
                        //JSONObject face_landmarks_json_object = getFaceLandmarkJsonObject(landmarks_json_object);
                        //Log.d("face", String.valueOf(landmarks_json_object));
                        publishJsonMessage(landmarks_json_object);
                        //json_message = face_landmarks_json_object.toString();
                    } catch (InvalidProtocolBufferException | JSONException e) {
                        e.printStackTrace();
                    }
                });

/*        processor.addPacketCallback(
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

    private JSONObject getFaceLandmarkJsonObject(JSONObject landmarks_json_object) throws JSONException {
        JSONObject face_landmarks_json_object = new JSONObject();

        String head_turn = getHeadTurn(landmarks_json_object);
        face_landmarks_json_object.put("head_turn", head_turn); //get head rotation

        String head_rotation = getHeadRotation(landmarks_json_object);
        face_landmarks_json_object.put("head_rotation", head_rotation); //get head rotation

        String mouth = getMouthDistance(landmarks_json_object);
        face_landmarks_json_object.put("open_mouth", mouth); //get mouth distance

        String left_eye = getLeftEyeDistance(landmarks_json_object);
        face_landmarks_json_object.put("blink_left_eye", left_eye); //get left_eye

        String right_eye = getRightEyeDistance(landmarks_json_object);
        face_landmarks_json_object.put("blink_right_eye", right_eye); //get right_eye

        String left_eyebrow = getLeftEyebrow(landmarks_json_object);
        face_landmarks_json_object.put("left_eyebrow", left_eyebrow); //get left_eyebrow

        String right_eyebrow = getRightEyebrow(landmarks_json_object);
        face_landmarks_json_object.put("right_eyebrow", right_eyebrow); //get right_eyebrow

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
        return mouth_distance_rank; //0~49
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
        return blink_left_eye;  //0:open 1:blink
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
        return blink_right_eye;  //0:open 1:blink
    }

    private String getLeftEyebrow(JSONObject landmarks_json_object){
        String left_eyebrow = "0";
        try {
            JSONObject Z = landmarks_json_object.getJSONObject("face_landmark[168]");
            double center = Z.getDouble("Z");
            JSONObject left_up_Y = landmarks_json_object.getJSONObject("face_landmark[336]");
            double left_up_eye_Y = left_up_Y.getDouble("Y");
            JSONObject left_down_Y = landmarks_json_object.getJSONObject("face_landmark[413]");
            double left_down_eye_Y = left_down_Y.getDouble("Y");
            double left_eye_distance = left_down_eye_Y - left_up_eye_Y;
            double center_Z = Double.parseDouble(String.format("%.3f", center));

            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_y2 = y2.getDouble("Y");
            JSONObject center_y = landmarks_json_object.getJSONObject("face_landmark[168]");
            double nose_center_y = center_y.getDouble("Y");
            double ear_center_y = (left_ear_y1 + right_ear_y2)/2;
            double y = Double.parseDouble(String.format("%.2f", nose_center_y - ear_center_y));

            /*Log.v("left", String.valueOf(left_eye_distance));
            Log.v("left_center", String.valueOf(center_Z));
            Log.v("y", String.valueOf(y+0.03));*/
            double c = center_Z * -1 + 0.01;
            if (left_eye_distance - c >= 0.013){
                left_eyebrow = "1";
            }else if (left_eye_distance - c <= 0.001){
                left_eyebrow = "-1";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return left_eyebrow;
    }

    private String getRightEyebrow(JSONObject landmarks_json_object){
        String right_eyebrow = "0";
        try {
            JSONObject Z = landmarks_json_object.getJSONObject("face_landmark[168]");
            double center = Z.getDouble("Z");
            JSONObject right_up_Y = landmarks_json_object.getJSONObject("face_landmark[107]");
            double right_up_eye_Y = right_up_Y.getDouble("Y");
            JSONObject right_down_Y = landmarks_json_object.getJSONObject("face_landmark[189]");
            double right_down_eye_Y = right_down_Y.getDouble("Y");
            double right_eye_distance = right_down_eye_Y - right_up_eye_Y;
            double center_Z = Double.parseDouble(String.format("%.3f", center));

            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_y2 = y2.getDouble("Y");
            JSONObject center_y = landmarks_json_object.getJSONObject("face_landmark[168]");
            double nose_center_y = center_y.getDouble("Y");
            double ear_center_y = (left_ear_y1 + right_ear_y2)/2;
            double y = Double.parseDouble(String.format("%.2f", nose_center_y - ear_center_y));

            /*Log.v("right", String.valueOf(right_eye_distance));
            Log.v("right_center", String.valueOf(center_Z));
            Log.v("y", String.valueOf(y+0.03));*/
            double c = center_Z * -1 + 0.01;
            if (right_eye_distance - c >= 0.013){
                right_eyebrow = "1";
            }else if (right_eye_distance - c <= 0.001){
                right_eyebrow = "-1";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return right_eyebrow;
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
        return mouth_angle;  //0~3
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
        return mouth_angle;  //0~3
    }

    private String getHeadTurn(JSONObject landmarks_json_object){
        String head_turn = "";
        try {
            JSONObject x1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_x1 = x1.getDouble("X");
            JSONObject x2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_x2 = x2.getDouble("X");
            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_y2 = y2.getDouble("Y");
            JSONObject z1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_z1 = z1.getDouble("Z");
            JSONObject z2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_z2 = z2.getDouble("Z");

            JSONObject center_x = landmarks_json_object.getJSONObject("face_landmark[168]");
            double nose_center_x = center_x.getDouble("X");
            JSONObject center_y = landmarks_json_object.getJSONObject("face_landmark[168]");
            double nose_center_y = center_y.getDouble("Y");
            JSONObject center_z = landmarks_json_object.getJSONObject("face_landmark[168]");
            double nose_center_z = center_z.getDouble("Z");

            double ear_center_x = (left_ear_x1 + right_ear_x2)/2;
            double ear_center_y = (left_ear_y1 + right_ear_y2)/2;
            double ear_center_z = (left_ear_z1 + right_ear_z2)/2;

            double x = Double.parseDouble(String.format("%.2f", nose_center_x - ear_center_x));
            double y = Double.parseDouble(String.format("%.2f", nose_center_y - ear_center_y));
            double z = Double.parseDouble(String.format("%.2f", nose_center_z - ear_center_z));
            head_turn = x + "," + y + "," + z;
            //Log.v("head_turn", String.valueOf(head_turn));

        } catch (JSONException e) {
            e.printStackTrace();
        }
        return head_turn;
    }

    private String getHeadRotation(JSONObject landmarks_json_object){
        String head_rotation = "";
        try {
            JSONObject x1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_x1 = x1.getDouble("X");
            JSONObject x2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_x2 = x2.getDouble("X");
            JSONObject y1 = landmarks_json_object.getJSONObject("face_landmark[356]");
            double left_ear_y1 = y1.getDouble("Y");
            JSONObject y2 = landmarks_json_object.getJSONObject("face_landmark[127]");
            double right_ear_y2 = y2.getDouble("Y");
            double angle = Double.parseDouble(String.format("%.2f",(right_ear_x2 - left_ear_x1)/(right_ear_y2 - left_ear_y1)));
            //Log.v("head_rotation", String.valueOf(angle));
            if (angle <= 13 & angle >= -13){
                head_rotation = String.valueOf(angle);
            }
            else{
                head_rotation = "0";
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return head_rotation;
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
                List<String> list = new ArrayList<String>();
                list.add(String.format("%.8f", landmark.getX()));
                list.add(String.format("%.8f", landmark.getY()));
                list.add(String.format("%.8f", landmark.getZ()));

                /*JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());*/
                String tag = "face_landmark[" + landmarkIndex + "]";
                landmarks_json_object.put(tag, list);
                ++landmarkIndex;
            }
        }
        else if(location == "right_hand"){
            int rlandmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                List<String> list = new ArrayList<String>();
                list.add(String.format("%.8f", landmark.getX()));
                list.add(String.format("%.8f", landmark.getY()));
                list.add(String.format("%.8f", landmark.getZ()));
                /*JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());*/
                String tag = "right_hand_landmark[" + rlandmarkIndex + "]";
                landmarks_json_object.put(tag, list);
                ++rlandmarkIndex;
            }
        }
        else if(location == "left_hand"){
                int llandmarkIndex = 0;
                for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                    List<String> list = new ArrayList<String>();
                    list.add(String.format("%.8f", landmark.getX()));
                    list.add(String.format("%.8f", landmark.getY()));
                    list.add(String.format("%.8f", landmark.getZ()));
                    /*JSONObject landmarks_json_object_part = new JSONObject();
                    landmarks_json_object_part.put("X", landmark.getX());
                    landmarks_json_object_part.put("Y", landmark.getY());
                    landmarks_json_object_part.put("Z", landmark.getZ());*/
                    String tag = "left_hand_landmark[" + llandmarkIndex + "]";
                    landmarks_json_object.put(tag, list);
                    ++llandmarkIndex;
                }
        }
        else if(location == "pose"){
            int plandmarkIndex = 0;
            for (NormalizedLandmark landmark : landmarks.getLandmarkList()) {
                List<String> list = new ArrayList<String>();
                list.add(String.format("%.8f", landmark.getX()));
                list.add(String.format("%.8f", landmark.getY()));
                list.add(String.format("%.8f", landmark.getZ()));
                /*JSONObject landmarks_json_object_part = new JSONObject();
                landmarks_json_object_part.put("X", landmark.getX());
                landmarks_json_object_part.put("Y", landmark.getY());
                landmarks_json_object_part.put("Z", landmark.getZ());*/
                String tag = "pose_landmark[" + plandmarkIndex + "]";
                landmarks_json_object.put(tag, list);
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

    //private final BlockingDeque<String> queue = new LinkedBlockingDeque();
    private final BlockingDeque<JSONObject> json_queue = new LinkedBlockingDeque();
    /*void publishMessage(String message) {
        //Adds a message to internal blocking queue
        try {
            //Log.d("","[q] " + message);
            queue.putLast(message);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }*/

    void publishJsonMessage(JSONObject message) {
        //Adds a message to internal blocking queue
        try {
            Log.d("","[q] " + message);
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

    public void send_TCP(String serverAddress, int port){
        publishThread = new Thread(new Runnable() {
            @Override
            public void run() {
                    try {
                        Socket s = new Socket(serverAddress, port);
                        OutputStream out = s.getOutputStream();
                        while (true) {
                            JSONObject message = json_queue.takeFirst();
                            out.write(message.toString().getBytes());
                            Log.v("", "[s] " + message);
                            out.flush();
                            //Log.v("", "[s] " + message);
                        }
                    } catch (Exception e) {
                        Log.d("", "Connection broken: " + e.getClass().getName());

                    }

            }
        });
        publishThread.start();
    }

    private void bindView(){
        enter_ip = (Button) findViewById(R.id.enter_ip);
        enter_ip.setOnClickListener(this);
    }

    @Override
    public void onClick(View v){
        if (v.getId() == R.id.enter_ip){
            AlertDialog.Builder dialog = new AlertDialog.Builder(MainActivity.this);
            dialog.setTitle("Please enter TCP server IP address");
            final View view = View.inflate(MainActivity.this, R.layout.tcpserver_ip, null);
            EditText et = view.findViewById(R.id.server_ip);
            dialog.setView(view);
            dialog.setPositiveButton("confirm", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialogInterface, int i) {
                    String ip = et.getText().toString();
                    Toast.makeText(MainActivity.this, "TCP server ip:" + ip, Toast.LENGTH_SHORT).show();
                    json_queue.clear();
                    serveraddress = ip;
                    send_TCP(serveraddress, port);
                    dialogInterface.cancel();
                }
            });

            dialog.setNegativeButton("cancel", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialogInterface, int i) {
                    dialogInterface.cancel();
                }
            });
            dialog.show();
        }
    }

    private void call() {
        List<PeerConnection.IceServer> iceServers = new ArrayList<>();
        iceServers.add(PeerConnection.IceServer.builder("stun:stun.l.google.com:19302").createIceServer());
        peerConnection = peerConnectionFactory.createPeerConnection(iceServers, new PeerConnectionAdapter("localconnection") {
            @Override
            public void onIceCandidate(IceCandidate iceCandidate) {
                super.onIceCandidate(iceCandidate);
                SignalingClient.get().sendIceCandidate(iceCandidate);
            }

            @Override
            public void onAddStream(MediaStream mediaStream) {
                super.onAddStream(mediaStream);
                VideoTrack remoteVideoTrack = mediaStream.videoTracks.get(0);
                runOnUiThread(() -> {
                    remoteVideoTrack.addSink(remoteView);
                });
            }
        });

        peerConnection.addStream(mediaStream);
    }

    private VideoCapturer createCameraCapturer(boolean isFront) {
        Camera1Enumerator enumerator = new Camera1Enumerator(false);
        final String[] deviceNames = enumerator.getDeviceNames();

        // First, try to find front facing camera
        for (String deviceName : deviceNames) {
            if (isFront ? enumerator.isFrontFacing(deviceName) : enumerator.isBackFacing(deviceName)) {
                VideoCapturer videoCapturer = enumerator.createCapturer(deviceName, null);

                if (videoCapturer != null) {
                    return videoCapturer;
                }
            }
        }

        return null;
    }

    @Override
    public void onCreateRoom() {

    }

    @Override
    public void onPeerJoined() {

    }

    @Override
    public void onSelfJoined() {
        peerConnection.createOffer(new SdpAdapter("local offer sdp") {
            @Override
            public void onCreateSuccess(SessionDescription sessionDescription) {
                super.onCreateSuccess(sessionDescription);
                peerConnection.setLocalDescription(new SdpAdapter("local set local"), sessionDescription);
                SignalingClient.get().sendSessionDescription(sessionDescription);
            }
        }, new MediaConstraints());
    }

    @Override
    public void onPeerLeave(String msg) {

    }

    @Override
    public void onOfferReceived(JSONObject data) {
        runOnUiThread(() -> {
            peerConnection.setRemoteDescription(new SdpAdapter("localSetRemote"),
                    new SessionDescription(SessionDescription.Type.OFFER, data.optString("sdp")));
            peerConnection.createAnswer(new SdpAdapter("localAnswerSdp") {
                @Override
                public void onCreateSuccess(SessionDescription sdp) {
                    super.onCreateSuccess(sdp);
                    peerConnection.setLocalDescription(new SdpAdapter("localSetLocal"), sdp);
                    SignalingClient.get().sendSessionDescription(sdp);
                }
            }, new MediaConstraints());

        });
    }

    @Override
    public void onAnswerReceived(JSONObject data) {
        peerConnection.setRemoteDescription(new SdpAdapter("localSetRemote"),
                new SessionDescription(SessionDescription.Type.ANSWER, data.optString("sdp")));
    }

    @Override
    public void onIceCandidateReceived(JSONObject data) {
        peerConnection.addIceCandidate(new IceCandidate(
                data.optString("id"),
                data.optInt("label"),
                data.optString("candidate")
        ));
    }

}