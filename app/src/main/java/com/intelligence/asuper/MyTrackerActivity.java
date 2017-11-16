package com.intelligence.asuper;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.RelativeLayout;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MyTrackerActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifier;
    //图像人脸小于高度的多少就不检测
    private int absoluteFaceSize;
    //临时图像对象
    private Mat matLin;
    //最终图像对象
    private Mat mat;

    //前置摄像头
    public static int CAMERA_FRONT = 0;
    //后置摄像头
    public static int CAMERA_BACK = 1;

    private int camera_scene = CAMERA_BACK;

    private void initializeOpenCVDependencies() {
        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();


            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }


        // And we are ready to go
        openCvCameraView.enableView();
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_my_tracker);

        final RelativeLayout relativeLayout = (RelativeLayout) findViewById(R.id.relative);

        openCvCameraView = new JavaCameraView(this, CameraBridgeViewBase.CAMERA_ID_FRONT);
        openCvCameraView.setCvCameraViewListener(this);

        final Button button = new Button(MyTrackerActivity.this);
        button.setText("切换摄像头");
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (camera_scene == CAMERA_FRONT) {//如果是前置摄像头就切换成后置

                    relativeLayout.removeAllViews();
                    openCvCameraView.disableView();
                    openCvCameraView = null;
                    cascadeClassifier = null;

                    openCvCameraView = new JavaCameraView(MyTrackerActivity.this, CameraBridgeViewBase.CAMERA_ID_BACK);
                    openCvCameraView.setCvCameraViewListener(MyTrackerActivity.this);

                    openCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);//后置摄像头
                    camera_scene = CAMERA_BACK;

                    relativeLayout.addView(openCvCameraView);
                    relativeLayout.addView(button);

                    initializeOpenCVDependencies();
                } else {

                    relativeLayout.removeAllViews();
                    openCvCameraView.disableView();
                    openCvCameraView = null;
                    cascadeClassifier = null;

                    openCvCameraView = new JavaCameraView(MyTrackerActivity.this, CameraBridgeViewBase.CAMERA_ID_FRONT);
                    openCvCameraView.setCvCameraViewListener(MyTrackerActivity.this);

                    openCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);//前置摄像头
                    camera_scene = CAMERA_FRONT;

                    relativeLayout.addView(openCvCameraView);
                    relativeLayout.addView(button);

                    initializeOpenCVDependencies();
                }
            }
        });

        relativeLayout.addView(openCvCameraView);
        relativeLayout.addView(button);

        if (camera_scene == CAMERA_FRONT) {
            openCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);//前置摄像头
        } else if (camera_scene == CAMERA_BACK) {
            openCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);//后置摄像头
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matLin = new Mat(height, width, CvType.CV_8UC4);//临时图像

        // 人脸小于高度的百分之30就不检测
        absoluteFaceSize = (int) (height * 0.1);
    }


    @Override
    public void onCameraViewStopped() {
    }


    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        //转置函数,将图像翻转（顺时针90度）
        Core.transpose(aInputFrame, matLin);

        if (camera_scene == CAMERA_FRONT) {//前置摄像头
            //转置函数,将图像翻转（对换）
            Core.flip(matLin, aInputFrame, 1);
            //转置函数,将图像顺时针顺转（对换）
            Core.flip(aInputFrame, matLin, 0);
            mat = matLin;
        } else if (camera_scene == CAMERA_BACK) {//后置摄像头
            //转置函数,将图像翻转（对换）
            Core.flip(matLin, aInputFrame, 1);
            mat = aInputFrame;
        }
        MatOfRect faces = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(mat, faces, 1.1, 1, 1, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        // 检测出多少个
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Log.i("123456", "facesArray[i].tl()坐上坐标 == " + facesArray[i].tl() + "      facesArray[i].br() == 右下坐标" + facesArray[i].br());
            Imgproc.rectangle(mat, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        }
        return mat;
    }


    @Override
    public void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.e("log_wons", "OpenCV init error");
            // Handle initialization error
        }
        initializeOpenCVDependencies();
        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }
}