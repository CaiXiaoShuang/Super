package com.intelligence.asuper;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

import static android.content.ContentValues.TAG;

public class MyTrackerActivity extends Activity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2 {
    // 公共模块
    private CameraBridgeViewBase mOpenCvCameraView;
    private Scalar mColor;
    private Mat mRgba;
    // 跟踪模块
    private boolean mIsColorSelected = false;
    private int winWidth = 130;
    private MatOfFloat rangesH, rangesS, rangesV;// HSV限定值范围
    private MatOfInt chansH, chansS, chansV;// 分别代表h、s、v通道
    private int H_Min = 0, H_Max = 180;
    private int S_Min = 0, S_Max = 180;
    private int V_Min = 0, V_Max = 180;
    private Mat mHist;
    private RotatedRect mRect;
    private Mat mHue;
    private Rect mTrackWindow;
    private Mat mHueImage;
    private List<Mat> planes;
    private MatOfInt histSize;
    private Mat backproject;
    private ArrayList<Mat> images;
    private MatOfInt moi;
    private MatOfFloat mof;
    private TermCriteria term;
    private SeekBar HminBar, HmaxBar;
    private SeekBar SminBar, SmaxBar;
    private SeekBar VminBar, VmaxBar;
    private SeekBar trackBox;
    // 转换摄像头
    private boolean chgCam = false;
    private Button chg_camBtn;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MyTrackerActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        if (mIsColorSelected) {
            // 获取hsv颜色空间
            if (mHueImage == null)
                mHueImage = new Mat(mRgba.size(), mRgba.type());
            Imgproc.cvtColor(mRgba, mHueImage, Imgproc.COLOR_RGB2HSV);
            // 获取hsv中的hue分量
            Core.extractChannel(mHueImage, mHue, 0);
            rangesH = new MatOfFloat(H_Min, H_Max);
            rangesS = new MatOfFloat(S_Min, S_Max);
            rangesV = new MatOfFloat(V_Min, V_Max);
            planes.add(mHueImage);
            // 得到直方图
            // Imgproc.calcHist(planes, chans0, new Mat(), mHist, histSize,
            // ranges);
            Imgproc.calcHist(planes, chansH, new Mat(), mHist, histSize,
                    rangesH);
            Imgproc.calcHist(planes, chansS, new Mat(), mHist, histSize,
                    rangesS);
            Imgproc.calcHist(planes, chansV, new Mat(), mHist, histSize,
                    rangesV);
            // 直方图标准化
            Core.normalize(mHist, mHist, 0, 255, Core.NORM_MINMAX);
            // 反向投影图
            images.add(mHue);
            Imgproc.calcBackProject(images, moi, mHist, backproject, mof, 1.0);
            // 获取跟踪框并用椭圆轮廓画出来
            mRect = Video.CamShift(backproject, mTrackWindow, term);
            Imgproc.ellipse(mRgba, mRect, mColor);
            // 通知系统释放内存,但不是实时的，所以并没什么用？
            System.gc();
        } else {
            mHue = new Mat(mRgba.size(), CvType.CV_8UC1);
            backproject = mHue;
        }
        if (mRect != null) {
            int y = (int) mRect.center.y - winWidth / 2;
            int x = (int) mRect.center.x - winWidth / 2;
            mTrackWindow.x = x;
            mTrackWindow.y = y;
            mTrackWindow.width = winWidth;
            mTrackWindow.height = winWidth;
        }
        return mRgba;
    }
    public void onCameraViewStarted(int width, int height) {
        // 公共模块
        mColor = new Scalar(255, 0, 0);
        // 跟踪模块
        mTrackWindow = new Rect();
        mHist = new Mat();
        planes = new ArrayList<Mat>();
        // 设置hsv通道
        chansH = new MatOfInt(0);
        chansS = new MatOfInt(1);
        chansV = new MatOfInt(2);
        histSize = new MatOfInt(10);
        images = new ArrayList<Mat>();
        // 用H分量作为跟踪对象.其余两个做限定条件提高跟踪准确率
        moi = chansH;
        mof = new MatOfFloat(0, 180);
        term = new TermCriteria(TermCriteria.COUNT, 200, 1);
    }
    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;
        int x = (int) event.getX() - xOffset;
        int y = (int) event.getY() - yOffset;
        if ((x < 0) || (y < 0) || (x > cols) || (y > rows))
            return false;
        mTrackWindow = new Rect(x - winWidth / 2, y - winWidth / 2, winWidth,
                winWidth);
        mIsColorSelected = true;
        return false; // don't need subsequent touch events
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_my_tracker);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        // mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        // 获取滚动条,通过设置滑动条监听改变v通道值的范围
        MyseekbarListener sbListen = new MyseekbarListener();
        // H
        HminBar = (SeekBar) findViewById(R.id.HseekBarMin);
        HmaxBar = (SeekBar) findViewById(R.id.HseekBarMax);
        HminBar.setOnSeekBarChangeListener(sbListen);
        HmaxBar.setOnSeekBarChangeListener(sbListen);
        // S
        SminBar = (SeekBar) findViewById(R.id.SseekBarMin);
        SmaxBar = (SeekBar) findViewById(R.id.SseekBarMax);
        SminBar.setOnSeekBarChangeListener(sbListen);
        SmaxBar.setOnSeekBarChangeListener(sbListen);
        // V
        VminBar = (SeekBar) findViewById(R.id.VseekBarMin);
        VmaxBar = (SeekBar) findViewById(R.id.VseekBarMax);
        VminBar.setOnSeekBarChangeListener(sbListen);
        VmaxBar.setOnSeekBarChangeListener(sbListen);
        // 可设置跟踪框的大小
        trackBox = (SeekBar) findViewById(R.id.seekBarTracBox);
        trackBox.setOnSeekBarChangeListener(sbListen);
        // 摄像头转换按钮
        MyOnClickListen click = new MyOnClickListen();
        chg_camBtn = (Button) findViewById(R.id.deal_btn);
        chg_camBtn.setOnClickListener(click);
    }
    // 按钮监听器
    class MyOnClickListen implements View.OnClickListener {
        @Override
        public void onClick(View v) {
            if (v.equals(chg_camBtn)) {
                if (chgCam) {
                    mOpenCvCameraView
                            .setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                } else {
                    mOpenCvCameraView
                            .setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                }
                chgCam = !chgCam;
            }
        }
    }
    // 创建一个滑动条监听内部类
    class MyseekbarListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onProgressChanged(SeekBar sb, int progress, boolean arg2) {
            // H
            if (sb.equals(HminBar)) {
                if (progress < H_Max)
                    H_Min = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "H分量设置错误:H_Min必须小于H_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            if (sb.equals(HmaxBar)) {
                if (progress > H_Min)
                    H_Max = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "H分量设置错误:H_Min必须小于H_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            // S
            if (sb.equals(SminBar)) {
                if (progress < S_Max)
                    S_Min = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "S分量设置错误:S_Min必须小于S_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            if (sb.equals(SmaxBar)) {
                if (progress > S_Min)
                    S_Max = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "S分量设置错误:S_Min必须小于S_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            // V
            if (sb.equals(VminBar)) {
                if (progress < V_Max)
                    V_Min = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "V分量设置错误:V_Min必须小于V_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            if (sb.equals(VmaxBar)) {
                if (progress > V_Min)
                    V_Max = progress;
                else {
                    Toast.makeText(getApplicationContext(),
                            "V分量设置错误:V_Min必须小于V_Max", Toast.LENGTH_SHORT)
                            .show();
                }
            }
            if (sb.equals(trackBox)) {
                if (progress > 0)
                    winWidth = progress;
            }
        }
        @Override
        public void onStartTrackingTouch(SeekBar arg0) {
        }
        @Override
        public void onStopTrackingTouch(SeekBar arg0) {
        }
    }
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV library not found!");
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    public void onCameraViewStopped() {
        mRgba.release();
    }
}