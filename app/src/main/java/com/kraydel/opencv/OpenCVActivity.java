package com.kraydel.opencv;


import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.WindowManager;

import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import static org.opencv.core.CvType.CV_8UC1;

public class OpenCVActivity extends Activity
        implements CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier eyeCascadeClassifier,faceCascadeClassifier, upperBodyCascadeClassifier,fullBodyCascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteObjectSize,cascadeSelect = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    initializeOpenCVDependencies();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private void initializeOpenCVDependencies() {
        eyeCascadeClassifier = cascadeSetup("haarcascade_eye.xml");
        faceCascadeClassifier = cascadeSetup("lbpcascade_frontalface.xml");
        upperBodyCascadeClassifier = cascadeSetup("haarcascade_upperbody.xml");
        fullBodyCascadeClassifier = cascadeSetup("haarcascade_fullbody.xml");

        // And we are ready to go
        openCvCameraView.enableView();
    }

    private CascadeClassifier cascadeSetup(String fileName) {
        try {

            // Copy the resource into a temp file so OpenCV can load it
            Log.d("OpenCVActivity", "Initialising dependencies");
            InputStream is = getResources().getAssets().open(fileName);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            Log.d("OpenCVActivity", "cascadeDir " + cascadeDir); //Debug log message
            File mCascadeFile = new File(cascadeDir, fileName);
            FileOutputStream os = new FileOutputStream(mCascadeFile); // writes to the mCascadeFile

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            return new CascadeClassifier(mCascadeFile.getAbsolutePath());

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
            return  null;
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        openCvCameraView = new JavaCameraView(this, 98); //98 Front camera, 99 Back camera
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CV_8UC1);
        // The faces will be a 20% of the height of the screen
        absoluteObjectSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if(event.getAction() == MotionEvent.ACTION_DOWN ){
            if(cascadeSelect < 3){
                cascadeSelect++;
            }else{
                cascadeSelect = 0;
            }
        }
        return super.onTouchEvent(event);
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGB2GRAY);

        Core.flip(aInputFrame, grayscaleImage.t(), 1);

        switch(cascadeSelect){
            case 0: detectObjects(aInputFrame, faceCascadeClassifier, new Scalar(128,0,128, 255)); break;
            case 1: detectObjects(aInputFrame, faceCascadeClassifier, new Scalar(128,0,128, 255));
                    detectObjects(aInputFrame, eyeCascadeClassifier, new Scalar(0, 255, 0, 255)); break;
            case 2: detectObjects(aInputFrame, fullBodyCascadeClassifier, new Scalar(255, 0, 0, 255)); break;
            case 3: detectObjects(aInputFrame, upperBodyCascadeClassifier, new Scalar(255, 0, 0, 255));
        }
        return aInputFrame;
    }

    //Passes in the current image from the camera, and a CascadeClassifier (Which will detect a certain pattern passed in using the xml file)
    public void detectObjects(Mat aInputFrame, CascadeClassifier cascadeClassifier ,Scalar colour){
        MatOfRect objects = new MatOfRect();

        // Use the classifier to detect objects
        //Returns detected objects in objects array
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, objects, 1.1, 4, 2,
                    new Size(absoluteObjectSize, absoluteObjectSize), new Size());
        }

        // If there are any objects found, draw a rectangle around it
        Rect[] faceArray = objects.toArray();
        for (Rect anObjectArray : faceArray) {
            Imgproc.rectangle(aInputFrame, anObjectArray.tl(), anObjectArray.br(), colour, 3);
            //Draws a rectangle based off the array of Rects given in the objects array
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }
}