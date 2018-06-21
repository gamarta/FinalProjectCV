//
// Created by Marta Galvan on 21/06/18.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include "TrackbarsManager.h"
#include "peopleCounter.h"

using namespace std;
using namespace cv;

double TrackbarsManager::th1;
double TrackbarsManager::th2;

double TrackbarsManager::thHough;

double TrackbarsManager::circRes;
double TrackbarsManager::min_rad;
double TrackbarsManager::max_rad;


TrackbarsManager::TrackbarsManager(peopleCounter *sh){

    const string trackbarNameCa1 = "Threshold 1";
    const string trackbarNameCa2 = "Threshold 2";

    const string trackbarNameCircle1 = "Min Circle radius";
    const string trackbarNameCircle2 = "Max Circle radius";
    const string trackbarNameCircle3 = "Circle resolution";

    // Create trackbars for Canny image

    namedWindow("Canny image", WINDOW_AUTOSIZE);

    createTrackbar(trackbarNameCa1, "Canny image", NULL, maxSlider, trackbarCanny1, sh);
    createTrackbar(trackbarNameCa2, "Canny image", NULL, maxSlider, trackbarCanny2, sh);

    // Create trackbars for Hough circles transform

    namedWindow("Hough Circles", WINDOW_AUTOSIZE);

    createTrackbar(trackbarNameCircle1, "Hough Circles", NULL, maxSlider, trackbarHoughCircle1, sh);
    createTrackbar(trackbarNameCircle2, "Hough Circles", NULL, maxSlider, trackbarHoughCircle2, sh);
    createTrackbar(trackbarNameCircle3, "Hough Circles", NULL, maxSlider, trackbarHoughCircle3, sh);

    //Set the default positions of trackbars

    setTrackbarPos(trackbarNameCa1, "Canny image", 53);
    setTrackbarPos(trackbarNameCa2, "Canny image", 44);

    setTrackbarPos(trackbarNameCircle1, "Hough Circles", 18);
    setTrackbarPos(trackbarNameCircle2, "Hough Circles", 29);
    setTrackbarPos(trackbarNameCircle3, "Hough Circles", 86);

    th1 = 712;
    th2 = 658;

    circRes = 3.58;
    min_rad = 2.08;
    max_rad = 10.03;


    // Final image

    namedWindow("Final image", WINDOW_AUTOSIZE);

    createImage(sh);

}

// Callback functions of all trackbars

void TrackbarsManager::trackbarCanny1(int value, void *obj) {
    double ratio = value/ (double) maxSlider;
    th1 = (thresh12_max - thresh12_min) * ratio + thresh12_min;
    createImage(obj);
}

void TrackbarsManager::trackbarCanny2(int value, void *obj) {
    double ratio = value/ (double) maxSlider;
    th2 = (thresh12_max - thresh12_min) * ratio + thresh12_min;
    createImage(obj);
}

void TrackbarsManager::trackbarHoughLines3(int value, void *obj){
    double ratio = value/ (double) maxSlider;
    thHough = (thr_hough_max - thr_hough_min) * ratio + thr_hough_min;
    createImage(obj);
}

void TrackbarsManager::trackbarHoughCircle1(int value, void *obj) {
    double ratio = value/ (double) maxSlider;
    min_rad = (min_rad_max - min_rad_min) * ratio + min_rad_min;
    createImage(obj);
}

void TrackbarsManager::trackbarHoughCircle2(int value, void *obj) {
    double ratio = value/ (double) maxSlider;
    max_rad = (max_rad_max - max_rad_min) * ratio + max_rad_min;
    createImage(obj);
}

void TrackbarsManager::trackbarHoughCircle3(int value, void *obj) {
    double ratio = value/ (double) maxSlider;
    circRes = (circRes_max - circRes_min) * ratio + circRes_min;
    createImage(obj);
}

void TrackbarsManager::createImage(void *obj) {

    cout << "Final parameters are: " << endl;

    cout << "Threshold 1: " << th1 <<endl;
    cout << "Threshold 2: " << th2 <<endl;

    cout << "Hough Threshold: " << thHough <<endl;

    cout << "Circle resolution: " << circRes <<endl;
    cout << "Min Circle radius: " << min_rad <<endl;
    cout << "Max Circle radius: " << max_rad <<endl;

    peopleCounter *sh = (peopleCounter *) obj;

    // Get Canny image

    Mat cannyImg = sh->getCanny(th1, th2);
    imshow("Canny image",cannyImg );
    //imwrite( "./images/cannyImage.png", cannyImg );


    // Get Hough Transform for circles

    Mat houghCimg;
    vector<Vec3f> houghCircles;
    sh->getHoughCircles(houghCimg, houghCircles, th2, thHough, circRes, min_rad, max_rad);

    imshow("Hough Circles", houghCimg);
    //imwrite( "./images/houghCircleImage.png", houghCimg );

    // Get the final image

    Mat finalImg = sh->getFinalImage(houghCircles);
    imshow("Final image", finalImg);
    imwrite( "./images/finalImage.png", finalImg );


}



