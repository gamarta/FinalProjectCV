//
// Created by Marta Galvan on 12/06/18.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>
#include "peopleCounter.h"

#include <sstream>

#include <string.h>
#include <histedit.h>

using namespace std;
using namespace cv;

peopleCounter::peopleCounter(string filename) {

    //cvtColor(image, image, CV_8UC2, 0);
    image = imread(filename);
    imshow("Original image ", image);
    cvtColor(image, image, CV_BGR2GRAY);
    imshow("Grayscale image ", image);

}

Mat peopleCounter::getCanny(int thresh1, int thresh2){

    Mat cannyImg;
    Canny(image, cannyImg, thresh1, thresh2);
    //imshow("Canny image", cannyImg);
    return cannyImg;

}

void peopleCounter::getHoughCircles(Mat &houghCimg, vector<Vec3f> &circles, double thresh2, double th_hough, double dp, double minRad, double maxRad) {

    HoughCircles(image, circles, HOUGH_GRADIENT, dp, image.cols/2, thresh2, th_hough, minRad, maxRad);
    houghCimg = image.clone();

    // Draw the circles
    for( size_t i = 0; i < circles.size(); i++){
        circle(houghCimg, Point(cvRound(circles[i][0]), cvRound(circles[i][1])), cvRound(circles[i][2]), Scalar(0,255,0), -1, CV_AA);
    }

}

Mat peopleCounter::getFinalImage(vector<Vec3f> &circles) {

    Mat finalImg = image.clone();

        // Draw the circles

        for( size_t i = 0; i < circles.size(); i++){
            circle(finalImg, Point(cvRound(circles[i][0]), cvRound(circles[i][1])), cvRound(circles[i][2]), Scalar(0,255,0), -1, CV_AA);
        }

        return finalImg;

}


