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
using namespace cuda;

peopleCounter::peopleCounter(string filename) {

    //cvtColor(image, image, CV_8UC2, 0);
    image = imread(filename);
    imshow("Original image ", image);
    cvtColor(image, image, CV_BGR2GRAY);
    imshow("Grayscale image ", image);

}

void peopleCounter::getHistogram() {

    // Set histogram bins count
    int bins = 256;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    normalize(hist, hist, 0, hist.rows, NORM_MINMAX, -1, Mat());

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
                ( hist_image
                        , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
                        , cv::Scalar::all(255)
                );
    }
    cv::imshow("Histogram", hist_image);

}


void peopleCounter::backgroudSubtract(const string filename) {
/*
    Mat mask;

    adaptiveThreshold(image, mask, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 33, 0);

    namedWindow("Foreground mask", CV_WINDOW_AUTOSIZE);
    imshow("Foreground mask", mask);*/



}


