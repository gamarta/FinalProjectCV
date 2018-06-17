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

    const int numBins = 64;
    //const int numBins = 256;

    // Get three black images
    //int histWidth = 512;
    //int histHeight = 400;
    int histWidth = 320;
    int histHeight = 240;
    int binWidth = cvRound((double) histWidth/numBins);

    // Display histograms
    const Scalar blackColor = Scalar(0, 0, 0);

    Mat hist( histHeight, histWidth, CV_8UC1, blackColor );

    float range[] = {0, 255};
    const float* ranges = {range};

    calcHist(&image, 1, 0, Mat(), hist, 1, &numBins, &ranges, true, false);

    // Normalization
    normalize(hist, hist, 0, hist.rows, NORM_MINMAX, -1, Mat());

    // Draw the three histograms
    for(int i = 1; i < numBins; i++) {

        line( hist, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist.at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
        //line(histImage, Point( bin_w*(i-1), hist_h - cvRound(depth_hist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(depth_hist.at<float>(i)) ), Scalar( 255), 2, 8, 0);

    }

    imshow("Histogram", hist);

}

void peopleCounter::backgroudSubtract(const string filename) {

    Ptr<BackgroundSubtractorMOG2> sub = createBackgroundSubtractorMOG2(500, 16, true);
    Mat mask;

    sub->apply(image, mask, -1);
    namedWindow("Foreground mask", CV_WINDOW_AUTOSIZE);
    imshow("Foreground mask", mask);

}


