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

void drawHistograms( vector<Mat> channels, const int &numBins, int &binWidth, vector<Mat> &hist, Mat &histImageB, Mat &histImageG, Mat &histImageR, int &histWidth, int &histHeight ) {

    float range[] = {0, 255};
    const float* ranges = {range};

    calcHist( &channels[0], 1, 0, noArray(), hist[0], 1, &numBins, &ranges, true, false );
    calcHist( &channels[1], 1, 0, noArray(), hist[1], 1, &numBins, &ranges, true, false );
    calcHist( &channels[2], 1, 0, noArray(), hist[2], 1, &numBins, &ranges, true, false );

    // Normalization
    normalize( hist[0], hist[0], 0, histImageB.rows, NORM_MINMAX, -1, noArray() );
    normalize( hist[1], hist[1], 0, histImageG.rows, NORM_MINMAX, -1, noArray() );
    normalize( hist[2], hist[2], 0, histImageR.rows, NORM_MINMAX, -1, noArray() );

    // Draw the three histograms
    for( int i = 1; i < numBins; i++ ) {
        line( histImageB, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist[0].at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImageG, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist[1].at<float>(i)) ), Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImageR, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist[2].at<float>(i)) ), Scalar( 0, 0, 255), 2, 8, 0  );
    }
}

void peopleCounter::getHistogram() {

    //Split a multichannel array into multiple single-channel arrays ( B, G and R )
    vector <Mat> channels(3);
    split(image, channels);

    const int numBins = 256;

    // Get three black images
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound( (double) histWidth/numBins );

    // Display histograms
    const Scalar blackColor = Scalar( 0, 0, 0 );

    vector<Mat> hist(3);
    Mat histB( histHeight, histWidth, CV_8UC3, blackColor );
    Mat histG( histHeight, histWidth, CV_8UC3, blackColor );
    Mat histR( histHeight, histWidth, CV_8UC3, blackColor );

    drawHistograms( channels, numBins, binWidth, hist, histB, histG, histR, histWidth, histHeight );

    imshow("B", histB);
    imshow("G", histG);
    imshow("R", histR);

}

void peopleCounter::backgroudSubtract(const string filename) {

    Ptr<BackgroundSubtractorMOG2> sub = createBackgroundSubtractorMOG2(500, 16, true);
    Mat mask;

    sub->apply(image, mask, -1);
    namedWindow("Foreground mask", CV_WINDOW_AUTOSIZE);
    imshow("Foreground mask", mask);

}


