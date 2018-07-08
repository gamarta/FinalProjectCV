#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>
#include "peopleCounter.h"

#include <iostream>
#include <sstream>
#include <string.h>
#include <histedit.h>

using namespace std;
using namespace cv;

peopleCounter::peopleCounter(string filename) {

    // Load input image
    image = imread(filename, CV_16U);

}

void peopleCounter::backgroudSubtract(const Mat &background, Mat& cleanForeground) {

    // Background subtraction

    Mat foreground;//(image.size(), image.type());
    absdiff(image, background, foreground);

    // Normalization

    double min, max;
    minMaxLoc(foreground, &min, &max, 0, 0, noArray());

    cout << "MAX: " << max << endl;

    double alpha = double( ((uint16_t)-1) )/max;
    double beta = (pow(2,16)-1)/max;

    cout << "alpha: " << alpha << endl;
    cout << "beta: " << beta << endl;

    Mat normImg= Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
                normImg.at<uint16_t>(y,x) =  alpha * foreground.at<uint16_t >(y,x);
        }
    }

    double min2, max2;
    minMaxLoc(normImg, &min2, &max2, 0, 0, noArray());

    cout << "MAX2: " << max2 << endl;

    // Opening operation to clean the obtained foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(normImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

}
/*
void peopleCounter::histEq(Mat &cleanForeground) {

    //Split a multichannel array into multiple single-channel arrays ( B, G and R )

    vector <Mat> channels(1);
    split(cleanForeground, channels);

    const int numBins = 256;
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double) histWidth/numBins);
    float range[] = {0, 255};
    const float* ranges = {range};

    // Display histograms

    vector<Mat> hist(1);
    Mat histG( histHeight, histWidth, CV_8UC1, Scalar(0, 0, 0) );

    calcHist( &channels[0], 1, 0, noArray(), hist[0], 1, &numBins, &ranges, true, false );

    // Normalization
    normalize( hist[0], hist[0], 0, histG.rows, NORM_MINMAX, -1, noArray() );
    // Draw the three histograms
    for( int i = 1; i < numBins; i++ ) {
        line( histG, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist[0].at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
    }

    imshow("Histogram", histG);

    vector<Mat> equal(1);
    equalizeHist(channels[0], equal[0]);
    vector<Mat> equalHist(1);
    Mat histEqG( histHeight, histWidth, CV_8UC1, Scalar(0, 0, 0) );
    // Draw the three histograms
    for( int i = 1; i < numBins; i++ ) {
        line( histEqG, Point( binWidth*(i-1), histHeight ) , Point( binWidth*(i), histHeight - cvRound(hist[0].at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
    }

    imshow("Equalized histogram", histEqG);

}*/

void peopleCounter::thresholding(Mat &cleanForeground, Mat &cleanBinaryImg) {

    // Binary Threshold

    double min, max;
    minMaxLoc(cleanForeground, &min, &max, 0, 0, noArray());
    double thresh = max/2.1;
    double whitePix = 65535;                //to get white pixel in 16bit image

    Mat binaryImg;
    threshold(cleanForeground, binaryImg, thresh, whitePix, THRESH_BINARY);

    // Opening operation to clean the binary image

    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
    morphologyEx(binaryImg, cleanBinaryImg, MORPH_OPEN, kernel, Point(-1, -1), 3, BORDER_CONSTANT, 0);

}

void peopleCounter::blobDetection(const Mat &cleanBinaryImg, Mat &colorBlobs, int &nComp, Mat &centroids) {

    Mat convertedImg; // = Mat::zeros(image.size(), CV_8UC1);
    cleanBinaryImg.convertTo(convertedImg, CV_8UC1, 1, 0);

    Mat labels, stats;

    nComp = connectedComponentsWithStats(convertedImg, labels, stats, centroids);
    cout << "Total Connected Components Detected: " << nComp-1 << endl;

    vector<Vec3b> colors(nComp+1);
    colors[0] = Vec3b(0,0,0);                                           // background pixels remain black


    for(int i = 1; i <= nComp; i++) {

        colors[i] = Vec3b(rand()%256, rand()%256, rand()%256);
        if(stats.at<int>(i-1, CC_STAT_AREA) < 100) {
            colors[i] = Vec3b(0, 0, 0);                                 // small regions are painted with black
        }

    }

    colorBlobs = Mat::zeros(convertedImg.size(), CV_8UC3);
    for( int y = 0; y < colorBlobs.rows; y++ ) {

        for (int x = 0; x < colorBlobs.cols; x++) {

            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nComp);
            colorBlobs.at<Vec3b>(y, x) = colors[label];

        }
    }

}


void peopleCounter::drawBox(Mat &cleanForeground, const Mat &centroids, const int &nComp) {

    // Return rectangle bounding the points

    cv::cvtColor(cleanForeground, cleanForeground, COLOR_GRAY2BGR);

    for(int i = 1; i <= nComp-1; i++) {

        rectangle(cleanForeground, Point(centroids.at<double>(i,0)-50, centroids.at<double>(i,1)-50), Point(centroids.at<double>(i,0)+50, centroids.at<double>(i,1)+50), Scalar(0, 0, 65535), 3);

    }
}


