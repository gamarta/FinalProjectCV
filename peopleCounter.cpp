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

void peopleCounter::backgroudSubtract(const Mat &background, Mat& cleanForeground, Mat &normImg) {

    // Background subtraction

    Mat foreground;
    absdiff(image, background, foreground);

    // Normalization

    double min, max;
    minMaxLoc(foreground, &min, &max, 0, 0, noArray());

    double alpha = double( ((uint16_t)-1) )/max;

     normImg= Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
                normImg.at<uint16_t>(y,x) =  alpha * foreground.at<uint16_t >(y,x);
        }
    }

    // Opening operation to clean the obtained foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(normImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

}

void peopleCounter::thresholding(Mat &cleanForeground, Mat &cleanBinaryImg, Mat &binaryImg) {

    // Binary Thresholding

    double min, max;
    minMaxLoc(cleanForeground, &min, &max, 0, 0, noArray());

    double thresh = max/2.1;
    double whitePix = 65535;

     binaryImg;
    threshold(cleanForeground, binaryImg, thresh, whitePix, THRESH_BINARY);

    // Opening operation to clean the binary image

    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
    morphologyEx(binaryImg, cleanBinaryImg, MORPH_OPEN, kernel, Point(-1, -1), 3, BORDER_CONSTANT, 0);

}

void peopleCounter::blobDetection(Mat &cleanBinaryImg, Mat &coloredBlobs, int &nComp, Mat &centroids) {

    // Conversion to 8 bit image

    cleanBinaryImg.convertTo(cleanBinaryImg, CV_8UC1, 1, 0);

    // Connected components computation

    Mat labels, stats;

    nComp = connectedComponentsWithStats(cleanBinaryImg, labels, stats, centroids);


    // Random color generation to fill connected components

    vector<Vec3b> colors(nComp+1);
    // First component will be the background (black colored)
    colors[0] = Vec3b(0, 0, 0);

    for(int i = 1; i <= nComp; i++) {
        // Define randomly colors vector
        colors[i] = Vec3b(rand()%256, rand()%256, rand()%256);

    }

    coloredBlobs = Mat::zeros(cleanBinaryImg.size(), CV_8UC3);
    for( int y = 0; y < coloredBlobs.rows; y++ ) {
        for (int x = 0; x < coloredBlobs.cols; x++) {

            int label = labels.at<int>(y, x);
            coloredBlobs.at<Vec3b>(y, x) = colors[label];

        }
    }

}


void peopleCounter::drawBox(Mat &cleanForeground, const Mat &centroids, const int &nComp) {

    // Draw a rectangle box bounding detected heads

    cv::cvtColor(cleanForeground, cleanForeground, COLOR_GRAY2BGR);

    for(int i = 1; i <= nComp-1; i++) {

        rectangle(cleanForeground, Point(centroids.at<double>(i,0)-50, centroids.at<double>(i,1)-50), Point(centroids.at<double>(i,0)+50, centroids.at<double>(i,1)+50), Scalar(0, 0, 65535), 3);

    }

    cout << "Total number of people detected: " << nComp-1 << endl;
}


