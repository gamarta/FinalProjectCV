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

    image = imread(filename, CV_16U);

}

void peopleCounter::backgroudSubtract(Mat background, Mat cleanForeground) {

    // Background subtraction

    Mat foreground;
    absdiff(image, background, foreground);

    // Gamma transform

    // alpha = 2^16 / MAX //2^8

    double min, max;
    minMaxLoc(foreground, &min, &max, 0, 0, noArray());

    double alpha = double(((uint16_t)-1)) /max;             /*< Simple contrast control */
    int beta = min;                                         /*< Simple brightness control */

    Mat brightImg = Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
                brightImg.at<uint16_t>(y,x) =  alpha * foreground.at<uint16_t >(y,x);
        }
    }

    // Opening operation to clean the foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(brightImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

}

void peopleCounter::thresholding(Mat cleanForeground, Mat cleanBinaryImg) {

    Mat binaryImg = Mat::zeros(image.size(), image.type());

    // Binary Threshold

    double min, max;
    minMaxLoc(cleanForeground, &min, &max, 0, 0, noArray());

    double thresh = max/2.06;
    double whitePix = 65535;    //to get white pixel in 16bit image

    threshold(cleanForeground, binaryImg, thresh, whitePix, THRESH_BINARY);

    // Opening operation to clean the binary image

    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
    morphologyEx(binaryImg, cleanBinaryImg, MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

}

void peopleCounter::blobDetection(Mat cleanBinaryImg, Mat convertedImg, Mat img_color, int nComp) {

    convertedImg = Mat::zeros(image.size(), CV_8UC1);
    cleanBinaryImg.convertTo(convertedImg, CV_8UC1, 1, 0);

    Mat labels, stats, centroids;
    int i;
    i, nComp = connectedComponentsWithStats(convertedImg, labels, stats, centroids);
    //cout << "Total Connected Components Detected: " << nComp << endl;

    vector<Vec3b> colors(nComp+1);
    colors[0] = Vec3b(0,0,0);                                       // background pixels remain black

    for( i = 1; i <= nComp; i++ ) {

        colors[i] = Vec3b(rand()%256, rand()%256, rand()%256);
        if( stats.at<int>(i-1, CC_STAT_AREA) < 100 )
            colors[i] = Vec3b(0,0,0);                               // small regions are painted with black too

    }

    img_color = Mat::zeros(convertedImg.size(), CV_8UC3);
    for( int y = 0; y < img_color.rows; y++ ) {

        for (int x = 0; x < img_color.cols; x++) {

            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nComp);
            img_color.at<Vec3b>(y, x) = colors[label];

        }
    }
}
