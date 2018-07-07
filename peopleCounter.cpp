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

    // Load input image in 8 bit

    image = imread(filename, CV_8U);

}

void peopleCounter::backgroudSubtract(const Mat &background, Mat& cleanForeground) {

    // Background subtraction

    Mat foreground;//(image.size(), image.type());
    absdiff(image, background, foreground);

    // Normalization

    double min, max;
    minMaxLoc(foreground, &min, &max, 0, 0, noArray());

    double alpha = double(((uint8_t)-1))/max;

    Mat brightImg= Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
                brightImg.at<uint8_t>(y,x) =  alpha * foreground.at<uint8_t >(y,x);
        }
    }

    // Opening operation to clean the obtained foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(brightImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);


}

void peopleCounter::thresholding(const Mat &cleanForeground, Mat &cleanBinaryImg) {

    Mat binaryImg;

    // Binary Threshold

    double min, max;
    minMaxLoc(cleanForeground, &min, &max, 0, 0, noArray());

    double thresh = 125;
    double whitePix = 256;

    threshold(cleanForeground, binaryImg, thresh, whitePix, THRESH_BINARY);

    // Opening operation to clean the binary image

    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
    morphologyEx(binaryImg, cleanBinaryImg, MORPH_OPEN, kernel, Point(-1, -1), 3, BORDER_CONSTANT, 0);

}

void peopleCounter::blobDetection(const Mat &cleanBinaryImg, Mat &colorBlobs, int &nComp, Mat &centroids) {

    Mat labels, stats;

    nComp = connectedComponentsWithStats(cleanBinaryImg, labels, stats, centroids);
    cout << "Total Connected Components Detected: " << nComp-1 << endl;

    vector<Vec3b> colors(nComp+1);
    colors[0] = Vec3b(0,0,0);                                           // background pixels remain black


    for(int i = 1; i <= nComp; i++) {

        colors[i] = Vec3b(rand()%256, rand()%256, rand()%256);
        if(stats.at<int>(i-1, CC_STAT_AREA) < 100) {
            colors[i] = Vec3b(0, 0, 0);                                 // small regions are painted with black
        }

    }

    colorBlobs = Mat::zeros(cleanBinaryImg.size(), CV_8UC3);
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

        rectangle(cleanForeground, Point(centroids.at<double>(i,0)-50, centroids.at<double>(i,1)-50), Point(centroids.at<double>(i,0)+50, centroids.at<double>(i,1)+50), Scalar(0, 0, 255), 3);

    }
}


