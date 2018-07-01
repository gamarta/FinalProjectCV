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
    //imshow("Original image", image);

    //cout << "Channels original depth image: " << image.channels() << endl;


}

void peopleCounter::backgroudSubtract(Mat background, Mat cleanForeground) {

    // Background subtraction

    Mat foreground;
    absdiff(image, background, foreground);

    // Alpha transform

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

    //imshow("Foreground image" + , brightImg);

    // Opening operation to clean the foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(brightImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);
    //imshow("Foreground cleaned image", cleanForeground);

    //cout << cleanForeground <<endl;

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

    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
    morphologyEx(binaryImg, cleanBinaryImg, MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

    //imshow("Binary image", binaryImg);

}

void peopleCounter::blobDetection(Mat cleanBinaryImg, Mat convertedImg) {

    convertedImg = Mat::zeros(image.size(), CV_8UC1);
    cleanBinaryImg.convertTo(convertedImg, CV_8UC1, 1, 0);

/*

    Mat blobs = Mat::zeros(image.size(), image.type());
    Mat stats, centroids;
    connectedComponentsWithStats(binaryImg, blobs, stats, centroids, 8, CV_16U);
    // Show blobs
    imshow("Blobs image", blobs );


    Mat output = Mat::zeros(image.size(), image.type());
    vector <vector<Point2i>> blobs;

    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binaryImg.convertTo(label_image, CV_32SC1);

    int label_count = 2;        // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }


    // Randomly color the blobs

    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));

        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;

            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }

    imshow("Labelled blobs", output);
*/
}

