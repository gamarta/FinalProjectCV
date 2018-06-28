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
    imshow("Original image ", image);

    cout << "Channels original depth image: " << image.channels() << endl;


}

void peopleCounter::backgroudSubtract(Mat background, Mat cleanForeground) {

    // Background subtraction

    Mat foreground;
    absdiff(image, background, foreground);

    // Alpha transform

    // alpha = 2^16 / MAX //2^8

    double min, max;
    Point min_loc, max_loc;
    minMaxLoc(foreground, &min, &max, &min_loc, &max_loc);

    double alpha = double(((uint16_t)-1)) /max;         /*< Simple contrast control */
    int beta = 0;                                       /*< Simple brightness control */

            cout << alpha << endl;

    Mat brightImg = Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                brightImg.at<uint16_t>(y,x) =  alpha*( foreground.at<uint16_t >(y,x) );
            }
        }
    }

    imshow("Foreground image", brightImg);

    // Opening operation to clean the foreground image

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(brightImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);
    imshow("Foreground cleaned image", cleanForeground);

    //cout << cleanForeground <<endl;

}

void getBinary(Mat cleanForeground, Mat binaryImg) {

    threshold(cleanForeground, binaryImg, 200, 1, THRESH_BINARY);
    imshow("Binary image", binaryImg);

}

