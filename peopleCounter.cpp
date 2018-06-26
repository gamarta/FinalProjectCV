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

    //cvtColor(image, image, CV_8UC2, 0);
    image = imread(filename);
    imshow("Original image ", image);
    //cvtColor(image, image, CV_BGR2GRAY);
    //imshow("Grayscale image ", image);

}

void peopleCounter::backgroudSubtract(Mat background, Mat cleanForeground) {

    Mat foreground;
    absdiff(image, background, foreground);

    double alpha = 2.0; /*< Simple contrast control */
    int beta = 40;       /*< Simple brightness control */

    Mat brightImg = Mat::zeros(image.size(), image.type());

    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                brightImg.at<Vec3b>(y,x)[c] =
                        saturate_cast<uchar>( alpha*( foreground.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }

    imshow("Foreground image", brightImg);

    cleanForeground = Mat::zeros(image.size(), image.type());
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    morphologyEx(brightImg, cleanForeground,MORPH_OPEN, kernel, Point(-1, -1), 2, BORDER_CONSTANT, 0);

    imshow("After morphological operations", cleanForeground);

}


void peopleCounter::histEqualization(Mat cleanForeground, Mat hist) {

    cout << "Channels image: " << cleanForeground.channels() << endl;

    // Set histogram bins count
    int numBins = 256;
    int histSize[] = {numBins};

    int const histWidth = 512;
    int const histHeight = 256;
    //int binWidth = cvRound( (double) histWidth/numBins );

    Mat hist( histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0) );

    // Set ranges for histogram bins
    float range[] = {0, 255};
    const float* ranges = {range};

    int channels[] = {0};

    // create matrix for histogram
    cv::Mat3b hist_image = cv::Mat3b::zeros(histHeight, numBins);

    calcHist( &cleanForeground, 1, channels, Mat(), hist, 1, histSize, &ranges, true, false );

    double max_val=0;
    minMaxLoc(hist, 0, &max_val);

    // visualize each bin
    for(int b = 0; b < numBins; b++) {

        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*histHeight/max_val);
        line( hist_image, cv::Point(b, histHeight-height), cv::Point(b, histHeight), Scalar::all(255));

    }

    imshow("Histogram", hist_image);


}

