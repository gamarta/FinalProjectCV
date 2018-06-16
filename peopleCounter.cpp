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
    //cvtColor(image, image, CV_BGR2GRAY);
    //imshow("Grayscale image ", image);

}

void peopleCounter::backgroudSubtract() {

    Ptr<BackgroundSubtractorMOG2> sub = createBackgroundSubtractorMOG2(500, 16, true);
    Mat mask;

    sub->apply(image, mask, -1);
    namedWindow("Foreground mask", CV_WINDOW_AUTOSIZE);
    imshow("Foreground mask", mask);

}


