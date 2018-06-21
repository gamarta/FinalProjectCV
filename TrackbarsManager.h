//
// Created by Marta Galvan on 21/06/18.
//

#ifndef FINALPROJECTCV_TRACKBARSMANAGER_H
#define FINALPROJECTCV_TRACKBARSMANAGER_H

#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "peopleCounter.h"


#define maxSlider 100

//Canny image
#define thresh12_min 400
#define thresh12_max 1000

#define thr_hough_min 30
#define thr_hough_max 90

//Hough circles transform
#define circRes_min 1
#define circRes_max 4

#define min_rad_min 1
#define min_rad_max 7

#define max_rad_min 8
#define max_rad_max 15


class TrackbarsManager {


private:

    static double th1;
    static double th2;

    static double thHough;

    static double circRes;
    static double min_rad;
    static double max_rad;

    static void trackbarCanny1(int value, void *obj);
    static void trackbarCanny2(int value, void *obj);

    static void trackbarHoughLines3(int value, void *obj);

    static void trackbarHoughCircle1(int value, void *obj);
    static void trackbarHoughCircle2(int value, void *obj);
    static void trackbarHoughCircle3(int value, void *obj);

public:

    TrackbarsManager(peopleCounter *sh);
    static void createImage(void *obj);


};

#endif //FINALPROJECTCV_TRACKBARSMANAGER_H


