#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "peopleCounter.h"

#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace cv;

void read_directory(const string &name, vector<string> &files) {
    DIR *dir = opendir(name.c_str());
    struct dirent *dp;

    while ((dp = readdir(dir)) != NULL) {
        if (!strcmp(dp->d_name, "color0") || !strcmp(dp->d_name, "Depth-")) {}
        else {
            files.push_back(dp->d_name);
        }
    }

    closedir(dir);
}

int main() {

    // Read folders
/*
    const string folder = "./dataset/";
    vector<string> images;
    read_directory(folder, images);
*/
    const string images = "./dataset/Depth-*.png";
    vector<String> imagesNames;
    glob(images, imagesNames);

    cout << imagesNames.size() << endl;

    Mat backImg = imread(imagesNames[0], CV_16U);

    for (int i = 1; i < imagesNames.size(); ++i) {

        peopleCounter depth = peopleCounter(imagesNames[i]);
        Mat foreImg(backImg.size(), backImg.type());
        Mat histImg(backImg.size(), backImg.type());
        Mat binary(backImg.size(), backImg.type());
        Mat converted;

        depth.backgroudSubtract(backImg, foreImg);
        imshow("Foreground image " + imagesNames[i], foreImg);

        depth.thresholding(foreImg, binary);
        //imshow("Binary image " + imagesNames[i], binary);

        depth.blobDetection(binary, converted);
        imshow("Converted image " + imagesNames[i], converted);

    }


    waitKey(0);

    cout << "Done" << endl;
    return 0;

}