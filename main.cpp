#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "peopleCounter.h"

#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace cv;

int main() {

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
        Mat converted(backImg.size(), CV_8UC1);
        Mat blobs(backImg.size(), CV_8UC3);
        int nPeople;

        depth.backgroudSubtract(backImg, foreImg);
        imshow("Foreground image " + imagesNames[i], foreImg);
        //imwrite("./output/foreground" + imagesNames[i] + ".png", foreImg);

        depth.thresholding(foreImg, binary);

        depth.blobDetection(binary, converted, blobs, nPeople);
        //cout << "Total Connected Components in image " + imagesNames[i] + ": " << to_string(nPeople) << endl;
        imshow("Heads detected " + imagesNames[i], blobs);
        //imwrite("./output/blobs" + imagesNames[i] + ".png", blobs);


    }

    waitKey(0);
    return 0;

}