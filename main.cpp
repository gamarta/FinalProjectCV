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

    Mat backImg = imread(imagesNames[0], CV_16U);
    string window_name_blob;
    string window_name_box;

    for (int i = 1; i < imagesNames.size(); ++i) {

        peopleCounter depth = peopleCounter(imagesNames[i]);
        Mat foreImg(backImg.size(), backImg.type());
        Mat histImg(backImg.size(), backImg.type());
        Mat binary(backImg.size(), backImg.type());
        Mat converted(backImg.size(), CV_8UC1);
        Mat blobs(backImg.size(), CV_8UC3);
        int nPeople;
        Mat Bcentre;

        depth.backgroudSubtract(backImg, foreImg);
        depth.thresholding(foreImg, binary);
        depth.blobDetection(binary, blobs, nPeople, foreImg, Bcentre);
        depth.drawBox(foreImg, Bcentre, nPeople);

        // Final view
        window_name_blob = "Blobs in " + imagesNames[i];
        namedWindow(window_name_blob);
        imshow(window_name_blob, blobs);
        //imwrite(window_name_blob, blobs);

        window_name_box = "Boxes in " + imagesNames[i];
        namedWindow(window_name_box);
        imshow(window_name_box, foreImg);
        waitKey(0);
        //imwrite(window_name_box, foreImg);
        destroyAllWindows();


    }

    return 0;

}