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

    string window_name_bin, window_name_blob, window_name_box, window_name_tresh;

    for (int i = 1; i < imagesNames.size(); ++i) {

        peopleCounter depth = peopleCounter(imagesNames[i]);

        Mat foreImg, binary, blobs, Bcenters;
        int nPeople;

        depth.backgroudSubtract(backImg, foreImg);
        //depth.histEq(foreImg);
        depth.thresholding(foreImg, binary);
        depth.blobDetection(binary, blobs, nPeople, Bcenters);
        depth.drawBox(foreImg, Bcenters, nPeople);


        // Final view

        window_name_bin = "Binary image of " + imagesNames[i];
        namedWindow(window_name_bin);
        imshow(window_name_bin, binary);
        //imwrite(window_name_bin, binary);


        window_name_blob = "Blob image of " + imagesNames[i];
        namedWindow(window_name_blob);
        imshow(window_name_blob, blobs);
        //imwrite(window_name_blob, blobs);

        window_name_box = "Heads detected in " + imagesNames[i];
        namedWindow(window_name_box);
        imshow(window_name_box, foreImg);
        //imwrite(window_name_box, foreImg);

        waitKey(0);
        destroyAllWindows();

    }

    return 0;

}