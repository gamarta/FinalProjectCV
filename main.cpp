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

    string window_name_bin, window_name_blob, window_name_box, window_name_thresh, window_name_back;
    string output_name_bin, output_name_blob, output_name_box, output_name_thresh, output_name_back;

    string window_name_norm, output_name_norm, window_name_b, output_name_b;

    for (int i = 1; i < imagesNames.size(); ++i) {

        peopleCounter depth = peopleCounter(imagesNames[i]);

        Mat foreImg, binImg, blobImg, Bcenters;
        int nPeople;

        Mat norm, bin;

        depth.backgroudSubtract(backImg, foreImg, norm);

        window_name_back = "Cleaned foreground image of " + imagesNames[i];
        namedWindow(window_name_back);
        imshow(window_name_back, foreImg);
        output_name_back = "./output/cleanedForeground" + to_string(i) + ".png";
        imwrite(output_name_back, foreImg);

        window_name_norm = "Foreground image of " + imagesNames[i];
        namedWindow(window_name_norm);
        imshow(window_name_norm, norm);
        output_name_norm = "./output/foreground" + to_string(i) + ".png";
        imwrite(output_name_norm, norm);

        depth.thresholding(foreImg, binImg, bin);

        window_name_bin = "Cleaned binary image of " + imagesNames[i];
        namedWindow(window_name_bin);
        imshow(window_name_bin, binImg);
        output_name_bin = "./output/cleanedBinary" + to_string(i) + ".png";
        imwrite(output_name_bin, binImg);

        window_name_b = "Binary image of " + imagesNames[i];
        namedWindow(window_name_b);
        imshow(window_name_b, bin);
        output_name_b = "./output/binary" + to_string(i) + ".png";
        imwrite(output_name_b, bin);

        depth.blobDetection(binImg, blobImg, nPeople, Bcenters);

        window_name_blob = "Connected components of " + imagesNames[i];
        namedWindow(window_name_blob);
        imshow(window_name_blob, blobImg);
        output_name_blob = "./output/blob" + to_string(i) + ".png";
        imwrite(output_name_blob, blobImg);

        depth.drawBox(foreImg, Bcenters, nPeople);

        window_name_box = "Heads detected in " + imagesNames[i];
        namedWindow(window_name_box);
        imshow(window_name_box, foreImg);
        output_name_box = "./output/final" + to_string(i) + ".png";
        imwrite(output_name_box, foreImg);


        // Final view





        waitKey(0);
        destroyAllWindows();

    }

    return 0;

}