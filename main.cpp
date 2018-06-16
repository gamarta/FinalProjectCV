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

    const string background = imagesNames[0];

    cout << imagesNames[1] << endl;
    peopleCounter depth = peopleCounter(imagesNames[1]);
    depth.backgroudSubtract();






    waitKey(0);

    cout << "Done" << endl;
    return 0;

}