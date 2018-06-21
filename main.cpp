#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "peopleCounter.h"
#include "TrackbarsManager.h"

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
    vector<string> imagesNames;

    read_directory(folder, imagesNames);

*/
    const string images = "./dataset/color*.png";
    vector<String> imagesNames;
    glob(images, imagesNames);

    //peopleCounter depth = peopleCounter(imagesNames[1]);

    peopleCounter *sh = new peopleCounter(imagesNames[1]);

    TrackbarsManager *track = new TrackbarsManager(sh);

    waitKey(0);

    cout << "Done" << endl;
    return 0;

}