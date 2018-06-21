//
// Created by Marta Galvan on 12/06/18.
//

#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    cv::Mat getCanny(int thresh1, int thresh2);
    void getHoughCircles(cv::Mat &houghCimg, std::vector<cv::Vec3f> &circles, double thresh2, double th_hough, double dp, double minRad, double maxRad);
    cv::Mat getFinalImage(std::vector<cv::Vec3f> &circles);


private:
    cv::Mat image;

};


#endif //PROJECTCV_PEOPLECOUNTER_H
