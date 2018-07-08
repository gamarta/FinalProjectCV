#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    void backgroudSubtract(const cv::Mat &background, cv::Mat &cleanForeground);
    void histEq(cv::Mat &cleanForeground);
    void thresholding(cv::Mat &cleanForeground, cv::Mat &cleanBinaryImg);
    void blobDetection(const cv::Mat &cleanBinaryImg, cv::Mat &colorBlobs, int &nComp, cv::Mat &centroids);
    void drawBox(cv::Mat &cleanForeground, const cv::Mat &centroids, const int &nComp);


private:
    cv::Mat image;

};


#endif //PROJECTCV_PEOPLECOUNTER_H