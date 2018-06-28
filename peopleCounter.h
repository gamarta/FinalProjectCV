#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    void backgroudSubtract(cv::Mat background, cv::Mat cleanForeground);
    void thresholding(cv::Mat cleanForeground, cv::Mat binaryImg);
    void blobDetection(cv::Mat binaryImg);


private:
    cv::Mat image;

};


#endif //PROJECTCV_PEOPLECOUNTER_H