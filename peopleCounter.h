#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    void backgroudSubtract(cv::Mat background, cv::Mat cleanForeground);
    //void histEqualization(cv::Mat cleanForeground, cv::Mat histogram);
    void getBinary(cv::Mat cleanForeground, cv::Mat binaryImg);


private:
    cv::Mat image;
    cv::Mat backg;

};


#endif //PROJECTCV_PEOPLECOUNTER_H