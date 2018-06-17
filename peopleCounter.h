//
// Created by Marta Galvan on 12/06/18.
//

#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    //void peopleCounter::drawHistograms(std::vector<cv::Mat> channels, const int &numBins, int &binWidth, std::vector<cv::Mat> &hist, cv::Mat &histImageB, cv::Mat &histImageG, cv::Mat &histImageR, int &histWidth, int &histHeight);
    void getHistogram();
    void backgroudSubtract(const std::string filename);


private:
    cv::Mat image;
    cv::Mat backg;

};


#endif //PROJECTCV_PEOPLECOUNTER_H
