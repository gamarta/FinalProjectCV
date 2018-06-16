//
// Created by Marta Galvan on 12/06/18.
//

#ifndef PROJECTCV_PEOPLECOUNTER_H
#define PROJECTCV_PEOPLECOUNTER_H



class peopleCounter {

public:
    peopleCounter(const std::string filename);
    void processImages(char* fistFrameFilename);
    void backgroudSubtract();


private:
    cv::Mat image;
    cv::Mat backg;

};


#endif //PROJECTCV_PEOPLECOUNTER_H
