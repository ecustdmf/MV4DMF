#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/droplet_tracking.h"


int main(int argc, char *argv[]) {
    char path[] = "../videos/blank-300ms.mp4";
    openVideo(path);
    //openCamera();  

    initTracker();

    int *labels;
    while (true) {
        labels = updateTracker();
        if (!labels) break;
        for (int i = 0; i < 5; ++i) {
            std::cout << labels[i] << ", ";
        }
        std::cout << std::endl;
        char key = cv::waitKey(1);
        if (key == 27) break;
    }
    closeTracker();

    return 0;
}