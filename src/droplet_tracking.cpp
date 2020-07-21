#include "opencv2/imgproc/imgproc.hpp"

#include "droplet_tracking.h"
#include "camera/camera.h"
#include "preprocessing/preprocessing.h"
#include "detection/detect.h"
#include "tracking/tracking.h"

// globals
cv::Mat frame;
cv::Rect bbox;
int labels[5];  // center upper lower left right
std::string winname{ "Droplet Tracking" };

// functions
void openCamera(int cameraIndex, double width, double height, double fps) {
    camera::open(cameraIndex, width, height, fps);
    recorder::open();
}

void openVideo(char* path) {
    camera::openVideo(path);
    recorder::open();
}

void setCameraParams(double brightness, double contrast, double saturation, double hue, double gain, double exposure) {
    camera::setParams(brightness, contrast, saturation, hue, gain, exposure);
}

void imshow() {
    cv::rectangle(frame, bbox, cv::Scalar(50, 180, 20), 3, 8, 0);
    cv::imshow(winname, frame);
    cv::waitKey(1);
}

void initTracker() {
    camera::read(frame);
    preprocessing(frame);
    // manuDetect(frame, bbox);
    bool ret = circleDetect(frame, 25, 50, bbox);
    if(!ret) exit(-1);
    tracker::init(frame, bbox);
    imshow();
}

int* updateTracker() {
    if (! camera::read(frame)) return nullptr;
    perspective(frame);
    tracker::update(frame, bbox);
    getLabels(bbox, labels);
    imshow();
    return labels;
}

void closeTracker(){
    cv::destroyAllWindows();
    recorder::close();
    camera::close();
}