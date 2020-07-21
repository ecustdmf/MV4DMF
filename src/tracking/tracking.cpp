#include "tracking.h"
#include "CF Tracker/utils.hpp"

KCF_Tracker t;

void tracker::init(cv::Mat& frame, cv::Rect& bbox) {
    t.use_colorname = true;
    t.use_gray = true;
    t.init(frame, bbox);
}

void tracker::update(cv::Mat& frame, cv::Rect& bbox) {
    t.track(frame);
    bbox = t.getBBox();
}


