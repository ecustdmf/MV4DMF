#pragma once
#include "CF Tracker/KCF_Tracker.hpp"

namespace tracker {
    void init(cv::Mat&, cv::Rect&);
    void update(cv::Mat&, cv::Rect&);
}