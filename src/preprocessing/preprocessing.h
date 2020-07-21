#pragma once
#include <iostream>
#include "opencv2/core/core.hpp"

extern void preprocessing(cv::Mat& frame, bool existBG=true);
extern void perspective(cv::Mat&);
extern void getLabels(cv::Rect&, int*);
