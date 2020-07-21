#pragma once

#include <iostream>
#include "opencv2/core/core.hpp"

void manuDetect(cv::Mat &, cv::Rect &);

bool colorDetect(cv::Mat &, cv::Scalar, cv::Scalar, double, double, cv::Rect &);

bool circleDetect(cv::Mat &img, double min_r, double max_r, cv::Rect &bbox);


