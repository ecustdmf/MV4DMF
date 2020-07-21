#pragma once
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

extern int  getLabel(const int& x, const int& y);
extern void segment(const cv::Mat&);
extern void getLabelMatrix();