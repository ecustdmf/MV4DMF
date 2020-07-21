#pragma once
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


extern void _perspective(cv::Mat& img);
extern void selectAnchors(const cv::Mat&);
extern void getTransformMatrix();