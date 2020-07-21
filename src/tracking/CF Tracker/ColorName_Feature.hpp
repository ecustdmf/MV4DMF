#pragma once
#include "opencv2/core.hpp"

class ColorName_Feature
{
public:
    static std::vector<cv::Mat> extract(const cv::Mat& patch_rgb);

private:
    inline static int rgb2id(int r, int g, int b);
    static const int p_cn_channels = 10;
    static float p_id2feat[32768][10];
};

