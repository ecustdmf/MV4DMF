#include "detect.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

void manuDetect(cv::Mat &frame, cv::Rect &bbox) {
    bbox = cv::selectROI(frame);
    cv::destroyAllWindows();
}

bool colorDetect(cv::Mat &img, cv::Scalar lower_thres, cv::Scalar upper_thres, double min_r, double max_r,
                 cv::Rect &bbox) {
    cv::Mat imhsv, imbin;
    cv::cvtColor(img, imhsv, cv::COLOR_BGR2HSV);
    cv::inRange(imhsv, lower_thres, upper_thres, imbin);
    cv::Mat SE = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(imbin, imbin, cv::MORPH_OPEN, SE);
    cv::morphologyEx(imbin, imbin, cv::MORPH_CLOSE, SE);

    vector<vector<cv::Point>> contours;
    cv::findContours(imbin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double min_area = min_r * min_r * CV_PI;
    double max_area = max_r * max_r * CV_PI;
    for (auto contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= min_area && area <= max_area) {
            bbox = cv::boundingRect(contour);
            return true;
        }
    }
    return false;
}

bool circleDetect(cv::Mat &img, double min_r, double max_r, cv::Rect &bbox) {
    cv::Mat imgray = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, imgray, cv::COLOR_BGR2GRAY);
    }

    cv::Point2d center(0, 0);
    float radius = 0;
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(imgray, circles, cv::HOUGH_GRADIENT, 2, min_r / 4, 100, 80, min_r, max_r);
    auto num = circles.size();
    if (num) {
        num = (num > 5) ? 5 : num;
        for (int i = 0; i < num; i++) {
            // cv::Point2f c = cv::Point2f(circles[i][0], circles[i][1]);
            // cv::circle(img, c, (int) circles[i][2], cv::Scalar(255), 1, 8, 0);
            // cv::imshow("circles", img);
            center.x += circles[i][0];
            center.y += circles[i][1];
            radius += circles[i][2];
        }
        center.x /= num, center.y /= num, radius /= num;
        bbox = cv::Rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
        return true;
    }
    return false;
}
