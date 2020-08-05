#include "perspective_transform.h"

using namespace std;
using namespace cv;

cv::Mat transformMatrix;
cv::Point2f srcAnchors[4];
cv::Point2f dstCorners[4] = {
        cv::Point(20, 15),
        cv::Point(880, 15),
        cv::Point(880, 535),
        cv::Point(20, 535)
};
cv::Size BOARD_SIZE(900, 550);
string win = "Select Anchors";
int cx(0), cy(0);
int d = 40;
Scalar fontColor(255, 255, 255);

cv::Point2i getCrossPoint(vector<int>& lineA, vector<int>& lineB) {
    cv::Point2i crossPt;
    auto k1 = ((double)lineA[3] - lineA[1]) / ((double)lineA[2] - lineA[0] + 1e-4);
    auto k2 = ((double)lineB[3] - lineB[1]) / ((double)lineB[2] - lineB[0] + 1e-4);
    k1 = abs(k1) <= 0.2 ? 0 : k1;
    k2 = abs(k2) <= 0.2 ? 0 : k2;
    auto b1 = lineA[1] - k1 * lineA[0];
    auto b2 = lineB[1] - k2 * lineB[0];
    if (cvRound(abs(k1)) >= 10) {
        crossPt.x = lineA[0];
        crossPt.y = cvRound(k2 * crossPt.x + b2);
    }
    else if (cvRound(abs(k2)) >= 10) {
        crossPt.x = lineB[0];
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    else {
        crossPt.x = cvRound(-(b1 - b2) / (k1 - k2));
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    return crossPt;
}


void onMouse(int event, int x, int y, int flag, void* userdata) {
    cv::Mat& src = *(cv::Mat*) userdata;
    cv::Mat temp = src.clone();
    switch (event) {
    case cv::EVENT_MOUSEMOVE:       // move
        break;

    case cv::EVENT_LBUTTONDOWN:     // left -> select
        cx = x, cy = y;

        break;

    case cv::EVENT_RBUTTONDOWN:     // right -> delete
        cx = 0, cy = 0;
    }
    if (cx)
        cv::circle(temp, Point2i(cx, cy), d / 2, Scalar(80, 200, 80), -1);
    cv::circle(temp, Point2i(x, y), d / 2, Scalar(20, 200, 255), -1);
    cv::putText(temp, "Select anchors in turn: ", Point(30, 30), 0, 0.8, fontColor, 1, 8);
    cv::putText(temp, "top left -> top right -> bottom right -> bottom left", Point(60, 70), 0, 0.8,
        fontColor, 1, 8);
    cv::putText(temp, "Left  click: select", Point(30, 110), 0, 0.8, fontColor, 1, 8);
    cv::putText(temp, "Space press: next", Point(30, 150), 0, 0.8, fontColor, 1, 8);
    cv::imshow(win, temp);
    cv::waitKey(1);
}

void locate(const cv::Mat& src) {
    int x1 = cx - d / 2, y1 = cy - d / 2, w = d, h = d;
    cv::Rect bbox(x1, y1, w, h);
    cv::Mat roi, dst, SE;
    src(bbox).copyTo(roi);
    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(roi, roi, cv::Size(5, 5), 2.5);
    cv::threshold(roi, roi, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);

    dst = roi.clone();
    SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, d / 5));
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, SE);
    vector<int> lineA;
    for (int i = 0; i < h; i += h / 5) {
        int j1 = 0, j2 = 0;
        for (int j = 1; j < w - 1; j++) {
            if ((int)dst.at<uchar>(i, j) == 0 && (int)dst.at<uchar>(i, j - 1) == 255) {
                j1 = j;
            }
            if ((int)dst.at<uchar>(i, j) == 0 && (int)dst.at<uchar>(i, j + 1) == 255) {
                j2 = j;
                break;
            }
        }
        if (j1 > 0 && j2 > 0) {
            lineA.push_back(x1 + (j1 + j2) / 2);
            lineA.push_back(y1 + i);
            if (lineA.size() == 4) break;
        }
    }
    assert(lineA.size() == 4);
    //printf("(%d,%d)  (%d,%d)\n", lineA[0], lineA[1], lineA[2], lineA[3]);

    dst = roi.clone();
    SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(d / 4, 1));
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, SE);
    vector<int> lineB;
    for (int j = 0; j < w; j += w / 5) {
        int i1 = 0, i2 = 0;
        for (int i = 1; i < h - 1; i++) {
            if ((int)dst.at<uchar>(i, j) == 0 && (int)dst.at<uchar>(i - 1, j) == 255) {
                i1 = i;
            }
            if ((int)dst.at<uchar>(i, j) == 0 && (int)dst.at<uchar>(i + 1, j) == 255) {
                i2 = i;
                break;
            }
        }
        if (i1 > 0 && i2 > 0) {
            lineB.push_back(x1 + j);
            lineB.push_back(y1 + (i1 + i2) / 2);
            if (lineB.size() == 4) break;
        }
    }
    assert(lineB.size() == 4);
    //printf("(%d,%d)  (%d,%d)\n", lineB[0], lineB[1], lineB[2], lineB[3]);

    cv::Point pt = getCrossPoint(lineA, lineB);
    cx = pt.x, cy = pt.y;
}

inline void drawCross(cv::Mat& src, cv::Point center, cv::Scalar color, int length, int thickness) {
    static int x1, y1, x2, y2;
    x1 = center.x - length / 2, y1 = center.y;
    x2 = center.x + length / 2, y2 = center.y;
    cv::line(src, Point(x1, y1), Point(x2, y2), color, thickness, 8, 0);
    x1 = center.x, y1 = center.y - length / 2;
    x2 = center.x, y2 = center.y + length / 2;
    cv::line(src, Point(x1, y1), Point(x2, y2), color, thickness, 8, 0);
}


void selectAnchors(const cv::Mat& img) {
    cv::Mat drawing = img.clone();
    vector<Point> anchors;
    while (true) {
        cx = 0, cy = 0;
        cv::imshow(win, drawing);
        cv::setMouseCallback(win, onMouse, &drawing);
        int key = cv::waitKey(0);

        if (key == 32 && anchors.size() < 4 && cx) {
            locate(img);
            drawCross(drawing, Point(cx, cy), cv::Scalar(20, 180, 20), 30, 2);
            anchors.push_back(Point(cx, cy));
        }
        else if ((key == int('z') || key == int('Z')) && anchors.size() > 0) {
            anchors.pop_back();
            drawing = img.clone();
            for (auto pt : anchors) {
                drawCross(drawing, pt, cv::Scalar(20, 180, 20), 30, 2);
            }
        }
        else if (anchors.size() == 4) break;
        else continue;
    }
    cv::destroyAllWindows();
    copy(anchors.begin(), anchors.end(), srcAnchors);
    cout << srcAnchors[3] << endl;
}

void getTransformMatrix() {
    transformMatrix = cv::getPerspectiveTransform(srcAnchors, dstCorners);
}

void _perspective(cv::Mat& img) {
    cv::warpPerspective(img, img, transformMatrix, BOARD_SIZE);
}

cv::Point2i getCrossPoint(cv::Vec4i& lineA, cv::Vec4i& lineB) {
    cv::Point2i crossPt;
    auto k1 = ((double)lineA[3] - lineA[1]) / ((double)lineA[2] - lineA[0] + 1e-4);
    auto k2 = ((double)lineB[3] - lineB[1]) / ((double)lineB[2] - lineB[0] + 1e-4);
    auto b1 = lineA[1] - k1 * lineA[0];
    auto b2 = lineB[1] - k2 * lineB[0];
    if (abs(k1) > 1e4) {
        crossPt.x = lineA[0];
        crossPt.y = cvRound(k2 * crossPt.x + b2);
    }
    else if (abs(k2) > 1e4) {
        crossPt.x = lineB[0];
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    else {
        crossPt.x = cvRound(-(b1 - b2) / (k1 - k2));
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    return crossPt;
}