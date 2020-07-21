#include "perspective_transform.h"

using namespace std;

cv::Mat transformMatrix;
cv::Point2f anchors[4];
cv::Point2f dstCorners[4] = {
        cv::Point(20, 15),
        cv::Point(880, 15),
        cv::Point(880, 535),
        cv::Point(20, 535)
};
cv::Size BOARD_SIZE(900, 550);


cv::Point2i getCrossPoint(vector<int> &lineA, vector<int> &lineB) {
    cv::Point2i crossPt;
    auto k1 = ((double) lineA[3] - lineA[1]) / ((double) lineA[2] - lineA[0] + 1e-4);
    auto k2 = ((double) lineB[3] - lineB[1]) / ((double) lineB[2] - lineB[0] + 1e-4);
    k1 = abs(k1) <= 0.2 ? 0 : k1;
    k2 = abs(k2) <= 0.2 ? 0 : k2;
    auto b1 = lineA[1] - k1 * lineA[0];
    auto b2 = lineB[1] - k2 * lineB[0];
    if (cvRound(abs(k1)) >= 10) {
        crossPt.x = lineA[0];
        crossPt.y = cvRound(k2 * crossPt.x + b2);
    } else if (cvRound(abs(k2)) >= 10) {
        crossPt.x = lineB[0];
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    } else {
        crossPt.x = cvRound(-(b1 - b2) / (k1 - k2));
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    return crossPt;
}


namespace Anchor {
    int x{-1}, y{-1}, d{50};
    bool SELECTED = false;
    string win = "Select Anchors";

    void onMouse(int event, int _x, int _y, int flag, void *userdata) {
        cv::Mat &src = *(cv::Mat *) userdata;
        cv::Mat temp = src.clone();
        switch (event) {
            case cv::EVENT_LBUTTONDOWN:     // select anchor
            {
                SELECTED = true;
                x = (int) _x, y = (int) _y;
                break;
            }

            case cv::EVENT_MOUSEMOVE: {
                break;
            }
        }
        cv::Point center(_x, _y);
        if (SELECTED) center = cv::Point(x, y);
        cv::circle(temp, center, d / 2, cv::Scalar(180, 255, 180), -1);
        cv::putText(temp, "Select anchors in turn: ", cv::Point(30, 30), 0, 0.8, cv::Scalar(255, 255, 255), 1, 8);
        cv::putText(temp, "top left -> top right -> bottom right -> bottom left", cv::Point(60, 70), 0, 0.8,
                    cv::Scalar(255, 255, 255), 1, 8);
        cv::putText(temp, "Left  click: select", cv::Point(30, 110), 0, 0.8, cv::Scalar(255, 255, 255), 1, 8);
        cv::putText(temp, "Space press: next", cv::Point(30, 150), 0, 0.8, cv::Scalar(255, 255, 255), 1, 8);
        cv::imshow(win, temp);
        cv::waitKey(1);
    }

    void selectROI(cv::Mat &drawing) {
        cv::imshow(win, drawing);
        cv::setMouseCallback(win, onMouse, &drawing);
        cv::waitKey(0);
        if (!SELECTED) {
            cerr << "Not select anchors!" << endl;
            exit(-4);
        }
    }

    void locate(const cv::Mat &src) {
        int x1 = x - d / 2, y1 = y - d / 2, w = d, h = d;
        cv::Rect bbox(x1, y1, w, h);
        cv::Mat roi, dst, SE;
        src(bbox).copyTo(roi);
        cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(roi, roi, cv::Size(5, 5), 2.5);
        cv::threshold(roi, roi, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY);

        // ˮƽɨ��
        dst = roi.clone();
        SE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, d / 5));
        cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, SE);
        vector<int> lineA;
        for (int i = 0; i < h; i += h / 5) {
            int j1 = 0, j2 = 0;
            for (int j = 1; j < w - 1; j++) {
                if ((int) dst.at<uchar>(i, j) == 0 && (int) dst.at<uchar>(i, j - 1) == 255) {
                    j1 = j;
                }
                if ((int) dst.at<uchar>(i, j) == 0 && (int) dst.at<uchar>(i, j + 1) == 255) {
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
                if ((int) dst.at<uchar>(i, j) == 0 && (int) dst.at<uchar>(i - 1, j) == 255) {
                    i1 = i;
                }
                if ((int) dst.at<uchar>(i, j) == 0 && (int) dst.at<uchar>(i + 1, j) == 255) {
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
        x = pt.x, y = pt.y;
    }

    void select(const cv::Mat &img, cv::Mat &drawing, float &_x, float &_y) {
        x = -1, y = -1, SELECTED = false;
        selectROI(drawing);
        //printf("(%d, %d) ... ", x, y);
        locate(img);
        _x = x, _y = y;
        //printf("(%d, %d)\n", x, y);
    }
};


inline void drawCross(cv::Mat &src, cv::Point center, cv::Scalar color, int length, int thickness) {
    //���ƺ���
    cv::line(src, cv::Point(center.x - length / 2, center.y), cv::Point(center.x + length / 2, center.y), color,
             thickness, 8, 0);
    //��������
    cv::line(src, cv::Point(center.x, center.y - length / 2), cv::Point(center.x, center.y + length / 2), color,
             thickness, 8, 0);
    return;
}


void selectAnchors(const cv::Mat &img) {
    cv::Mat drawing = img.clone();
    for (int i = 0; i < 4; ++i) {
        Anchor::select(img, drawing, anchors[i].x, anchors[i].y);
        drawCross(drawing, anchors[i], cv::Scalar(20, 180, 20), 30, 2);
    }
    cv::imshow("Select Anchors", drawing);
    cv::destroyAllWindows();
}

void getTransformMatrix() {
    transformMatrix = cv::getPerspectiveTransform(anchors, dstCorners);
}

void _perspective(cv::Mat &img) {
    cv::warpPerspective(img, img, transformMatrix, BOARD_SIZE);
}

cv::Point2i getCrossPoint(cv::Vec4i &lineA, cv::Vec4i &lineB) {
    cv::Point2i crossPt;
    auto k1 = ((double) lineA[3] - lineA[1]) / ((double) lineA[2] - lineA[0] + 1e-4);
    auto k2 = ((double) lineB[3] - lineB[1]) / ((double) lineB[2] - lineB[0] + 1e-4);
    auto b1 = lineA[1] - k1 * lineA[0];
    auto b2 = lineB[1] - k2 * lineB[0];
    if (abs(k1) > 1e4) {
        crossPt.x = lineA[0];
        crossPt.y = cvRound(k2 * crossPt.x + b2);
    } else if (abs(k2) > 1e4) {
        crossPt.x = lineB[0];
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    } else {
        crossPt.x = cvRound(-(b1 - b2) / (k1 - k2));
        crossPt.y = cvRound(k1 * crossPt.x + b1);
    }
    return crossPt;
}