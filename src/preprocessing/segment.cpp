#include "segment.h"
#include <set>

using namespace std;

cv::Mat segImg;
cv::Mat labelMatrix;

struct Electrode
{
    int x;      // center x
    int y;      // center y
    int label;
    Electrode(int _x, int _y, int _l) : x(_x), y(_y), label(_l) {}
};

vector<cv::Vec3b> getColors(int num)
{
    int step = 256 / num;
    vector<int> px;
    for (int i = 0; i < 256; i += step)
        px.push_back(i);

    vector<cv::Vec3b> colors(num);
    for (int j = 0; j < 3; j++)
    {
        random_shuffle(px.begin(), px.end());
        for (int i = 0; i < num; i++)
        {
            colors[i][j] = px[i];
        }
    }
    return colors;
}


void color(cv::Mat& img, cv::Mat& labels, int num, int startLabel, int bg = 255)
{
    auto colors = getColors(num);
    cv::Mat drawing = cv::Mat(img.size(), CV_8UC3, cv::Scalar(bg, bg, bg));
    for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++) {
            int index = labels.at<int>(i, j);
            if (index >= startLabel) {
                drawing.at<cv::Vec3b>(i, j) = colors[index];
            }
        }
    }
    cv::imshow("colored image", drawing);
    cv::waitKey(0);
    cv::destroyAllWindows();
}


void segment(const cv::Mat& src)
{
    cv::Mat dst;
    int m{ src.size().height }, n{ src.size().width };

    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(dst, dst, cv::Size(5, 5), 1.5);
    int bs = m / 3 / 2 * 2 + 1;
    cv::adaptiveThreshold(dst, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, bs, 20);

    dst = 255 - dst;
    cv::Mat labels, stats, centroids;  //labels CV_32S; stats CV_32S; centroids CV64F;
    int num = cv::connectedComponentsWithStats(dst, labels, stats, centroids);
    dst = cv::Mat(dst.size(), CV_8UC1, cv::Scalar(255));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (labels.at<int>(i, j) == 1) {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    num = cv::connectedComponentsWithStats(dst, labels, stats, centroids);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (labels.at<int>(i, j) == 1) {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
    segImg = dst;
    //cv::imshow("segment", segImg);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
}


void getLabelMatrix()
{
    cv::Mat labels, stats, centroids;  //labels CV_32S; stats CV_32S; centroids CV64F;
    int num = cv::connectedComponentsWithStats(segImg, labels, stats, centroids);
    //color(segImg, labels, num, 1, 0);

    vector<Electrode> e;
    int x0, y0;
    for (int k = 1; k < num; k++) {
        x0 = centroids.at<double>(k, 0);
        y0 = centroids.at<double>(k, 1);
        e.push_back(Electrode(x0, y0, k));
    }
    num = e.size();

    // sort
    for (int i = 0; i < num; i++) {
        int m = i;
        for (int j = i + 1; j < num; j++){
            if (e[j].y < e[m].y) m = j;
        }
        auto temp = e[i];
        e[i] = e[m];
        e[m] = temp;
    }
    
    // filter
    for (int i = 1; i < num; i++) {
        if (abs(e[i].y - e[i - 1].y) < 10) e[i].y = e[i - 1].y;
    }

    // resort
    for (int i = 0; i < num; i++) {
        int m = i;
        for (int j = i + 1; j < num; j++) {
            if (e[j].y < e[m].y ||
                (e[j].y == e[m].y && e[j].x < e[m].x)) {
                m = j;
            }
        }
        auto temp = e[i];
        e[i] = e[m];
        e[m] = temp;
    }

    // map labels
    cv::Mat table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        int val = 0;
        for (int j = 0; j < num; j++) {
            if (i == e[j].label) {
                val = j + 1;
                break;
            }
        }
        table.at<uchar>(0, i) = val;
    }
    labels.convertTo(labels, CV_8UC1);
    cv::LUT(labels, table, labelMatrix);
    //cv::imshow("labels", labelMatrix*3);
    //cv::waitKey(0);
}


int  getLabel(const int& x, const int& y)
{
    return (int)labelMatrix.at<uchar>(y, x);
}