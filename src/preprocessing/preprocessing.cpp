#include "preprocessing.h"
#include "perspective_transform.h"
#include "segment.h"

using namespace std;

void preprocessing(cv::Mat& frame, bool existBG)
{
    selectAnchors(frame);
    getTransformMatrix();
    perspective(frame);
    cv::Mat bg = existBG ? cv::imread("../src/background.png") : frame;
    assert(! bg.empty());
    segment(bg);
    getLabelMatrix();
}

void perspective(cv::Mat& frame)
{
    _perspective(frame);
}


void getLabels(cv::Rect& bbox, int* labels)
{
    for (int i = 0; i < 5; i++) labels[i] = 0;

    static int c_x, c_y, label;
    c_x = bbox.x + bbox.width / 2;
    c_y = bbox.y + bbox.height / 2;
    labels[0] = getLabel(c_x, c_y); // center

    for (int y = c_y - 1; y >= 0; y -= 10) {
        label = getLabel(c_x, y);
        if (label != labels[0] && label) {
            labels[1] = label;      // upper
            break;
        }
    }

    for (int y = c_y + 1; y < bbox.y + bbox.height; y += 10) {
        label = getLabel(c_x, y);
        if (label != labels[0] && label) {
            labels[2] = label;      // lower
            break;
        }
    }

    for (int x = c_x - 1; x >= 0; x -= 10) {
        label = getLabel(x, c_y);
        if (label != labels[0] && label) {
            labels[3] = label;      // left
            break;
        }
    }

    for (int x = c_x + 1; x < bbox.x + bbox.width; x += 10) {
        label = getLabel(x, c_y);
        if (label != labels[0] && label) {
            labels[4] = label;      // right
            break;
        }
    }
}