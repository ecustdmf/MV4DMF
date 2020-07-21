#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ColorName_Feature.hpp"
#include "complexmat.hpp"

struct BBox_c
{
    double cx, cy, w, h;

    inline void scale(double factor)
    {
        cx *= factor;   // center x
        cy *= factor;   // center y
        w *= factor;
        h *= factor;
    }

    inline cv::Rect get_rect()
    {
        return cv::Rect(cx - w / 2., cy - h / 2., w, h);
    }

};

class KCF_Tracker
{
public:
    bool use_linearkernel{ false };
    bool use_colorname{ true }; 
    bool use_gray{ true };
    bool use_rgb{ false };
    bool use_gradiant{ false }; 
    /*
    padding             ... extra area surrounding the target           (1.5)
    kernel_sigma        ... gaussian kernel bandwidth                   (0.5)
    lambda              ... regularization                              (1e-4)
    interp_factor       ... linear interpolation factor for adaptation  (0.02)
    output_sigma_factor ... spatial bandwidth (proportional to target)  (0.1)
    cell_size           ... hog cell size                               (4)
    */
    KCF_Tracker() {}
    KCF_Tracker(double padding, double kernel_sigma, double lambda, double interp_factor, double output_sigma_factor, int cell_size) :
        padding(padding), output_sigma_factor(output_sigma_factor), kernel_sigma(kernel_sigma),
        lambda(lambda), interp_factor(interp_factor), cell_size(cell_size) {}
    
    // Init/re-init methods
    void init(cv::Mat& img, const cv::Rect& bbox);
    void setTrackerPose(BBox_c& bbox, cv::Mat& img);
    void updateTrackerPosition(BBox_c& bbox);

    // frame-to-frame object tracking
    void track(cv::Mat& img);
    cv::Rect getBBox();

private:
    BBox_c roi;
    double max_patch_size = 100. * 100.;
    bool resize_image = false;

    double padding = 2.0;
    double output_sigma_factor = 0.1;
    double output_sigma;
    double kernel_sigma = 0.5;    //def = 0.5
    double lambda = 1e-4;         //regularization in learning step
    double interp_factor = 0.02;  //def = 0.02, linear interpolation factor for adaptation
    int cell_size = 4;            //4 for hog (= bin_size)
    int windows_size[2];
    cv::Mat cos_window;

    //model
    ComplexMat yf;
    ComplexMat model_alphaf;
    ComplexMat model_alphaf_num;
    ComplexMat model_alphaf_den;
    ComplexMat model_xf;

    //functions
    cv::Mat get_subwindow(const cv::Mat& input, int cx, int cy, int size_x, int size_y);
    cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
    ComplexMat gaussian_correlation(const ComplexMat& xf, const ComplexMat& yf, double sigma, bool auto_correlation = false);
    cv::Mat circshift(const cv::Mat& patch, int x_rot, int y_rot);
    cv::Mat cosine_window_function(int dim1, int dim2);
    ComplexMat fft2(const cv::Mat& input);
    ComplexMat fft2(const std::vector<cv::Mat>& input, const cv::Mat& cos_window);
    cv::Mat ifft2(const ComplexMat& inputf);
    std::vector<cv::Mat> get_features(cv::Mat& input_rgb, cv::Mat& input_gray, int cx, int cy, int size_x, int size_y);
    cv::Point2f sub_pixel_peak(cv::Point& max_loc, cv::Mat& response);
};

