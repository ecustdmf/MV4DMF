#include "KCF_Tracker.hpp"


void KCF_Tracker::init(cv::Mat& img, const cv::Rect& bbox)
{
    //check boundary, enforce min size
    double x1 = bbox.x < 0 ? 0 : bbox.x;
    double y1 = bbox.y < 0 ? 0 : bbox.y;
    double x2 = bbox.x + bbox.width > img.cols - 1 ? img.cols - 1. : x1 + bbox.width;
    double y2 = bbox.y + bbox.height > img.rows - 1 ? img.rows - 1. : y1 + bbox.height;

    if (x2 - x1 < 2. * cell_size)
    {
        double diff = (2. * cell_size - x2 + x1) / 2.;
        if (x1 - diff >= 0 && x2 + diff < img.cols) {
            x1 -= diff;
            x2 += diff;
        }
        else if (x1 - 2 * diff >= 0) {
            x1 -= 2 * diff;
        }
        else {
            x2 += 2 * diff;
        }
    }
    if (y2 - y1 < 2. * cell_size)
    {
        double diff = (2. * cell_size - y2 + y1) / 2.;
        if (y1 - diff >= 0 && y2 + diff < img.rows) {
            y1 -= diff;
            y2 += diff;
        }
        else if (y1 - 2 * diff >= 0) {
            y1 -= 2 * diff;
        }
        else {
            y2 += 2 * diff;
        }
    }

    // ROI region
    roi.w = x2 - x1;
    roi.h = y2 - y1;
    roi.cx = x1 + roi.w / 2.;
    roi.cy = y1 + roi.h / 2.;

    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, cv::COLOR_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    }
    else
        img.convertTo(input_gray, CV_32FC1);

    // resize the ROI and image
    if (roi.w * roi.h > max_patch_size) {
        std::cout << "resizing image by factor of 2" << std::endl;
        resize_image = true;
        roi.scale(0.5);
        cv::resize(input_gray, input_gray, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
    }

    // resize win & fit to fhog cell size
    windows_size[0] = round(roi.w * (1. + padding) / cell_size) * cell_size;
    windows_size[1] = round(roi.h * (1. + padding) / cell_size) * cell_size;
    // std::cout << "init: img size " << img.cols << " " << img.rows << std::endl;
    // std::cout << "init: roi size " << roi.w << " " << roi.h << std::endl;

    // window weights, i.e. labels
    output_sigma = std::sqrt(roi.w * roi.h) * output_sigma_factor / static_cast<double>(cell_size);
    yf = fft2(gaussian_shaped_labels(output_sigma, windows_size[0] / cell_size, windows_size[1] / cell_size));
    cos_window = cosine_window_function(yf.cols, yf.rows);

    //obtain a sub-window for training initial model
    std::vector<cv::Mat> patch_feat = get_features(input_rgb, input_gray, roi.cx, roi.cy, windows_size[0], windows_size[1]);
    model_xf = fft2(patch_feat, cos_window);

    if (use_linearkernel) {
        ComplexMat xfconj = model_xf.conj();
        model_alphaf_num = xfconj.mul(yf);
        model_alphaf_den = (model_xf * xfconj);
        model_alphaf = model_alphaf_num / model_alphaf_den;
    }
    else {
        //Kernel Ridge Regression, calculate alphas (in Fourier domain)
        ComplexMat kf = gaussian_correlation(model_xf, model_xf, kernel_sigma, true);
        model_alphaf_num = yf * kf;
        model_alphaf_den = kf * (kf + lambda);
        model_alphaf = model_alphaf_num / model_alphaf_den;
    }
}

void KCF_Tracker::setTrackerPose(BBox_c& bbox, cv::Mat& img)
{
    init(img, bbox.get_rect());
}

void KCF_Tracker::updateTrackerPosition(BBox_c& bbox)
{
    if (resize_image) {
        BBox_c tmp = bbox;
        tmp.scale(0.5);
        roi.cx = tmp.cx;
        roi.cy = tmp.cy;
    }
    else {
        roi.cx = bbox.cx;
        roi.cy = bbox.cy;
    }
}

cv::Rect KCF_Tracker::getBBox()
{
    BBox_c tmp = roi;
    if (resize_image)
        tmp.scale(2);
    return tmp.get_rect();
}

void KCF_Tracker::track(cv::Mat& img)
{
    cv::Mat input_gray, input_rgb = img.clone();
    if (img.channels() == 3) {
        cv::cvtColor(img, input_gray, cv::COLOR_BGR2GRAY);
        input_gray.convertTo(input_gray, CV_32FC1);
    }
    else
        img.convertTo(input_gray, CV_32FC1);

    // don't need too large image
    if (resize_image) {
        cv::resize(input_gray, input_gray, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
        cv::resize(input_rgb, input_rgb, cv::Size(0, 0), 0.5, 0.5, cv::INTER_AREA);
    }

    std::vector<cv::Mat> patch_feat;
    cv::Point2i max_response_pt;

    patch_feat = get_features(input_rgb, input_gray, roi.cx, roi.cy, windows_size[0], windows_size[1]);
    ComplexMat zf = fft2(patch_feat, cos_window);
    cv::Mat response;
    if (use_linearkernel)
        response = ifft2((model_alphaf * zf).sum_over_channels());
    else {
        ComplexMat kzf = gaussian_correlation(zf, model_xf, kernel_sigma);
        response = ifft2(model_alphaf * kzf);
    }

    /* target location is at the maximum response. we must take into
    account the fact that, if the target doesn't move, the peak
    will appear at the top-left corner, not at the center (this is
    discussed in the paper). the responses wrap around cyclically. */
    double min_val, max_val;
    cv::Point2i min_loc, max_loc;
    cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);
    max_response_pt = max_loc;

    //sub pixel quadratic interpolation from neighbours
    if (max_response_pt.y > response.rows / 2) //wrap around to negative half-space of vertical axis
        max_response_pt.y = max_response_pt.y - response.rows;
    if (max_response_pt.x > response.cols / 2) //same for horizontal axis
        max_response_pt.x = max_response_pt.x - response.cols;

    cv::Point2d new_location(max_response_pt.x, max_response_pt.y);
    new_location = sub_pixel_peak(max_response_pt, response);

    roi.cx += double(cell_size * new_location.x);
    roi.cy += double(cell_size * new_location.y);
    if (roi.cx < 0) roi.cx = 0;
    if (roi.cx > img.cols - 1.) roi.cx = img.cols - 1.;
    if (roi.cy < 0) roi.cy = 0;
    if (roi.cy > img.rows - 1.) roi.cy = img.rows - 1.;

    //obtain a subwindow for training at newly estimated target position
    patch_feat = get_features(input_rgb, input_gray, roi.cx, roi.cy, windows_size[0], windows_size[1]);
    ComplexMat xf = fft2(patch_feat, cos_window);

    //subsequent frames, interpolate model
    model_xf = model_xf * (1. - interp_factor) + xf * interp_factor;

    ComplexMat alphaf_num, alphaf_den;

    if (use_linearkernel) {
        ComplexMat xfconj = xf.conj();
        alphaf_num = xfconj.mul(yf);
        alphaf_den = (xf * xfconj);
    }
    else {
        //Kernel Ridge Regression, calculate alphas (in Fourier domain)
        ComplexMat kf = gaussian_correlation(xf, xf, kernel_sigma, true);
        //        ComplexMat alphaf = p_yf / (kf + p_lambda); //equation for fast training
        //        p_model_alphaf = p_model_alphaf * (1. - p_interp_factor) + alphaf * p_interp_factor;
        alphaf_num = yf * kf;
        alphaf_den = kf * (kf + lambda);
    }

    model_alphaf_num = model_alphaf_num * (1. - interp_factor) + alphaf_num * interp_factor;
    model_alphaf_den = model_alphaf_den * (1. - interp_factor) + alphaf_den * interp_factor;
    model_alphaf = model_alphaf_num / model_alphaf_den;
}

// ****************************************************************************

std::vector<cv::Mat> KCF_Tracker::get_features(cv::Mat& input_rgb, cv::Mat& input_gray, int cx, int cy, int size_x, int size_y)
{
    std::vector<cv::Mat> features;

    cv::Mat patch_rgb = get_subwindow(input_rgb, cx, cy, size_x, size_y);
    cv::Mat patch_gray = get_subwindow(input_gray, cx, cy, size_x, size_y);
    cv::resize(patch_rgb, patch_rgb, cv::Size(size_x / cell_size, size_y / cell_size), 0., 0., cv::INTER_LINEAR);
    cv::resize(patch_gray, patch_gray, cv::Size(size_x / cell_size, size_y / cell_size), 0., 0., cv::INTER_LINEAR);
    cv::Mat patch_rgb_norm(patch_rgb.size(), CV_32F);
    cv::Mat patch_gray_norm(patch_gray.size(), CV_32F);
    patch_rgb.convertTo(patch_rgb_norm, CV_32F, 1. / 255., -0.5);
    patch_gray.convertTo(patch_gray_norm, CV_32F, 1. / 255., -0.5);

    // get colorname feature
    if (use_colorname && input_rgb.channels() == 3) {
        std::vector<cv::Mat> colorname = ColorName_Feature::extract(patch_rgb);
        features.insert(features.end(), colorname.begin(), colorname.end());
    }

    // get gray feature
    if (use_gray && input_gray.channels() == 1) {
        std::vector<cv::Mat> gray = { patch_gray_norm };
        features.insert(features.end(), gray.begin(), gray.end());
    }

    // get rgb feature
    if (use_rgb && input_rgb.channels() == 3) {
        //use rgb color space
        cv::Mat ch1(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch2(patch_rgb_norm.size(), CV_32FC1);
        cv::Mat ch3(patch_rgb_norm.size(), CV_32FC1);
        std::vector<cv::Mat> rgb = { ch1, ch2, ch3 };
        cv::split(patch_rgb_norm, rgb);
        features.insert(features.end(), rgb.begin(), rgb.end());
    }

    // get gradiant feature
    if (use_gradiant) {
        cv::Mat dx, dy, imgradiant;
        cv::Scharr(patch_gray_norm, dx, -1, 1, 0);
        cv::Scharr(patch_gray_norm, dy, -1, 0, 1);
        cv::addWeighted(cv::abs(dx), 0.5, cv::abs(dy), 0.5, 0, imgradiant);
        std::vector<cv::Mat> gradiant = { imgradiant };
        features.insert(features.end(), gradiant.begin(), gradiant.end());
    }

    return features;
}

cv::Mat KCF_Tracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
    cv::Mat labels(dim2, dim1, CV_32FC1);
    int range_y[2] = { -dim2 / 2, dim2 - dim2 / 2 };
    int range_x[2] = { -dim1 / 2, dim1 - dim1 / 2 };

    double sigma_s = sigma * sigma;

    for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
        float* row_ptr = labels.ptr<float>(j);
        double y_s = y * y;
        for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
            row_ptr[i] = std::exp(-0.5 * (y_s + x * x) / sigma_s);
        }
    }

    //rotate so that 1 is at top-left corner (see KCF paper for explanation)
    cv::Mat rot_labels = circshift(labels, range_x[0], range_y[0]);
    //sanity check, 1 at top left corner
    assert(rot_labels.at<float>(0, 0) >= 1.f - 1e-10f);

    return rot_labels;
}

cv::Mat KCF_Tracker::circshift(const cv::Mat& patch, int x_rot, int y_rot)
{
    cv::Mat rot_patch(patch.size(), CV_32FC1);
    cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

    //circular rotate x-axis
    if (x_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-x_rot, patch.cols);
        cv::Range rot_range(0, patch.cols - (-x_rot));
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(0, -x_rot);
        rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }
    else if (x_rot > 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols - x_rot);
        cv::Range rot_range(x_rot, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

        //rotated part
        orig_range = cv::Range(patch.cols - x_rot, patch.cols);
        rot_range = cv::Range(0, x_rot);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }
    else {    //zero rotation
       //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.cols);
        cv::Range rot_range(0, patch.cols);
        patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
    }

    //circular rotate y-axis
    if (y_rot < 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(-y_rot, patch.rows);
        cv::Range rot_range(0, patch.rows - (-y_rot));
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(0, -y_rot);
        rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }
    else if (y_rot > 0) {
        //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows - y_rot);
        cv::Range rot_range(y_rot, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

        //rotated part
        orig_range = cv::Range(patch.rows - y_rot, patch.rows);
        rot_range = cv::Range(0, y_rot);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }
    else { //zero rotation
       //move part that does not rotate over the edge
        cv::Range orig_range(0, patch.rows);
        cv::Range rot_range(0, patch.rows);
        tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
    }

    return rot_patch;
}

ComplexMat KCF_Tracker::fft2(const cv::Mat& input)
{
    cv::Mat complex_result;
    //    cv::Mat padded;                            //expand input image to optimal size
    //    int m = cv::getOptimalDFTSize( input.rows );
    //    int n = cv::getOptimalDFTSize( input.cols ); // on the border add zero pixels
    //    copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    //    cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
    //    return ComplexMat(complex_result(cv::Range(0, input.rows), cv::Range(0, input.cols)));

    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    return ComplexMat(complex_result);
}

ComplexMat KCF_Tracker::fft2(const std::vector<cv::Mat>& input, const cv::Mat& cos_window)
{
    int n_channels = input.size();
    ComplexMat result(input[0].rows, input[0].cols, n_channels);
    for (int i = 0; i < n_channels; ++i) {
        cv::Mat complex_result;
        //        cv::Mat padded;                            //expand input image to optimal size
        //        int m = cv::getOptimalDFTSize( input[0].rows );
        //        int n = cv::getOptimalDFTSize( input[0].cols ); // on the border add zero pixels

        //        copyMakeBorder(input[i].mul(cos_window), padded, 0, m - input[0].rows, 0, n - input[0].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        //        cv::dft(padded, complex_result, cv::DFT_COMPLEX_OUTPUT);
        //        result.set_channel(i, complex_result(cv::Range(0, input[0].rows), cv::Range(0, input[0].cols)));

        cv::dft(input[i].mul(cos_window), complex_result, cv::DFT_COMPLEX_OUTPUT);
        result.set_channel(i, complex_result);
    }
    return result;
}

cv::Mat KCF_Tracker::ifft2(const ComplexMat& inputf)
{

    cv::Mat real_result;
    if (inputf.n_channels == 1) {
        cv::dft(inputf.to_cv_mat(), real_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    }
    else {
        std::vector<cv::Mat> mat_channels = inputf.to_cv_mat_vector();
        std::vector<cv::Mat> ifft_mats(inputf.n_channels);
        for (int i = 0; i < inputf.n_channels; ++i) {
            cv::dft(mat_channels[i], ifft_mats[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        }
        cv::merge(ifft_mats, real_result);
    }
    return real_result;
}

//hann window actually (Power-of-cosine windows)
cv::Mat KCF_Tracker::cosine_window_function(int dim1, int dim2)
{
    cv::Mat m1(1, dim1, CV_32FC1), m2(dim2, 1, CV_32FC1);
    double N_inv = 1. / (static_cast<double>(dim1) - 1.);
    for (int i = 0; i < dim1; ++i)
        m1.at<float>(i) = 0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i)* N_inv));
    N_inv = 1. / (static_cast<double>(dim2) - 1.);
    for (int i = 0; i < dim2; ++i)
        m2.at<float>(i) = 0.5 * (1. - std::cos(2. * CV_PI * static_cast<double>(i)* N_inv));
    cv::Mat ret = m2 * m1;
    return ret;
}

// Returns sub-window of image input centered at [cx, cy] coordinates),
// with size [width, height]. If any pixels are outside of the image,
// they will replicate the values at the borders.
cv::Mat KCF_Tracker::get_subwindow(const cv::Mat& input, int cx, int cy, int width, int height)
{
    cv::Mat patch;

    int x1 = cx - width / 2;
    int y1 = cy - height / 2;
    int x2 = cx + width / 2;
    int y2 = cy + height / 2;

    //out of image
    if (x1 >= input.cols || y1 >= input.rows || x2 < 0 || y2 < 0) {
        patch.create(height, width, input.type());
        patch.setTo(0.f);
        return patch;
    }

    int top = 0, bottom = 0, left = 0, right = 0;

    //fit to image coordinates, set border extensions;
    if (x1 < 0) {
        left = -x1;
        x1 = 0;
    }
    if (y1 < 0) {
        top = -y1;
        y1 = 0;
    }
    if (x2 >= input.cols) {
        right = x2 - input.cols + width % 2;
        x2 = input.cols;
    }
    else
        x2 += width % 2;

    if (y2 >= input.rows) {
        bottom = y2 - input.rows + height % 2;
        y2 = input.rows;
    }
    else
        y2 += height % 2;

    if (x2 - x1 == 0 || y2 - y1 == 0)
        patch = cv::Mat::zeros(height, width, CV_32FC1);
    else
        cv::copyMakeBorder(input(cv::Range(y1, y2), cv::Range(x1, x2)), patch, top, bottom, left, right, cv::BORDER_REPLICATE);

    //sanity check
    assert(patch.cols == width && patch.rows == height);

    return patch;
}

ComplexMat KCF_Tracker::gaussian_correlation(const ComplexMat& xf, const ComplexMat& yf, double sigma, bool auto_correlation)
{
    float xf_sqr_norm = xf.sqr_norm();
    float yf_sqr_norm = auto_correlation ? xf_sqr_norm : yf.sqr_norm();

    ComplexMat xyf = auto_correlation ? xf.sqr_mag() : xf * yf.conj();

    //ifft2 and sum over 3rd dimension, we dont care about individual channels
    ComplexMat xyf_sum = xyf.sum_over_channels();

    cv::Mat ifft2_res = ifft2(xyf_sum);

    float numel_xf_inv = 1.f / (xf.cols * xf.rows * xf.n_channels);
    cv::Mat tmp;
    cv::exp(-1.f / (sigma * sigma) * cv::max((xf_sqr_norm + yf_sqr_norm - 2 * ifft2_res) * numel_xf_inv, 0), tmp);

    return fft2(tmp);
}

float get_response_circular(cv::Point2i& pt, cv::Mat& response)
{
    int x = pt.x;
    int y = pt.y;
    if (x < 0)
        x = response.cols + x;
    if (y < 0)
        y = response.rows + y;
    if (x >= response.cols)
        x = x - response.cols;
    if (y >= response.rows)
        y = y - response.rows;

    return response.at<float>(y, x);
}

cv::Point2f KCF_Tracker::sub_pixel_peak(cv::Point& max_loc, cv::Mat& response)
{
    //find neighbourhood of max_loc (response is circular)
    // 1 2 3
    // 4   5
    // 6 7 8
    cv::Point2i p1(max_loc.x - 1, max_loc.y - 1), p2(max_loc.x, max_loc.y - 1), p3(max_loc.x + 1, max_loc.y - 1);
    cv::Point2i p4(max_loc.x - 1, max_loc.y), p5(max_loc.x + 1, max_loc.y);
    cv::Point2i p6(max_loc.x - 1, max_loc.y + 1), p7(max_loc.x, max_loc.y + 1), p8(max_loc.x + 1, max_loc.y + 1);

    // fit 2d quadratic function f(x, y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
    cv::Mat A = (cv::Mat_<float>(9, 6) <<
        p1.x * p1.x, p1.x * p1.y, p1.y * p1.y, p1.x, p1.y, 1.f,
        p2.x * p2.x, p2.x * p2.y, p2.y * p2.y, p2.x, p2.y, 1.f,
        p3.x * p3.x, p3.x * p3.y, p3.y * p3.y, p3.x, p3.y, 1.f,
        p4.x * p4.x, p4.x * p4.y, p4.y * p4.y, p4.x, p4.y, 1.f,
        p5.x * p5.x, p5.x * p5.y, p5.y * p5.y, p5.x, p5.y, 1.f,
        p6.x * p6.x, p6.x * p6.y, p6.y * p6.y, p6.x, p6.y, 1.f,
        p7.x * p7.x, p7.x * p7.y, p7.y * p7.y, p7.x, p7.y, 1.f,
        p8.x * p8.x, p8.x * p8.y, p8.y * p8.y, p8.x, p8.y, 1.f,
        max_loc.x * max_loc.x, max_loc.x * max_loc.y, max_loc.y * max_loc.y, max_loc.x, max_loc.y, 1.f);
    cv::Mat fval = (cv::Mat_<float>(9, 1) <<
        get_response_circular(p1, response),
        get_response_circular(p2, response),
        get_response_circular(p3, response),
        get_response_circular(p4, response),
        get_response_circular(p5, response),
        get_response_circular(p6, response),
        get_response_circular(p7, response),
        get_response_circular(p8, response),
        get_response_circular(max_loc, response));
    cv::Mat x;
    cv::solve(A, fval, x, cv::DECOMP_SVD);

    double a = x.at<float>(0), b = x.at<float>(1), c = x.at<float>(2),
        d = x.at<float>(3), e = x.at<float>(4);

    cv::Point2f sub_peak(max_loc.x, max_loc.y);
    if (4 * a * c - b * b > 1e-5) {
        sub_peak.y = ((2.f * a * e) / b - d) / (b - (4 * a * c) / b);
        sub_peak.x = (-2 * c * sub_peak.y - e) / b;
        if (fabs(sub_peak.x - max_loc.x) > 1 ||
            fabs(sub_peak.y - max_loc.y) > 1)
            sub_peak = max_loc;
    }

    return sub_peak;
}

