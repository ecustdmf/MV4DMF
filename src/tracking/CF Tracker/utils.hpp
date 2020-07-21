#pragma once
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <opencv2/core.hpp>

double get_distance(const cv::Point2d& A, const cv::Point2d& B)
{
    return cv::sqrt(cv::pow(A.x - B.x, 2) + cv::pow(A.y - B.y, 2));
}

template <class T>
std::string num2str(T number, int prewidth, int precision, char filled = ' ')
{
    std::stringstream ss;
    int bitwidth = prewidth > 0 ? prewidth + precision + 1 : -1;
    ss << std::setw(bitwidth) << std::setfill(filled)
        << std::setprecision(precision) << std::fixed << number;
    return ss.str();
}

template <class T>
class Computer
{
public:
    Computer() = default;
    void run(const std::vector<T>& data) { 
        m_mean = std::accumulate(data.begin(), data.end(), 0.) / data.size();
        m_var = 0;
        for (auto x : data)
        {
            m_var += pow(x - m_mean, 2);
        }
        m_var /= data.size();
        m_std_dev = sqrt(m_var);
    }
    double mean() { return m_mean; }
    double std_dev() { return m_std_dev; }
    double var() { return m_var; }
private:
    double m_mean = 0;
    double m_std_dev = 0;
    double m_var = 0;
};