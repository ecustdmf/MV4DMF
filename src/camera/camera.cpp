#include "camera.h"
#include <direct.h>     // _mkdir
#include <io.h>         // _access
#include <ctime>        // time

using namespace std;

cv::VideoCapture capturer;
cv::VideoWriter writer;
double frameWidth, frameHeight, fps;

void camera::open(int cameraIndex, double _width, double _height, double _fps)
{
    capturer.open(cameraIndex, cv::CAP_DSHOW);
    assert(capturer.isOpened());
    capturer.set(cv::CAP_PROP_FRAME_WIDTH, _width);
    capturer.set(cv::CAP_PROP_FRAME_HEIGHT, _height);
    capturer.set(cv::CAP_PROP_FPS, _fps);
    frameWidth = capturer.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight = capturer.get(cv::CAP_PROP_FRAME_HEIGHT);
    get_webcam_fps();
    printf("Camera ON: %d��%d, %.2f FPS \n\n", (int)frameWidth, (int)frameHeight, fps);
}

void camera::setParams(double brightness, double contrast, double saturation, double hue, double gain, double exposure)
{
    assert(capturer.isOpened());
    capturer.set(cv::CAP_PROP_BRIGHTNESS, brightness);
    capturer.set(cv::CAP_PROP_CONTRAST, contrast);
    capturer.set(cv::CAP_PROP_SATURATION, saturation);
    capturer.set(cv::CAP_PROP_HUE, hue);
    capturer.set(cv::CAP_PROP_GAIN, gain);
    capturer.set(cv::CAP_PROP_EXPOSURE, exposure);
    cout << "Camera Params: " << endl;
    printf("brightness = %.2f\n", capturer.get(cv::CAP_PROP_BRIGHTNESS));
    printf("contrast = %.2f\n", capturer.get(cv::CAP_PROP_CONTRAST));
    printf("saturation = %.2f\n", capturer.get(cv::CAP_PROP_SATURATION));
    printf("hue = %.2f\n", capturer.get(cv::CAP_PROP_HUE));
    printf("gain = %.2f\n", capturer.get(cv::CAP_PROP_GAIN));
    printf("exposure = %.2f\n", capturer.get(cv::CAP_PROP_EXPOSURE));
    cout << endl;
}

static void camera::get_webcam_fps()
{
    cv::Mat frame;
    int num_frames = fps * 10;
    double tick_count = cv::getTickCount();
    for (int i = 0; i < num_frames; i++) {
        capturer >> frame;
    }
    tick_count = cv::getTickCount() - tick_count;
    double t = tick_count / cv::getTickFrequency();   // s
    fps = num_frames / t;
    //printf("Estimated FPS: %.2f\n\n", fps);
}

void camera::openVideo(char* path)
{
    capturer.open(path);
    assert(capturer.isOpened());
    int frameCount = (int)capturer.get(cv::CAP_PROP_FRAME_COUNT);
    frameWidth = capturer.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight = capturer.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = capturer.get(cv::CAP_PROP_FPS);
    printf("Video INFO: about %d frames, %dx%d, %.2f FPS \n\n", frameCount, (int)frameWidth, (int)frameHeight, fps);
}

bool camera::read(cv::Mat& frame)
{
    capturer >> frame;
    if (frame.empty()) return false;
    recorder::record(frame);
    return true;
}

void camera::close()
{
    capturer.release();
}

static string getVideoName()
{
    const char* folder = "../videos";
    if (0 != _access(folder, 0))
    {
        int ret = _mkdir(folder);
    }
    time_t unixtime = time(nullptr);
    tm now;
    localtime_s(&now, &unixtime);
    char* name = new char[50];
    sprintf_s(name, 50, "%s/record-%4d-%02d-%02d %02d.%02d.mp4", folder,
        1900 + now.tm_year, 1 + now.tm_mon, now.tm_mday,
        now.tm_hour, now.tm_min);
    return name;
}

void recorder::open(cv::Size size)
{
    string videoName = getVideoName();
    int fourcc = writer.fourcc('m', 'p', '4', 'v');   // mp4
    if (size.width == -1) size = cv::Size(frameWidth, frameHeight);
    // cout << size << endl;
    writer.open(videoName, fourcc, fps, size);
    assert(writer.isOpened());
    printf("Start recording ... Path: %s\n\n", (videoName).c_str());
}

void recorder::record(cv::Mat& frame){
    writer << frame;
}

void recorder::close(){
    writer.release();
}


