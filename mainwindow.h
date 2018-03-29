#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <string>

// April tags detector and various families that can be selected by command line option
#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag16h5.h"
#include "AprilTags/Tag25h7.h"
#include "AprilTags/Tag25h9.h"
#include "AprilTags/Tag36h9.h"
#include "AprilTags/Tag36h11.h"
using namespace cv;

#include <iostream>
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    VideoCapture cap;

    //--------------------------------------------------------------

    AprilTags::TagDetector* m_tagDetector;

    bool m_draw = true; // draw image and April tag detections?
    bool m_arduino; // send tag detections to serial port?
    bool m_timing = false; // print timing information for each tag extraction call

    int m_width; // image size in pixels
    int m_height;
    double m_tagSize; // April tag side length in meters of square black frame
    double m_fx; // camera focal length in pixels
    double m_fy;
    double m_px; // camera principal point
    double m_py;


    int m_deviceId; // camera id (in case of multiple cameras)

    list<string> m_imgNames;

    cv::VideoCapture m_cap;

    int m_exposure;
    int m_gain;
    int m_brightness;
    //-------------------------------------------------------------------------

private slots:
    void on_pushButton_open_webcam_clicked();

    void on_pushButton_noAprilTags_clicked();

    void on_pushButton_close_webcam_clicked();

    void update_window();
    void update_window2();

    void loop();

    void processImage(cv::Mat& image, cv::Mat& image_gray);

    void print_detection(AprilTags::TagDetection& detection) const;

    void on_pushButton_crosshair_clicked();

    void on_pushButton_TEST_clicked();

private:
    Ui::MainWindow *ui;

    QTimer *timer;
    //VideoCapture cap;

    Mat frame;
    QImage qt_image;

    bool crosshair;
    int i;
};

#endif // MAINWINDOW_H
