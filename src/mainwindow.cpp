//Standard Includes for QT
#include "mainwindow.h"
#include "ui_mainwindow.h"

// OpenCV library for easy access to USB camera and drawing of images
// on screen
#include "opencv2/opencv.hpp"
#include <sys/time.h>

using namespace cv;

#include <cmath>

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

/**
 * Convert rotation matrix to Euler angles
 */
void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
}

double tic() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return ((double)t.tv_sec + ((double)t.tv_usec)/1000000.);
}


class TagOptimization {

public:
    int count;
    double time;
    int tagID;
    double X, Y, Z; //Meters
    double pitch, roll, yaw; //Radians
    double QR_X, QR_Y; // Meters
    double QR_Z = 8.1; //Meters
    double CAMERA_X, CAMERA_Y, CAMERA_Z; //Meters
    double QR_X_Rot, QR_Y_Rot;

                                         //Unique coordinate system, You must precisely measure the location of your coordinates in 3D space.
                                         //The index of a coordinate applys to the tag id. Currently only accounts for [X,Y]
    double coordinates1[9][2] = { { 4.117,1.217 },{ 1.446,3.442 },{ 4.141,5.089 },{ 0,0 },{ 0,0 },{ 0,-.7 },{ 0,1 },{ 0,0 },{ 0,0 } };

    //Wall coordinates
    double coordinates[9][2] = { { 0,0 },{ -1.5,0 },{ 1.5,0 },{ 0,0 },{ 0,1 },{ 0,-.755 },{ 0,1 },{ 0,0 },{ 0,0 } };

    TagOptimization(double TIME, int TAGID, double x, double y, double PITCH, double ROLL, double YAW)
    {
        time = TIME;
        tagID = TAGID;
        X = x;
        Y = y;
        pitch = PITCH;
        roll = ROLL;
        yaw = YAW;
        //Determine QR's coordinates given ID
        QR_X = coordinates[tagID][0];
        QR_Y = coordinates[tagID][1];
        //Determine Camera Position in relation to QR Coordinate
        //convert radians to degrees
        pitch = pitch * 180 / M_PI;
        roll = roll * 180 / M_PI;
        yaw = yaw * 180 / M_PI;
        //Coordinate System has to be rotated to account for orientation of camera
        QR_X_Rot = QR_X*cos(yaw) - QR_Y*sin(yaw);
        QR_Y_Rot = QR_Y*cos(yaw) + QR_X*sin(yaw);
        CAMERA_X = QR_X_Rot - X;
        CAMERA_Y = QR_Y_Rot - Y;

    }

    friend ostream& operator<<(ostream& os, const TagOptimization& t);
};

//Allows you to print a TagOptimization object using cout <<
ostream& operator<<(ostream& os, const TagOptimization& t)
{
    os << "," << t.time << "," << t.tagID << "," << t.X << "," << t.Y << ","
        << t.pitch << "," << t.roll << "," << t.yaw << "," << t.QR_X_Rot << "," << t.QR_Y_Rot << ","
        << t.CAMERA_X << "," << t.CAMERA_Y << "\n";
    return os;
}



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    timer = new QTimer(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_open_webcam_clicked()
{
    m_tagDetector = new AprilTags::TagDetector(AprilTags::tagCodes36h11);
    cap.open(0);

    if(!cap.isOpened())  // Check if we succeeded
    {
        cout << "camera is not open" << endl;
    }
    else
    {
        cout << "camera is open" << endl;
        i=0;

        connect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
        timer->start(20);
    }
}

void MainWindow::on_pushButton_noAprilTags_clicked()
{
    cap.open(0);

    if(!cap.isOpened())  // Check if we succeeded
    {
        cout << "camera is not open" << endl;
    }
    else
    {
        cout << "camera is open" << endl;
        i=0;

        connect(timer, SIGNAL(timeout()), this, SLOT(update_window2()));
        timer->start(20);
    }
}

void MainWindow::on_pushButton_close_webcam_clicked()
{
    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
    cap.release();

    Mat image = Mat::zeros(frame.size(),CV_8UC3);

    qt_image = QImage((const unsigned char*) (image.data), image.cols, image.rows, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qt_image));

    ui->label->resize(ui->label->pixmap()->size());

    cout << "camera is closed" << endl;
}

void MainWindow::loop() {

  cv::Mat image = frame;
  cv::Mat image_gray;

  int frame1 = 0;
  double last_t = tic();
  //while (true) {

    // capture frame
    //m_cap >> image;

    processImage(image, image_gray);

    // print out the frame rate at which image frames are being processed
    frame1++;
    if (frame1 % 10 == 0) {
      double t = tic();
      cout << "  " << 10./(t-last_t) << " fps" << endl;
      last_t = t;
    }

}

void MainWindow::processImage(cv::Mat& image, cv::Mat& image_gray) {
  cv::cvtColor(image, image_gray, CV_BGR2GRAY);
  double t0;
  if (m_timing) {
    t0 = tic();
  }
  vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(image_gray);
  if (m_timing) {
    double dt = tic()-t0;
    cout << "Extracting tags took " << dt << " seconds." << endl;
  }

  // print out each detection
  cout << detections.size() << " tags detected:" << endl;
  for (int i=0; i<detections.size(); i++) {
    print_detection(detections[i]);
  }

  // show the current image including any detections
  if (m_draw) {
    for (int i=0; i<detections.size(); i++) {
      // also highlight in the image
      detections[i].draw(image);
    }
    //line(image, Point(image.cols / 2, (image.rows / 2) - 50), Point(image.cols / 2, (image.rows/2)+50), Scalar(255, 255, 5), 1);
    if (crosshair){
    line(image, Point((frame.cols / 2) - 50, frame.rows / 2), Point((frame.cols / 2) + 50, frame.rows / 2), Scalar(0, 0, 255), 3);
    line(image, Point(frame.cols / 2, (frame.rows / 2) - 50), Point(frame.cols / 2, (frame.rows / 2) + 50), Scalar(0, 0, 255), 3);
    }

    qt_image = QImage((const unsigned char*) (image.data), image.cols, image.rows, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qt_image));

    ui->label->resize(ui->label->pixmap()->size());

    //char *intStr = itoa(i);

    std::string s = std::to_string(i);
    ui->updateText->setText(QString::fromStdString(s));
    //TagOptimization t(i);
    cout << i << endl;
    i++;
  }

}

void MainWindow::print_detection(AprilTags::TagDetection& detection) const {
    cout << "  Id: " << detection.id
         << " (Hamming: " << detection.hammingDistance << ")";

    // recovering the relative pose of a tag:

    // NOTE: for this to be accurate, it is necessary to use the
    // actual camera parameters here as well as the actual tag size
    // (m_fx, m_fy, m_px, m_py, m_tagSize)

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    detection.getRelativeTranslationRotation(m_tagSize, m_fx, m_fy, m_px, m_py,
                                             translation, rotation);

    Eigen::Matrix3d F;
    F <<
      1, 0,  0,
      0,  -1,  0,
      0,  0,  1;
    Eigen::Matrix3d fixed_rot = F*rotation;
    double yaw, pitch, roll;
    wRo_to_euler(fixed_rot, yaw, pitch, roll);

    cout << "  distance=" << translation.norm()
         << "m, x=" << translation(0)
         << ", y=" << translation(1)
         << ", z=" << translation(2)
         << ", yaw=" << yaw
         << ", pitch=" << pitch
         << ", roll=" << roll
     << ", \n THIS IS A FUCKING TEST!!!!"
         << endl;

    // Also note that for SLAM/multi-view application it is better to
    // use reprojection error of corner points, because the noise in
    // this relative pose is very non-Gaussian; see iSAM source code
    // for suitable factors.
  }

void MainWindow::update_window()
{
    cap >> frame;
    cvtColor(frame, frame, CV_BGR2RGB);
    loop();
}

void MainWindow::update_window2()
{
    cap >> frame;
    cvtColor(frame, frame, CV_BGR2RGB);

    if (crosshair){
    line(frame, Point((frame.cols / 2) - 50, frame.rows / 2), Point((frame.cols / 2) + 50, frame.rows / 2), Scalar(0, 0, 255), 3);
    line(frame, Point(frame.cols / 2, (frame.rows / 2) - 50), Point(frame.cols / 2, (frame.rows / 2) + 50), Scalar(0, 0, 255), 3);
    }

    qt_image = QImage((const unsigned char*) (frame.data), frame.cols, frame.rows, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qt_image));

    ui->label->resize(ui->label->pixmap()->size());

    //char *intStr = itoa(i);

    std::string s = std::to_string(i);
    ui->updateText->setText(QString::fromStdString(s));
    //TagOptimization t(i);
    cout << i << endl;
    i++;
}

void MainWindow::on_pushButton_crosshair_clicked()
{
    if (crosshair){
        crosshair = false;
        cout << "Crosshair Enabled." << endl;
    }
    else{
        crosshair = true;
        cout << "Crosshair Disabled." << endl;
    }
}

void MainWindow::on_pushButton_TEST_clicked()
{
    cap.open(0);

    if(!cap.isOpened())  // Check if we succeeded
    {
        cout << "camera is not open" << endl;
    }
    else
    {
        cout << "camera is open" << endl;
        i=0;

        connect(timer, SIGNAL(timeout()), this, SLOT(update_window2()));
        timer->start(20);
    }
}
