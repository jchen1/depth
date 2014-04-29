#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <queue>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>

#define NUM_CHANNELS 3
#define WIDTH 640
#define HEIGHT 480
#define BLOCK_SIZE 3                               // Must be positive, odd
#define MAX_DISPARITY (((WIDTH / 8) + 15) & -16)   // Must be multiple of 16
#define SCALE 1

int main()
{
    static const std::string intrinsic_filename = "intrinsics.yml";
    static const std::string extrinsic_filename = "extrinsics.yml";

    // reading intrinsic parameters
    cv::FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsic_filename);
        return -1;
    }

    cv::Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    M1 *= SCALE;
    M2 *= SCALE;

    fs.open(extrinsic_filename, CV_STORAGE_READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsic_filename);
        return -1;
    }

    cv::Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    cv::Rect roi1, roi2;
    cv::Mat Q;
    cv::Size img_size(WIDTH, HEIGHT);
    stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

    cv::Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    cv::VideoCapture cap0(1);
    cv::VideoCapture cap1(2);
    if (!cap0.isOpened() || !cap1.isOpened())
    {
        std::cerr << "Webcam could not be opened." << std::endl;
        return -1;
    }

    cap0.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap0.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap1.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap1.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

    cv::StereoSGBM sgbm;

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = BLOCK_SIZE;
    sgbm.P1 = 8 * NUM_CHANNELS*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32 * NUM_CHANNELS*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = MAX_DISPARITY;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 32;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = 0;

    for (;;)
    {
        cv::Mat src0, src1, disp;
        cap0 >> src0;
        cap1 >> src1;

        cv::Mat img1r, img2r;
        remap(src0, img1r, map11, map12, cv::INTER_LINEAR);
        remap(src1, img2r, map21, map22, cv::INTER_LINEAR);

        src0 = img1r;
        src1 = img2r;

        sgbm(src0, src1, disp);

        imshow("cam0", src0);
        imshow("cam1", src1);
        imshow("disp", disp);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }
    return 0;
}