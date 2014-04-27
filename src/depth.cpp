#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <queue>
#include <vector>
#include <cstdint>
#include <iostream>

#define NUM_CHANNELS 3
#define WIDTH 640
#define HEIGHT 480
#define BLOCK_SIZE 3                               // Must be positive, odd
#define MAX_DISPARITY (((WIDTH / 8) + 15) & -16)   // Must be multiple of 16

int main()
{
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