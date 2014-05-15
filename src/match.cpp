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

// Start from 0 and count up. Each webcam has a unique id.
// Find these by trial and error.
// NOTE: Must be configured for target machine.
#define LEFT_CAM 2
#define RIGHT_CAM 1

static inline int load_camera_config(cv::Mat& map11,
    cv::Mat& map12,
    cv::Mat& map21,
    cv::Mat& map22)
{
    static const std::string intrinsic_filename = "intrinsics.yml";
    static const std::string extrinsic_filename = "extrinsics.yml";

    // reading intrinsic parameters
    cv::FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsic_filename.c_str());
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
        printf("Failed to open file %s\n", extrinsic_filename.c_str());
        return -1;
    }

    cv::Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    cv::Rect roi1, roi2;
    cv::Mat Q;
    cv::Size img_size(WIDTH, HEIGHT);
    stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

    return 0;
}


int main()
{
    cv::Mat map11, map12, map21, map22;
    if (load_camera_config(map11, map12, map21, map22) == -1)
        return -1;

    cv::VideoCapture left_cam(LEFT_CAM), right_cam(RIGHT_CAM);
    if (!left_cam.isOpened() || !right_cam.isOpened())
    {
        std::cerr << "Webcam could not be opened." << std::endl;
        return -1;
    }

    left_cam.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH * SCALE);
    left_cam.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT * SCALE);
    right_cam.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH * SCALE);
    right_cam.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT * SCALE);

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

    int blockSize = BLOCK_SIZE / 2;
    int numDisparities = MAX_DISPARITY / 16;

    cv::namedWindow("controls");
    cv::createTrackbar("Block Size", "controls", &blockSize, 16);
    cv::createTrackbar("Speckle Range", "controls", &sgbm.speckleRange, 100);
    cv::createTrackbar("Prefilter Cap", "controls", &sgbm.preFilterCap, 255);
    cv::createTrackbar("Uniqueness Ratio", "controls", &sgbm.uniquenessRatio, 100);
    cv::createTrackbar("Speckle Size", "controls", &sgbm.speckleWindowSize, 255);
    cv::createTrackbar("Minimum Disparity", "controls", &sgbm.minDisparity, 100);
    cv::createTrackbar("Disparity Number", "controls", &numDisparities, 64);

    for (;;)
    {
        cv::Mat left_raw, right_raw, left_corrected, right_corrected, disp, disp8;

        left_cam >> left_raw;
        right_cam >> right_raw;

        if (left_raw.channels() != 3 || right_raw.channels() != 3)
            continue;

        remap(left_raw, left_corrected, map11, map12, cv::INTER_LINEAR);
        remap(right_raw, right_corrected, map21, map22, cv::INTER_LINEAR);

        sgbm.SADWindowSize = blockSize *2+1;
        sgbm.numberOfDisparities = numDisparities * 16;

        sgbm(left_corrected, right_corrected, disp);

        disp.convertTo(disp8, CV_8U, 255 / (MAX_DISPARITY*16.));

        imshow("Left Eye", left_corrected);
        imshow("Right Eye", right_corrected);
        imshow("Disparity", disp8);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }
    return 0;
}