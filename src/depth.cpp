#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

#define CHECKERBOARD_WIDTH 9
#define CHECKERBOARD_HEIGHT 6

// Start from 0 and count up. Each webcam has a unique id.
// Find these by trial and error.
// NOTE: Must be configured for target machine.
#define LEFT_CAM 2
#define RIGHT_CAM 1

static bool
calibrate(Mat left, Mat right, Size boardSize)
{
    bool displayCorners = false;
    const int maxScale = 2;
    const float squareSize = 1.f;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f>> imagePoints[2];
    vector<vector<Point3f>> objectPoints(1);
    Size imageSize;

    int i, j, k, nimages = 1;

    imagePoints[0].resize(1);
    imagePoints[1].resize(1);
    vector<string> goodImageList;

    bool foundLeft = false, foundRight = false;
    vector<Point2f>& cornersLeft = imagePoints[0][0];
    vector<Point2f>& cornersRight = imagePoints[1][0];

    foundLeft = findChessboardCorners(left, boardSize, cornersLeft,
        CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_ADAPTIVE_THRESH);
    foundRight = findChessboardCorners(right, boardSize, cornersRight,
        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

    if (!foundLeft || !foundRight) {
        return false;
    }

    cornerSubPix(left, cornersLeft, Size(11, 11), Size(-1, -1),
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
    cornerSubPix(right, cornersRight, Size(11, 11), Size(-1, -1),
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));

    for (i = 0; i < boardSize.height; i++) {
        for (j = 0; j < boardSize.width; j++) {
            objectPoints[0].push_back(Point3f(i*squareSize, j*squareSize, 0));
        }
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    left.size(), R, T, E, F,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5),
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5 );
    cout << "done with RMS error=" << rms << endl;


// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    int npt = (int)imagePoints[0][0].size();
    Mat imgpt[2];
    for( k = 0; k < 2; k++ )
    {
        imgpt[k] = Mat(imagePoints[k][0]);
        undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
        computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
    }
    for( j = 0; j < npt; j++ )
    {
        double errij = fabs(imagePoints[0][0][j].x*lines[1][j][0] +
                            imagePoints[0][0][j].y*lines[1][j][1] + lines[1][j][2]) +
                       fabs(imagePoints[1][0][j].x*lines[0][j][0] +
                            imagePoints[1][0][j].y*lines[0][j][1] + lines[0][j][2]);
        err += errij;
    }
    npoints += npt;
    cout << "average reprojection err = " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";


   return true;

}

int main(int argc, char** argv)
{
    cv::Mat mat_left, mat_right;
    cv::VideoCapture cap_left(LEFT_CAM), cap_right(RIGHT_CAM);

    if (!cap_left.isOpened() || !cap_right.isOpened()) {
        cerr << "couldn't open cameras" << endl;
        return 0;
    }

    vector<Mat> images;
    Size boardSize(CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT);

    while (true) {
        cap_left >> mat_left;
        cap_right >> mat_right;
        if (mat_left.channels() != 3 || mat_right.channels() != 3)
            continue;
        cv::cvtColor(mat_left, mat_left, CV_BGR2GRAY);
        cv::cvtColor(mat_right, mat_right, CV_BGR2GRAY);
        switch ((char)waitKey(1)) {
            case 'p':
                if (calibrate(mat_left.clone(), mat_right.clone(), boardSize)) {
                    cout << "Saved extrinsics and intrinsics." << endl;
                }
                break;
            case 'q':
                // quit
                return 0;
                break;
        }
        imshow("left", mat_left);
        imshow("right", mat_right);
    }
}
