#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <vector>
#include <cstdint>


int main()
{
    cv::VideoCapture cap0(0);
    cv::VideoCapture cap1(1);
    if (!cap0.isOpened() || !cap1.isOpened())
    {
        std::cerr << "Webcam could not be opened." << std::endl;
        return -1;
    }

    for (;;)
    {
        cv::Mat src0, src1;
        cap0 >> src0;
        cap1 >> src1;
        cv::resize(src0, src0, cv::Size(640, 480));
        cv::resize(src1, src1, cv::Size(640, 480));

        imshow("cam0", src0);
        imshow("cam1", src1);

        if (cv::waitKey(30) >= 0)
        {
            break;
        }
    }
    return 0;
}