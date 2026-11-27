#include <disparityEstimation.h>

namespace Fenrir {
    using namespace std;

    cv::Mat drawDepthFrame(const vector<StereoBlockMatch>& disparityMeasurements,
                        int searchBound);

    cv::Mat drawDepthColorKey(double focalLength, double baseline, int sbmSearchBound);

    void updateDepthDemoFrame(
        int sbmSearchBound,
        vector<StereoBlockMatch>& depthResults,
        cv::Mat& leftImage,
        cv::Mat& rightImage,
        cv::Mat& image
    );
}