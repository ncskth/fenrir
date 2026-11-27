#include <visualization.h>
#include <disparityEstimation.h>

namespace Fenrir {

    using namespace std;

    pair<CvDepth, int> decodeCvType(int typeCode) {
        int depth = typeCode & 7;
        int channels = (typeCode >> 3) + 1;
        return { static_cast<CvDepth>(depth), channels };
    }

    int cvTypeFromDepth(CvDepth d) {
        switch (d) {
            case CvDepth::U8:  return CV_8U;
            case CvDepth::I8:  return CV_8S;
            case CvDepth::U16: return CV_16U;
            case CvDepth::I16: return CV_16S;
            case CvDepth::I32: return CV_32S;
            case CvDepth::F32: return CV_32F;
            case CvDepth::F64: return CV_64F;
        }
        throw runtime_error("bad depth");
    }

    cv::Mat drawDepthFrame(const vector<StereoBlockMatch>& disparityMeasurements,
                        int searchBound) {
        const int H = 480, W = 640;
        cv::Mat result = cv::Mat::zeros(H, W, CV_8UC3);
        if (disparityMeasurements.empty()) {
            return result;
        }

        const size_t n = disparityMeasurements.size();

        // Build float HSV, then convert once.
        cv::Mat hsvImg((int)(n), 1, CV_8UC3);
        for (size_t i = 0; i < n; ++i) {
            double d = (double)disparityMeasurements[i].pixelDisparity;
            float h = (uint8_t)(120.0 * (d / searchBound));
            hsvImg.at<cv::Vec3b>((int)(i), 0) = cv::Vec3b(h, 255, 255);
        }

        // reshape to 3-channel single-column  image then convert to RGB
        cv::Mat bgrImg;
        cv::cvtColor(hsvImg.reshape(3, 1), bgrImg, cv::COLOR_HSV2BGR);

        // Paint 3x3 patches.
        for (size_t i = 0; i < n; ++i) {
            int x = disparityMeasurements[i].x;
            int y = disparityMeasurements[i].y;
            cv::Vec3f c = bgrImg.at<cv::Vec3b>(0, (int)(i));

            int x0 = max(0, x - 1), x1 = min(W, x + 2);
            int y0 = max(0, y - 1), y1 = min(H, y + 2);
            for (int xi = x0; xi < x1; ++xi) {
                for (int yi = y0; yi < y1; ++yi) {
                    result.at<cv::Vec3b>(yi, xi) = c;
                }
            }
        }
        return result;
    }

    cv::Mat drawDepthColorKey(const cv::Mat& Q, int sbmSearchBound) {
        // Get focal length and baseline from Q
        double f = Q.at<double>(2, 3);
        double Tx = -1.0 / Q.at<double>(3, 2);

        // Create key image (vertical color bar)
        int height = 480;
        int width = 200;
        cv::Mat key = cv::Mat::zeros(height, width, CV_8UC3);

        // Fill color bar
        for(int y = height - 1; y >= 0; y--) {
            // Normalized disparity from 0 to sbmSearchBound
            float disp = 1.0 - (float)(height - y - 1) / height;

            // Hue from 0 to 120 (red to green in HSV)
            uint8_t hue = (uint8_t)(120.0f * disp);

            // HSV to BGR
            cv::Mat hsv(1, 1, CV_8UC3);
            hsv.at<cv::Vec3b>(0,0) = cv::Vec3b(hue, 255, 255); // OpenCV HSV: H/2
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

            // Fill half of the row
            for(int x = 0; x < width/2; x++) {
                key.at<cv::Vec3b>(y, x) = bgr.at<cv::Vec3b>(0,0);
            }
        }

        // Add ticks and labels
        int numTicks = 5;
        for(int i = numTicks - 1; i >= 0; i--) {
            // Disparity from 2 to sbmSearchBound
            float disp = 10.0f + (float)i / (numTicks - 1) * (sbmSearchBound - 20);
            float depth = f * Tx / disp;

            int y = (int)((disp / sbmSearchBound) * height);
            if(y < 0) y = 0;
            if(y >= height) y = height - 1;

            // Draw tick mark
            cv::line(key, cv::Point(width - 10, y), cv::Point(width, y), cv::Scalar(255,255,255), 2);

            // Draw label
            string label = cv::format("%.2f m", depth);
            cv::putText(key, label, cv::Point(100, y+5), cv::FONT_HERSHEY_SIMPLEX,
                        0.4, cv::Scalar(255,255,255), 1);
        }

        // Add title
        //cv::putText(key, "Depth", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX,
        //            0.6, cv::Scalar(255,255,255), 1);
        cv::putText(key, "Near", cv::Point(20, height-10), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(200,200,200), 1);
        cv::putText(key, "Far", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(200,200,200), 1);

        return key;
    }
}