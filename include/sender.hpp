#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <disparityEstimation.h>

namespace Fenrir {

    using namespace std;

    class Sender {
        public:
            Sender(const string& addr) : ctx_(1), sock_(ctx_, zmq::socket_type::push) {
                // e.g. "tcp://*:5555"
                sock_.connect(addr);
            }

            void sendFullDepthFrame(
                const cv::Mat& leftImg,
                const cv::Mat& rightImg,
                const vector<StereoBlockMatch>& depthResults
            ) {
                cv::Mat leftContiguous = leftImg.isContinuous() ? leftImg : leftImg.clone();
                cv::Mat rightContiguous = rightImg.isContinuous() ? rightImg : rightImg.clone();

                int32_t numEvents = (int32_t)(depthResults.size());
                vector<int32_t> depthContiguous;
                depthContiguous.reserve(numEvents);
                for (const auto& match : depthResults) {
                    depthContiguous.push_back(match.x);
                    depthContiguous.push_back(match.y);
                    depthContiguous.push_back(match.pixelDisparity);
                }

                // 2 VGA Grayscale images + 3 ints for each disparity measurement
                zmq::message_t message(2*480*640 + 3*4*numEvents);
                memcpy((char*)message.data(), leftContiguous.data, 480*640);
                memcpy((char*)message.data() + 480*640, rightContiguous.data, 480*640);
                memcpy((char*)message.data() + 2*480*640, depthContiguous.data(), 3*4*numEvents);

                sock_.send(message, zmq::send_flags::none);
            }

        private:
            zmq::context_t ctx_;
            zmq::socket_t sock_;
    };
}