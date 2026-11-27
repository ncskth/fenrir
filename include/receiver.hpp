#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <disparityEstimation.h>
#include <sender.hpp>

namespace Fenrir {

    using namespace std;

    struct RecvResult {
        cv::Mat leftImage;
        cv::Mat rightImage;
        std::vector<StereoBlockMatch> disparityEvents;
    };

    class Receiver {
        public:
            Receiver(const string& addr) : ctx_(1), sock_(ctx_, zmq::socket_type::pull) {
                // e.g. "tcp://*:5555"
                sock_.bind(addr);
            }

            RecvResult receiveFullDepthFrame() {
                zmq::message_t message;
                auto result = sock_.recv(message, zmq::recv_flags::none);

                if (!result.has_value()) {
                    throw runtime_error("recv vailed");
                }

                if (message.size() < 2*480*640) {
                    throw runtime_error("Depth Demo Receiver expects two 8-bit VGA images");
                }

                string bytes;
                bytes.assign(static_cast<char*>(message.data()), message.size());

                cv::Mat leftImage(480, 640, CV_8UC1);
                memcpy(leftImage.data, bytes.data(), 480*640);
                cv::Mat rightImage(480, 640, CV_8UC1);
                memcpy(rightImage.data, bytes.data() + 480*640, 480*640);

                size_t numEventBytes = message.size() - 2*480*640;
                if (numEventBytes%12 != 0) {
                    throw runtime_error("Bad number of bytes in event data");
                }
                size_t numEvents = numEventBytes/(3*4);

                vector<StereoBlockMatch> disparityEvents;
                disparityEvents.reserve(numEvents);
                for (size_t i = 0; i < 3*4*numEvents; i += 3*4) {
                    int32_t x, y, pixelDisparity;
                    memcpy(&x, bytes.data() + 2*480*640 + i, 4);
                    memcpy(&y, bytes.data() + 2*480*640 + i + 4, 4);
                    memcpy(&pixelDisparity, bytes.data() + 2*480*640 + i + 8, 4);
                    disparityEvents.push_back({x, y, pixelDisparity});
                }

                return {
                    leftImage,
                    rightImage,
                    disparityEvents
                };
            }

            void shutdown() {
                sock_.close();
                ctx_.shutdown();
            }

        private:
            zmq::context_t ctx_;
            zmq::socket_t sock_;
    };
}