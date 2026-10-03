#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class ZMQImageSender {
public:
    ZMQImageSender(const std::string& addr) : ctx_(1), sock_(ctx_, zmq::socket_type::push) {
        // e.g. "tcp://*:5555"
        sock_.connect(addr);
    }

    // Send three images in one logical message
    void sendThreeImages(const cv::Mat& left, const cv::Mat& right, const cv::Mat& depthColor) {
        sendOneImage(left);
        sendOneImage(right);
        sendOneImage(depthColor);
    }

    void sendOneImage(const cv::Mat& img) {
        // Ensure contiguous memory so we can send img.data as one block
        cv::Mat contiguous = img.isContinuous() ? img : img.clone();

        // Metadata: rows, cols, type, total bytes
        int32_t rows = contiguous.rows;
        int32_t cols = contiguous.cols;
        int32_t type = contiguous.type();
        int64_t bytes = static_cast<int64_t>(contiguous.total() * contiguous.elemSize());

        zmq::message_t meta(4 * 3 + 8);   // 3 int32 + 1 int64
        std::memcpy(meta.data(),      &rows,  4);
        std::memcpy((char*)meta.data() + 4,  &cols,  4);
        std::memcpy((char*)meta.data() + 8,  &type,  4);
        std::memcpy((char*)meta.data() + 12, &bytes, 8);
        sock_.send(meta, zmq::send_flags::sndmore);

        // Raw pixel data — no encoding
        zmq::message_t data(bytes);
        std::memcpy(data.data(), contiguous.data, bytes);
        sock_.send(data, zmq::send_flags::none);
    }

private:

    zmq::context_t ctx_;
    zmq::socket_t sock_;
};