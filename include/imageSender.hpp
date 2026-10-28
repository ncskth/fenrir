#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;

class ZMQImageSender {
public:
    ZMQImageSender(const string& addr) : ctx_(1), sock_(ctx_, zmq::socket_type::push) {
        // e.g. "tcp://*:5555"
        sock_.connect(addr);
    }

    static constexpr uint8_t TAG_LEFT  = 0;
    static constexpr uint8_t TAG_RIGHT  = 1;
    static constexpr uint8_t TAG_TRIPLES = 2;

    void sendOneImage(const cv::Mat& img, bool isLeft) {
        // Ensure contiguous memory so we can send img.data as one block
        cv::Mat contiguous = img.isContinuous() ? img : img.clone();

        // Metadata: rows, cols, type, total bytes
        int32_t rows = contiguous.rows;
        int32_t cols = contiguous.cols;
        int32_t type = contiguous.type();
        int64_t bytes = static_cast<int64_t>(contiguous.total() * contiguous.elemSize());

        zmq::message_t meta(4 * 3 + 8);   // 3 int32 + 1 int64
        memcpy(meta.data(),      &rows,  4);
        memcpy((char*)meta.data() + 4,  &cols,  4);
        memcpy((char*)meta.data() + 8,  &type,  4);
        memcpy((char*)meta.data() + 12, &bytes, 8);

        zmq::message_t tag(sizeof(uint8_t));
        *static_cast<uint8_t*>(tag.data()) = isLeft ? TAG_LEFT : TAG_RIGHT;
        sock_.send(tag, zmq::send_flags::sndmore);

        sock_.send(meta, zmq::send_flags::sndmore);
        // Raw pixel data — no encoding
        zmq::message_t pixeldata(bytes);
        std::memcpy(pixeldata.data(), contiguous.data, bytes);
        sock_.send(pixeldata, zmq::send_flags::none);
    }

    void sendTriples(const vector<std::tuple<int, int, int>>& triples) {
        zmq::message_t tag(sizeof(uint8_t));
        *static_cast<uint8_t*>(tag.data()) = TAG_TRIPLES;
        sock_.send(tag, zmq::send_flags::sndmore);

        uint64_t count = triples.size();
        zmq::message_t countMsg(sizeof(count));
        std::memcpy(countMsg.data(), &count, sizeof(count));
        sock_.send(countMsg, zmq::send_flags::sndmore);

        for (size_t i = 0; i < triples.size(); ++i) {
            auto [a, b, c] = triples[i];
            int32_t vals[3] = {a, b, c};
            zmq::message_t parts[3];
            for (int k = 0; k < 3; ++k) {
                parts[k] = zmq::message_t(sizeof(int32_t));
                std::memcpy(parts[k].data(), &vals[k], sizeof(int32_t));
            }
            bool last = (i + 1 == triples.size());
            sock_.send(parts[0], zmq::send_flags::sndmore);
            sock_.send(parts[1], zmq::send_flags::sndmore);
            sock_.send(parts[2], last ? zmq::send_flags::none : zmq::send_flags::sndmore);
        }
        if (triples.empty()) {
            zmq::message_t empty(0);
            sock_.send(empty, zmq::send_flags::none);
        }
    }

    void sendFullVisFrame(
        const cv::Mat& leftImg,
        const cv::Mat& rightImg,
        const vector<tuple<int, int, int>>& depthResults) {
            sendOneImage(leftImg, true);
            sendOneImage(rightImg, false);
            sendTriples(depthResults);
    }

private:

    zmq::context_t ctx_;
    zmq::socket_t sock_;
};