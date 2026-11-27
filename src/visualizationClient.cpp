#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include <visualization.h>

using namespace std;
using namespace Fenrir;

struct RecvResult {
    std::string kind;
    cv::Mat mat;                        // for 'left' / 'right'
    std::vector<StereoBlockMatch> triples;  // for 'triples'
};

RecvResult recvMessage(zmq::socket_t& sock) {
    const string TAG_LEFT    = string("\x00", 1);
    const string TAG_RIGHT   = string("\x01", 1);
    const string TAG_TRIPLES = string("\x02", 1);

    zmq::message_t tagMsg;
    if (!sock.recv(tagMsg, zmq::recv_flags::none)) {
        throw runtime_error("recv failed");
    }
    string tag(static_cast<const char*>(tagMsg.data()), tagMsg.size());

    RecvResult out;

    auto recvStr = [&](string& s) {
        zmq::message_t m;
        sock.recv(m, zmq::recv_flags::none);
        s.assign(static_cast<const char*>(m.data()), m.size());
    };

    if (tag == TAG_LEFT) {
        string meta, data;
        recvStr(meta);
        recvStr(data);
        out.kind = "left";
        out.mat  = frameToMat(meta, data);
        return out;
    }

    if (tag == TAG_RIGHT) {
        string meta, data;
        recvStr(meta);
        recvStr(data);
        out.kind = "right";
        out.mat  = frameToMat(meta, data);
        return out;
    }

    if (tag == TAG_TRIPLES) {
        string countBytes;
        recvStr(countBytes);
        uint64_t count = 0;
        memcpy(&count, countBytes.data(), sizeof(uint64_t));

        out.kind = "triples";
        out.triples.reserve(static_cast<size_t>(count));

        for (uint64_t i = 0; i < count; ++i) {
            int32_t a, b, c;
            string buf;
            recvStr(buf); memcpy(&a, buf.data(), sizeof(int32_t));
            recvStr(buf); memcpy(&b, buf.data(), sizeof(int32_t));
            recvStr(buf); memcpy(&c, buf.data(), sizeof(int32_t));
            out.triples.emplace_back(a, b, c);
        }
        return out;
    }

    throw runtime_error("unknown tag");
}

// ---------------------------------------------------------------------------
// receive_and_update_full_frame
// ---------------------------------------------------------------------------
cv::Mat receiveAndUpdateFullFrame(zmq::socket_t& sock,
                                cv::Mat& image,
                                int searchBound) {
    for (int i = 0; i < 3; ++i) {
        RecvResult r = recvMessage(sock);
        if (r.kind == "left") {
            cv::Mat three;
            cv::cvtColor(r.mat, three, cv::COLOR_GRAY2BGR);
            cv::Rect roi(0, 480, 640, 480);
            if (three.type() != image.type() || three.size() != roi.size()) {
                cv::Mat converted;
                three.convertTo(converted, image.type());
                converted.copyTo(image(roi));
            } else {
                three.copyTo(image(roi));
            }
        } else if (r.kind == "right") {
            cv::Mat three;
            cv::cvtColor(r.mat, three, cv::COLOR_GRAY2BGR);
            cv::Rect roi(640, 480, 640, 480);
            if (three.type() != image.type() || three.size() != roi.size()) {
                cv::Mat converted;
                three.convertTo(converted, image.type());
                converted.copyTo(image(roi));
            } else {
                three.copyTo(image(roi));
            }
        } else {
            cv::Mat depthRender = drawDepthFrame(r.triples, searchBound);
            cv::Rect roi(0, 0, 640, 480);
            depthRender.copyTo(image(roi));
        }
    }
    return image;
}

int main(int argc, char* argv[]) {

    namespace po = boost::program_options;

    string calibJSONPath;
    string videoName;
    int fps;
    int sbmSearchBound;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("video-filename", po::value<string>(&videoName)->default_value(""), "Filename for video recording. Don't record video if not provided")
        ("fps", po::value<int>(&fps)->default_value(10), "Default frame rate of recorded video");
        ("sbm-search-bound", po::value<int>(&sbmSearchBound)->default_value(100), "Matching a block from the left camera will check at most this many blocks from the right camera.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    bool recordVideo = !videoName.empty();

    if (calibJSONPath.empty()) {
        cerr << "error: --calibration-json is required\n";
        return 1;
    }

    auto calibration = dv::camera::CalibrationSet::LoadFromFile(calibJSONPath);

    // Get calibration data
    auto leftCameraCalib = calibration.getCameraCalibration("C0").value();
    auto rightCameraCalib = calibration.getCameraCalibration("C1").value();
    const cv::Size resolution = leftCameraCalib.resolution;

    cv::Mat camMatL_mat = cv::Mat(leftCameraCalib.getCameraMatrix());  // CV_32F
    cv::Mat camMatR_mat = cv::Mat(rightCameraCalib.getCameraMatrix()); // CV_32F

    cv::Mat distCoeffsL_mat = cv::Mat(leftCameraCalib.distortion);     // CV_32F
    cv::Mat distCoeffsR_mat = cv::Mat(rightCameraCalib.distortion);    // CV_32F

    auto Reigen = rightCameraCalib.transformationToC0.getRotationMatrix();
    auto Teigen = rightCameraCalib.transformationToC0.getTranslation();
    cv::Mat R, T;
    cv::eigen2cv(Reigen, R);
    cv::eigen2cv(Teigen, T);

    // Convert everything to CV_64F for stereoRectify
    camMatL_mat.convertTo(camMatL_mat, CV_64F);
    camMatR_mat.convertTo(camMatR_mat, CV_64F);
    distCoeffsL_mat.convertTo(distCoeffsL_mat, CV_64F);
    distCoeffsR_mat.convertTo(distCoeffsR_mat, CV_64F);
    R.convertTo(R, CV_64F);
    T.convertTo(T, CV_64F);

    // Stereo rectify (CV_64F)
    cv::Mat R1, R2, P1, P2, Q;
    stereoRectify(camMatL_mat, distCoeffsL_mat, camMatR_mat, distCoeffsR_mat,
                resolution, R, T, R1, R2, P1, P2, Q);

    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, zmq::socket_type::pull);
    sock.bind("tcp://*:5555");
    cout << "INTIALIZED SOCKET" << endl;

    cv::namedWindow("Depth Demo", cv::WINDOW_NORMAL);
    cout << "INTIALIZED WINDOW" << endl;

    // use rectified camera matrix to compute disparity-to-depth conversion rate
    cv::Mat depthColorKey = drawDepthColorKey(Q, sbmSearchBound);
    cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
    // Place the color key at [0:480, 640:840].
    depthColorKey.copyTo(image(cv::Rect(640, 0, depthColorKey.cols, depthColorKey.rows)));

    vector<cv::Mat> videoFrames;

    while (true) {
        try {
            image = receiveAndUpdateFullFrame(sock, image, sbmSearchBound);

            if (recordVideo) {
                videoFrames.push_back(image.clone());
            }

            cv::imshow("Depth Demo", image);
            cv::waitKey(2);
        } catch (const exception& e) {
            // Ctrl-C typically surfaces here as an exception from recv().
            if (string(e.what()).find("Interrupted") != string::npos ||
                string(e.what()) == "recv failed") {
                if (recordVideo) {
                    cout << "Saving video . . ." << endl;
                    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                    cv::VideoWriter writer(videoName, fourcc, fps,
                                        cv::Size(1280, 960));
                    if (!writer.isOpened()) {
                        writer.open(videoName,
                                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                    fps, cv::Size(1280, 960));
                    }
                    if (!writer.isOpened()) {
                        throw runtime_error("no usable VideoWriter backend");
                    }
                    for (const auto& f : videoFrames) {
                        writer.write(f);
                    }
                    writer.release();
                }
                cv::destroyAllWindows();
                sock.close();
                ctx.shutdown();
                break;
            }
            throw;
        }
    }

    return 0;
}
