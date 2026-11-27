#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/camera/calibration_set.hpp>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include <visualization.h>
#include <receiver.hpp>
#include <calibData.h>

using namespace std;
using namespace Fenrir;

int depthDemoMain(
    string calibJSONPath,
    string videoName,
    bool recordVideo,
    int fps,
    int sbmSearchBound
) {

    if (calibJSONPath.empty()) {
        cerr << "error: --calibration-json is required for depth estimation demo\n";
        return 1;
    }

    CalibData calibData = loadCalibData(calibJSONPath);

    Receiver receiver = Receiver("tcp://*:5555");

    cv::namedWindow("Depth Demo", cv::WINDOW_NORMAL);

    // use rectified camera matrix to compute disparity-to-depth conversion rate
    cv::Mat depthColorKey = drawDepthColorKey(
        calibData.focalLength,
        calibData.baseline,
        sbmSearchBound
    );
    cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
    // Place the color key at [0:480, 640:840].
    depthColorKey.copyTo(image(cv::Rect(640, 0, depthColorKey.cols, depthColorKey.rows)));

    vector<cv::Mat> videoFrames;

    while (true) {
        try {
            RecvResult r = receiver.receiveFullDepthFrame();
            updateDepthDemoFrame(
                sbmSearchBound,
                r.disparityEvents,
                r.leftImage,
                r.rightImage,
                ref(image)
            );

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
                receiver.shutdown();
                break;
            }
            throw;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {

    namespace po = boost::program_options;

    string calibJSONPath = "";
    string videoName = "";
    int fps = 10;
    int sbmSearchBound = 100;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("calibration-json", po::value<string>(&calibJSONPath), "Camera calibration file written in JSON according to inivation standards")
        ("video-filename", po::value<string>(&videoName)->default_value(""), "Filename for video recording. Don't record video if not provided")
        ("fps", po::value<int>(&fps), "Default frame rate of recorded video");
        ("sbm-search-bound", po::value<int>(&sbmSearchBound), "Matching a block from the left camera will check at most this many blocks from the right camera.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    bool recordVideo = !videoName.empty();

    return depthDemoMain(
        calibJSONPath,
        videoName,
        recordVideo,
        fps,
        sbmSearchBound
    );
}
