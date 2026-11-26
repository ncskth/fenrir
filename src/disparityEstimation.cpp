#include <disparityEstimation.h>

namespace Fenrir
{
    using namespace std::chrono_literals;
    using namespace std;

    tuple<cv::Mat, cv::Mat> edgeKernels(int width) {
        double sigma = width / 2.0;
        double s2 = sigma * sigma;
        double center = (width - 1) / 2.0;
        cv::Mat kernelx(width, width, CV_64FC1);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                double x = i - center;
                double y = j - center;
                kernelx.at<double>(i, j) = -x * exp(-0.5 * (x*x + y*y) / s2) / s2;
            }
        }
        cv::Mat kernely;
        cv::transpose(kernelx, kernely);
        return {kernelx, kernely};
    }

    bool isHorizontalEdge(const cv::Mat &img,
                          const cv::Mat &kernelx,
                          const cv::Mat &kernely,
                          int y, int x)   // y = row, x = col
    {
        double dx = 0.0, dy = 0.0;
        int w2 = (kernelx.rows - 1) / 2;
        for (int i = 0; i < kernelx.rows; i++) {
            for (int j = 0; j < kernelx.cols; j++) {
                int row = y + i - w2;
                int col = x + j - w2;
                double v = img.at<double>(col, row);
                dx += kernelx.at<double>(i, j) * v;
                dy += kernely.at<double>(i, j) * v;
            }
        }
        // Horizontal edge => gradient mostly vertical => |dy| >> |dx|
        return abs(dy) > 4.0 * abs(dx);
    }

    StereoBlockMatch matchSingleBlock(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const int centerX,
        const int centerY,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight)
    {
        const int blockWidth = 2 * halfBlockWidth + 1;
        const int blockHeight = 2 * halfBlockHeight + 1;
        const int blockArea = blockWidth * blockHeight;
        const int leftX = centerX - halfBlockWidth;
        const int topY = centerY - halfBlockHeight;

        // Bounds check
        if (leftX < 0 || topY < 0 ||
            leftX + blockWidth > combinedTSLeft.cols ||
            topY + blockHeight > combinedTSLeft.rows)
        {
            return {centerX, centerY, -1};
        }

        // If the local neighborhood is a horizontal edge, stereo block matching will be unreliable
        if (isHorizontalEdge(combinedTSLeft, kernelx, kernely, centerX, centerY))
        {
            return {centerX, centerY, -1};
        }

        // Access left patch directly from matrix data
        const double *leftData = combinedTSLeft.ptr<double>(topY) + leftX;
        const size_t leftStep = combinedTSLeft.step / sizeof(double);

        // Compute left mean and variance directly
        double leftSum = 0, leftSumSq = 0;
        for (int y = 0; y < blockHeight; y++)
        {
            const double *row = leftData + y * leftStep;
            for (int x = 0; x < blockWidth; x++)
            {
                const double val = row[x];
                leftSum += val;
                leftSumSq += val * val;
            }
        }
        const double leftMean = leftSum / blockArea;
        const double leftVariance = (leftSumSq - leftSum * leftSum / blockArea) / (blockArea - 1);

        if (leftVariance < minVariance)
        {
            return {centerX, centerY, -1};
        }

        const int maxDisparity = std::min(searchBound, leftX + 1);
        if (maxDisparity < 0)
        {
            return {centerX, centerY, -1};
        }

        // Initial right patch at disparity 0
        const double *rightData = combinedTSRight.ptr<double>(topY) + leftX;
        const size_t rightStep = combinedTSRight.step / sizeof(double);

        // Compute initial statistics
        double rightSum = 0, rightSumSq = 0, sumXY = 0;
        for (int y = 0; y < blockHeight; y++)
        {
            const double *leftRow = leftData + y * leftStep;
            const double *rightRow = rightData + y * rightStep;
            for (int x = 0; x < blockWidth; x++)
            {
                const double lv = leftRow[x];
                const double rv = rightRow[x];
                rightSum += rv;
                rightSumSq += rv * rv;
                sumXY += lv * rv;
            }
        }

        double variance = (rightSumSq - rightSum * rightSum / blockArea) / (blockArea - 1);
        double covariance = (sumXY - leftSum * rightSum / blockArea) / (blockArea - 1);

        double bestCorrelation = minCorrelation;
        int pixelDisparity = -1;

        if (variance > minVariance)
        {
            const double correlation = covariance / std::sqrt(leftVariance * variance);
            if (correlation > bestCorrelation)
            {
                bestCorrelation = correlation;
                pixelDisparity = 0;
            }
        }

        // Store last column sums for sliding window
        double lastColSum = 0, lastColSqSum = 0;
        for (int y = 0; y < blockHeight; y++)
        {
            const double *rightRow = rightData + y * rightStep;
            const double val = rightRow[blockWidth - 1];
            lastColSum += val;
            lastColSqSum += val * val;
        }

        // Search disparities
        for (int disparity = 1; disparity <= maxDisparity; disparity++)
        {
            const int rightX = leftX - disparity;
            if (rightX < 0 || rightX + blockWidth > combinedTSRight.cols)
            {
                break;
            }

            const double *newRightData = combinedTSRight.ptr<double>(topY) + rightX;

            // Compute new column sums
            double newColSum = 0, newColSqSum = 0;
            for (int y = 0; y < blockHeight; y++)
            {
                const double *row = newRightData + y * rightStep;
                const double val = row[0];
                newColSum += val;
                newColSqSum += val * val;
            }

            // Update sliding statistics
            rightSum += newColSum - lastColSum;
            rightSumSq += newColSqSum - lastColSqSum;

            // Update last column for next iteration
            lastColSum = 0;
            lastColSqSum = 0;
            for (int y = 0; y < blockHeight; y++)
            {
                const double *row = newRightData + y * rightStep;
                const double val = row[blockWidth - 1];
                lastColSum += val;
                lastColSqSum += val * val;
            }

            // Compute covariance for current disparity
            sumXY = 0;
            for (int y = 0; y < blockHeight; y++)
            {
                const double *leftRow = leftData + y * leftStep;
                const double *rightRow = newRightData + y * rightStep;
                for (int x = 0; x < blockWidth; x++)
                {
                    sumXY += leftRow[x] * rightRow[x];
                }
            }

            variance = (rightSumSq - rightSum * rightSum / blockArea) / (blockArea - 1);
            covariance = (sumXY - leftSum * rightSum / blockArea) / (blockArea - 1);

            if (variance > minVariance)
            {
                const double correlation = covariance / std::sqrt(leftVariance * variance);
                if (correlation > bestCorrelation)
                {
                    bestCorrelation = correlation;
                    pixelDisparity = disparity;
                }
            }
        }

        StereoBlockMatch result = {centerX, centerY, pixelDisparity};

        return result;
    }

    vector<StereoBlockMatch> stereoBlockMatchingSequential(
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight,
        const vector<tuple<int, int>> &centers,
        const int start,
        const int end)
    {
        int numBlocks = end - start;
        vector<StereoBlockMatch> matches;
        matches.reserve(numBlocks);

        for (int i = start; i < end; i++)
        {
            matches.push_back(matchSingleBlock(
                minVariance,
                minCorrelation,
                resolution,
                halfBlockWidth,
                halfBlockHeight,
                searchBound,
                get<0>(centers[i]),
                get<1>(centers[i]),
                kernelx,
                kernely,
                combinedTSLeft,
                combinedTSRight));
        }

        return matches;
    }

    vector<vector<StereoBlockMatch>> stereoBlockMatchingParallel(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int searchBound,
        const cv::Mat &kernelx,
        const cv::Mat &kernely,
        const cv::Mat &combinedTSLeft,
        const cv::Mat &combinedTSRight,
        const vector<tuple<int, int>> &centers)
    {
        size_t numBlocks = centers.size();
        vector<vector<StereoBlockMatch>> matches;
        matches.reserve(numThreads);
        vector<future<vector<StereoBlockMatch>>> futures;
        futures.reserve(numThreads);

        int blocksPerThread = numBlocks / numThreads;

        for (int i = 0; i < numThreads; i++)
        {
            int start = blocksPerThread * i;
            int end = min((int)numBlocks, blocksPerThread * (i + 1));
            futures.push_back(async(launch::async, [&]
                                    { return stereoBlockMatchingSequential(
                                          minVariance,
                                          minCorrelation,
                                          resolution,
                                          halfBlockWidth,
                                          halfBlockHeight,
                                          searchBound,
                                          kernelx,
                                          kernely,
                                          combinedTSLeft,
                                          combinedTSRight,
                                          centers,
                                          start,
                                          end); }));
        }

        for (int i = 0; i < numThreads; i++)
        {
            auto theseMatches = futures[i].get();
            matches.push_back(theseMatches);
        }

        return matches;
    }

    void depthEstimationLoop(
        const int numThreads,
        const double minVariance,
        const double minCorrelation,
        const cv::Size resolution,
        const int halfBlockWidth,
        const int halfBlockHeight,
        const int downsampling,
        const int searchBound,
        queue<vector<tuple<int, int>>> &incomingLeftEvents,
        cv::Mat &leftImage,
        cv::Mat &rightImage,
        queue<vector<StereoBlockMatch>> &outgoingEstimates)
    {
        auto kernels = edgeKernels(min(2*halfBlockWidth, 2*halfBlockHeight));
        cv::Mat kernelx = get<0>(kernels);
        cv::Mat kernely = get<1>(kernels);

        while (true)
        {
            if (!incomingLeftEvents.empty())
            {
                vector<tuple<int, int>> centers = incomingLeftEvents.front();
                incomingLeftEvents.pop();
                int numEvents = incomingLeftEvents.size();

                vector<vector<StereoBlockMatch>> matchResult = stereoBlockMatchingParallel(
                    numThreads,
                    minVariance,
                    minCorrelation,
                    resolution,
                    halfBlockWidth,
                    halfBlockHeight,
                    searchBound,
                    ref(kernelx),
                    ref(kernely),
                    ref(leftImage),
                    ref(rightImage),
                    ref(centers));

                vector<StereoBlockMatch> result;
                result.reserve(numEvents);
                for(vector<StereoBlockMatch> outer : matchResult) {
                    for(StereoBlockMatch match : outer) {
                        if(match.pixelDisparity > -1) {
                            result.push_back(match);
                        }
                    }
                }

                outgoingEstimates.push(result);
            }
            else
            {
                this_thread::sleep_for(2ms);
            }
        }
    }
}