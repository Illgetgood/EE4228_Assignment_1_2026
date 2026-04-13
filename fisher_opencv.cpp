/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
 
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/objdetect.hpp"
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <map>
#include <set>
#include <vector>
 
using namespace cv;
using namespace cv::face;
using namespace std;

Ptr<Facemark> facemark = FacemarkLBF::create();
 
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// Tan and Triggs Illumination Normalization
// Robust preprocessing for face recognition to minimize lighting effects.
Mat tanTriggsPreprocessing(InputArray _src,
    float gamma = 0.2, float sigma0 = 1.0,
    float sigma1 = 2.0, int sz = 11,
    float alpha = 0.1, float tau = 10.0) {
    Mat src = _src.getMat();
    Mat img;
    src.convertTo(img, CV_32F);
    
    // 1. Gamma Correction
    pow(img, gamma, img);
    
    // 2. Difference of Gaussian (DoG)
    Mat img0, img1;
    GaussianBlur(img, img0, Size(sz, sz), sigma0);
    GaussianBlur(img, img1, Size(sz, sz), sigma1);
    img = img0 - img1;
    
    // 3. Contrast Equalization
    Mat abs_img;
    absdiff(img, Scalar(0), abs_img);
    pow(abs_img, alpha, abs_img);
    float mean_a = mean(abs_img).val[0];
    img = img / pow(mean_a, 1.0f / alpha);
    
    absdiff(img, Scalar(0), abs_img);
    min(abs_img, tau, abs_img);
    pow(abs_img, alpha, abs_img);
    float mean_t = mean(abs_img).val[0];
    img = img / pow(mean_t, 1.0f / alpha);
    
    // Final normalization to [0, 255] for FisherFace
    Mat dst;
    normalize(img, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// ADDED function: Read CSV, resize images, populate labels, and build label->name map
map<int, string> buildLabelNameMap(const string& csvFile, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    map<int, string> labelNames;
    ifstream file(csvFile.c_str(), ifstream::in);
    if (!file) {
        cerr << "Error opening CSV file: " << csvFile << endl;
        return labelNames;
    }

    string line, path, labelStr;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, labelStr);
        if(!path.empty() && !labelStr.empty()) {
            int label = stoi(labelStr);

            Mat img = imread(path, 0); // load grayscale
            if (img.empty()) {
                cout << "Could not read image: " << path << endl;
                continue;
            }

            resize(img, img, Size(200,200));  // force same size for all images
            // Apply Tan-Triggs lighting normalization (replaced CLAHE for better robustness)
            img = tanTriggsPreprocessing(img);
            images.push_back(img);
            labels.push_back(label);

            // Extract folder name as person's name
            size_t pos1 = path.find_last_of("/\\");             // last slash
            size_t pos2 = path.find_last_of("/\\", pos1 - 1);  // second last slash
            string name = path.substr(pos2 + 1, pos1 - pos2 - 1);

            // Map label to name (first occurrence only)
            if (labelNames.find(label) == labelNames.end()) {
                labelNames[label] = name;
            }
        }
    }

    return labelNames;
}

// Print per-class and overall training accuracy
void printTrainingAccuracy(
    Ptr<FisherFaceRecognizer> model,
    const vector<Mat>& images,
    const vector<int>& labels,
    const map<int,string>& labelNames
) {
    map<int, int> correct, total;
    for (size_t i = 0; i < images.size(); i++) {
        int pred = model->predict(images[i]);
        total[labels[i]]++;
        if (pred == labels[i]) correct[labels[i]]++;
    }
    int totalAll = 0, correctAll = 0;
    cout << "\n--- Training Set Accuracy ---" << endl;
    for (auto& kv : total) {
        int lbl = kv.first;
        string name = labelNames.count(lbl) ? labelNames.at(lbl) : "?";
        int c = correct.count(lbl) ? correct[lbl] : 0;
        int t = kv.second;
        cout << format("  %-12s | %d / %d correct (%.0f%%)",
                       name.c_str(), c, t, 100.0*c/t) << endl;
        correctAll += c; totalAll += t;
    }
    cout << format("  TOTAL        | %d / %d correct (%.0f%%)",
                   correctAll, totalAll, 100.0*correctAll/totalAll) << endl;
    cout << "  NOTE: Low training accuracy = bad data quality or too few images per person." << endl;
    cout << "----------------------------\n" << endl;
}

int hashFace(const Rect &face) {
    return face.x + face.y * 1000;  // simple, works for small videos
}

// Compute a recognition threshold using an 80/20 per-person train/val split.
double computeThreshold(
    Ptr<FisherFaceRecognizer>,
    const vector<Mat>& allImages,
    const vector<int>& allLabels,
    const map<int,string>& labelNames,
    double multiplier = 1.5
) {
    map<int, vector<int>> labelIdx;
    for (size_t i = 0; i < allLabels.size(); i++)
        labelIdx[allLabels[i]].push_back((int)i);

    vector<int> trainIdx, valIdx;
    for (auto& kv : labelIdx) {
        vector<int>& idx = kv.second;
        int nVal = max(1, (int)(idx.size() * 0.2));
        for (int i = 0; i < (int)idx.size() - nVal; i++) trainIdx.push_back(idx[i]);
        for (int i = (int)idx.size() - nVal; i < (int)idx.size(); i++) valIdx.push_back(idx[i]);
    }

    set<int> trainSet;
    for (int i : trainIdx) trainSet.insert(allLabels[i]);
    if ((int)trainSet.size() < 2) {
        cout << "Not enough data for threshold calibration — using default 3000.0" << endl;
        return 3000.0;
    }

    vector<Mat> trainImgs; vector<int> trainLbls;
    for (int i : trainIdx) { trainImgs.push_back(allImages[i]); trainLbls.push_back(allLabels[i]); }

    Ptr<FisherFaceRecognizer> tmpModel = FisherFaceRecognizer::create();
    tmpModel->train(trainImgs, trainLbls);

    map<int, vector<double>> labelDistances;
    for (int i : valIdx) {
        int pred = -1; double dist = 0.0;
        tmpModel->predict(allImages[i], pred, dist);
        labelDistances[allLabels[i]].push_back(dist);
    }

    cout << "\n--- Per-Person Distance Statistics (held-out 20%, lower = closer) ---" << endl;
    vector<double> perLabelMax;
    for (auto& kv : labelDistances) {
        int lbl = kv.first;
        vector<double>& dists = kv.second;
        double sum = 0, maxD = dists[0], minD = dists[0];
        for (double d : dists) { sum += d; maxD = max(maxD, d); minD = min(minD, d); }
        double mean = sum / dists.size();
        double sq = 0;
        for (double d : dists) sq += (d - mean) * (d - mean);
        double stddev = dists.size() > 1 ? sqrt(sq / (dists.size() - 1)) : 0.0;
        string name = labelNames.count(lbl) ? labelNames.at(lbl) : "?";
        cout << format("  %-12s | min: %7.1f  mean: %7.1f  max: %7.1f  stddev: %6.1f  (n=%d)",
                       name.c_str(), minD, mean, maxD, stddev, (int)dists.size()) << endl;
        perLabelMax.push_back(maxD);
    }

    double avgMaxDist = 0;
    for (double d : perLabelMax) avgMaxDist += d;
    avgMaxDist /= perLabelMax.size();

    double threshold = avgMaxDist * multiplier;
    cout << format("\nAuto threshold = %.1f  (avg worst-case dist %.1f x %.1fx margin)",
                   threshold, avgMaxDist, multiplier) << endl;
    cout << "----------------------------------------------------------------------\n" << endl;
    return threshold;
}

// Align a face crop using eye positions to match the CropFace training preprocessing.
// CropFace target: offset_pct=(0.3,0.3), dest_sz=(200,200)
//   => left eye at (60,60), right eye at (140,60) in 200x200 output
Mat alignFace(const Mat& gray, Point2f eyeLeft, Point2f eyeRight, int outSize = 200) {
    const float offsetW = 0.3f * outSize;   // 60px
    const float offsetH = 0.3f * outSize;   // 60px
    const float desiredDist = outSize - 2.0f * offsetW;  // 80px

    float dx = eyeRight.x - eyeLeft.x;
    float dy = eyeRight.y - eyeLeft.y;
    float actualDist = sqrt(dx*dx + dy*dy);
    if (actualDist < 1.0f) return Mat();

    float scale = desiredDist / actualDist;
    float angle = atan2(dy, dx) * 180.0f / (float)CV_PI;

    Point2f eyeCenter((eyeLeft.x + eyeRight.x) * 0.5f,
                      (eyeLeft.y + eyeRight.y) * 0.5f);

    Mat M = getRotationMatrix2D(eyeCenter, angle, scale);
    // Shift so eye center lands at (outSize/2, offsetH)
    M.at<double>(0, 2) += outSize * 0.5 - eyeCenter.x;
    M.at<double>(1, 2) += offsetH - eyeCenter.y;

    Mat aligned;
    warpAffine(gray, aligned, M, Size(outSize, outSize), INTER_LINEAR, BORDER_REPLICATE);
    return aligned;
}

// Live face recognition using FisherFace with eye-landmark alignment
void run_live_recognition(
    Ptr<FisherFaceRecognizer> model,
    const map<int,string>& labelNames,
    int face_height = 200,
    int historyLength = 10,
    double confidenceThreshold = 3000.0  // FisherFace distance: lower = better
) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading Haar cascade detector." << endl;
        return;
    }

    // Load facial landmark detector for eye-aligned crops (matching training pipeline)
    Ptr<face::FacemarkLBF> fm = face::FacemarkLBF::create();
    bool facemarkLoaded = false;
    try {
        fm->loadModel("lbfmodel.yaml");
        facemarkLoaded = true;
        cout << "Facemark loaded: using eye-aligned crops." << endl;
    } catch (...) {
        cout << "Warning: lbfmodel.yaml not found — landmark alignment disabled." << endl;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera." << endl;
        return;
    }

    cout << "Press ESC to exit live recognition." << endl;

    Mat frame, gray;
    map<int, deque<int>> faceHistories;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Viola-Jones face detection (Haar Cascade)
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        // Try landmark detection on all faces at once
        vector<vector<Point2f>> landmarks;
        bool gotLandmarks = false;
        if (facemarkLoaded && !faces.empty()) {
            vector<Rect> facesForLM = faces;
            gotLandmarks = fm->fit(gray, facesForLM, landmarks);
        }

        map<int, bool> seenFaces;

        for (size_t i = 0; i < faces.size(); i++) {
            Rect face = faces[i];

            Mat faceImg;
            bool aligned = false;

            // Try eye-landmark alignment to match CropFace training preprocessing
            if (gotLandmarks && i < landmarks.size() && landmarks[i].size() >= 68) {
                Point2f eyeLeft  = landmarks[i][36];
                Point2f eyeRight = landmarks[i][45];
                faceImg = alignFace(gray, eyeLeft, eyeRight, face_height);
                if (!faceImg.empty()) aligned = true;
            }

            // Apply Tan-Triggs lighting normalization (matching training pipeline)
            faceImg = tanTriggsPreprocessing(faceImg);

            // No CLAHE — matching training pipeline (now using Tan-Triggs)
            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(faceImg, predictedLabel, confidence);

            int faceId = hashFace(face);
            seenFaces[faceId] = true;

            deque<int>& history = faceHistories[faceId];
            history.push_back(predictedLabel);
            if ((int)history.size() > historyLength) history.pop_front();

            map<int,int> count;
            for (int l : history) count[l]++;

            int stableLabel = -1, maxCount = 0;
            for (auto& p : count)
                if (p.second > maxCount) { maxCount = p.second; stableLabel = p.first; }

            // Predicted name (raw per-frame)
            string predictedName = "?";
            auto itPred = labelNames.find(predictedLabel);
            if (itPred != labelNames.end()) predictedName = itPred->second;

            // Stable name (majority vote over history)
            string stableName = predictedName;
            if (stableLabel >= 0) {
                auto itS = labelNames.find(stableLabel);
                if (itS != labelNames.end()) stableName = itS->second;
            }

            bool isConfident = (confidence < confidenceThreshold && stableLabel >= 0);
            string personName = isConfident ? stableName : ("? " + predictedName);
            Scalar boxColor   = isConfident ? Scalar(0, 200, 0) : Scalar(0, 140, 255);

            rectangle(frame, face, boxColor, 2, LINE_AA);
            string alignTag = aligned ? "" : " [raw]";
            string text = format("%s (dist: %.0f)%s", personName.c_str(), confidence, alignTag.c_str());
            putText(frame, text, Point(face.x, max(face.y - 10, 20)),
                    FONT_HERSHEY_SIMPLEX, 0.75, boxColor, 2, LINE_AA);
        }

        for (auto it = faceHistories.begin(); it != faceHistories.end(); ) {
            if (!seenFaces[it->first]) it = faceHistories.erase(it);
            else ++it;
        }

        imshow("Live FisherFace Recognition", frame);
        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 3) {
        output_folder = string(argv[2]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    map<int, string> labelNames;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        labelNames = buildLabelNameMap(fn_csv, images, labels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // The following lines create an Fisherfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // If you just want to keep 10 Fisherfaces, then call
    // the factory method like this:
    //
    //      FisherFaceRecognizer::create(10);
    //
    // However it is not useful to discard Fisherfaces! Please
    // always try to use _all_ available Fisherfaces for
    // classification.
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0) and use _all_
    // Fisherfaces, then call it with:
    //
    //      FisherFaceRecognizer::create(0, 123.0);
    //
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    model->train(images, labels);

    // Print training set accuracy as a data quality diagnostic
    printTrainingAccuracy(model, images, labels, labelNames);

    // The following line predicts the label of a given
    // test image:
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    // Here is how to get the eigenvalues of this Eigenfaces model:
    Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    Mat W = model->getEigenVectors();
    // Get the sample mean from the training data
    Mat mean = model->getMean();
    // Display or save:
    if(argc == 2) {
        imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    } else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }
    // Display or save the first, at most 16 Fisherfaces:
    for (int i = 0; i < min(16, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Bone colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
        // Display or save:
        if(argc == 2) {
            imshow(format("fisherface_%d", i), cgrayscale);
        } else {
            imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }
    // Display or save the image reconstruction at some predefined steps:
    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {
        // Slice the Fisherface from the model:
        Mat ev = W.col(num_component);
        Mat projection = LDA::subspaceProject(ev, mean, images[0].reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        if(argc == 2) {
            imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
        } else {
            imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
        }
    }
    // Display if we are not writing to an output folder:
    if(argc == 2) {
        waitKey(0);
    }
    // Compute data-driven threshold from FisherFace model on 80/20 split
    // multiplier=1.5: aligned crops from facemark should produce distances
    //  similar to training. Increase if names still show in orange.
    double autoThreshold = computeThreshold(model, images, labels, labelNames, 1.5);
    run_live_recognition(model, labelNames, height, 10, autoThreshold);
    return 0;
}