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
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <map>
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

// Simple hash function for a face rectangle to assign a temporary ID
int hashFace(const Rect &face) {
    return face.x + face.y * 1000;  // simple, works for small videos
}

// Live face recognition with per-face history
void run_live_recognition(
    Ptr<EigenFaceRecognizer> model,
    const map<int,string>& labelNames,
    int face_height = 200,
    int historyLength = 10  // number of frames to smooth over
) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading Haar cascade." << endl;
        return;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera." << endl;
        return;
    }

    cout << "Press ESC to exit live recognition." << endl;

    Mat frame, gray;

    // Map: face ID → deque of last N predictions
    map<int, deque<int>> faceHistories;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(
            gray,
            faces,
            1.2,
            6,
            0 | CASCADE_SCALE_IMAGE,
            Size(100,100)
        );

        map<int, bool> seenFaces;  // track which face IDs are present this frame

        for (size_t i = 0; i < faces.size(); i++) {
            Rect face = faces[i];

            // Tighter crop
            int x = face.x + face.width * 0.1;
            int y = face.y + face.height * 0.1;
            int w = face.width * 0.8;
            int h = face.height * 0.8;
            Rect tightFace(x, y, w, h);
            tightFace &= Rect(0, 0, gray.cols, gray.rows); // inside image

            Mat faceROI = gray(tightFace);

            // Preprocessing
            GaussianBlur(faceROI, faceROI, Size(3,3), 0);
            equalizeHist(faceROI, faceROI);

            Mat resized;
            resize(faceROI, resized, Size(face_height, face_height));

            // Predict label and confidence
            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(resized, predictedLabel, confidence);

            // Compute stable label using history
            int faceId = hashFace(face);
            seenFaces[faceId] = true;

            deque<int> &history = faceHistories[faceId];
            history.push_back(predictedLabel);
            if ((int)history.size() > historyLength)
                history.pop_front();

            map<int,int> count;
            for (int l : history)
                count[l]++;

            int stableLabel = predictedLabel;
            int maxCount = 0;
            for (auto &p : count) {
                if (p.second > maxCount) {
                    maxCount = p.second;
                    stableLabel = p.first;
                }
            }

            // Get person name
            string personName = "Unknown";
            auto itName = labelNames.find(stableLabel);
            if (itName != labelNames.end())
                personName = itName->second;

            // Draw rectangle and label
            rectangle(frame, face, Scalar(0,255,0), 2, LINE_AA);
            string text = format("ID %d: %s (%.2f)", stableLabel, personName.c_str(), confidence);
            putText(frame, text, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        }

        // Remove histories of faces not seen in current frame
        for (auto it = faceHistories.begin(); it != faceHistories.end(); ) {
            if (!seenFaces[it->first])
                it = faceHistories.erase(it);
            else
                ++it;
        }

        imshow("Live FisherFace Recognition", frame);
        if (waitKey(30) == 27) break;  // ESC
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
        // Build label->name map AND load images/labels
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
    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    // This here is a full PCA, if you just want to keep
    // 10 principal components (read Eigenfaces), then call
    // the factory method like this:
    //
    //      EigenFaceRecognizer::create(10);
    //
    // If you want to create a FaceRecognizer with a
    // confidence threshold (e.g. 123.0), call it with:
    //
    //      EigenFaceRecognizer::create(10, 123.0);
    //
    // If you want to use _all_ Eigenfaces and have a threshold,
    // then call the method like this:
    //
    //      EigenFaceRecognizer::create(0, 123.0);
    //
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create(25);
    model->train(images, labels);
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
    // Display or save the Eigenfaces:
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
        if(argc == 2) {
            imshow(format("eigenface_%d", i), cgrayscale);
        } else {
            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }

    // Display or save the image reconstruction at some predefined steps:
    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        if(argc == 2) {
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        } else {
            imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }
    // Display if we are not writing to an output folder:
    if(argc == 2) {
        waitKey(0);
    }
    // ADDED:Start live face recognition
    run_live_recognition(model, labelNames, height);

    return 0;
}