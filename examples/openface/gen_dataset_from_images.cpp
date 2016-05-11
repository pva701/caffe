// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false, "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false, "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");

DEFINE_bool(check_size, false, "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false, "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "", "Optional: What type should we encode the image as ('png','jpg',...).");

DEFINE_bool(flip, false, "Reflect images for increasing dataset");
DEFINE_int32(rotate, 0, "Image can be rotate on angle from range [-rotate, rotate]");
DEFINE_int32(resize, 0, "Image can be resized on percent from range [-resize, resize]");

using cv::Mat;
using cv::Point;
using cv::getRotationMatrix2D;
using cv::flip;
using cv::resize;

Mat rotateImage(const Mat& image, int grad_angle) {
    //if (grad_angle == 0) grad_angle = 1;
    Point src_center(image.cols/2, image.rows/2);
    Mat rot_mat = getRotationMatrix2D(src_center, grad_angle, 1.0);
    Mat dst;
    warpAffine(image, dst, rot_mat, image.size(), CV_INTER_LINEAR, 1);
    return dst;
}

Mat flipImage(const Mat& image) {
    Mat dst;
    flip(image, dst, 1);
    return dst;
}

Mat resizeImage(const Mat& image, double percent, bool horizontally) {
    int interpolation = (percent > 0 ? CV_INTER_LINEAR: CV_INTER_AREA);
    Mat dst;
    resize(image, dst, cv::Size(), 1 - horizontally * percent, 1 - !horizontally * percent, interpolation);
    return dst;
}


const int BATCH_SIZE = 3*10000;

void putToDb(scoped_ptr<db::DB>& db, scoped_ptr<db::Transaction>& txn, int& count, const string& key_str, const Datum& datum) {
    string val_str;
    datum.SerializeToString(&val_str);
    txn->Put(key_str, val_str);
    if (++count % BATCH_SIZE == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db->NewTransaction());
    }
}

void loadDataset(const string& root_folder, bool is_color,
                 const std::vector<std::pair<std::string, int> >& lines,
                 vector<vector<Mat> >& classedImage) {
    Mat mat;
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        mat = ReadImageToCVMat(root_folder + lines[line_id].first, is_color);
        classedImage[lines[line_id].second].push_back(mat);
    }
}

int rnd(int n) {
    return caffe::caffe_rng_rand() % n;
}
int rnd(int l, int r) {
    return caffe::caffe_rng_rand() % (r - l + 1) + l;
}

struct FaceTransformation {
    int angle;
    double percent;
    bool horizontally;

    static FaceTransformation randomTransformation(int max_angle, int max_percent) {
        int ang = rnd(-max_angle, max_angle);
        double perc = rnd(-max_percent, max_percent) / 100.0;
        return {ang, perc, rnd(2) == 0};
    }

    Mat apply(const Mat& image) {
        //return resizeImage(rotateImage(image, angle), percent, horizontally);
        return rotateImage(image, angle);
    }
};

void printGenerated(int count) {
    count /= 3;
    std::string suffix;
    while (count >= 1000) {
        count /= 1000;
        suffix += "K";
    }
    LOG(INFO) << "Generated " << suffix << count << " triplets.";
}

Datum toDatum(const Mat &mat) {
    static Datum common_datum;
    CVMatToDatum(mat, &common_datum);
    return common_datum;
}

template<class T>
int rnd(const vector<T>& vec) {
    return vec[rnd(vec.size())];
}

int main(int argc, char **argv) {
#ifdef USE_OPENCV
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of images to the triplet leveldb/lmdb\n"
                                    "format used as input for Caffe.\n"
                                    "Usage:\n"
                                    "gen_dataset_from_images [FLAGS] ROOT_FOLDER/ LISTFILE DB_NAME NUM_TRIPLETS\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc < 5) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/gen_dataset_from_images");
        return 1;
    }

    const bool is_color = !FLAGS_gray;
    int num_triplets = std::atoi(argv[4]);

    std::ifstream infile(argv[2]);
    std::vector<std::pair<std::string, int> > lines;
    std::string filename;
    int label;
    int num_classes = 0;
    while (infile >> filename >> label) {
        lines.push_back(std::make_pair(filename, label));
        if (label > num_classes)
            num_classes = label;
    }
    printf("A total of %d images\n", (int)lines.size());

    // Storing to db
    std::string root_folder(argv[1]);
    int count = 0;
    ++num_classes;
    vector<vector<Mat> > classedImage(num_classes + FLAGS_flip * num_classes);
    loadDataset(root_folder, is_color, lines, classedImage);
    vector<int> not_zero_classes;
    for (int cl = 0; cl < num_classes; ++cl) {
        for (int i = 0; i < classedImage[cl].size(); ++i)
            if (FLAGS_flip)
                classedImage[cl + num_classes].push_back(flipImage(classedImage[cl][i]));

        if (classedImage[cl].size() != 0) {
            not_zero_classes.push_back(cl);
            if (FLAGS_flip)
                not_zero_classes.push_back(cl + num_classes);
        }
    }
    num_classes += FLAGS_flip * num_classes;
    printf("Number of classes = %d\n", num_classes);
    printf("Required number of triplet = %d\n", num_triplets);

    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[3], db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    for (int tr = 0; tr < num_triplets; ++tr) {
        int anchor_class = rnd(not_zero_classes);
        int anchor_in_class = rnd(classedImage[anchor_class].size());
        FaceTransformation anchor_ft = FaceTransformation::randomTransformation(FLAGS_rotate, FLAGS_resize);
        int positive_in_class = rnd(classedImage[anchor_class].size());
        FaceTransformation positive_ft = FaceTransformation::randomTransformation(FLAGS_rotate, FLAGS_resize);

        int negative_class = rnd(not_zero_classes);
        while (negative_class == anchor_class)
            negative_class = rnd(not_zero_classes);
        int negative_in_class = rnd(classedImage[negative_class].size());
        FaceTransformation negative_ft = FaceTransformation::randomTransformation(FLAGS_rotate, FLAGS_resize);

        Mat anchor = anchor_ft.apply(classedImage[anchor_class][anchor_in_class]);
        Mat positive = positive_ft.apply(classedImage[anchor_class][positive_in_class]);
        Mat negative = negative_ft.apply(classedImage[negative_class][negative_in_class]);

        /*printf("ang, perc = %d %.3lf\n", anchor_ft.angle, anchor_ft.percent);
        cv::imshow("src", classedImage[anchor_class][anchor_in_class]);
        cv::imshow("dst", anchor);
        cv::waitKey(0);*/

        putToDb(db, txn, count, caffe::format_int(count, 8), toDatum(anchor));
        putToDb(db, txn, count, caffe::format_int(count, 8), toDatum(positive));
        putToDb(db, txn, count, caffe::format_int(count, 8), toDatum(negative));
        if (count % BATCH_SIZE == 0)
            printGenerated(count);
    }

    // write the last batch
    if (count % BATCH_SIZE != 0)
        txn->Commit();
    printGenerated(count);
#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV

    return 0;
}
