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

DEFINE_bool(gray, false,
            "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");

DEFINE_bool(check_size, false,
            "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
            "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
              "Optional: What type should we encode the image as ('png','jpg',...).");

void putToDb(scoped_ptr<db::DB>& db, scoped_ptr<db::Transaction>& txn, int& count, const string& key_str, const Datum& datum) {
    string val_str;
    datum.SerializeToString(&val_str);
    txn->Put(key_str, val_str);
    if (++count % 1000 == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db->NewTransaction());
        LOG(INFO) << "Processed " << count << " files.";
    }
}

void loadDataset(const string& root_folder, bool is_color, const std::vector<std::pair<std::string, int> >& lines,
                 vector<vector<Datum> >& classedImage) {
    Datum datum;
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        std::string enc = "";
        bool status = ReadImageToDatum(root_folder + lines[line_id].first,
                                       lines[line_id].second, 0, 0, is_color,
                                       enc, &datum);
        if (!status) continue;
        classedImage[lines[line_id].second].push_back(datum);
    }
}

int rnd(int n) {
    return caffe::caffe_rng_rand() % n;
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
                                    "gen_dataset_from_images [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc < 4) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/gen_dataset_from_images");
        return 1;
    }

    const bool is_color = !FLAGS_gray;

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
    printf("A total of %d images\n", lines.size());

    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[3], db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    std::string root_folder(argv[1]);
    int count = 0;

    ++num_classes;
    vector<vector<Datum> > classedImage(num_classes);
    loadDataset(root_folder, is_color, lines, classedImage);
    vector<Datum*> dataset;

    for (int cl = 0; cl < num_classes; ++cl) {
        for (int i = 0; i < classedImage[cl].size(); ++i)
            dataset.push_back(&classedImage[cl][i]);
    }

    int glob_index = 0;
    for (int cl = 0; cl < num_classes; glob_index += classedImage[cl].size(), ++cl)
	if (classedImage[cl].size() != 1)
        for (int i = 0; i < classedImage[cl].size(); ++i) {
            int anchor = glob_index + i;
            int positive = rnd(classedImage[cl].size());//relative index
            while (positive == i)
                positive = rnd(classedImage[cl].size());
            positive += glob_index;
            int negative = rnd(dataset.size());//absolute index
            while (glob_index <= negative && negative < glob_index + classedImage[cl].size())
                negative = rnd(dataset.size());

            putToDb(db, txn, count, caffe::format_int(count, 8), *dataset[anchor]);
            putToDb(db, txn, count, caffe::format_int(count, 8), *dataset[positive]);
            putToDb(db, txn, count, caffe::format_int(count, 8), *dataset[negative]);
        }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }
    printf("Generated dataset size = %d\n", count);
#else
    LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
    return 0;
}
