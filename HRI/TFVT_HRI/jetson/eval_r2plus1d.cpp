#include "multimodal_act.hpp"
#include "paddle/include/paddle_inference_api.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef SERVER_MODE
#include "eval_server.grpc.pb.h"
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <thread>
#endif

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXi;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, 1> VectorXi64;

const double eps = 0.000001;
const int NUM_ACT = 317;
const int IMG_RESIZE_0 = 416;
const int IMG_RESIZE = 224;
const int VIEW_H = 360; // 720 / 2
const int VIEW_W = 640; // 1280 / 2
const int OB_WINDOW_LEN = 8;
const int SAFE_ACTS[] = {1, 3, 4, 5, 6, 7, 8, 10};

struct EvalEntry {
  int label;
  int waeID;
  std::string dir;
  std::vector<std::string> frameIDs;
  std::vector<Eigen::Vector4f> bboxLst;
};

std::deque<cv::Mat> obWindow;
std::deque<Eigen::VectorXf> processedImgWindow;

DEFINE_string(dirname, "./baseline_r2plus1d",
              "Directory of the inference model and params.");
DEFINE_double(tau, 1.0, "Softmax temperature hyperparameter.");
DEFINE_int32(topK, 50, "Number of top-k multimodal actions.");
DEFINE_bool(gpu, false, "Whether to use GPU config.");
DEFINE_string(logdir, "./log",
              "Directory of the log, include observations, response JSON.");
DEFINE_string(dataTxt, "/mnt/xueyang/Code/xiaodu-hi/data/final_eval.txt",
              "Path to evaluation dataset txt file.");
DEFINE_string(outputType, "wae",
              "Output type of R(2+1)D model, options: wae, scenario.");

#ifdef SERVER_MODE
DEFINE_int32(port, 8888, "Port of gRPC server to bind.");
#endif

template <typename T> void PrintVectorX(std::string name, T v, size_t n) {
  std::cout << "[" << name << "] ";
  std::cout << "size: " << v.size() << ", first " << n << ": ";
  for (size_t i = 0; i < n; i++)
    std::cout << v(i) << " ";
  std::cout << std::endl;
}

template <typename T>
std::vector<size_t> ArgSort(const std::vector<T> &v, bool ascending = false) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  if (ascending)
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  else
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

std::vector<size_t> ArgSort(const Eigen::VectorXf &v, bool ascending = false) {
  std::vector<float> vec;
  vec.resize(v.size());
  for (int i = 0; i < v.size(); i++)
    vec[i] = v(i);

  return ArgSort<float>(vec, ascending);
}

int GetCurrentHour() {
  auto now = std::chrono::system_clock::now();
  std::time_t t_now = std::chrono::system_clock::to_time_t(now);
  tm *date = localtime(&t_now);
  return static_cast<int>(date->tm_hour);
}

void PreprocessImage(cv::Mat &image, Eigen::VectorXf &flattenOutput,
                     std::string outputImgFile = "") {
  double aspectRatio = std::min(IMG_RESIZE_0 * 1.0 / image.rows,
                                IMG_RESIZE_0 * 1.0 / image.cols);
  int newH = static_cast<int>(std::floor(image.rows * aspectRatio));
  int newW = static_cast<int>(std::floor(image.cols * aspectRatio));
  cv::resize(image, image, cv::Size(newW, newH));

  cv::Mat boxedImg(IMG_RESIZE_0, IMG_RESIZE_0, CV_8UC3,
                   cv::Scalar(128, 128, 128));

  int yOffset = static_cast<int>(std::floor(IMG_RESIZE_0 - newH) / 2.0);
  int xOffset = static_cast<int>(std::floor(IMG_RESIZE_0 - newW) / 2.0);
  image.copyTo(boxedImg(cv::Rect(xOffset, yOffset, newW, newH)));
  cv::resize(boxedImg, boxedImg, cv::Size(IMG_RESIZE, IMG_RESIZE));

  if (outputImgFile != "")
    cv::imwrite(outputImgFile, boxedImg);

  cv::Mat rgbImg;
  cv::cvtColor(boxedImg, rgbImg, cv::COLOR_BGRA2RGB);

  cv::Mat rgbMat[3];
  cv::split(rgbImg, rgbMat);

  Eigen::MatrixXf rMat, gMat, bMat;
  cv::cv2eigen(rgbMat[0], rMat);
  cv::cv2eigen(rgbMat[1], gMat);
  cv::cv2eigen(rgbMat[2], bMat);

  rMat /= 255.0;
  gMat /= 255.0;
  bMat /= 255.0;

  float rMean = 0.43216, gMean = 0.394666, bMean = 0.37645;
  float rStd = 0.22803, gStd = 0.22145, bStd = 0.216989;
  rMat = rMat - rMean * Eigen::MatrixXf::Ones(rMat.rows(), rMat.cols());
  gMat = gMat - gMean * Eigen::MatrixXf::Ones(gMat.rows(), gMat.cols());
  bMat = bMat - bMean * Eigen::MatrixXf::Ones(bMat.rows(), bMat.cols());
  rMat /= rStd;
  gMat /= gStd;
  bMat /= bStd;

  // Flatten RGB mode data with shape [C, H, W]
  RowMajorMatrixXf t_rMat(rMat), t_gMat(gMat), t_bMat(bMat);
  flattenOutput.resize(t_rMat.size() + t_gMat.size() + t_bMat.size());
  flattenOutput << Eigen::Map<Eigen::VectorXf>(t_rMat.data(), t_rMat.size()),
      Eigen::Map<Eigen::VectorXf>(t_gMat.data(), t_gMat.size()),
      Eigen::Map<Eigen::VectorXf>(t_bMat.data(), t_bMat.size());
}

void PrepareMultimodalActions(std::string filename,
                              std::vector<MultimodalAction> &multimodalActs) {
  std::ifstream infile(filename);
  std::string talk, exp, act;
  while (true) {
    if (!std::getline(infile, talk))
      break;
    std::getline(infile, exp);
    std::getline(infile, act);

    MultimodalAction ma(talk, exp, act);
    multimodalActs.push_back(ma);
  }
}

namespace paddle {
using paddle::AnalysisConfig;

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double TimeDiff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareTRTConfig(AnalysisConfig *config, std::string name) {
  std::string paramsFile = FLAGS_dirname + "/" + name + "_params";
  std::string modelFile = FLAGS_dirname + "/" + name + "_model";
  bool modelOnly = !boost::filesystem::exists(paramsFile);

  if (modelOnly)
    config->SetModel(modelFile);
  else
    config->SetModel(modelFile, paramsFile);

  if (FLAGS_gpu) {
    // Init GPU memory: 1000MB, GPU id: 0
    config->EnableUseGpu(1000, 0);
  } else {
    config->DisableGpu();
    config->SetCpuMathLibraryNumThreads(8);
  }
  config->SwitchUseFeedFetchOps(false);
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrOptim(true);
}

void RunCtrl(PaddlePredictor *predictor,
             const std::vector<Eigen::VectorXf> &imgArray,
             Eigen::VectorXf &flattenActPred,
             Eigen::VectorXf &flattenActTopKSample) {
  auto time1 = time();

  if (imgArray.size() != OB_WINDOW_LEN)
    throw std::runtime_error("Image array should have size " +
                             std::to_string(OB_WINDOW_LEN));

  flattenActPred.resize(NUM_ACT);
  flattenActTopKSample.resize(1);

  auto imgInput = predictor->GetInputTensor("x2paddle_0");
  // For [1, 3, 8, 224, 224] input
  // imgInput->Reshape({1, 3, OB_WINDOW_LEN, IMG_RESIZE, IMG_RESIZE});
  // int lastDim = IMG_RESIZE * IMG_RESIZE;
  // RowMajorMatrixXf imgTensor(3 * OB_WINDOW_LEN, lastDim);
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < OB_WINDOW_LEN; j++) {
  //     int k = i * OB_WINDOW_LEN + j;
  //     imgTensor.row(k) = imgArray[j].segment(i * lastDim, lastDim);
  //   }
  // }

  // For [1, 8, 3, 224, 224] input
  imgInput->Reshape({1, OB_WINDOW_LEN, 3, IMG_RESIZE, IMG_RESIZE});
  RowMajorMatrixXf imgTensor(OB_WINDOW_LEN, imgArray[0].size());
  for (int i = 0; i < OB_WINDOW_LEN; i++)
    imgTensor.row(i) = imgArray[i];

  imgInput->copy_from_cpu(imgTensor.data());

  auto softmaxTemp = predictor->GetInputTensor("softmax_temp");
  softmaxTemp->Reshape({1});
  Eigen::VectorXf tau;
  tau.resize(1);
  tau(0) = FLAGS_tau;
  softmaxTemp->copy_from_cpu(tau.data());

  auto topKInput = predictor->GetInputTensor("top_k");
  topKInput->Reshape({1});
  VectorXi64 topK;
  topK.resize(1);
  topK(0) = FLAGS_topK;
  topKInput->copy_from_cpu(topK.data());

  CHECK(predictor->ZeroCopyRun());

  auto outputNames = predictor->GetOutputNames();
  auto actPred = predictor->GetOutputTensor(outputNames[0]);
  auto actTopKSample = predictor->GetOutputTensor(outputNames[1]);

  actPred->copy_to_cpu(flattenActPred.data());
  actTopKSample->copy_to_cpu(flattenActTopKSample.data());

  auto time2 = time();
  LOG(INFO) << "[RunCtrl] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
}

bool ConvertPredToJsons(int reqID, bool useSkill,
                        const Eigen::VectorXf &actPred,
                        const Eigen::VectorXf &actTopKSample,
                        const std::vector<MultimodalAction> &multimodalActs,
                        std::string &resJson, float &resActScore) {
  auto time1 = time();
  size_t nullActAt = ArgSort(actPred)[0];
  // std::cout << "=========== " << nullActAt << std::endl;
  if (nullActAt == 0) {
    resJson = "{}";
    return false;
  }

  std::default_random_engine rnd;
  rnd.seed(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> uniform(0, 1);

  int safeActNum = sizeof(SAFE_ACTS) / sizeof(SAFE_ACTS[0]);
  int sampleID = static_cast<int>(actTopKSample(actTopKSample.size() - 1));
  std::string talk = multimodalActs[sampleID].talk;

  if (talk.find("C") != talk.npos) {
    // ignore the salutation.
    sampleID = SAFE_ACTS[static_cast<int>(uniform(rnd) * safeActNum)];
  }

  int hour = GetCurrentHour();
  MultimodalAction ma = multimodalActs[sampleID];
  resJson = ma.to_json(hour, reqID, useSkill, "", "你");
  resActScore = actPred(sampleID);

  auto time2 = time();
  LOG(INFO) << "[ConvertPredToJsons] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
  return true;
}

} // namespace paddle

void RunEval(paddle::PaddlePredictor *ctrlPredictor,
             const std::vector<MultimodalAction> &multimodalActs, EvalEntry ee,
             bool &hasAct, std::string &resJson, float &annoActScore) {
  obWindow.clear();
  processedImgWindow.clear();

  int offset = ee.frameIDs.size() - OB_WINDOW_LEN;
  if (offset < 0)
    throw "Evaluation entry has frames less than " +
        std::to_string(OB_WINDOW_LEN);

  for (int i = offset; i < ee.frameIDs.size(); i++) {
    std::string imgFile = ee.dir + "/" + ee.frameIDs[i] + ".jpg";
    cv::Mat img = cv::imread(imgFile);

    Eigen::VectorXf flattenImg;
    PreprocessImage(img, flattenImg);

    obWindow.push_back(img);
    processedImgWindow.push_back(flattenImg);
  }

  Eigen::VectorXf flattenActPred, flattenActTopKSample;
  paddle::RunCtrl(ctrlPredictor,
                  {processedImgWindow.begin(), processedImgWindow.end()},
                  flattenActPred, flattenActTopKSample);

  float resActScore;
  hasAct =
      paddle::ConvertPredToJsons(0, false, flattenActPred, flattenActTopKSample,
                                 multimodalActs, resJson, resActScore);
  annoActScore = flattenActPred(ee.waeID);
}

#ifdef SERVER_MODE
using evalserver::EvalRequest;
using evalserver::EvalResponse;
using evalserver::EvalServer;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

void DecodeFrames(const EvalRequest *request, std::vector<cv::Mat> &frames) {
  int nframe = request->nframe();
  std::string bytesStr = request->frames();
  cv::Mat merge(VIEW_H * nframe, VIEW_W, CV_8UC3,
                const_cast<char *>(bytesStr.c_str()));

  for (int i = 0; i < nframe; i++) {
    cv::Rect rect(0, VIEW_H * i, VIEW_W, VIEW_H);
    frames.push_back(merge(rect).clone());
  }
}

class EvalServiceImpl final : public EvalServer::Service {
public:
  EvalServiceImpl(paddle::PaddlePredictor *ctrlPredictor)
      : ctrlPredictor(ctrlPredictor) {
    std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
    PrepareMultimodalActions(actPath, multimodalActs);
  }

  Status infer(ServerContext *context, const EvalRequest *request,
               EvalResponse *reply) override {
    std::vector<cv::Mat> frames;
    DecodeFrames(request, frames);

    obWindow.clear();
    processedImgWindow.clear();

    for (auto img : frames) {
      Eigen::VectorXf flattenImg;
      PreprocessImage(img, flattenImg);

      obWindow.push_back(img);
      processedImgWindow.push_back(flattenImg);
    }

    Eigen::VectorXf flattenActPred, flattenActTopKSample;
    paddle::RunCtrl(ctrlPredictor,
                    {processedImgWindow.begin(), processedImgWindow.end()},
                    flattenActPred, flattenActTopKSample);

    float resActScore;
    std::string resJson;
    bool hasAct = paddle::ConvertPredToJsons(
        0, false, flattenActPred, flattenActTopKSample, multimodalActs, resJson,
        resActScore);
    int nullActAt = ArgSort(flattenActPred)[0];

    reply->set_response(resJson);
    reply->set_response_score(resActScore);
    reply->set_trigger_pred(0.0);
    reply->set_nullact_score(flattenActPred(0));
    reply->set_nullact_id(nullActAt);
    return Status::OK;
  }

private:
  paddle::PaddlePredictor *ctrlPredictor;
  std::vector<MultimodalAction> multimodalActs;
};

void RunServer(paddle::PaddlePredictor *ctrlPredictor) {
  std::string server_address("0.0.0.0:" + std::to_string(FLAGS_port));
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  EvalServiceImpl service(ctrlPredictor);
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(8 * 1024 * 1024); // larger than 6912007
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

#endif // end of server mode

using namespace std;

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Usage : ./eval_r2plus1d ");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Directory of the inference model and params: " << FLAGS_dirname
            << endl;

  std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(actPath, multimodalActs);

  paddle::AnalysisConfig ctrlCfg;
  paddle::PrepareTRTConfig(&ctrlCfg, "ctrl");
  auto ctrlPredictor = paddle::CreatePaddlePredictor(ctrlCfg);
  LOG(INFO) << "Created controller model predictor" << endl;

#ifndef SERVER_MODE
  // ==================================================
  // start of evaluation using -dataTxt
  // ==================================================

  int eeID = 0, nullActTP = 0, nullActFP = 0, nullActFN = 0;
  float actNLL = 0.0;

  if (!boost::filesystem::exists(FLAGS_logdir))
    boost::filesystem::create_directories(FLAGS_logdir);

  ofstream nullActFPLog(FLAGS_logdir + "/null_act_fp.txt");
  ofstream nullActFNLog(FLAGS_logdir + "/null_act_fn.txt");
  ofstream metricLog(FLAGS_logdir + "/metric.txt");

  ifstream infile(FLAGS_dataTxt);
  string line;
  while (true) {
    if (!getline(infile, line))
      break;

    istringstream ss(line);
    EvalEntry ee;
    ss >> ee.label;
    ss >> ee.dir;

    string tmp, frameID;
    ss >> tmp;
    istringstream tt(tmp);
    while (getline(tt, frameID, ',')) {
      ee.frameIDs.push_back(frameID);
    }

    if (ee.label == 1) {
      int scenarioID; // only useful for baselines
      ss >> ee.waeID;
      ss >> scenarioID;

      while (ss) {
        ss >> tmp;
        istringstream bb(tmp);

        string str;
        Eigen::Vector4f bbox;
        int i = 0;
        while (getline(bb, str, ',')) {
          bbox(i) = stof(str);
          i++;
        }

        if (ss)
          ee.bboxLst.push_back(bbox);
      }

    } else if (ee.label == 0) {
      ee.waeID = 0;
    }

    bool hasAct;
    string resJson;
    float annoActScore;
    RunEval(static_cast<paddle::PaddlePredictor *>(ctrlPredictor.get()),
            multimodalActs, ee, hasAct, resJson, annoActScore);

    if (hasAct && ee.label == 1)
      nullActTP++;
    else if (hasAct && ee.label == 0) {
      nullActFP++;
      nullActFPLog << line << endl;
    } else if (!hasAct && ee.label == 1) {
      nullActFN++;
      nullActFNLog << line << endl;
    }

    actNLL += -log(annoActScore);

    eeID++;
    cout << "=========== " << eeID << endl;

    // cout << "===== Eval entry =====" << endl;
    // cout << ee.label << " " << ee.waeID << endl;
    // cout << ee.dir << endl;
    // for (auto i : ee.frameIDs)
    //   cout << i << " ";
    // cout << endl;
    // for (auto i : ee.bboxLst)
    //   cout << "[" << i(0) << ", " << i(1) << ", " << i(2) << ", " << i(3) <<
    //   "]"
    //        << endl;
  }

  double nullActPrecision = (nullActTP + eps) / (nullActTP + nullActFP + eps);
  double nullActRecall = (nullActTP + eps) / (nullActTP + nullActFN + eps);
  cout << "==============================" << endl;
  cout << "Null Act" << endl;
  cout << "Precision: " << nullActPrecision << endl;
  cout << "Recall: " << nullActRecall << endl;
  cout << "==============================" << endl;

  double avgActNLL = actNLL / eeID;
  cout << "Act Average NLL: " << avgActNLL << endl;
  cout << "==============================" << endl;

  metricLog << "#NullAct" << endl;
  metricLog << "Precision Recall" << endl;
  metricLog << nullActPrecision << " " << nullActRecall << endl;
  metricLog << "\nactNLL: " << avgActNLL << endl;
  metricLog << "eeID: " << eeID << endl;

  nullActFPLog.close();
  nullActFNLog.close();
  metricLog.close();

#else
  // ==================================================
  // start of eval server
  // ==================================================
  RunServer(static_cast<paddle::PaddlePredictor *>(ctrlPredictor.get()));

#endif
}
