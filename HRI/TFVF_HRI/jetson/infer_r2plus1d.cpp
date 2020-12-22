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
#include "proactive_greeting.grpc.pb.h"
#include <atomic>
#include <boost/lockfree/spsc_queue.hpp>
#include <grpcpp/grpcpp.h>
#include <thread>
#endif

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXi;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, 1> VectorXi64;

const int NUM_ACT = 317;
const int IMG_RESIZE_0 = 416;
const int IMG_RESIZE = 224;
const int VIEW_H = 360; // 720 / 2
const int VIEW_W = 640; // 1280 / 2
const int OB_WINDOW_LEN = 8;
const int SAFE_ACTS[] = {1, 3, 4, 5, 6, 7, 8, 10};

#ifdef SERVER_MODE
const int Q_SIZE = 100;

struct Request {
  int id;
  cv::Mat ob;
  std::string wakeup;
};

struct Log {
  int id;
  std::vector<cv::Mat> obs;
  Eigen::VectorXf actPred;
  std::string jsonStr;
};

struct Response {
  int id;
  std::string jsonStr;
};

boost::lockfree::spsc_queue<Request> requestQ(Q_SIZE);
boost::lockfree::spsc_queue<Response> ctrlQ(Q_SIZE);
boost::lockfree::spsc_queue<Log> logQ(Q_SIZE * 5);
std::atomic<bool> robotWakeup(false);
#endif

std::deque<cv::Mat> obWindow;
std::deque<Eigen::VectorXf> processedImgWindow;

DEFINE_string(dirname, "./baseline_r2plus1d",
              "Directory of the inference model and params.");
DEFINE_double(tau, 1.0, "Softmax temperature hyperparameter.");
DEFINE_int32(topK, 50, "Number of top-k multimodal actions.");
DEFINE_double(occupy, 5000.0, "Robot occpuy time in ms.");
DEFINE_string(logdir, "./log",
              "Directory of the log, include observations, response JSON.");
DEFINE_bool(gpu, false, "Whether to use GPU config.");
DEFINE_int32(timeout, 2, "Number of frames that are tolerant for timeout.");

#ifdef SERVER_MODE
DEFINE_int32(port, 8888, "Port of gRPC server to bind.");
#else
DEFINE_string(video, "../video.mp4", "Path to video file for local inference.");
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

#ifdef TESTCASE_ONLY
  std::ifstream infile("/mnt/xueyang/Code/xiaodu-hi/jetson/a_t.txt");
  Eigen::VectorXf flatten(3 * OB_WINDOW_LEN * IMG_RESIZE * IMG_RESIZE);
  int i = 0;
  while (i < flatten.size()) {
    float f;
    infile >> f;
    flatten(i) = f;
    i++;
  }
#endif

#ifndef TESTCASE_ONLY
  imgInput->copy_from_cpu(imgTensor.data());
#else
  imgInput->copy_from_cpu(flatten.data());
#endif

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
                        std::string &resJson) {
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

  auto time2 = time();
  LOG(INFO) << "[ConvertPredToJsons] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
  return true;
}

} // namespace paddle

void TestMatchPytorch(paddle::PaddlePredictor *ctrlPredictor) {
  // std::string dir =
  // "/mnt/xueyang/Code/xiaodu-hi/data/full_neg_data/134167_9";
  // std::vector<std::string> ids({"5", "7", "10", "12", "15", "17", "20",
  // "23"});
  std::string dir = "/mnt/xueyang/Code/xiaodu-hi/data/xiaodu_clips_v2/2741_12";
  std::vector<std::string> ids(
      {"104", "106", "109", "111", "114", "116", "119", "122"});

  std::vector<Eigen::VectorXf> imgArray;
  for (size_t i = 0; i < ids.size(); i++) {
    std::string imgFile = dir + "/" + ids[i] + ".jpg";

    cv::Mat img = cv::imread(imgFile);
    Eigen::VectorXf flattenImg;
    PreprocessImage(img, flattenImg);
    imgArray.push_back(flattenImg);
  }

  // For [1, 3, 8, 224, 224] input
  // int lastDim = IMG_RESIZE * IMG_RESIZE;
  // for (int i = 0; i < 3; i++) {
  //   for (int j = 0; j < OB_WINDOW_LEN; j++) {
  //     int k = i * OB_WINDOW_LEN + j;
  //     imgTensor.row(k) = imgArray[j].segment(i * lastDim, lastDim);
  //   }
  // }

  // Eigen::VectorXf flatten(3 * OB_WINDOW_LEN * lastDim);
  // for (int i = 0; i < 3 * OB_WINDOW_LEN; i++)
  //   flatten.segment(i * lastDim, lastDim) = imgTensor.row(i);

  RowMajorMatrixXf imgTensor(OB_WINDOW_LEN, imgArray[0].size());
  for (int i = 0; i < OB_WINDOW_LEN; i++)
    imgTensor.row(i) = imgArray[i];

  Eigen::VectorXf flatten(OB_WINDOW_LEN * imgArray[0].size());
  for (int i = 0; i < OB_WINDOW_LEN; i++)
    flatten.segment(i * imgArray[0].size(), imgArray[0].size()) = imgArray[i];

  std::ofstream outfile("/mnt/xueyang/Code/xiaodu-hi/jetson/b.txt");
  if (outfile.is_open()) {
    for (int i = 0; i < flatten.size(); i++)
      outfile << flatten(i) << std::endl;
    outfile.close();
  }
  // PrintVectorX<Eigen::VectorXf>("imgTensor", flatten, flatten.size());

  Eigen::VectorXf flattenActPred, flattenActTopKSample;
  paddle::RunCtrl(ctrlPredictor, imgArray, flattenActPred,
                  flattenActTopKSample);
  PrintVectorX<Eigen::VectorXf>("actPred", flattenActPred,
                                flattenActPred.size());
}

void TestVideo(paddle::PaddlePredictor *predictor, std::string videoPath) {
  std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(actPath, multimodalActs);

  cv::VideoCapture capture(videoPath);
  if (!capture.isOpened()) {
    std::cout << "Error opening video file: " << videoPath << std::endl;
    return;
  }

  auto time1 = paddle::time();
  int nframe = 0;
  while (true) {
    auto t1 = paddle::time();
    cv::Mat img;
    capture >> img;
    if (img.empty())
      break;
    nframe++;
    auto t2 = paddle::time();
    LOG(INFO) << "[read video] cost: " << paddle::TimeDiff(t1, t2) << "ms"
              << std::endl;

    t1 = paddle::time();
    Eigen::VectorXf flattenImg;
    PreprocessImage(img, flattenImg);
    t2 = paddle::time();
    LOG(INFO) << "[PreprocessImage] cost: " << paddle::TimeDiff(t1, t2) << "ms"
              << std::endl;

    if (obWindow.size() < OB_WINDOW_LEN) {
      obWindow.push_back(img);
      processedImgWindow.push_back(flattenImg);
    } else {
      obWindow.pop_front();
      processedImgWindow.pop_front();

      obWindow.push_back(img);
      processedImgWindow.push_back(flattenImg);
    }

    if (obWindow.size() < OB_WINDOW_LEN)
      continue;

    Eigen::VectorXf flattenActPred, flattenActTopKSample;
    paddle::RunCtrl(predictor,
                    {processedImgWindow.begin(), processedImgWindow.end()},
                    flattenActPred, flattenActTopKSample);
    // PrintVectorX<Eigen::VectorXf>("actPred", flattenActPred,
    //                               flattenActPred.size());

    std::string resJson;
    paddle::ConvertPredToJsons(0, false, flattenActPred, flattenActTopKSample,
                               multimodalActs, resJson);
    std::cout << resJson << std::endl;
  }

  auto time2 = paddle::time();
  double ms = paddle::TimeDiff(time1, time2);
  double fps = nframe / (ms / 1000);
  std::cout << "Avg fps: " << fps << std::endl;
}

#ifdef SERVER_MODE
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using grpc::WriteOptions;

void ProcessLog() {
  std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(actPath, multimodalActs);

  Log log;
  while (true) {
    bool hasLog = logQ.pop(log);
    if (!hasLog) {
      // log writer is not so urgent.
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    std::string logdir = FLAGS_logdir + "/" + std::to_string(log.id);
    if (boost::filesystem::create_directories(logdir)) {
      for (size_t i = 0; i < log.obs.size(); i++)
        cv::imwrite(logdir + "/" + std::to_string(i) + ".jpg", log.obs[i]);

      std::string jsonfile = logdir + "/res.txt";
      std::ofstream outfile(jsonfile);
      if (outfile.is_open()) {
        outfile << log.jsonStr << std::endl;

        for (size_t i : ArgSort(log.actPred))
          outfile << log.actPred(i) << " " << multimodalActs[i].to_json()
                  << std::endl;
        outfile.close();
      } else {
        LOG(WARNING) << "Cannot create " + jsonfile << std::endl;
      }

    } else {
      LOG(WARNING) << "Cannot create " + logdir << std::endl;
    }
  }
}

void ProcessRequest(paddle::PaddlePredictor *ctrlPredictor) {
  // random engine for diversity
  // std::default_random_engine rnd;
  // rnd.seed(std::chrono::system_clock::now().time_since_epoch().count());
  // std::uniform_real_distribution<double> uniform(0, 1);

  std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(actPath, multimodalActs);

  while (true) {
    Request req;
    bool hasRequest = requestQ.pop(req);
    if (!hasRequest)
      continue;

    if (robotWakeup) {
      obWindow.clear();
      processedImgWindow.clear();
      continue;
    }

    auto t1 = paddle::time();
    Eigen::VectorXf flattenImg;
    // cv::imwrite("infer_img" + std::to_string(req.id) + ".jpg", req.ob);
    PreprocessImage(req.ob, flattenImg);

    auto t2 = paddle::time();
    LOG(INFO) << "[PreprocessImage] cost: " << paddle::TimeDiff(t1, t2) << "ms"
              << std::endl;

    if (obWindow.size() < OB_WINDOW_LEN) {
      obWindow.push_back(req.ob);
      processedImgWindow.push_back(flattenImg);
    } else {
      obWindow.pop_front();
      processedImgWindow.pop_front();

      obWindow.push_back(req.ob);
      processedImgWindow.push_back(flattenImg);
    }

    if (obWindow.size() < OB_WINDOW_LEN)
      continue;

    Eigen::VectorXf flattenActPred, flattenActTopKSample;
    bool paddleError = false;
    try {
      paddle::RunCtrl(ctrlPredictor,
                      {processedImgWindow.begin(), processedImgWindow.end()},
                      flattenActPred, flattenActTopKSample);
      // PrintVectorX<Eigen::VectorXf>("actPred", flattenActPred,
      //                               flattenActPred.size());
    } catch (const std::runtime_error &error) {
      paddleError = true;
    }

    if (paddleError)
      continue;

    std::string resJson;
    bool hasAct = paddle::ConvertPredToJsons(req.id, false, flattenActPred,
                                             flattenActTopKSample,
                                             multimodalActs, resJson);

    if (hasAct) {
      Response res;
      res.id = req.id;
      res.jsonStr = resJson;
      ctrlQ.push(res);

      Log log;
      log.id = req.id;
      log.jsonStr = resJson;
      log.obs = {obWindow.begin(), obWindow.end()};
      log.actPred = flattenActPred;
      logQ.push(log);
    }
  }
}

cv::Mat DecodeImage(grpc::VideoRequest &request) {
  std::string bytesStr = request.curframe();
  cv::Mat frame(VIEW_H, VIEW_W, CV_8UC3, const_cast<char *>(bytesStr.c_str()));
  return frame;
}

class ProactiveGreetingServiceImpl final
    : public grpc::ProactiveGreeting::Service {
  Status infer(ServerContext *context,
               ServerReaderWriter<grpc::InferResponse, grpc::VideoRequest>
                   *stream) override {
    grpc::VideoRequest request;
    auto lastValidResTime = paddle::time();
    while (stream->Read(&request)) {
      Request req;
      req.ob = DecodeImage(request).clone();
      req.id = static_cast<int>(request.reqid());
      req.wakeup = request.wakeup();
      robotWakeup = req.wakeup == "1";
      requestQ.push(req);

      grpc::InferResponse res;
      Response resp;
      bool hasAct = ctrlQ.pop(resp);
      bool respSet = hasAct;
      while (hasAct) {
        // Get last one
        hasAct = ctrlQ.pop(resp);
        respSet |= hasAct;
      }

      if (respSet) {
        double deltaT = paddle::TimeDiff(lastValidResTime, paddle::time());
        if (req.id - resp.id > FLAGS_timeout || deltaT < FLAGS_occupy) {
          robotWakeup = true;
          Response r;
          while (ctrlQ.pop(r))
            continue;

          std::string jsonStr = "{\"QueryID\": " + std::to_string(req.id) + "}";
          res.set_response(jsonStr);

          if (context->IsCancelled()) {
            stream->WriteLast(res, WriteOptions().set_last_message());
            return Status::CANCELLED;
          }

          stream->Write(res);
          LOG(INFO) << jsonStr << std::endl;
          LOG(WARNING) << "Timeout..." << std::endl;
        } else {
          res.set_response(resp.jsonStr);
          if (context->IsCancelled()) {
            stream->WriteLast(res, WriteOptions().set_last_message());
            return Status::CANCELLED;
          }

          stream->Write(res);
          LOG(INFO) << resp.jsonStr << std::endl;
          lastValidResTime = paddle::time();
        }

      } else {
        std::string jsonStr = "{\"QueryID\": " + std::to_string(req.id) + "}";
        res.set_response(jsonStr);
        if (context->IsCancelled()) {
          stream->WriteLast(res, WriteOptions().set_last_message());
          return Status::CANCELLED;
        }

        stream->Write(res);
        LOG(INFO) << jsonStr << std::endl;
      }
    }
    return Status::OK;
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:" + std::to_string(FLAGS_port));
  ProactiveGreetingServiceImpl service;
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

#endif // end of grpc server

using namespace std;

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Usage : ./infer_r2plus1d ");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Directory of the inference model and params: " << FLAGS_dirname
            << endl;

  paddle::AnalysisConfig ctrlCfg;
  paddle::PrepareTRTConfig(&ctrlCfg, "ctrl");
  auto ctrlPredictor = paddle::CreatePaddlePredictor(ctrlCfg);
  LOG(INFO) << "Created controller model predictor" << endl;

#ifdef TESTCASE_ONLY
  TestMatchPytorch(static_cast<paddle::PaddlePredictor *>(ctrlPredictor.get()));
#else // TESTCASE_ONLY else

#ifndef SERVER_MODE
  TestVideo(static_cast<paddle::PaddlePredictor *>(ctrlPredictor.get()),
            FLAGS_video);
#else
  thread worker(ProcessRequest,
                static_cast<paddle::PaddlePredictor *>(ctrlPredictor.get()));
  worker.detach();

  thread logger(ProcessLog);
  logger.detach();

  RunServer();
#endif // end of server mode inference

#endif // TESTCASE_ONLY endif
}
