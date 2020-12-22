#include "instance.hpp"
#include "multimodal_act.hpp"
#include "paddle/include/paddle_inference_api.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <mutex>
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

#ifdef SERVER_MODE
#include "proactive_greeting.grpc.pb.h"
#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <grpcpp/grpcpp.h>
#include <thread>
#endif

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMajorMatrixXi;
typedef Eigen::Matrix<int64_t, Eigen::Dynamic, 1> VectorXi64;
typedef std::vector<std::vector<size_t>> LoD;
typedef std::vector<Instance> FrameInstances;

const double PI = 3.141592653589793;
const int NUM_ACT = 317;
const int IMG_RESIZE = 416;
const int VIEW_H = 360; // 720 / 2
const int VIEW_W = 640; // 1280 / 2
const int OB_WINDOW_LEN = 10;
const int ROI_FEAT_RESOLUTION = 5;
const int TOKENS_PER_FRAME = 20;
const int FM_SCALE = 32;
const int ROI_FEAT_DIM = 512;
const int VISUAL_TOKEN_DIM = 562;
const int PRED_DIM = 6;
const int INTERESTED_CALSS[] = {0, 24, 26, 28, 27, 67};
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
  float confidence;
  std::vector<cv::Mat> obs;
  std::vector<FrameInstances> frameInstArray;
  Eigen::VectorXf objPred;
  Eigen::VectorXf actPred;
  std::string jsonStr;
};

struct Response {
  int id;
  std::string jsonStr;
  bool lagSensitive;
};

boost::lockfree::spsc_queue<Request> requestQ(Q_SIZE);
boost::lockfree::spsc_queue<Response> ctrlQ(Q_SIZE);
boost::lockfree::spsc_queue<Log> logQ(Q_SIZE * 5);
std::atomic<bool> robotWakeup(false);
#endif

#if defined(SERVER_MODE) && defined(ASYNC_INFER)

struct PreprocessRes {
  int id;
  cv::Mat ob; // just for easy logging
  Eigen::VectorXf flattenImg;
};

struct DetectorRes {
  int id;
  LoD lod;
  cv::Mat ob; // just for easy logging
  int objCount;
  Eigen::VectorXf flattenPred;
  Eigen::VectorXf flattenFeatureMap;
};

boost::lockfree::spsc_queue<PreprocessRes> preprocessQ(Q_SIZE);
boost::lockfree::spsc_queue<DetectorRes> detectorQ(Q_SIZE);
#endif // end of data def for async inference

std::deque<cv::Mat> obWindow;
std::deque<Eigen::VectorXf> visualTokensWindow;
std::deque<Eigen::VectorXf> paddingMaskWindow;
std::deque<Eigen::VectorXi> sharedFrameIds;
std::deque<FrameInstances> frameInstWindow;
// std::deque<Eigen::VectorXf> recentInterests;

DEFINE_string(dirname, "./xiaodu_hi_v3",
              "Directory of the inference model and params.");
DEFINE_double(th, 0.5, "Threshold for interaction trigger.");
DEFINE_double(tau, 1.0, "Softmax temperature hyperparameter.");
DEFINE_int32(topK, 50, "Number of top-k multimodal actions.");
DEFINE_double(occupy, 5000.0, "Robot occpuy time in ms.");
DEFINE_int32(minInstSize, 16000, "Minimum instance view size.");
DEFINE_string(logdir, "./log",
              "Directory of the log, include observations, response JSON.");
DEFINE_bool(gpu, false, "Whether to use GPU config.");
DEFINE_int32(timeout, 2, "Number of frames that are tolerant for timeout.");

DEFINE_bool(salutation, false, "Whether to integrate salutation classifier.");
DEFINE_double(
    saluL1, 0.4,
    "Confendence gap to trust root prediction of saluation classifier.");
DEFINE_double(saluL2, 0.4,
              "Confendence gap to trust left or right prediction of saluation "
              "classifier.");
DEFINE_double(
    objTH, 0.3,
    "Threshold to check whether the person has interaction intention.");

DEFINE_bool(
    ensemble, false,
    "Whether to ensemble trigger, object detector and null action controller.");

#ifdef LOCAL_INFER
DEFINE_string(video, "../video.mp4", "Path to video file for local inference.");
#endif

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
  // See perception/common/utils.py::yolov4_img_preprocess
  double aspectRatio =
      std::min(IMG_RESIZE * 1.0 / image.rows, IMG_RESIZE * 1.0 / image.cols);
  int newH = static_cast<int>(std::floor(image.rows * aspectRatio));
  int newW = static_cast<int>(std::floor(image.cols * aspectRatio));
  cv::resize(image, image, cv::Size(newW, newH));

  cv::Mat boxedImg(IMG_RESIZE, IMG_RESIZE, CV_8UC3, cv::Scalar(128, 128, 128));

  int yOffset = static_cast<int>(std::floor(IMG_RESIZE - newH) / 2.0);
  int xOffset = static_cast<int>(std::floor(IMG_RESIZE - newW) / 2.0);
  image.copyTo(boxedImg(cv::Rect(xOffset, yOffset, newW, newH)));

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

void GetSalutation(const Eigen::VectorXf objPred,
                   const FrameInstances &instances, std::string &salu,
                   int &objCount) {
  float maxObj = 0.0;
  objCount = 0;
  salu = "";
  for (size_t i = 0; i < instances.size(); i++) {
    if (instances[i].classID == 0 && objPred(i) > FLAGS_objTH) {
      objCount++;
      if (objPred(i) > maxObj) {
        maxObj = objPred(i);
        salu = instances[i].get_salutation(FLAGS_saluL1, FLAGS_saluL2);
      }
    }
  }

  if (objCount > 1)
    salu = "你们"; // or "大家"
  else if (salu == "")
    salu = "你";
}

std::string GetPronoun(int objCount) {
  if (objCount > 1)
    return "大家";
  else
    return "你";
}

bool CheckNearField(const FrameInstances &instances, double areaTH = 0.30) {
  bool isNear = false;
  double viewArea = static_cast<double>(VIEW_H * VIEW_W);
  for (auto it = instances.begin(); it != instances.end(); it++) {
    if (it->get_area_size() / viewArea > areaTH) {
      isNear = true;
      break;
    }
  }
  return isNear;
}

bool CheckLagSensitive(const FrameInstances &instances) {
  // TODO: only consider the potential objects
  bool isSensitive = false;
  for (size_t i = 0; i < instances.size(); i++) {
    if (instances[i].classID != 0)
      continue;

    float h = instances[i].bbox(3) - instances[i].bbox(1);
    float x1 = std::abs(instances[i].bbox(0) - 0);
    float x2 = std::abs(VIEW_W - instances[i].bbox(2));
    float x = std::min(x1, x2);
    // std::cout << "================" << std::endl;
    // std::cout << "h / VIEW_H: " << h / VIEW_H << std::endl;
    // std::cout << "x / VIEW_W: " << x / VIEW_W << std::endl;
    if (h / VIEW_H > 0.9 && x / VIEW_W < 0.1) {
      isSensitive = true;
      break;
    }
  }
  return isSensitive;
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
    // config->EnableUseGpu(1000, 5);
  } else {
    config->DisableGpu();
    config->SetCpuMathLibraryNumThreads(8);
  }
  config->SwitchUseFeedFetchOps(false);
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrOptim(true);
}

void GenerateSizeInputData(int seqLen, int h, int w,
                           Eigen::VectorXi &flattenSizeData) {
  flattenSizeData.resize(seqLen * 2);
  for (int i = 0; i < seqLen; i++) {
    flattenSizeData(2 * i) = h;
    flattenSizeData(2 * i + 1) = w;
  }
}

bool CompareInstances(Instance &i, Instance &j) {
  return (i.get_area_size() > j.get_area_size());
}

void ConvertPredToInstances(int nframe, const LoD &predLod,
                            const std::vector<Eigen::VectorXf> &preds,
                            std::vector<FrameInstances> &frameInstArray) {
  Eigen::VectorXf flattenPred = preds[0];
  Eigen::VectorXf flattenRoisFeats = preds[1];

  Eigen::VectorXf flattenRootPred, flattenLeftPred, flattenRightPred;
  if (FLAGS_salutation) {
    flattenRootPred = preds[2];
    flattenLeftPred = preds[3];
    flattenRightPred = preds[4];
  }

  frameInstArray.resize(nframe);
  for (int i = 0; i < nframe; i++)
    frameInstArray[i].resize(TOKENS_PER_FRAME);

  int nclass = sizeof(INTERESTED_CALSS) / sizeof(INTERESTED_CALSS[0]);
  std::set<int> interested(INTERESTED_CALSS, INTERESTED_CALSS + nclass);

  for (int i = 0; i < nframe; i++) {
    // Filter, see interaction/common/data_v2.py::filter_instances
    int npred = predLod[0][i + 1] - predLod[0][i];
    FrameInstances initFrameInst;
    for (int j = 0; j < npred; j++) {
      int k = predLod[0][i] + j;
      Instance inst(flattenPred.segment(k * PRED_DIM, PRED_DIM),
                    flattenRoisFeats.segment(k * ROI_FEAT_DIM, ROI_FEAT_DIM));

      if (FLAGS_salutation)
        inst.update_salutation(
            flattenRootPred.segment(k * 2, 2),
            flattenLeftPred.segment(k * SALU_LEFT_DIM, SALU_LEFT_DIM),
            flattenRightPred.segment(k * SALU_RIGHT_DIM, SALU_RIGHT_DIM));

      if (interested.find(inst.classID) != interested.end()) {
        double instSize =
            (inst.bbox(2) - inst.bbox(0)) * (inst.bbox(3) - inst.bbox(1));
        if (inst.classID == 0 && instSize > FLAGS_minInstSize)
          initFrameInst.push_back(inst);
        else if (inst.classID != 0)
          initFrameInst.push_back(inst);
      }
    }

    std::cout << "frame " << i
              << ", found interested instances: " << initFrameInst.size()
              << std::endl;

    if (npred == 0) {
      for (int j = 0; j < TOKENS_PER_FRAME; j++)
        frameInstArray[i][j] = Instance();
    } else if (initFrameInst.size() <= TOKENS_PER_FRAME) {
      for (size_t j = 0; j < initFrameInst.size(); j++)
        frameInstArray[i][j] = initFrameInst[j];
      for (size_t j = initFrameInst.size(); j < TOKENS_PER_FRAME; j++)
        frameInstArray[i][j] = Instance();
    } else {
      int nperson = 0;
      for (size_t j = 0; j < initFrameInst.size(); j++)
        if (initFrameInst[j].classID == 0)
          nperson++;

      if (nperson < TOKENS_PER_FRAME) {
        int mis = TOKENS_PER_FRAME - nperson;
        for (size_t j = 0; j < initFrameInst.size(); j++)
          if (initFrameInst[j].classID == 0)
            frameInstArray[i][j] = initFrameInst[j];

        // TODO: shuffle the instances that are not persons before added
        size_t j = 0, k = nperson;
        while (mis > 0 && j < initFrameInst.size()) {
          if (initFrameInst[j].classID != 0) {
            frameInstArray[i][k] = initFrameInst[j];
            k++;
            mis--;
          }
          j++;
        }
      } else {
        std::sort(initFrameInst.begin(), initFrameInst.end(), CompareInstances);
        for (size_t j = 0, k = 0;
             j < initFrameInst.size() && k < TOKENS_PER_FRAME; j++) {
          if (initFrameInst[j].classID == 0) {
            frameInstArray[i][k] = initFrameInst[j];
            k++;
          }
        }
      }
    }
  } // loop frames
}

void Meshgrid(const Eigen::VectorXf &x, const Eigen::VectorXf &y,
              RowMajorMatrixXf &X, RowMajorMatrixXf &Y) {
  int nx = x.size(), ny = y.size();
  X.resize(ny, nx);
  Y.resize(ny, nx);
  for (int i = 0; i < ny; i++)
    X.row(i) = x;
  for (int i = 0; i < ny; i++)
    Y.col(i) = y;
}

void GetPosEmb(const Instance &inst, Eigen::VectorXf &posEmb) {
  // See perception/common/utils.py::get_bbox_pos_emb
  posEmb.resize(2 * ROI_FEAT_RESOLUTION * ROI_FEAT_RESOLUTION);

  if (inst.classID == -1) {
    posEmb.setZero();
    return;
  }

  Eigen::Vector4f halfWH(
      {VIEW_W + 0.0, VIEW_H + 0.0, VIEW_W + 0.0, VIEW_H + 0.0});
  halfWH = halfWH / 2.0;

  Eigen::Vector4f bbox = (inst.bbox - halfWH) * (PI / 2.0);
  bbox = bbox.array() / halfWH.array();

  Eigen::VectorXf xPos, yPos;
  xPos.setLinSpaced(ROI_FEAT_RESOLUTION, bbox(0), bbox(2));
  yPos.setLinSpaced(ROI_FEAT_RESOLUTION, bbox(1), bbox(3));
  xPos = xPos.array().sin();
  yPos = yPos.array().sin();

  RowMajorMatrixXf xPosEmb, yPosEmb;
  Meshgrid(xPos, yPos, xPosEmb, yPosEmb);

  posEmb << Eigen::Map<Eigen::VectorXf>(xPosEmb.data(), xPosEmb.size()),
      Eigen::Map<Eigen::VectorXf>(yPosEmb.data(), yPosEmb.size());
}

void GetVisualToken(const Instance &inst, Eigen::VectorXf &visualToken) {
  // See interaction/common/data_v2.py::convert_instances_to_visual_tokens
  Eigen::VectorXf posEmb;
  GetPosEmb(inst, posEmb);

  visualToken.resize(VISUAL_TOKEN_DIM);
  visualToken << posEmb, inst.feat;
}

void GetAttnMask(int nframe, Eigen::VectorXf &flattenAttnMask) {
  int seqLen = nframe * TOKENS_PER_FRAME;
  flattenAttnMask.resize(seqLen * seqLen);
  flattenAttnMask.setZero();

  for (int i = 0; i < seqLen; i++) {
    int len = (i / TOKENS_PER_FRAME + 1) * TOKENS_PER_FRAME;
    int offset = i * seqLen;
    flattenAttnMask.segment(offset, len) = Eigen::VectorXf::Ones(len);
  }
}

void GetObjMask(const FrameInstances &instances, Eigen::VectorXf &objMask) {
  objMask.resize(instances.size());
  for (size_t i = 0; i < instances.size(); i++) {
    if (instances[i].classID == 0)
      objMask(i) = 1.0;
    else
      objMask(i) = 0.0;
  }
}

int RunDetector(PaddlePredictor *predictor,
                const std::vector<Eigen::VectorXf> &imgArray, LoD &predLod,
                Eigen::VectorXf &flattenPred,
                Eigen::VectorXf &flattenFeatureMap) {
  auto time1 = time();

  int nframe = imgArray.size();

  RowMajorMatrixXf imgTensor(nframe, imgArray[0].size());
  for (size_t i = 0; i < imgArray.size(); i++)
    imgTensor.row(i) = imgArray[i];

  auto imgInput = predictor->GetInputTensor("image");
  imgInput->Reshape({nframe, 3, IMG_RESIZE, IMG_RESIZE});
  imgInput->copy_from_cpu(imgTensor.data());

  auto imSizeInput = predictor->GetInputTensor("im_size");
  Eigen::VectorXi flattenImSizeInput;
  GenerateSizeInputData(nframe, VIEW_H, VIEW_W, flattenImSizeInput);
  imSizeInput->Reshape({nframe, 2});
  imSizeInput->copy_from_cpu(flattenImSizeInput.data());

  auto inSizeInput = predictor->GetInputTensor("in_size");
  Eigen::VectorXi flattenInSizeInput;
  GenerateSizeInputData(nframe, IMG_RESIZE, IMG_RESIZE, flattenInSizeInput);
  inSizeInput->Reshape({nframe, 2});
  inSizeInput->copy_from_cpu(flattenInSizeInput.data());

  CHECK(predictor->ZeroCopyRun());

  auto outputNames = predictor->GetOutputNames();
  auto pred = predictor->GetOutputTensor(outputNames[0]);
  auto fm = predictor->GetOutputTensor(outputNames[1]);

  predLod = pred->lod();

  auto predShape = pred->shape();
  int predSize = std::accumulate(predShape.begin(), predShape.end(), 1,
                                 std::multiplies<int>());
  flattenPred.resize(predSize);
  pred->copy_to_cpu(flattenPred.data());

  auto fmShape = fm->shape();
  int fmSize = std::accumulate(fmShape.begin(), fmShape.end(), 1,
                               std::multiplies<int>());
  flattenFeatureMap.resize(fmSize);
  fm->copy_to_cpu(flattenFeatureMap.data());

  auto time2 = time();
  LOG(INFO) << "[RunDetector] nframe: " << nframe
            << ", cost: " << TimeDiff(time1, time2) << "ms" << std::endl;

  if (predSize == 1)
    return 0;
  else
    return predShape[0];
}

void RunVisualTokenizer(PaddlePredictor *predictor, int nframe,
                        int frameIdOffset, int npred, const LoD &predLod,
                        const Eigen::VectorXf &flattenPred,
                        const Eigen::VectorXf &flattenFeatureMap,
                        Eigen::VectorXf &flattenVisualTokens,
                        Eigen::VectorXf &flattenPaddingMask,
                        Eigen::VectorXi &flattenFrameIds,
                        std::vector<FrameInstances> &frameInstArray) {
  auto time1 = time();

  flattenVisualTokens.resize(nframe * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM);
  flattenPaddingMask.resize(nframe * TOKENS_PER_FRAME);
  flattenFrameIds.resize(nframe * TOKENS_PER_FRAME);
  frameInstArray.resize(nframe);
  for (int i = 0; i < nframe; i++)
    frameInstArray[i].resize(TOKENS_PER_FRAME);

  flattenVisualTokens.setZero();
  flattenPaddingMask.setZero();
  for (int i = 0; i < flattenFrameIds.size(); i++)
    flattenFrameIds(i) = frameIdOffset + i / TOKENS_PER_FRAME;

  if (flattenPred.size() > 1) {
    auto fmInput = predictor->GetInputTensor("fm");
    fmInput->Reshape(
        {nframe, ROI_FEAT_DIM, IMG_RESIZE / FM_SCALE, IMG_RESIZE / FM_SCALE});
    fmInput->copy_from_cpu(flattenFeatureMap.data());

    auto predInput = predictor->GetInputTensor("pred");
    predInput->Reshape({npred, PRED_DIM});
    predInput->SetLoD(predLod);
    predInput->copy_from_cpu(flattenPred.data());

    CHECK(predictor->ZeroCopyRun());

    auto outputNames = predictor->GetOutputNames();
    auto roisFeats = predictor->GetOutputTensor(outputNames[0]);
    auto roisFeatsShape = roisFeats->shape();
    int roisFeatsSize =
        std::accumulate(roisFeatsShape.begin(), roisFeatsShape.end(), 1,
                        std::multiplies<int>());
    Eigen::VectorXf flattenRoisFeats;
    flattenRoisFeats.resize(roisFeatsSize);
    roisFeats->copy_to_cpu(flattenRoisFeats.data());

    if (FLAGS_salutation) {
      auto rootPred = predictor->GetOutputTensor(outputNames[1]);
      auto rootPredShape = rootPred->shape();
      int rootPredSize =
          std::accumulate(rootPredShape.begin(), rootPredShape.end(), 1,
                          std::multiplies<int>());

      auto leftPred = predictor->GetOutputTensor(outputNames[2]);
      auto leftPredShape = leftPred->shape();
      int leftPredSize =
          std::accumulate(leftPredShape.begin(), leftPredShape.end(), 1,
                          std::multiplies<int>());

      auto rightPred = predictor->GetOutputTensor(outputNames[3]);
      auto rightPredShape = rightPred->shape();
      int rightPredSize =
          std::accumulate(rightPredShape.begin(), rightPredShape.end(), 1,
                          std::multiplies<int>());

      Eigen::VectorXf flattenRootPred, flattenLeftPred, flattenRightPred;
      flattenRootPred.resize(rootPredSize);
      flattenLeftPred.resize(leftPredSize);
      flattenRightPred.resize(rightPredSize);

      rootPred->copy_to_cpu(flattenRootPred.data());
      leftPred->copy_to_cpu(flattenLeftPred.data());
      rightPred->copy_to_cpu(flattenRightPred.data());

      std::vector<Eigen::VectorXf> preds{flattenPred, flattenRoisFeats,
                                         flattenRootPred, flattenLeftPred,
                                         flattenRightPred};
      ConvertPredToInstances(nframe, predLod, preds, frameInstArray);
    } else {
      std::vector<Eigen::VectorXf> preds{flattenPred, flattenRoisFeats};
      ConvertPredToInstances(nframe, predLod, preds, frameInstArray);
    }

    for (size_t i = 0; i < frameInstArray.size(); i++) {
      for (size_t j = 0; j < frameInstArray[i].size(); j++) {
        Eigen::VectorXf visualToken;
        GetVisualToken(frameInstArray[i][j], visualToken);
        // std::cout << "==========" << i << ", " << j << "==========" <<
        // std::endl; PrintVectorX<Eigen::VectorXf>(
        //     "instance feat", frameInstArray[i][j].feat, 10);
        // PrintVectorX<Eigen::VectorXf>("visualToken", visualToken, 60);
        flattenVisualTokens.segment(
            (i * TOKENS_PER_FRAME + j) * VISUAL_TOKEN_DIM, VISUAL_TOKEN_DIM) =
            visualToken;

        if (frameInstArray[i][j].classID != -1)
          flattenPaddingMask(i * TOKENS_PER_FRAME + j) = 1.0;
      }
    }
  }

  auto time2 = time();
  LOG(INFO) << "[RunVisualTokenizer] nframe: " << nframe
            << ", cost: " << TimeDiff(time1, time2) << "ms" << std::endl;
}

void RunAttnCtrl(PaddlePredictor *predictor, int nframe, int naction,
                 const Eigen::VectorXf &flattenVisualTokens,
                 const Eigen::VectorXf &flattenPaddingMask,
                 const Eigen::VectorXi &flattenFrameIds,
                 Eigen::VectorXf &flattenTriggerPred,
                 Eigen::VectorXf &flattenObjPred,
                 Eigen::VectorXf &flattenActPred,
                 Eigen::VectorXf &flattenActTopKSample) {
  auto time1 = time();

  int seqLen = nframe * TOKENS_PER_FRAME;
  flattenTriggerPred.resize(nframe);
  flattenObjPred.resize(seqLen);
  flattenActPred.resize(nframe * naction);
  flattenActTopKSample.resize(nframe);

  auto visualTokenInput = predictor->GetInputTensor("visual_tokens");
  visualTokenInput->Reshape({1, seqLen, VISUAL_TOKEN_DIM});
  visualTokenInput->copy_from_cpu(flattenVisualTokens.data());

  auto paddingMaskInput = predictor->GetInputTensor("padding_mask");
  paddingMaskInput->Reshape({1, seqLen});
  paddingMaskInput->copy_from_cpu(flattenPaddingMask.data());

  auto frameIdsInput = predictor->GetInputTensor("frame_ids");
  frameIdsInput->Reshape({1, seqLen});
  VectorXi64 ids = flattenFrameIds.cast<int64_t>();
  frameIdsInput->copy_from_cpu(ids.data());

  Eigen::VectorXf flattenAttnMask;
  GetAttnMask(nframe, flattenAttnMask);
  auto attnMaskInput = predictor->GetInputTensor("attn_mask");
  attnMaskInput->Reshape({1, seqLen, seqLen});
  attnMaskInput->copy_from_cpu(flattenAttnMask.data());

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
  auto triggerPred = predictor->GetOutputTensor(outputNames[0]);
  auto objPred = predictor->GetOutputTensor(outputNames[1]);
  auto actPred = predictor->GetOutputTensor(outputNames[2]);
  auto actTopKSample = predictor->GetOutputTensor(outputNames[3]);

  triggerPred->copy_to_cpu(flattenTriggerPred.data());
  objPred->copy_to_cpu(flattenObjPred.data());
  actPred->copy_to_cpu(flattenActPred.data());
  actTopKSample->copy_to_cpu(flattenActTopKSample.data());

  auto time2 = time();
  LOG(INFO) << "[RunAttnCtrl] nframe: " << nframe
            << ", cost: " << TimeDiff(time1, time2) << "ms" << std::endl;
}

bool ConvertPredToJsons(float triggerPred, int reqID, bool useSkill,
                        const Eigen::VectorXf &objPred,
                        const Eigen::VectorXf &actPred,
                        const Eigen::VectorXf &actTopKSample,
                        const FrameInstances &instances,
                        const std::vector<MultimodalAction> &multimodalActs,
                        std::string &resJson) {
  auto time1 = time();
  resJson = "{}";
  if (!FLAGS_ensemble && triggerPred < FLAGS_th)
    return false;

  int objCount;
  std::string salu;
  GetSalutation(objPred, instances, salu, objCount);

  if (FLAGS_ensemble) {
    // TODO: investigate better ensemble strategy, e.g. weighted voting
    size_t nullActAt = ArgSort(actPred)[0];
    int supports = static_cast<int>(triggerPred > FLAGS_th);
    supports += static_cast<int>(objCount > 0);
    supports += static_cast<int>(nullActAt != 0);
    if (supports < 3.0 / 2)
      return false;
    // if (supports < 3.0)
    //   return false; // all supports!
  }

  std::default_random_engine rnd;
  rnd.seed(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> uniform(0, 1);

  int safeActNum = sizeof(SAFE_ACTS) / sizeof(SAFE_ACTS[0]);
  int sampleID = static_cast<int>(actTopKSample(actTopKSample.size() - 1));
  std::string talk = multimodalActs[sampleID].talk;
  // LOG(INFO) << "req id: " << std::to_string(reqID) << ", init talk: " << talk
  // << std::endl;

  if (!FLAGS_salutation && talk.find("C") != talk.npos) {
    // ignore the salutation.
    sampleID = SAFE_ACTS[static_cast<int>(uniform(rnd) * safeActNum)];
  }

  if (talk.find("拍照") != talk.npos) {
    // TODO: update this!
    // Check the camera/cell phone exist
    bool hasPhone = false;
    for (auto it = instances.begin(); it != instances.end(); it++) {
      if (it->classID == 67) {
        hasPhone = true;
        break;
      }
    }

    if (!hasPhone)
      sampleID = SAFE_ACTS[static_cast<int>(uniform(rnd) * safeActNum)];
  }

  std::string pronoun = GetPronoun(objCount);
  int hour = GetCurrentHour();

  MultimodalAction ma = multimodalActs[sampleID];
  resJson = ma.to_json(hour, reqID, useSkill, salu, pronoun);

  auto time2 = time();
  LOG(INFO) << "[ConvertPredToJsons] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
  return true;
}

} // namespace paddle

// ==============================
// Test Cases
// ==============================
void TestPreprocessImage(std::string filename) {
  cv::Mat image = cv::imread(filename);
  Eigen::VectorXf flattenImg;
  PreprocessImage(image, flattenImg, "../test_preprocess_image.jpg");
  PrintVectorX<Eigen::VectorXf>("flattenImg", flattenImg, 20);
}

void TestGenerateSizeInputData() {
  int seqLen = 5, h = VIEW_H, w = VIEW_W;
  Eigen::VectorXi sizeData;
  paddle::GenerateSizeInputData(seqLen, h, w, sizeData);
  PrintVectorX<Eigen::VectorXi>("sizeData", sizeData, seqLen * 2);
}

void TestRunDetector(paddle::PaddlePredictor *predictor, std::string posFile,
                     std::string negFile) {
  cv::Mat posImg = cv::imread(posFile);
  cv::Mat negImg = cv::imread(negFile);

  Eigen::VectorXf posFlattenImg, negFlattenImg;
  PreprocessImage(posImg, posFlattenImg);
  PreprocessImage(negImg, negFlattenImg);

  Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
  Eigen::VectorXi flattenFrameIds;

  std::vector<Eigen::VectorXf> imgArray({posFlattenImg, negFlattenImg});
  LoD predLod;
  Eigen::VectorXf flattenPred, flattenFeatureMap;
  int objCount = paddle::RunDetector(predictor, imgArray, predLod, flattenPred,
                                     flattenFeatureMap);
  std::cout << "Detect on pos and neg images... Found " << objCount
            << " objects" << std::endl;

  int steps = 100;
  while (steps) {
    std::vector<Eigen::VectorXf> imgArray2({negFlattenImg});
    objCount = paddle::RunDetector(predictor, imgArray2, predLod, flattenPred,
                                   flattenFeatureMap);
    std::cout << "Detect on neg image... Found " << objCount << " objects"
              << std::endl;
    steps--;
  }
}

void TestGetPosEmb() {
  Eigen::VectorXf posEmb;
  Instance padInst;
  paddle::GetPosEmb(padInst, posEmb);
  PrintVectorX<Eigen::VectorXf>("pad pos emb", posEmb, posEmb.size());

  Eigen::VectorXf pred;
  pred.resize(6);
  pred << 0, 0.9, 300.0, 160.0, 340.0, 200.0;
  Instance randInst(pred, Eigen::VectorXf::Random(ROI_FEAT_DIM));
  paddle::GetPosEmb(randInst, posEmb);
  PrintVectorX<Eigen::VectorXf>("pos emb", posEmb, posEmb.size());
  std::cout << "expected pos emb: -0.09801714, -0.04906767,  0.        ,  "
               "0.04906767,  0.09801714 ..."
            << std::endl;
}

void TestGetVisualToken() {
  Eigen::VectorXf visualToken;
  Instance padInst;
  paddle::GetVisualToken(padInst, visualToken);
  bool allZero = true;
  for (int i = 0; i < visualToken.size(); i++) {
    if (visualToken(i) != 0)
      allZero = false;
  }

  if (!allZero)
    std::cout << "WRONG! For padding instance, its visual token is not zeros."
              << std::endl;

  Eigen::VectorXf pred;
  pred.resize(6);
  pred << 0, 0.9, 300.0, 160.0, 340.0, 200.0;
  Eigen::VectorXf feat = Eigen::VectorXf::Random(ROI_FEAT_DIM);
  Instance randInst(pred, feat);
  paddle::GetVisualToken(randInst, visualToken);
  PrintVectorX<Eigen::VectorXf>("token", visualToken, visualToken.size());
  bool allEqual = true;
  for (int i = 2 * ROI_FEAT_RESOLUTION * ROI_FEAT_RESOLUTION;
       i < visualToken.size(); i++) {
    if (visualToken(i) !=
        feat(i - 2 * ROI_FEAT_RESOLUTION * ROI_FEAT_RESOLUTION))
      allEqual = false;
  }
  if (!allEqual)
    std::cout << "WRONG! For random instance, its feat does not match"
              << std::endl;

  if (allZero && allEqual)
    std::cout << "TestGetVisualToken passed!" << std::endl;
}

void TestRunVisualTokenizer(paddle::PaddlePredictor *detectorPredictor,
                            paddle::PaddlePredictor *predictor,
                            std::string posFile, std::string negFile) {
  cv::Mat posImg = cv::imread(posFile);
  cv::Mat negImg = cv::imread(negFile);

  Eigen::VectorXf posFlattenImg, negFlattenImg;
  PreprocessImage(posImg, posFlattenImg);
  PreprocessImage(negImg, negFlattenImg);

  std::vector<Eigen::VectorXf> imgArray({posFlattenImg, negFlattenImg});
  LoD predLod;
  Eigen::VectorXf flattenPred, flattenFeatureMap;
  int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                     flattenPred, flattenFeatureMap);
  if (objCount > 0) {
    std::cout << "Found " << objCount << " instances" << std::endl;
    Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;
    paddle::RunVisualTokenizer(predictor, 2, 1, objCount, predLod, flattenPred,
                               flattenFeatureMap, flattenVisualTokens,
                               flattenPaddingMask, flattenFrameIds,
                               frameInstArray);
    PrintVectorX<Eigen::VectorXf>("flattenFeatureMap", flattenFeatureMap, 20);
    PrintVectorX<Eigen::VectorXf>("flattenPaddingMask", flattenPaddingMask,
                                  flattenPaddingMask.size());
    PrintVectorX<Eigen::VectorXi>("flattenFrameIds", flattenFrameIds,
                                  flattenFrameIds.size());
  } else {
    std::cout << "No instances found" << std::endl;
  }
}

void TestGetAttnMask() {
  int nframe = 2;
  Eigen::VectorXf attnMask;
  paddle::GetAttnMask(nframe, attnMask);
  PrintVectorX<Eigen::VectorXf>("attnMask", attnMask, attnMask.size());
}

void TestRunAttnCtrl(paddle::PaddlePredictor *detectorPredictor,
                     paddle::PaddlePredictor *visualTokenizerPredictor,
                     paddle::PaddlePredictor *predictor, std::string posFile,
                     std::string negFile) {
  int nframe = 10;
  RowMajorMatrixXf visualTokenTensor(nframe,
                                     TOKENS_PER_FRAME * VISUAL_TOKEN_DIM);
  RowMajorMatrixXf paddingMaskTensor(nframe, TOKENS_PER_FRAME);
  RowMajorMatrixXi frameIdsTensor(nframe, TOKENS_PER_FRAME);

  for (int i = 0; i < nframe; i++) {
    Eigen::VectorXf flattenImg;
    if (i < nframe / 2) {
      cv::Mat img = cv::imread(negFile);
      PreprocessImage(img, flattenImg);
    } else {
      cv::Mat img = cv::imread(posFile);
      PreprocessImage(img, flattenImg);
    }

    std::vector<Eigen::VectorXf> imgArray({flattenImg});
    LoD predLod;
    Eigen::VectorXf flattenPred, flattenFeatureMap;
    int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                       flattenPred, flattenFeatureMap);

    Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;
    paddle::RunVisualTokenizer(visualTokenizerPredictor, 1, i + 1, objCount,
                               predLod, flattenPred, flattenFeatureMap,
                               flattenVisualTokens, flattenPaddingMask,
                               flattenFrameIds, frameInstArray);

    visualTokenTensor.row(i) = flattenVisualTokens;
    paddingMaskTensor.row(i) = flattenPaddingMask;
    frameIdsTensor.row(i) = flattenFrameIds;
  }

  Eigen::VectorXf flattenVisualTokens(Eigen::Map<Eigen::VectorXf>(
      visualTokenTensor.data(), nframe * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM));
  Eigen::VectorXf flattenPaddingMask(Eigen::Map<Eigen::VectorXf>(
      paddingMaskTensor.data(), nframe * TOKENS_PER_FRAME));
  Eigen::VectorXi flattenFrameIds(Eigen::Map<Eigen::VectorXi>(
      frameIdsTensor.data(), nframe * TOKENS_PER_FRAME));

  Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
      flattenActTopKSample;
  paddle::RunAttnCtrl(predictor, nframe, NUM_ACT, flattenVisualTokens,
                      flattenPaddingMask, flattenFrameIds, flattenTriggerPred,
                      flattenObjPred, flattenActPred, flattenActTopKSample);
  PrintVectorX<Eigen::VectorXf>("flattenTriggerPred", flattenTriggerPred,
                                flattenTriggerPred.size());
  PrintVectorX<Eigen::VectorXf>(
      "flattenObjPred",
      flattenObjPred.segment((nframe - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME),
      TOKENS_PER_FRAME);
  PrintVectorX<Eigen::VectorXf>(
      "flattenActPred", flattenActPred.segment((nframe - 1) * NUM_ACT, NUM_ACT),
      NUM_ACT);
  PrintVectorX<Eigen::VectorXf>("flattenActTopKSample", flattenActTopKSample,
                                flattenActTopKSample.size());
}

void TestPrepareMultimodalActions() {
  std::string filename = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(filename, multimodalActs);
  for (size_t i = 0; i < multimodalActs.size(); i++) {
    std::cout << i << ": " << multimodalActs[i] << std::endl;
  }

  std::cout << "Selected safe actions:" << std::endl;
  int safeActNum = sizeof(SAFE_ACTS) / sizeof(SAFE_ACTS[0]);
  for (int i = 0; i < safeActNum; i++)
    std::cout << SAFE_ACTS[i] << ": " << multimodalActs[SAFE_ACTS[i]]
              << std::endl;
}

void TestVideo(paddle::PaddlePredictor *detectorPredictor,
               paddle::PaddlePredictor *visualTokenizerPredictor,
               paddle::PaddlePredictor *attnCtrlPredictor,
               std::string videoPath) {
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

    std::vector<Eigen::VectorXf> imgArray({flattenImg});
    LoD predLod;
    Eigen::VectorXf flattenPred, flattenFeatureMap;
    int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                       flattenPred, flattenFeatureMap);

    Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;
    paddle::RunVisualTokenizer(
        visualTokenizerPredictor, 1, obWindow.size() + 1, objCount, predLod,
        flattenPred, flattenFeatureMap, flattenVisualTokens, flattenPaddingMask,
        flattenFrameIds, frameInstArray);

    if (obWindow.size() < OB_WINDOW_LEN) {
      obWindow.push_back(img);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      sharedFrameIds.push_back(flattenFrameIds);
      frameInstWindow.push_back(frameInstArray[0]);
    } else {
      obWindow.pop_front();
      visualTokensWindow.pop_front();
      paddingMaskWindow.pop_front();
      frameInstWindow.pop_front();

      obWindow.push_back(img);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      frameInstWindow.push_back(frameInstArray[0]);
    }

    if (obWindow.size() < OB_WINDOW_LEN)
      continue;

    Eigen::VectorXf fullVisualTokens, fullPaddingMask;
    Eigen::VectorXi fullFrameIds;
    fullVisualTokens.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME *
                            VISUAL_TOKEN_DIM);
    fullPaddingMask.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);
    fullFrameIds.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);

    for (int i = 0; i < OB_WINDOW_LEN; i++) {
      fullVisualTokens.segment(i * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM,
                               TOKENS_PER_FRAME * VISUAL_TOKEN_DIM) =
          visualTokensWindow[i];
      fullPaddingMask.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          paddingMaskWindow[i];
      fullFrameIds.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          sharedFrameIds[i];
    }

    Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
        flattenActTopKSample;
    paddle::RunAttnCtrl(attnCtrlPredictor, OB_WINDOW_LEN, NUM_ACT,
                        fullVisualTokens, fullPaddingMask, fullFrameIds,
                        flattenTriggerPred, flattenObjPred, flattenActPred,
                        flattenActTopKSample);

    Eigen::VectorXf objMask;
    paddle::GetObjMask(frameInstArray[0], objMask);
    PrintVectorX<Eigen::VectorXf>("objMask", objMask, objMask.size());
    Eigen::VectorXf objPred = flattenObjPred.segment(
        (OB_WINDOW_LEN - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME);
    objPred = objPred.array() * objMask.array();
    PrintVectorX<Eigen::VectorXf>("objPred", objPred, objPred.size());

    std::string resJson;
    paddle::ConvertPredToJsons(
        flattenTriggerPred(OB_WINDOW_LEN - 1), 0, false, objPred,
        flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT),
        flattenActTopKSample, frameInstArray[0], multimodalActs, resJson);
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

#ifdef ASYNC_INFER
void AsyncRunDetector(paddle::PaddlePredictor *detectorPredictor) {
  PreprocessRes prepRes;
  while (true) {
    bool hasTask = preprocessQ.pop(prepRes);
    if (!hasTask)
      continue;

    std::vector<Eigen::VectorXf> imgArray({prepRes.flattenImg});

    DetectorRes detRes;
    detRes.id = prepRes.id;
    detRes.ob = prepRes.ob;
    detRes.objCount =
        paddle::RunDetector(detectorPredictor, imgArray, detRes.lod,
                            detRes.flattenPred, detRes.flattenFeatureMap);
    detectorQ.push(detRes);

    // LoD predLod;
    // Eigen::VectorXf flattenPred, flattenFeatureMap;
    // int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
    //                                    flattenPred, flattenFeatureMap);
  }
}

void AsyncRunVTokenizerAttnCtrl(
    paddle::PaddlePredictor *visualTokenizerPredictor,
    paddle::PaddlePredictor *attnCtrlPredictor) {

  std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
  std::vector<MultimodalAction> multimodalActs;
  PrepareMultimodalActions(actPath, multimodalActs);

  // random engine for diversity
  std::default_random_engine rnd;
  rnd.seed(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> uniform(0, 1);

  DetectorRes detRes;
  while (true) {
    bool hasTask = detectorQ.pop(detRes);
    if (!hasTask)
      continue;

    if (robotWakeup) {
      obWindow.clear();
      visualTokensWindow.clear();
      paddingMaskWindow.clear();
      frameInstWindow.clear();
      continue;
    }

    Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;
    paddle::RunVisualTokenizer(visualTokenizerPredictor, 1, obWindow.size() + 1,
                               detRes.objCount, detRes.lod, detRes.flattenPred,
                               detRes.flattenFeatureMap, flattenVisualTokens,
                               flattenPaddingMask, flattenFrameIds,
                               frameInstArray);

    if (obWindow.size() < OB_WINDOW_LEN) {
      obWindow.push_back(detRes.ob);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      sharedFrameIds.push_back(flattenFrameIds);
      frameInstWindow.push_back(frameInstArray[0]);
    } else {
      obWindow.pop_front();
      visualTokensWindow.pop_front();
      paddingMaskWindow.pop_front();
      frameInstWindow.pop_front();

      obWindow.push_back(detRes.ob);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      frameInstWindow.push_back(frameInstArray[0]);
    }

    if (obWindow.size() < OB_WINDOW_LEN)
      continue;

    Eigen::VectorXf fullVisualTokens, fullPaddingMask;
    Eigen::VectorXi fullFrameIds;
    fullVisualTokens.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME *
                            VISUAL_TOKEN_DIM);
    fullPaddingMask.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);
    fullFrameIds.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);

    for (int i = 0; i < OB_WINDOW_LEN; i++) {
      fullVisualTokens.segment(i * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM,
                               TOKENS_PER_FRAME * VISUAL_TOKEN_DIM) =
          visualTokensWindow[i];
      fullPaddingMask.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          paddingMaskWindow[i];
      fullFrameIds.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          sharedFrameIds[i];
    }

    Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
        flattenActTopKSample;
    paddle::RunAttnCtrl(attnCtrlPredictor, OB_WINDOW_LEN, NUM_ACT,
                        fullVisualTokens, fullPaddingMask, fullFrameIds,
                        flattenTriggerPred, flattenObjPred, flattenActPred,
                        flattenActTopKSample);

    bool useSkill = CheckNearField(frameInstArray[0]);

    Eigen::VectorXf objMask;
    paddle::GetObjMask(frameInstArray[0], objMask);
    Eigen::VectorXf objPred = flattenObjPred.segment(
        (OB_WINDOW_LEN - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME);
    objPred = objPred.array() * objMask.array();

    std::string resJson;
    bool hasAct = paddle::ConvertPredToJsons(
        flattenTriggerPred(OB_WINDOW_LEN - 1), detRes.id, useSkill, objPred,
        flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT),
        flattenActTopKSample, frameInstArray[0], multimodalActs, resJson);
    bool lagSensitive = CheckLagSensitive(frameInstArray[0]);

    if (hasAct) {
      Response res;
      res.id = detRes.id;
      res.jsonStr = resJson;
      res.lagSensitive = lagSensitive;
      ctrlQ.push(res);

      Log log;
      log.id = detRes.id;
      log.confidence = flattenTriggerPred(OB_WINDOW_LEN - 1);
      log.jsonStr = resJson;
      log.obs = {obWindow.begin(), obWindow.end()};
      log.frameInstArray.resize(frameInstWindow.size());
      for (size_t i = 0; i < frameInstWindow.size(); i++) {
        for (size_t j = 0; j < frameInstWindow[i].size(); j++)
          log.frameInstArray[i].push_back(frameInstWindow[i][j]);
      }
      log.actPred =
          flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT);
      log.objPred = objPred;
      logQ.push(log);
    }
  } // end of while loop
}
#endif // end of block for async functional pipeline

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

    int lastID = log.frameInstArray.size() - 1;
    bool isSensitive = CheckLagSensitive(log.frameInstArray[lastID]);
    if (isSensitive)
      continue;

    std::string logdir = FLAGS_logdir + "/" + std::to_string(log.id);
    if (boost::filesystem::create_directories(logdir)) {
      for (size_t i = 0; i < log.obs.size(); i++) {
        cv::imwrite(logdir + "/" + std::to_string(i) + ".jpg", log.obs[i]);

        for (int j = 0; j < TOKENS_PER_FRAME; j++) {
          if (log.frameInstArray[i][j].classID != 0)
            break;

          double aspectRatio = IMG_RESIZE / (VIEW_W + 0.0);
          int x1 =
              std::max(0, static_cast<int>(log.frameInstArray[i][j].bbox(0) *
                                           aspectRatio));
          int y1 =
              std::max(0, static_cast<int>(log.frameInstArray[i][j].bbox(1) *
                                           aspectRatio));
          int x2 = std::min(
              static_cast<int>(VIEW_W * aspectRatio),
              static_cast<int>(log.frameInstArray[i][j].bbox(2) * aspectRatio));
          int y2 = std::min(
              static_cast<int>(VIEW_H * aspectRatio),
              static_cast<int>(log.frameInstArray[i][j].bbox(3) * aspectRatio));

          cv::Rect bbox(x1, y1, x2 - x1, y2 - y1);
          cv::rectangle(log.obs[i], bbox, cv::Scalar(0, 255, 0), 2, 8);

          // TODO: display obj confidence
          cv::putText(log.obs[i], std::to_string(j), cv::Point(x1 + 2, y1 + 10),
                      cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
        }

        cv::imwrite(logdir + "/" + std::to_string(i) + "_vis.jpg", log.obs[i]);
      }

      std::string jsonfile = logdir + "/res.txt";
      std::ofstream outfile(jsonfile);
      if (outfile.is_open()) {
        outfile << log.confidence << std::endl;
        outfile << log.jsonStr << std::endl;

        // for (size_t i = 0; i < multimodalActs.size(); i++)
        for (size_t i : ArgSort(log.actPred))
          outfile << log.actPred(i) << " " << multimodalActs[i].to_json()
                  << std::endl;

        outfile.close();
      } else {
        LOG(WARNING) << "Cannot create " + jsonfile << std::endl;
      }

      std::string instfile = logdir + "/inst.txt";
      std::ofstream outfile2(instfile);
      if (outfile2.is_open()) {
        for (size_t i = 0; i < log.frameInstArray.size(); i++) {
          for (int j = 0; j < TOKENS_PER_FRAME; j++) {
            if (i == log.frameInstArray.size() - 1)
              outfile2 << "#" << i << "-" << j << ": " << log.objPred(j)
                       << std::endl;
            else
              outfile2 << "#" << i << "-" << j << std::endl;

            outfile2 << log.frameInstArray[i][j].classID << std::endl;
            outfile2 << log.frameInstArray[i][j].score << std::endl;

            for (int k = 0; k < 4; k++)
              outfile2 << log.frameInstArray[i][j].bbox(k) << " ";
            float bboxSize = (log.frameInstArray[i][j].bbox(3) -
                              log.frameInstArray[i][j].bbox(1)) *
                             (log.frameInstArray[i][j].bbox(2) -
                              log.frameInstArray[i][j].bbox(0));
            outfile2 << bboxSize << std::endl;

            if (FLAGS_salutation)
              outfile2 << log.frameInstArray[i][j].salutation_cls_tree()
                       << std::endl;

            for (int k = 0; k < ROI_FEAT_DIM - 1; k++)
              outfile2 << log.frameInstArray[i][j].feat(k) << " ";
            outfile2 << log.frameInstArray[i][j].feat(ROI_FEAT_DIM - 1)
                     << std::endl;
          }
        }

        outfile2.close();
      } else {
        LOG(WARNING) << "Cannot create " + instfile << std::endl;
      }
    } else {
      LOG(WARNING) << "Cannot create " + logdir << std::endl;
    }
  }
}

void ProcessRequest(paddle::PaddlePredictor *detectorPredictor,
                    paddle::PaddlePredictor *visualTokenizerPredictor,
                    paddle::PaddlePredictor *attnCtrlPredictor) {
  // random engine for diversity
  std::default_random_engine rnd;
  rnd.seed(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> uniform(0, 1);

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
      visualTokensWindow.clear();
      paddingMaskWindow.clear();
      frameInstWindow.clear();
      continue;
    }

    auto t1 = paddle::time();
    Eigen::VectorXf flattenImg;
    // cv::imwrite("infer_img" + std::to_string(req.id) + ".jpg", req.ob);
    PreprocessImage(req.ob, flattenImg);

    auto t2 = paddle::time();
    LOG(INFO) << "[PreprocessImage] cost: " << paddle::TimeDiff(t1, t2) << "ms"
              << std::endl;

#ifndef ASYNC_INFER
    std::vector<Eigen::VectorXf> imgArray({flattenImg});
    LoD predLod;
    Eigen::VectorXf flattenPred, flattenFeatureMap;
    int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                       flattenPred, flattenFeatureMap);

    Eigen::VectorXf flattenVisualTokens, flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;
    paddle::RunVisualTokenizer(
        visualTokenizerPredictor, 1, obWindow.size() + 1, objCount, predLod,
        flattenPred, flattenFeatureMap, flattenVisualTokens, flattenPaddingMask,
        flattenFrameIds, frameInstArray);

    if (obWindow.size() < OB_WINDOW_LEN) {
      obWindow.push_back(req.ob);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      sharedFrameIds.push_back(flattenFrameIds);
      frameInstWindow.push_back(frameInstArray[0]);
    } else {
      obWindow.pop_front();
      visualTokensWindow.pop_front();
      paddingMaskWindow.pop_front();
      frameInstWindow.pop_front();

      obWindow.push_back(req.ob);
      visualTokensWindow.push_back(flattenVisualTokens);
      paddingMaskWindow.push_back(flattenPaddingMask);
      frameInstWindow.push_back(frameInstArray[0]);
    }

    if (obWindow.size() < OB_WINDOW_LEN)
      continue;

    Eigen::VectorXf fullVisualTokens, fullPaddingMask;
    Eigen::VectorXi fullFrameIds;
    fullVisualTokens.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME *
                            VISUAL_TOKEN_DIM);
    fullPaddingMask.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);
    fullFrameIds.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);

    for (int i = 0; i < OB_WINDOW_LEN; i++) {
      fullVisualTokens.segment(i * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM,
                               TOKENS_PER_FRAME * VISUAL_TOKEN_DIM) =
          visualTokensWindow[i];
      fullPaddingMask.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          paddingMaskWindow[i];
      fullFrameIds.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          sharedFrameIds[i];
    }

    Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
        flattenActTopKSample;
    bool paddleError = false;
    try {
      paddle::RunAttnCtrl(attnCtrlPredictor, OB_WINDOW_LEN, NUM_ACT,
                          fullVisualTokens, fullPaddingMask, fullFrameIds,
                          flattenTriggerPred, flattenObjPred, flattenActPred,
                          flattenActTopKSample);
    } catch (const std::runtime_error &error) {
      paddleError = true;
    }

    if (paddleError)
      continue;

    bool useSkill = CheckNearField(frameInstArray[0]);

    Eigen::VectorXf objMask;
    paddle::GetObjMask(frameInstArray[0], objMask);
    Eigen::VectorXf objPred = flattenObjPred.segment(
        (OB_WINDOW_LEN - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME);
    objPred = objPred.array() * objMask.array();

    std::string resJson;
    bool hasAct = paddle::ConvertPredToJsons(
        flattenTriggerPred(OB_WINDOW_LEN - 1), req.id, useSkill, objPred,
        flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT),
        flattenActTopKSample, frameInstArray[0], multimodalActs, resJson);
    bool lagSensitive = CheckLagSensitive(frameInstArray[0]);

    if (hasAct) {
      Response res;
      res.id = req.id;
      res.jsonStr = resJson;
      res.lagSensitive = lagSensitive;
      ctrlQ.push(res);

      Log log;
      log.id = req.id;
      log.confidence = flattenTriggerPred(OB_WINDOW_LEN - 1);
      log.jsonStr = resJson;
      log.obs = {obWindow.begin(), obWindow.end()};
      log.frameInstArray.resize(frameInstWindow.size());
      for (size_t i = 0; i < frameInstWindow.size(); i++) {
        for (size_t j = 0; j < frameInstWindow[i].size(); j++)
          log.frameInstArray[i].push_back(frameInstWindow[i][j]);
      }
      log.actPred =
          flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT);
      log.objPred = objPred;
      logQ.push(log);
    }
#else // end of sync infer and begin of async infer

    PreprocessRes prepRes;
    prepRes.id = req.id;
    prepRes.ob = req.ob;
    prepRes.flattenImg = flattenImg;
    preprocessQ.push(prepRes);

#endif // end of async infer
  }    // end of while loop
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
      // cv::imwrite("infer_img_raw" + std::to_string(request.reqid()) + ".jpg",
      //             req.ob);

      grpc::InferResponse res;
      Response resp;
      bool hasAct = ctrlQ.pop(resp);
      bool respSet = hasAct;
      while (hasAct) {
        // Get last one
        hasAct = ctrlQ.pop(resp);
        respSet |= hasAct;
      }

      // Handle lag sensitive response
      if (respSet && resp.lagSensitive) {
        respSet = false;
        lastValidResTime = paddle::time();
        LOG(INFO) << "Ignore lag sensitive response..." << std::endl;
      }

      if (respSet) {
        double deltaT = paddle::TimeDiff(lastValidResTime, paddle::time());
        if (req.id - resp.id > FLAGS_timeout || deltaT < FLAGS_occupy) {
          // Process timeout or robot is occupied
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

}; // end of ProactiveGreetingServiceImpl

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
  gflags::SetUsageMessage("Usage : ./infer_v3 ");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Directory of the inference model and params: " << FLAGS_dirname
            << endl;

  // Init predictors
  paddle::AnalysisConfig detectorCfg;
  // detectorCfg.EnableProfile();
  paddle::PrepareTRTConfig(&detectorCfg, "detector");
  auto detectorPredictor = paddle::CreatePaddlePredictor(detectorCfg);
  LOG(INFO) << "Created detector model predictor" << endl;

  paddle::AnalysisConfig visualTokenizerCfg;
  paddle::PrepareTRTConfig(&visualTokenizerCfg, "visual_tokenizer");
  auto visualTokenizerPredictor =
      paddle::CreatePaddlePredictor(visualTokenizerCfg);
  LOG(INFO) << "Created visual tokenizer predictor" << endl;

  paddle::AnalysisConfig attnCtrlCfg;
  paddle::PrepareTRTConfig(&attnCtrlCfg, "attn_ctrl");
  auto attnCtrlPredictor = paddle::CreatePaddlePredictor(attnCtrlCfg);
  LOG(INFO) << "Created attention controller predictor" << endl;

#ifdef TESTCASE_ONLY // start of testcases
  TestPreprocessImage("../test.jpg");
  TestGenerateSizeInputData();
  TestRunDetector(
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      "../test.jpg", "../test_neg.jpg");
  TestGetPosEmb();
  TestGetVisualToken();
  TestRunVisualTokenizer(
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      "../test.jpg", "../test_neg.jpg");
  TestGetAttnMask();
  TestRunAttnCtrl(
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()),
      "../test.jpg", "../test_neg.jpg");
  TestPrepareMultimodalActions();

#else // start of inference code

// start of local mode inference
#if !defined(SERVER_MODE) && defined(LOCAL_INFER)
  TestVideo(
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()),
      FLAGS_video);
#else // start of server mode inference

  std::thread worker(
      ProcessRequest,
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()));
  worker.detach();

  std::thread logger(ProcessLog);
  logger.detach();

#ifdef ASYNC_INFER
  std::thread asyncDetector(
      AsyncRunDetector,
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()));
  asyncDetector.detach();

  std::thread asyncCtrl(
      AsyncRunVTokenizerAttnCtrl,
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()));
  asyncCtrl.detach();
#endif

  RunServer();

#endif // end of local or server mode inference

#endif // end of inference code

  return 0;
}
