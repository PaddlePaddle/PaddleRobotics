#include "instance.hpp"
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
typedef std::vector<std::vector<size_t>> LoD;
typedef std::vector<Instance> FrameInstances;

const double eps = 0.000001;
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
const int INST_FM_DIM = 512 * 5 * 5;
const int INST_CLS_DIM = 80;
const int INST_POS_DIM = 50;
const int INST_CROP_SIZE = 128;
const int ATTN_HEADS = 8;
const int ATTN_LAYERS = 6;
const int PRED_DIM = 6;
const int INTERESTED_CALSS[] = {0, 24, 26, 28, 27, 67};
const int SAFE_ACTS[] = {1, 3, 4, 5, 6, 7, 8, 10};

struct EvalEntry {
  int label;
  int waeID;
  std::string dir;
  std::vector<std::string> frameIDs;
  std::vector<Eigen::Vector4f> bboxLst;
};

struct Object {
  float score;
  Eigen::Vector4f bbox;
  std::string salu;
};

struct InstanceInputs {
  Eigen::VectorXf visualTokens;
  Eigen::VectorXf instFM;
  Eigen::VectorXf instCls;
  Eigen::VectorXf instPos;
  Eigen::VectorXf instCrop;
};

std::deque<cv::Mat> obWindow;
std::deque<InstanceInputs> instInputsWindow;
std::deque<Eigen::VectorXf> paddingMaskWindow;
std::deque<Eigen::VectorXi> sharedFrameIds;
std::deque<FrameInstances> frameInstWindow;

DEFINE_string(dirname, "./xiaodu_hi_v3",
              "Directory of the inference model and params.");
DEFINE_double(th, 0.5, "Threshold for interaction trigger.");
DEFINE_double(tau, 1.0, "Softmax temperature hyperparameter.");
DEFINE_int32(topK, 50, "Number of top-k multimodal actions.");
DEFINE_int32(minInstSize, 16000, "Minimum instance view size.");
DEFINE_string(logdir, "./log",
              "Directory of the log, include observations, response JSON.");
DEFINE_bool(gpu, false, "Whether to use GPU config.");

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

DEFINE_string(dataTxt, "/mnt/xueyang/Code/xiaodu-hi/data/final_eval.txt",
              "Path to evaluation dataset txt file.");

DEFINE_string(inputsType, "visual_token",
              "Inputs type of attention controller, options: visual_token, "
              "instance, without_inst_fm, without_inst_cls, without_inst_pos, "
              "inst_crop.");

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

std::string Now2Str() {
  auto now = std::chrono::system_clock::now();
  std::time_t t_now = std::chrono::system_clock::to_time_t(now);
  tm *date = localtime(&t_now);

  // format: 2020-09-22_18-00-00
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", date);

  std::string str(buf);
  return str;
}

void PreprocessImage(const cv::Mat &image, Eigen::VectorXf &flattenOutput,
                     std::string outputImgFile = "") {
  // See perception/common/utils.py::yolov4_img_preprocess
  double aspectRatio =
      std::min(IMG_RESIZE * 1.0 / image.rows, IMG_RESIZE * 1.0 / image.cols);
  int newH = static_cast<int>(std::floor(image.rows * aspectRatio));
  int newW = static_cast<int>(std::floor(image.cols * aspectRatio));
  cv::Mat newImage;
  cv::resize(image, newImage, cv::Size(newW, newH));

  cv::Mat boxedImg(IMG_RESIZE, IMG_RESIZE, CV_8UC3, cv::Scalar(128, 128, 128));

  int yOffset = static_cast<int>(std::floor(IMG_RESIZE - newH) / 2.0);
  int xOffset = static_cast<int>(std::floor(IMG_RESIZE - newW) / 2.0);
  newImage.copyTo(boxedImg(cv::Rect(xOffset, yOffset, newW, newH)));

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

void PreprocessInstCrop(const cv::Mat &image, const Eigen::Vector4f &bbox,
                        Eigen::VectorXf &crop) {
  int xmin = static_cast<int>(std::ceil(bbox(0)));
  int ymin = static_cast<int>(std::ceil(bbox(1)));
  int xmax = static_cast<int>(std::ceil(bbox(2)));
  int ymax = static_cast<int>(std::ceil(bbox(3)));

  xmin = std::max(0, xmin);
  ymin = std::max(0, ymin);
  xmax = std::min(VIEW_W - 1, xmax);
  ymax = std::min(VIEW_H - 1, ymax);

  cv::Rect rect(xmin, ymin, xmax - xmin, ymax - ymin);
  cv::Mat cropImg = image(rect).clone();
  cv::resize(cropImg, cropImg, cv::Size(INST_CROP_SIZE, INST_CROP_SIZE));

  cv::Mat rgbImg;
  cv::cvtColor(cropImg, rgbImg, cv::COLOR_BGRA2RGB);

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
  crop.resize(t_rMat.size() + t_gMat.size() + t_bMat.size());
  crop << Eigen::Map<Eigen::VectorXf>(t_rMat.data(), t_rMat.size()),
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
                   const FrameInstances instances, std::string &salu,
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

bool CheckNearField(const FrameInstances instances, double areaTH = 0.35) {
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

bool CompareInstances(Instance i, Instance j) {
  return (i.get_area_size() > j.get_area_size());
}

void ConvertPredToInstances(int nframe, const LoD predLod,
                            const std::vector<Eigen::VectorXf> preds,
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

  int featDim = ROI_FEAT_DIM;
  if (FLAGS_inputsType.find("inst") != std::string::npos) {
    featDim = INST_FM_DIM;
    if (FLAGS_inputsType.find("inst_crop") != std::string::npos)
      featDim = 3 * INST_CROP_SIZE * INST_CROP_SIZE;
  }

  for (int i = 0; i < nframe; i++) {
    // Filter, see interaction/common/data_v2.py::filter_instances
    int npred = predLod[0][i + 1] - predLod[0][i];
    FrameInstances initFrameInst;
    for (int j = 0; j < npred; j++) {
      int k = predLod[0][i] + j;
      Instance inst(flattenPred.segment(k * PRED_DIM, PRED_DIM),
                    flattenRoisFeats.segment(k * featDim, featDim));

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
        frameInstArray[i][j] = Instance(featDim);
    } else if (initFrameInst.size() <= TOKENS_PER_FRAME) {
      for (size_t j = 0; j < initFrameInst.size(); j++)
        frameInstArray[i][j] = initFrameInst[j];
      for (size_t j = initFrameInst.size(); j < TOKENS_PER_FRAME; j++)
        frameInstArray[i][j] = Instance(featDim);
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

void GetObjMask(const FrameInstances instances, Eigen::VectorXf &objMask) {
  objMask.resize(instances.size());
  for (size_t i = 0; i < instances.size(); i++) {
    if (instances[i].classID == 0)
      objMask(i) = 1.0;
    else
      objMask(i) = 0.0;
  }
}

int RunDetector(PaddlePredictor *predictor,
                const std::vector<Eigen::VectorXf> imgArray, LoD &predLod,
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
                        InstanceInputs &instInputs,
                        Eigen::VectorXf &flattenPaddingMask,
                        Eigen::VectorXi &flattenFrameIds,
                        std::vector<FrameInstances> &frameInstArray) {
  auto time1 = time();

  if (FLAGS_inputsType == "visual_token") {
    instInputs.visualTokens.resize(nframe * TOKENS_PER_FRAME *
                                   VISUAL_TOKEN_DIM);
    instInputs.visualTokens.setZero();
  } else {
    instInputs.instFM.resize(nframe * TOKENS_PER_FRAME * INST_FM_DIM);
    instInputs.instCls.resize(nframe * TOKENS_PER_FRAME * INST_CLS_DIM);
    instInputs.instPos.resize(nframe * TOKENS_PER_FRAME * INST_POS_DIM);
    instInputs.instFM.setZero();
    instInputs.instCls.setZero();
    instInputs.instPos.setZero();
    // NOTE: a tmp modif to test wheter padding mask works
    // instInputs.instFM +=
    //     Eigen::VectorXf::Ones(nframe * TOKENS_PER_FRAME * INST_FM_DIM);
  }

  flattenPaddingMask.resize(nframe * TOKENS_PER_FRAME);
  flattenFrameIds.resize(nframe * TOKENS_PER_FRAME);
  frameInstArray.resize(nframe);
  for (int i = 0; i < nframe; i++)
    frameInstArray[i].resize(TOKENS_PER_FRAME);

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
        int offset = i * TOKENS_PER_FRAME + j;

        if (FLAGS_inputsType == "visual_token") {
          Eigen::VectorXf visualToken;
          GetVisualToken(frameInstArray[i][j], visualToken);
          instInputs.visualTokens.segment(offset * VISUAL_TOKEN_DIM,
                                          VISUAL_TOKEN_DIM) = visualToken;
        } else {
          instInputs.instFM.segment(offset * INST_FM_DIM, INST_FM_DIM) =
              frameInstArray[i][j].feat;

          Eigen::VectorXf cls(INST_CLS_DIM);
          cls.setZero();
          if (frameInstArray[i][j].classID >= 0)
            cls(frameInstArray[i][j].classID) = 1.0;
          instInputs.instCls.segment(offset * INST_CLS_DIM, INST_CLS_DIM) = cls;

          Eigen::VectorXf pos;
          GetPosEmb(frameInstArray[i][j], pos);
          instInputs.instPos.segment(offset * INST_POS_DIM, INST_POS_DIM) = pos;
        }

        if (frameInstArray[i][j].classID != -1)
          flattenPaddingMask(i * TOKENS_PER_FRAME + j) = 1.0;
      }
    }
  }

  auto time2 = time();
  LOG(INFO) << "[RunVisualTokenizer] nframe: " << nframe
            << ", cost: " << TimeDiff(time1, time2) << "ms" << std::endl;
}

void RunInstCropExtractor(const cv::Mat &image, int frameIdOffset,
                          const LoD &predLod,
                          const Eigen::VectorXf &flattenPred,
                          InstanceInputs &instInputs,
                          Eigen::VectorXf &flattenPaddingMask,
                          Eigen::VectorXi &flattenFrameIds,
                          std::vector<FrameInstances> &frameInstArray) {
  auto time1 = time();

  int cropDim = 3 * INST_CROP_SIZE * INST_CROP_SIZE;
  instInputs.instCrop.resize(TOKENS_PER_FRAME * cropDim);
  instInputs.instCls.resize(TOKENS_PER_FRAME * INST_CLS_DIM);
  instInputs.instPos.resize(TOKENS_PER_FRAME * INST_POS_DIM);
  instInputs.instCrop.setZero();
  instInputs.instCls.setZero();
  instInputs.instPos.setZero();

  flattenPaddingMask.resize(TOKENS_PER_FRAME);
  flattenFrameIds.resize(TOKENS_PER_FRAME);
  frameInstArray.resize(1);
  frameInstArray[0].resize(TOKENS_PER_FRAME);

  flattenPaddingMask.setZero();
  for (int i = 0; i < flattenFrameIds.size(); i++)
    flattenFrameIds(i) = frameIdOffset + i / TOKENS_PER_FRAME;

  // PrintVectorX<Eigen::VectorXf>("flattenPred", flattenPred,
  // flattenPred.size());

  Eigen::VectorXf flattenCrop;
  if (flattenPred.size() > 1) {
    flattenCrop.resize(flattenPred.size() / 6 * cropDim);

    for (int i = 0; i < flattenPred.size() / 6; i++) {
      Eigen::Vector4f bbox = flattenPred.segment(i * 6 + 2, 4);
      Eigen::VectorXf crop;
      PreprocessInstCrop(image, bbox, crop);
      flattenCrop.segment(i * cropDim, cropDim) = crop;
    }

    std::vector<Eigen::VectorXf> preds{flattenPred, flattenCrop};
    ConvertPredToInstances(1, predLod, preds, frameInstArray);

    for (size_t i = 0; i < frameInstArray.size(); i++) {
      for (size_t j = 0; j < frameInstArray[i].size(); j++) {
        int offset = i * TOKENS_PER_FRAME + j;
        instInputs.instCrop.segment(offset * cropDim, cropDim) =
            frameInstArray[i][j].feat;

        Eigen::VectorXf cls(INST_CLS_DIM);
        cls.setZero();
        if (frameInstArray[i][j].classID >= 0)
          cls(frameInstArray[i][j].classID) = 1.0;
        instInputs.instCls.segment(offset * INST_CLS_DIM, INST_CLS_DIM) = cls;

        Eigen::VectorXf pos;
        GetPosEmb(frameInstArray[i][j], pos);
        instInputs.instPos.segment(offset * INST_POS_DIM, INST_POS_DIM) = pos;

        if (frameInstArray[i][j].classID != -1)
          flattenPaddingMask(i * TOKENS_PER_FRAME + j) = 1.0;
      }
    }
  }

  auto time2 = time();
  LOG(INFO) << "[RunInstCropExtractor] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
}

void RunAttnCtrl(PaddlePredictor *predictor, int nframe, int naction,
                 const std::vector<InstanceInputs> &instInputsArray,
                 const Eigen::VectorXf &flattenPaddingMask,
                 const Eigen::VectorXi &flattenFrameIds,
                 Eigen::VectorXf &flattenTriggerPred,
                 Eigen::VectorXf &flattenObjPred,
                 Eigen::VectorXf &flattenActPred,
                 Eigen::VectorXf &flattenActTopKSample,
                 Eigen::VectorXf &flattenAttnWeights) {
  if (instInputsArray.size() != nframe)
    throw std::runtime_error("Instance inputs size is not equal to given nframe"
                             "attention controller.");

  auto time1 = time();

  int seqLen = nframe * TOKENS_PER_FRAME;
  int cropDim = 3 * INST_CROP_SIZE * INST_CROP_SIZE;
  flattenTriggerPred.resize(nframe);
  flattenObjPred.resize(seqLen);
  flattenActPred.resize(nframe * naction);
  flattenActTopKSample.resize(nframe);

  if (FLAGS_inputsType == "visual_token") {
    Eigen::VectorXf flattenVisualTokens(nframe * TOKENS_PER_FRAME *
                                        VISUAL_TOKEN_DIM);
    for (int i = 0; i < nframe; i++)
      flattenVisualTokens.segment(i * TOKENS_PER_FRAME * VISUAL_TOKEN_DIM,
                                  TOKENS_PER_FRAME * VISUAL_TOKEN_DIM) =
          instInputsArray[i].visualTokens;
    auto visualTokenInput = predictor->GetInputTensor("visual_tokens");
    visualTokenInput->Reshape({1, seqLen, VISUAL_TOKEN_DIM});
    visualTokenInput->copy_from_cpu(flattenVisualTokens.data());
  } else if (FLAGS_inputsType == "instance") {
    Eigen::VectorXf flattenInstFM(nframe * TOKENS_PER_FRAME * INST_FM_DIM);
    Eigen::VectorXf flattenInstCls(nframe * TOKENS_PER_FRAME * INST_CLS_DIM);
    Eigen::VectorXf flattenInstPos(nframe * TOKENS_PER_FRAME * INST_POS_DIM);
    for (int i = 0; i < nframe; i++) {
      flattenInstFM.segment(i * TOKENS_PER_FRAME * INST_FM_DIM,
                            TOKENS_PER_FRAME * INST_FM_DIM) =
          instInputsArray[i].instFM;
      flattenInstCls.segment(i * TOKENS_PER_FRAME * INST_CLS_DIM,
                             TOKENS_PER_FRAME * INST_CLS_DIM) =
          instInputsArray[i].instCls;
      flattenInstPos.segment(i * TOKENS_PER_FRAME * INST_POS_DIM,
                             TOKENS_PER_FRAME * INST_POS_DIM) =
          instInputsArray[i].instPos;
    }

    auto instFMInput = predictor->GetInputTensor("inst_fm");
    instFMInput->Reshape({1, seqLen, INST_FM_DIM});
    instFMInput->copy_from_cpu(flattenInstFM.data());

    auto instClsInput = predictor->GetInputTensor("inst_cls");
    instClsInput->Reshape({1, seqLen, INST_CLS_DIM});
    instClsInput->copy_from_cpu(flattenInstCls.data());

    auto instPosInput = predictor->GetInputTensor("inst_pos_emb");
    instPosInput->Reshape({1, seqLen, INST_POS_DIM});
    instPosInput->copy_from_cpu(flattenInstPos.data());
  } else if (FLAGS_inputsType == "without_inst_fm") {
    Eigen::VectorXf flattenInstCls(nframe * TOKENS_PER_FRAME * INST_CLS_DIM);
    Eigen::VectorXf flattenInstPos(nframe * TOKENS_PER_FRAME * INST_POS_DIM);
    for (int i = 0; i < nframe; i++) {
      flattenInstCls.segment(i * TOKENS_PER_FRAME * INST_CLS_DIM,
                             TOKENS_PER_FRAME * INST_CLS_DIM) =
          instInputsArray[i].instCls;
      flattenInstPos.segment(i * TOKENS_PER_FRAME * INST_POS_DIM,
                             TOKENS_PER_FRAME * INST_POS_DIM) =
          instInputsArray[i].instPos;
    }

    auto instClsInput = predictor->GetInputTensor("inst_cls");
    instClsInput->Reshape({1, seqLen, INST_CLS_DIM});
    instClsInput->copy_from_cpu(flattenInstCls.data());

    auto instPosInput = predictor->GetInputTensor("inst_pos_emb");
    instPosInput->Reshape({1, seqLen, INST_POS_DIM});
    instPosInput->copy_from_cpu(flattenInstPos.data());
  } else if (FLAGS_inputsType == "without_inst_cls") {
    Eigen::VectorXf flattenInstFM(nframe * TOKENS_PER_FRAME * INST_FM_DIM);
    Eigen::VectorXf flattenInstPos(nframe * TOKENS_PER_FRAME * INST_POS_DIM);
    for (int i = 0; i < nframe; i++) {
      flattenInstFM.segment(i * TOKENS_PER_FRAME * INST_FM_DIM,
                            TOKENS_PER_FRAME * INST_FM_DIM) =
          instInputsArray[i].instFM;
      flattenInstPos.segment(i * TOKENS_PER_FRAME * INST_POS_DIM,
                             TOKENS_PER_FRAME * INST_POS_DIM) =
          instInputsArray[i].instPos;
    }

    auto instFMInput = predictor->GetInputTensor("inst_fm");
    instFMInput->Reshape({1, seqLen, INST_FM_DIM});
    instFMInput->copy_from_cpu(flattenInstFM.data());

    auto instPosInput = predictor->GetInputTensor("inst_pos_emb");
    instPosInput->Reshape({1, seqLen, INST_POS_DIM});
    instPosInput->copy_from_cpu(flattenInstPos.data());
  } else if (FLAGS_inputsType == "without_inst_pos") {
    Eigen::VectorXf flattenInstFM(nframe * TOKENS_PER_FRAME * INST_FM_DIM);
    Eigen::VectorXf flattenInstCls(nframe * TOKENS_PER_FRAME * INST_CLS_DIM);

    for (int i = 0; i < nframe; i++) {
      flattenInstFM.segment(i * TOKENS_PER_FRAME * INST_FM_DIM,
                            TOKENS_PER_FRAME * INST_FM_DIM) =
          instInputsArray[i].instFM;
      flattenInstCls.segment(i * TOKENS_PER_FRAME * INST_CLS_DIM,
                             TOKENS_PER_FRAME * INST_CLS_DIM) =
          instInputsArray[i].instCls;
    }

    auto instFMInput = predictor->GetInputTensor("inst_fm");
    instFMInput->Reshape({1, seqLen, INST_FM_DIM});
    instFMInput->copy_from_cpu(flattenInstFM.data());

    auto instClsInput = predictor->GetInputTensor("inst_cls");
    instClsInput->Reshape({1, seqLen, INST_CLS_DIM});
    instClsInput->copy_from_cpu(flattenInstCls.data());
  } else if (FLAGS_inputsType == "inst_crop") {
    Eigen::VectorXf flattenInstCrop(seqLen * cropDim);
    Eigen::VectorXf flattenInstCls(nframe * TOKENS_PER_FRAME * INST_CLS_DIM);
    Eigen::VectorXf flattenInstPos(nframe * TOKENS_PER_FRAME * INST_POS_DIM);

    for (int i = 0; i < nframe; i++) {
      flattenInstCrop.segment(i * TOKENS_PER_FRAME * cropDim,
                              TOKENS_PER_FRAME * cropDim) =
          instInputsArray[i].instCrop;
      flattenInstCls.segment(i * TOKENS_PER_FRAME * INST_CLS_DIM,
                             TOKENS_PER_FRAME * INST_CLS_DIM) =
          instInputsArray[i].instCls;
      flattenInstPos.segment(i * TOKENS_PER_FRAME * INST_POS_DIM,
                             TOKENS_PER_FRAME * INST_POS_DIM) =
          instInputsArray[i].instPos;
    }

    auto instCropInput = predictor->GetInputTensor("inst_crop");
    instCropInput->Reshape({1, seqLen, cropDim});
    instCropInput->copy_from_cpu(flattenInstCrop.data());

    auto instClsInput = predictor->GetInputTensor("inst_cls");
    instClsInput->Reshape({1, seqLen, INST_CLS_DIM});
    instClsInput->copy_from_cpu(flattenInstCls.data());

    auto instPosInput = predictor->GetInputTensor("inst_pos_emb");
    instPosInput->Reshape({1, seqLen, INST_POS_DIM});
    instPosInput->copy_from_cpu(flattenInstPos.data());
  } else {
    throw std::invalid_argument("Argument -inputsType has invalid value.");
  }

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

  if (outputNames.size() > 4) {
    int attnWeightsSize = ATTN_HEADS * ATTN_LAYERS *
                          std::pow(OB_WINDOW_LEN * TOKENS_PER_FRAME, 2);
    flattenAttnWeights.resize(attnWeightsSize);

    auto attnWeights = predictor->GetOutputTensor(outputNames[4]);
    attnWeights->copy_to_cpu(flattenAttnWeights.data());
  } else {
    flattenAttnWeights.resize(1);
    flattenAttnWeights(0) = 0.0;
  }

  auto time2 = time();
  LOG(INFO) << "[RunAttnCtrl] nframe: " << nframe
            << ", cost: " << TimeDiff(time1, time2) << "ms" << std::endl;
} // namespace paddle

bool ConvertPredToJsons(float triggerPred, int reqID, bool useSkill,
                        const Eigen::VectorXf objPred,
                        const Eigen::VectorXf actPred,
                        const Eigen::VectorXf actTopKSample,
                        const FrameInstances instances,
                        const std::vector<MultimodalAction> &multimodalActs,
                        std::string &resJson, float &resActScore) {
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
    supports += static_cast<int>(nullActAt != 0);
    // supports += static_cast<int>(objCount > 0);
    // if (supports < 3.0 / 2)
    //   return false;
    if (supports < 2)
      return false;
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
  resActScore = actPred(sampleID);

  auto time2 = time();
  LOG(INFO) << "[ConvertPredToJsons] cost: " << TimeDiff(time1, time2) << "ms"
            << std::endl;
  return true;
}

} // namespace paddle

void RunEval(paddle::PaddlePredictor *detectorPredictor,
             paddle::PaddlePredictor *visualTokenizerPredictor,
             paddle::PaddlePredictor *attnCtrlPredictor, EvalEntry ee,
             float &triggerPred, float &nullActScore, int &nullActID,
             float &annoActScore, std::vector<Object> &validObjs) {
  obWindow.clear();
  instInputsWindow.clear();
  paddingMaskWindow.clear();
  frameInstWindow.clear();
  sharedFrameIds.clear();

  for (auto i : ee.frameIDs) {
    std::string imgFile = ee.dir + "/" + i + ".jpg";
    cv::Mat img = cv::imread(imgFile);
    cv::resize(img, img, cv::Size(VIEW_W, VIEW_H));
    std::cout << "img size: " << img.size() << std::endl;

    Eigen::VectorXf flattenImg;
    PreprocessImage(img, flattenImg);

    std::vector<Eigen::VectorXf> imgArray({flattenImg});
    LoD predLod;
    Eigen::VectorXf flattenPred, flattenFeatureMap;
    int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                       flattenPred, flattenFeatureMap);

    InstanceInputs instInputs;
    Eigen::VectorXf flattenPaddingMask;
    Eigen::VectorXi flattenFrameIds;
    std::vector<FrameInstances> frameInstArray;

    if (FLAGS_inputsType.find("inst_crop") != FLAGS_inputsType.npos)
      paddle::RunInstCropExtractor(img, obWindow.size() + 1, predLod,
                                   flattenPred, instInputs, flattenPaddingMask,
                                   flattenFrameIds, frameInstArray);
    else
      paddle::RunVisualTokenizer(
          visualTokenizerPredictor, 1, obWindow.size() + 1, objCount, predLod,
          flattenPred, flattenFeatureMap, instInputs, flattenPaddingMask,
          flattenFrameIds, frameInstArray);

    obWindow.push_back(img);
    instInputsWindow.push_back(instInputs);
    paddingMaskWindow.push_back(flattenPaddingMask);
    sharedFrameIds.push_back(flattenFrameIds);
    frameInstWindow.push_back(frameInstArray[0]);
  }

  Eigen::VectorXf fullPaddingMask;
  Eigen::VectorXi fullFrameIds;
  fullPaddingMask.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);
  fullFrameIds.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);

  for (int i = 0; i < OB_WINDOW_LEN; i++) {
    fullPaddingMask.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
        paddingMaskWindow[i];
    fullFrameIds.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
        sharedFrameIds[i];
  }

  Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
      flattenActTopKSample, flattenAttnWeights;
  paddle::RunAttnCtrl(attnCtrlPredictor, OB_WINDOW_LEN, NUM_ACT,
                      {instInputsWindow.begin(), instInputsWindow.end()},
                      fullPaddingMask, fullFrameIds, flattenTriggerPred,
                      flattenObjPred, flattenActPred, flattenActTopKSample,
                      flattenAttnWeights);

  int winLen = frameInstWindow.size();
  FrameInstances instances = frameInstWindow[winLen - 1];

  Eigen::VectorXf objMask;
  paddle::GetObjMask(instances, objMask);
  Eigen::VectorXf objPred = flattenObjPred.segment(
      (OB_WINDOW_LEN - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME);
  objPred = objPred.array() * objMask.array();

  triggerPred = flattenTriggerPred(OB_WINDOW_LEN - 1);

  Eigen::VectorXf actPred =
      flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT);
  nullActScore = actPred(0);
  nullActID = ArgSort(actPred)[0];

  annoActScore = actPred(ee.waeID);

  for (size_t i = 0; i < instances.size(); i++) {
    if (instances[i].classID == 0) {
      Object obj;
      obj.score = objPred(i);
      obj.bbox = instances[i].bbox;
      obj.salu = instances[i].get_salutation(FLAGS_saluL1, FLAGS_saluL2);
      validObjs.push_back(obj);
    }
  }
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
  EvalServiceImpl(paddle::PaddlePredictor *detectorPredictor,
                  paddle::PaddlePredictor *visualTokenizerPredictor,
                  paddle::PaddlePredictor *attnCtrlPredictor)
      : detectorPredictor(detectorPredictor),
        visualTokenizerPredictor(visualTokenizerPredictor),
        attnCtrlPredictor(attnCtrlPredictor) {
    std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
    PrepareMultimodalActions(actPath, multimodalActs);
  }

  EvalServiceImpl(paddle::PaddlePredictor *detectorPredictor,
                  paddle::PaddlePredictor *attnCtrlPredictor)
      : detectorPredictor(detectorPredictor), visualTokenizerPredictor(nullptr),
        attnCtrlPredictor(attnCtrlPredictor) {
    std::string actPath = FLAGS_dirname + "/" + "multimodal_actions.txt";
    PrepareMultimodalActions(actPath, multimodalActs);
  }

  Status infer(ServerContext *context, const EvalRequest *request,
               EvalResponse *reply) override {
    std::vector<cv::Mat> frames;
    DecodeFrames(request, frames);

    obWindow.clear();
    instInputsWindow.clear();
    paddingMaskWindow.clear();
    frameInstWindow.clear();
    sharedFrameIds.clear();

    for (auto img : frames) {
      Eigen::VectorXf flattenImg;
      PreprocessImage(img, flattenImg);

      std::vector<Eigen::VectorXf> imgArray({flattenImg});
      LoD predLod;
      Eigen::VectorXf flattenPred, flattenFeatureMap;
      int objCount = paddle::RunDetector(detectorPredictor, imgArray, predLod,
                                         flattenPred, flattenFeatureMap);

      InstanceInputs instInputs;
      Eigen::VectorXf flattenPaddingMask;
      Eigen::VectorXi flattenFrameIds;
      std::vector<FrameInstances> frameInstArray;

      if (FLAGS_inputsType.find("inst_crop") != FLAGS_inputsType.npos)
        paddle::RunInstCropExtractor(
            img, obWindow.size() + 1, predLod, flattenPred, instInputs,
            flattenPaddingMask, flattenFrameIds, frameInstArray);
      else
        paddle::RunVisualTokenizer(
            visualTokenizerPredictor, 1, obWindow.size() + 1, objCount, predLod,
            flattenPred, flattenFeatureMap, instInputs, flattenPaddingMask,
            flattenFrameIds, frameInstArray);

      obWindow.push_back(img);
      instInputsWindow.push_back(instInputs);
      paddingMaskWindow.push_back(flattenPaddingMask);
      sharedFrameIds.push_back(flattenFrameIds);
      frameInstWindow.push_back(frameInstArray[0]);
    }

    Eigen::VectorXf fullPaddingMask;
    Eigen::VectorXi fullFrameIds;
    fullPaddingMask.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);
    fullFrameIds.resize(OB_WINDOW_LEN * TOKENS_PER_FRAME);

    for (int i = 0; i < OB_WINDOW_LEN; i++) {
      fullPaddingMask.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          paddingMaskWindow[i];
      fullFrameIds.segment(i * TOKENS_PER_FRAME, TOKENS_PER_FRAME) =
          sharedFrameIds[i];
    }

    Eigen::VectorXf flattenTriggerPred, flattenObjPred, flattenActPred,
        flattenActTopKSample, flattenAttnWeights;
    paddle::RunAttnCtrl(attnCtrlPredictor, OB_WINDOW_LEN, NUM_ACT,
                        {instInputsWindow.begin(), instInputsWindow.end()},
                        fullPaddingMask, fullFrameIds, flattenTriggerPred,
                        flattenObjPred, flattenActPred, flattenActTopKSample,
                        flattenAttnWeights);

    int winLen = frameInstWindow.size();
    FrameInstances instances = frameInstWindow[winLen - 1];

    Eigen::VectorXf objMask;
    paddle::GetObjMask(instances, objMask);
    Eigen::VectorXf objPred = flattenObjPred.segment(
        (OB_WINDOW_LEN - 1) * TOKENS_PER_FRAME, TOKENS_PER_FRAME);
    objPred = objPred.array() * objMask.array();

    float triggerPred = flattenTriggerPred(OB_WINDOW_LEN - 1);

    Eigen::VectorXf actPred =
        flattenActPred.segment((OB_WINDOW_LEN - 1) * NUM_ACT, NUM_ACT);
    float nullActScore = actPred(0);
    int nullActID = ArgSort(actPred)[0];

    std::string resJson;
    float resActScore;
    bool hasAct = paddle::ConvertPredToJsons(
        triggerPred, 0, false, objPred, actPred, flattenActTopKSample,
        instances, multimodalActs, resJson, resActScore);

    if (flattenAttnWeights.size() > 1)
      saveLog(flattenAttnWeights);

    reply->set_response(resJson);
    reply->set_response_score(resActScore);
    reply->set_trigger_pred(triggerPred);
    reply->set_nullact_score(nullActScore);
    reply->set_nullact_id(nullActID);
    return Status::OK;
  }

private:
  paddle::PaddlePredictor *detectorPredictor;
  paddle::PaddlePredictor *visualTokenizerPredictor;
  paddle::PaddlePredictor *attnCtrlPredictor;

  std::vector<MultimodalAction> multimodalActs;

  void saveLog(const Eigen::VectorXf &flattenAttnWeights) {
    std::string timestamp = Now2Str();
    std::string logdir = FLAGS_logdir + "/" + Now2Str();
    if (!boost::filesystem::exists(logdir))
      boost::filesystem::create_directories(logdir);

    std::string txt = logdir + "/inst_attn.txt";
    std::ofstream outfile(txt);
    if (!outfile.is_open()) {
      LOG(WARNING) << "Cannot create " + txt << std::endl;
      return;
    }

    for (int i = 0; i < obWindow.size(); i++) {
      cv::imwrite(logdir + "/" + std::to_string(i) + ".jpg", obWindow[i]);

      for (int j = 0; j < TOKENS_PER_FRAME; j++) {
        outfile << "#inst " << i << "-" << j << std::endl;
        outfile << frameInstWindow[i][j].classID << std::endl;
        outfile << frameInstWindow[i][j].score << std::endl;
        for (int k = 0; k < 4; k++)
          outfile << frameInstWindow[i][j].bbox(k) << " ";
        outfile << std::endl;
      }
    }

    // dim of single layer single head
    int slshDim = std::pow(OB_WINDOW_LEN * TOKENS_PER_FRAME, 2);
    for (int i = 0; i < ATTN_LAYERS; i++) {
      for (int j = 0; j < ATTN_HEADS; j++) {
        outfile << "#attn " << i << "-" << j << std::endl;
        for (int k = 0; k < slshDim; k++)
          outfile << flattenAttnWeights((i * ATTN_HEADS + j) * slshDim + k)
                  << " ";
        outfile << std::endl;
      }
    }

    outfile.close();
  }
};

void RunServer(paddle::PaddlePredictor *detectorPredictor,
               paddle::PaddlePredictor *visualTokenizerPredictor,
               paddle::PaddlePredictor *attnCtrlPredictor) {
  std::string server_address("0.0.0.0:" + std::to_string(FLAGS_port));
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  EvalServiceImpl service(detectorPredictor, visualTokenizerPredictor,
                          attnCtrlPredictor);
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetMaxReceiveMessageSize(8 * 1024 * 1024); // larger than 6912007
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

#endif // end of grpc server

using namespace std;

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Usage : ./eval ");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Directory of the inference model and params: " << FLAGS_dirname
            << endl;

  // Init predictors
  paddle::AnalysisConfig detectorCfg;
  // detectorCfg.EnableProfile();
  paddle::PrepareTRTConfig(&detectorCfg, "detector");
  auto detectorPredictor = paddle::CreatePaddlePredictor(detectorCfg);
  LOG(INFO) << "Created detector model predictor" << endl;

  unique_ptr<paddle::PaddlePredictor> visualTokenizerPredictor;
  if (FLAGS_inputsType.find("inst_crop") == FLAGS_inputsType.npos) {
    paddle::AnalysisConfig visualTokenizerCfg;
    paddle::PrepareTRTConfig(&visualTokenizerCfg, "visual_tokenizer");
    visualTokenizerPredictor =
        paddle::CreatePaddlePredictor(visualTokenizerCfg);
    LOG(INFO) << "Created visual tokenizer predictor" << endl;
  }

  paddle::AnalysisConfig attnCtrlCfg;
  paddle::PrepareTRTConfig(&attnCtrlCfg, "attn_ctrl");
  auto attnCtrlPredictor = paddle::CreatePaddlePredictor(attnCtrlCfg);
  LOG(INFO) << "Created attention controller predictor" << endl;

#ifndef SERVER_MODE
  // ==================================================
  // start of evaluation using -dataTxt
  // ==================================================

  int eeID = 0, triggerTP = 0, triggerFP = 0, triggerFN = 0, nullActTP = 0,
      nullActFP = 0, nullActFN = 0;
  float actNLL = 0.0;

  if (!boost::filesystem::exists(FLAGS_logdir))
    boost::filesystem::create_directories(FLAGS_logdir);

  ofstream triggerFPLog(FLAGS_logdir + "/trigger_fp.txt");
  ofstream triggerFNLog(FLAGS_logdir + "/trigger_fn.txt");
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

    float triggerPred, nullActScore, annoActScore;
    int nullActID;
    vector<Object> validObjs;

    if (FLAGS_inputsType.find("inst_crop") == FLAGS_inputsType.npos)
      RunEval(static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
              static_cast<paddle::PaddlePredictor *>(
                  visualTokenizerPredictor.get()),
              static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()),
              ee, triggerPred, nullActScore, nullActID, annoActScore,
              validObjs);
    else
      RunEval(static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
              static_cast<paddle::PaddlePredictor *>(nullptr),
              static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()),
              ee, triggerPred, nullActScore, nullActID, annoActScore,
              validObjs);

    if (triggerPred > FLAGS_th && ee.label == 1)
      triggerTP++;
    else if (triggerPred > FLAGS_th && ee.label == 0) {
      triggerFP++;
      triggerFPLog << line << endl;
    } else if (triggerPred <= FLAGS_th && ee.label == 1) {
      triggerFN++;
      triggerFNLog << line << endl;
    }

    if (nullActID != 0 && ee.label == 1)
      nullActTP++;
    else if (nullActID != 0 && ee.label == 0) {
      nullActFP++;
      nullActFPLog << line << endl;
    } else if (nullActID == 0 && ee.label == 1) {
      nullActFN++;
      nullActFNLog << line << endl;
    }

    actNLL += -log(annoActScore);

    // TODO: process validObjs

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

  double triggerPrecision = (triggerTP + eps) / (triggerTP + triggerFP + eps);
  double triggerRecall = (triggerTP + eps) / (triggerTP + triggerFN + eps);
  cout << "==============================" << endl;
  cout << "Trigger th: " << FLAGS_th << endl;
  cout << "Precision: " << triggerPrecision << endl;
  cout << "Recall: " << triggerRecall << endl;
  cout << "==============================" << endl;

  double nullActPrecision = (nullActTP + eps) / (nullActTP + nullActFP + eps);
  double nullActRecall = (nullActTP + eps) / (nullActTP + nullActFN + eps);
  cout << "Null Act" << endl;
  cout << "Precision: " << nullActPrecision << endl;
  cout << "Recall: " << nullActRecall << endl;
  cout << "==============================" << endl;

  double avgActNLL = actNLL / eeID;
  cout << "Act Average NLL: " << avgActNLL << endl;
  cout << "==============================" << endl;

  metricLog << "#Trigger" << endl;
  metricLog << "TH Precision Recall" << endl;
  metricLog << FLAGS_th << " " << triggerPrecision << " " << triggerRecall
            << endl;

  metricLog << "#NullAct" << endl;
  metricLog << "Precision Recall" << endl;
  metricLog << nullActPrecision << " " << nullActRecall << endl;

  metricLog << "\n\n" << endl;
  metricLog << "triggerTP: " << triggerTP << endl;
  metricLog << "triggerFP: " << triggerFP << endl;
  metricLog << "triggerFN: " << triggerFN << endl;
  metricLog << "\nnullActTP: " << nullActTP << endl;
  metricLog << "nullActFP: " << nullActFP << endl;
  metricLog << "nullActFN: " << nullActFN << endl;
  metricLog << "\nactNLL: " << avgActNLL << endl;
  metricLog << "eeID: " << eeID << endl;

  triggerFPLog.close();
  triggerFNLog.close();
  nullActFPLog.close();
  nullActFNLog.close();
  metricLog.close();
#else
  // ==================================================
  // start of eval server
  // ==================================================
  RunServer(
      static_cast<paddle::PaddlePredictor *>(detectorPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(visualTokenizerPredictor.get()),
      static_cast<paddle::PaddlePredictor *>(attnCtrlPredictor.get()));

#endif
}
