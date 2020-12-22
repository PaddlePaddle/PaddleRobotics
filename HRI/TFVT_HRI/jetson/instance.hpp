#include <Eigen/Dense>
#include <ostream>
#include <string>

const int FEAT_DIM = 512;
const int SALU_LEFT_DIM = 2;
const int SALU_RIGHT_DIM = 2;

class Instance {
public:
  int classID;
  float score;
  Eigen::Vector4f bbox;
  Eigen::VectorXf feat;

  Instance(int featDim = FEAT_DIM);
  Instance(const Eigen::VectorXf &pred, const Eigen::VectorXf &roiFeat);
  Instance(const Eigen::VectorXf &pred, const Eigen::VectorXf &roiFeat,
           const Eigen::VectorXf &saluRoot, const Eigen::VectorXf &saluLeft,
           const Eigen::VectorXf &saluRight);

  void update_salutation(const Eigen::VectorXf &saluRoot,
                         const Eigen::VectorXf &saluLeft,
                         const Eigen::VectorXf &saluRight);
  std::string salutation_cls_tree() const;
  std::string get_salutation(double l1, double l2) const;
  float get_area_size() const;

private:
  Eigen::Vector2f saluRoot;
  Eigen::VectorXf saluLeft;
  Eigen::VectorXf saluRight;

  double get_conf_gap(Eigen::VectorXf v) const;
};

Instance::Instance(int featDim) {
  classID = -1;
  score = 0.0;
  bbox.setZero();
  feat.resize(featDim);
  feat.setZero();

  saluLeft.resize(SALU_LEFT_DIM);
  saluRight.resize(SALU_RIGHT_DIM);
  saluRoot.setZero();
  saluLeft.setZero();
  saluRight.setZero();
}

Instance::Instance(const Eigen::VectorXf &pred,
                   const Eigen::VectorXf &roiFeat) {
  classID = static_cast<int>(pred(0));
  score = pred(1);
  for (int i = 0; i < 4; i++)
    bbox(i) = pred(i + 2);
  feat = roiFeat;

  saluLeft.resize(SALU_LEFT_DIM);
  saluRight.resize(SALU_RIGHT_DIM);
  saluRoot.setZero();
  saluLeft.setZero();
  saluRight.setZero();
}

Instance::Instance(const Eigen::VectorXf &pred, const Eigen::VectorXf &roiFeat,
                   const Eigen::VectorXf &root, const Eigen::VectorXf &left,
                   const Eigen::VectorXf &right) {
  classID = static_cast<int>(pred(0));
  score = pred(1);
  for (int i = 0; i < 4; i++)
    bbox(i) = pred(i + 2);
  feat = roiFeat;
  update_salutation(root, left, right);
}

void Instance::update_salutation(const Eigen::VectorXf &root,
                                 const Eigen::VectorXf &left,
                                 const Eigen::VectorXf &right) {
  saluRoot(0) = root(0);
  saluRoot(1) = root(1);
  saluLeft.resize(SALU_LEFT_DIM);
  saluRight.resize(SALU_RIGHT_DIM);
  for (int i = 0; i < SALU_LEFT_DIM; i++)
    saluLeft(i) = left(i);
  for (int i = 0; i < SALU_RIGHT_DIM; i++)
    saluRight(i) = right(i);
}

std::string Instance::salutation_cls_tree() const {
  std::string tree;

  tree += "[" + std::to_string(saluRoot(0)) + ", ( ";
  for (int i = 0; i < SALU_LEFT_DIM; i++)
    tree += std::to_string(saluLeft(i)) + " ";
  tree += ")], ";

  tree += "[" + std::to_string(saluRoot(1)) + ", ( ";
  for (int i = 0; i < SALU_RIGHT_DIM; i++)
    tree += std::to_string(saluRight(i)) + " ";
  tree += ")]";

  return tree;
}

std::string Instance::get_salutation(double l1, double l2) const {
  std::string salu = "";
  if (saluRoot(0) - saluRoot(1) > l1 && get_conf_gap(saluLeft) > l2) {
    if (saluLeft(0) > saluLeft(1))
      salu = "小哥哥";
    else
      salu = "叔叔";
  } else if (saluRoot(1) - saluRoot(0) > l1 && get_conf_gap(saluRight) > l2) {
    if (saluRight(0) > saluRight(1))
      salu = "小姐姐";
    else
      salu = "阿姨";
  }

  return salu;
}

float Instance::get_area_size() const {
  float xmin, ymin, xmax, ymax;
  xmin = bbox(0);
  ymin = bbox(1);
  xmax = bbox(2);
  ymax = bbox(3);

  return (ymax - ymin) * (xmax - xmin);
}

double Instance::get_conf_gap(Eigen::VectorXf v) const {
  double gap = 0.0;
  float maxV = 0.0, maxV2 = -1.0;
  for (int i = 0; i < v.size(); i++) {
    if (v(i) > maxV) {
      maxV2 = maxV;
      maxV = v(i);
    } else if (v(i) > maxV2) {
      maxV2 = v(i);
    }
  }

  if (maxV2 > 0)
    gap = maxV - maxV2;
  return gap;
}

std::ostream &operator<<(std::ostream &os, const Instance &inst) {
  os << "{cid: " << inst.classID << ",\n bbox: " << inst.bbox
     << ", \n salutation: " << inst.salutation_cls_tree()
     << ",\n feat: " << inst.feat.segment(0, 20) << "...}";
  return os;
}
