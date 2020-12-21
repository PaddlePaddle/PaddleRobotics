#include <Eigen/Dense>
#include <ostream>
#include <string>

class MultimodalAction {
public:
  std::string talk;
  std::string exp;
  std::string act;

  MultimodalAction();
  MultimodalAction(const std::string talk, const std::string exp,
                   const std::string act)
      : talk(talk), exp(exp), act(act) {}

  std::string to_json();
  std::string to_json(int hour, int reqID, bool useSkill,
                      std::string salu, std::string pronoun);

private:
  std::string convertHour2Time(int hour);
  std::string fillPlaceholder(std::string T, std::string P, std::string C);
};

MultimodalAction::MultimodalAction() {
  talk = "null";
  exp = "null";
  act = "null";
}

std::string MultimodalAction::to_json() {
  return "{\"Talk\": \"" + talk + "\", \"Expression\": \"" + exp +
         "\", \"Action\": \"" + act + "\"}";
}

std::string MultimodalAction::to_json(int hour, int reqID, bool useSkill,
                                      std::string salu, std::string pronoun) {
  std::string time = convertHour2Time(hour);
  std::string filledTalk = fillPlaceholder(time, pronoun, salu);
  std::string skill = "false";
  if (useSkill)
    skill = "true";

  return "{\"QueryID\": " + std::to_string(reqID) + ", \"Talk\": \"" +
         filledTalk + "\", \"Expression\": \"" + exp + "\", \"Action\": \"" +
         act + "\", \"UseSkill\": " + skill + "}";
}

std::string MultimodalAction::convertHour2Time(int hour) {
  std::string T;
  if (hour < 11)
    T = "早上";
  else if (hour < 14)
    T = "中午";
  else if (hour < 18)
    T = "下午";
  else
    T = "晚上";
  return T;
}

std::string MultimodalAction::fillPlaceholder(std::string T, std::string P, std::string C) {
  std::string res = talk;

  size_t pos = res.find("T");
  while (pos != res.npos) {
    res.replace(pos, 1, T);
    pos = res.find("T", pos + T.size());
  }

  pos = res.find("P");
  while (pos != res.npos) {
    res.replace(pos, 1, P);
    pos = res.find("P", pos + P.size());
  }

  pos = res.find("C");
  while (pos != res.npos) {
    res.replace(pos, 1, C);
    pos = res.find("C", pos + C.size());
  }
  return res;
}

std::ostream &operator<<(std::ostream &os, MultimodalAction &ma) {
  os << ma.to_json();
  return os;
}
