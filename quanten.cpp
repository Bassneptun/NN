#include <armadillo>
#include <memory>
#include <qenv/include/Confirm.hh>
#include <qenv/include/Engine.hh>
#include <sstream>
#include <string>
#include <vector>

namespace qenv {
#include "qenv/libqenv.h"
}

class Layer {};

class Encoding {
  virtual std::string get() = 0;
};

class AngleEncoding : public Encoding{
  int size;
  std::string get() override {
    std::stringstream ss;
    for(int i = 0; i < size; i++){
      ss << "QAL & 0 $" << " \"a" << i << "\"\n";
    }
    for(int i = 0; i < size; i++){
      ss << "RY \"a" << i << "\" ??" << i << "\n";
    }
    std::string out;
    ss.str(out);
    return out;
  }
};

class LiteralEncoding : public Encoding{
  int size;
  std::string get() override {
    std::stringstream ss;
    for(int i = 0; i < size; i++){
      ss << "QAL & 0 $" << " \"a" << i << "\"\n";
    }
    for(int i = 0; i < size; i++){
      ss << "RY \"a" << i << "\" ??" << i << "\n";
    }
    std::string out;
    ss.str(out);
    return out;
  }
};

class Network {
private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<Encoding> encoding;
  std::string buffer;
  Maybe bytecode;
  Engine engine;

public:
  int size() const { return layers.size(); }
  arma::vec forward(arma::vec inputs);
  arma::ivec forward_mes(arma::vec inputs);
};
