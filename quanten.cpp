#include <armadillo>
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <qenv/include/Confirm.hh>
#include <qenv/include/Engine.hh>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace env {
#include "qenv/libqenv.h"
}

class Layer {
public:
  virtual std::string code() = 0;
};

class RotationLayer : public Layer {
private:
  string gate;
  int n;
  bool entangled;
  int arg_pos;

public:
  RotationLayer(int n, int arg_pos, bool entangled = false, string gate = "RY")
      : n(n), gate(gate), entangled(entangled), arg_pos(arg_pos) {}
  string code() override {
    std::stringstream buffer;
    // s ist ein Rotationsgatter => RX/Y/Z
    assert(gate == "RX" || gate == "RY" || gate == "RZ");
    if (entangled) {
      buffer << "D" << this->gate << "$b  ??" << arg_pos << "\n";
    } else {
      for (int i = 0; i < this->n; i++) {
        buffer << this->gate << "$a" << i << " ??" << arg_pos << "\n";
      }
    }
    string out;
    buffer.str(out);
    return out;
  }
};

class EntaglementLayer : public Layer {
private:
  int n;

public:
  EntaglementLayer(int n) : n(n) {}

  string code() override {
    std::stringstream ss;
    ss << "DAL % # \"b\"";
    ss << "CMB $a" << 0 << " \%b";
    for (int i = 1; i < this->n; i++) {
      ss << "DCB $a" << i << " \%b\n";
    }
    string out;
    ss.str(out);
    return out;
  }
};

class DetanglementLayer : public Layer {
private:
  int n;
public:
  DetanglementLayer(int n) : n(n){}

  string code() override{
    std::stringstream ss;
    string out;
    for(int i = 0; i < this->n; i++){
      ss << "TR \%b $a" << i << " " << i << "\n";
    }
    ss.str(out);
    return out;
  }
};

class Encoding {
public:
  virtual std::string code() = 0;
};

class AngleEncoding : public Encoding {
  int size;
  std::string code() override {
    std::stringstream ss;
    for (int i = 0; i < size; i++) {
      ss << "QAL & 0 $" << " \"a" << i << "\"\n";
    }
    for (int i = 0; i < size; i++) {
      ss << "RX \"$a" << i << "\" ??" << i << "\n";
    }
    std::string out;
    ss.str(out);
    return out;
  }
};

class LiteralEncoding : public Encoding {
  int size;

public:
  std::string code() override {
    std::stringstream ss;
    for (int i = 0; i < size; i++) {
      ss << "QAL & 0 $" << " \"a" << i << "\"\n";
    }
    for (int i = 0; i < size; i++) {
      ss << "SET $a " << i << "?" << i << "\n";
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
  std::optional<Engine> engine;
  int n;

public:
  Network(std::vector<std::unique_ptr<Layer>> layers,
          std::unique_ptr<Encoding> encoding, int n)
      : encoding(std::move(encoding)), engine(std::nullopt) {
    for (int i = 0; i < layers.size(); i++) {
      this->layers.push_back(std::move(layers[i]));
    }
    this->engine = this->create_engine();
  }

  void generate_bytecode() {
    this->buffer = "";
    for (int i = 0; i < this->n; i++) {
      buffer += "QAL & 0 $ \"a" + std::to_string(i) + "\"\n";
    }
    this->buffer += this->encoding->code();
    for (int i = 0; i < this->layers.size(); i++) {
      this->buffer += this->layers[i]->code();
    }
  }

  Engine create_engine() {
    this->generate_bytecode();
    return Engine(buffer, false);
  }

  int size() const { return layers.size(); }
  arma::vec forward(arma::vec inputs);
  arma::ivec forward_mes(arma::vec inputs);
};

using ev_strategies = std::pair<double, double>;

template <int N>
class Generation{
private: 
  std::array<Network, N> member;
  std::array<ev_strategies, N> ev_strategies;
  int max_iteration;
  std::
public:
  Generation& get_generation(
}
