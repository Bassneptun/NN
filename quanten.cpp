#include <armadillo>
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <qenv/include/Confirm.hh>
#include <qenv/include/Engine.hh>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace env {
#include "qenv/libqenv.h"
}

std::random_device rd;
std::minstd_rand generator(rd());

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
  DetanglementLayer(int n) : n(n) {}

  string code() override {
    std::stringstream ss;
    string out;
    for (int i = 0; i < this->n; i++) {
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

public:
  AngleEncoding(int size) : size(size) {}
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
  LiteralEncoding(int size) : size(size) {}
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
public:
  struct ev_strategies {
    std::pair<double, double> params;
    std::function<void(Network &)> func;
    double ev_target = 0.4;
  };

  std::vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<Encoding> encoding;
  std::string buffer;
  std::optional<Engine> engine;
  std::vector<double> arguments;
  ev_strategies ev;
  int n;
  bool measure;

  Network(std::vector<std::unique_ptr<Layer>> layers,
          std::unique_ptr<Encoding> encoding, ev_strategies ev, int n,
          bool measure = false)
      : encoding(std::move(encoding)), engine(std::nullopt) {
    this->generate_bytecode();
    for (int i = 0; i < layers.size(); i++) {
      this->layers.push_back(std::move(layers[i]));
    }
    if(this->measure){
      for(int i = 0; i < n; i++){
        buffer.append("MES $" + std::to_string(i) + "\n");
      }
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
  arma::vec forward(std::vector<double> inputs) {
    this->engine->exe(inputs);
  }
};

int randint(int min, int max) {
  std::uniform_int_distribution<> select(min, max);
  return select(generator);
}

int randint(int max) {
  std::uniform_int_distribution<> select(0, max);
  return select(generator);
}

std::uniform_real_distribution<> s(0, 1.);

bool choice(double chance = 0.5) { return chance < s(generator); }

template <typename T> T choice(std::vector<T> container) {
  return container[randint(container.size())];
}

template <typename T> T &choice(std::pair<T, T> &container) {
  return choice() ? container.first : container.second;
}

template <typename dist> void mutate(Network &net) {
  dist mut(net.ev.params.first, net.ev.params.second);
  if (choice(net.ev.ev_target)) {
    net.arguments[randint(net.arguments.size())] += mut(generator);
  } else {
    choice(net.ev.params) += mut(generator);
    net.ev.ev_target += mut(generator);
  }
}

std::vector<std::function<void(Network &)>> mutation_operators = {
    [](Network &net) { mutate<std::normal_distribution<>>(net); },
    [](Network &net) { mutate<std::cauchy_distribution<>>(net); }};

template <int N2>
Network create_network(
    Network::ev_strategies ev,
    std::vector<std::unique_ptr<Layer>> layers =
        (std::vector<std::unique_ptr<
             Layer>>)(std::make_unique<Layer>(RotationLayer(N2, N2)),
                      std::make_unique<Layer>(EntaglementLayer(N2)),
                      std::make_unique<Layer>(RotationLayer(N2, 2 * N2, true)),
                      std::make_unique<Layer>(DetanglementLayer(N2)))) {
  // standard version: AngleEncoding -> RotationLayer -> Entanglement Layer ->
  // RotationLayer -> DetanglementLayer
  return Network(std::move(layers),
                 std::make_unique<Encoding>(AngleEncoding(N2)), ev, N2);
}

template <typename data, typename data2, typename out> constexpr out loss(data in, data2 expected){
  return (in - expected)*(in - expected);
}

template <typename T, typename U> concept Subtractable = requires(T a, U b){
  { loss(a, b) } -> std::convertible_to<double>;
};

template <int N, int N2, typename T, typename U> requires Subtractable<T, U> class Generation {
private:
  std::array<Network, N> members;
  int max_iteration;

public:
  Generation(int max_iteration) : max_iteration(max_iteration) {
    std::normal_distribution<double> gaussian_(-1., 1.);
    for (int i = 0; i < N; i++) {
      this->members[i].ev =
          std::make_pair(gaussian_(generator), gaussian_(generator));
      this->members[i] = create_network(this->members[i].ev);
    }
  }
  Network EP(T in, U expected){
    for(int i = 0; i < this->max_iteration; i++){

    }
  }
};
