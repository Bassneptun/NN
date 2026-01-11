#include <algorithm>
#include <armadillo>
#include <cassert>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <qenv/include/Confirm.hh>
#include <qenv/include/Engine.hh>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

std::random_device rd;
std::minstd_rand generator(rd());

class Layer {
public:
  virtual std::string code() = 0;
  virtual std::unique_ptr<Layer> clone() const = 0;
  virtual ~Layer() = default;
};

class RotationLayer : public Layer {
private:
  string gate;
  int n;
  bool entangled;
  int arg_pos;

public:
  RotationLayer(int n, int arg_pos, bool entangled = false, string gate = "RY")
      : gate(gate), n(n), entangled(entangled), arg_pos(arg_pos) {}
  string code() override {
    std::stringstream ss;
    // s ist ein Rotationsgatter => RX/Y/Z
    assert(gate == "RX" || gate == "RY" || gate == "RZ");
    if (entangled) {
      ss << "D" << this->gate << " \%b ??" << arg_pos << "\n";
    } else {
      for (int i = 0; i < this->n; i++) {
        ss << this->gate << " $a" << i << " ??" << arg_pos + i << "\n";
      }
    }
    return ss.str();
  }

  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<RotationLayer>(*this);
  }
};

class EntaglementLayer : public Layer {
private:
  int n;

public:
  EntaglementLayer(int n) : n(n) {}

  string code() override {
    std::stringstream ss;
    ss << "DAL % # \"b\"\n";
    ss << "CMB $a" << 0 << " $a" << 1 << " \%b\n";
    for (int i = 2; i < this->n; i++) {
      ss << "DCB $a" << i << " \%b\n";
    }
    return ss.str();
  }

  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<EntaglementLayer>(*this);
  }
};

class DetanglementLayer : public Layer {
private:
  int n;

public:
  DetanglementLayer(int n) : n(n) {}

  string code() override {
    std::stringstream ss;
    for (int i = 0; i < this->n; i++) {
      ss << "TR \%b $a" << i << " " << i << "\n";
    }
    return ss.str();
  }

  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<DetanglementLayer>(*this);
  }
};

class Encoding {
public:
  virtual std::string code() = 0;
  virtual std::unique_ptr<Encoding> clone() const = 0;
  virtual ~Encoding() = default;
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
      ss << "SET $a" << i << " 1 0" << "\n";
    }
    for (int i = 0; i < size; i++) {
      ss << "RY $a" << i << " ??" << i << "\n";
    }
    return ss.str();
  }

  std::unique_ptr<Encoding> clone() const override {
    return std::make_unique<AngleEncoding>(*this);
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
    return ss.str();
  }
  std::unique_ptr<Encoding> clone() const override {
    return std::make_unique<LiteralEncoding>(*this);
  }
};

class Network {
public:
  struct ev_strategies {
    std::pair<double, double> params;
    int func;
    double ev_target = 0.4;
    double mut_chance;
    bool double_mut;
  };

  std::vector<std::unique_ptr<Layer>> layers;
  std::unique_ptr<Encoding> encoding;
  std::string buffer;
  std::optional<Engine> engine;
  std::vector<double> arguments;
  ev_strategies ev;
  int n;

  Network &operator=(Network other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(Network &a, Network &b) noexcept {
    std::swap(a.layers, b.layers);
    std::swap(a.encoding, b.encoding);
    std::swap(a.buffer, b.buffer);
    std::swap(a.engine, b.engine);
    std::swap(a.arguments, b.arguments);
    std::swap(a.ev, b.ev);
    std::swap(a.n, b.n);
  }

  Network(const Network &other)
      : buffer(other.buffer), engine(other.engine), arguments(other.arguments),
        ev(other.ev), n(other.n) {

    // Copy encoding
    if (other.encoding) {
      encoding = other.encoding->clone();
    }

    // Copy layers
    layers.reserve(other.layers.size());
    for (const auto &layer : other.layers) {
      layers.push_back(layer ? layer->clone() : nullptr);
    }
  }

  Network(std::vector<std::unique_ptr<Layer>> layers,
          std::unique_ptr<Encoding> encoding, ev_strategies ev, int n)
      : encoding(std::move(encoding)), engine(std::nullopt), ev(ev), n(n) {
    for (size_t i = 0; i < layers.size(); i++) {
      this->layers.push_back(std::move(layers[i]));
    }
    this->generate_bytecode();
    this->engine = std::make_optional(this->create_engine());
  }

  void generate_bytecode() {
    this->buffer = "";
    this->buffer = this->encoding->code();
    for (size_t i = 0; i < this->layers.size(); i++) {
      this->buffer += this->layers[i]->code();
    }
  }

  Engine create_engine() { return Engine(this->buffer, false); }

  size_t size() const { return layers.size(); }

  std::vector<std::vector<Qbit>>
  forward(const std::vector<std::vector<double>> &inputs) {
    std::vector<std::vector<Qbit>> out;
    for (auto tmp : inputs) {
      tmp.insert(tmp.end(), this->arguments.begin(), this->arguments.end());
      this->engine->exe(tmp);
      out.push_back(this->engine->memory[0]);
      this->engine->clear();
    }
    return out;
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

std::uniform_real_distribution<double> s;

bool choice(double chance = 0.5) { return chance > s(generator); }

template <typename T> T choice(std::vector<T> container) {
  return container[randint(container.size())];
}

template <typename T> T &choice(std::pair<T, T> &container) {
  return choice() ? container.first : container.second;
}

template <typename dist> void mutate(Network &net) {
  if (choice(net.ev.mut_chance)) {
    dist mut(net.ev.params.first, net.ev.params.second);
    if (choice(net.ev.ev_target)) {
      net.arguments[randint(
          net.arguments.size() - 1)] += mut(generator);
    } else {
      choice(net.ev.params) += mut(generator);
      net.ev.ev_target += mut(generator);
      net.ev.mut_chance += mut(generator);
    }
    if (!net.ev.double_mut)
      return;
  }
}

std::vector<std::function<void(Network &)>> mutation_operators = {
    [](Network &net) { mutate<std::normal_distribution<>>(net); },
    [](Network &net) { mutate<std::cauchy_distribution<>>(net); }};

Network create_network(
    Network::ev_strategies ev, int N2 = 2,
    std::optional<std::vector<std::unique_ptr<Layer>>> layers = std::nullopt) {
  // standard version: AngleEncoding -> RotationLayer -> Entanglement Layer ->
  // RotationLayer -> DetanglementLayer
  std::vector<std::unique_ptr<Layer>> layers_;
  if (!layers) {
    layers_.push_back(std::make_unique<RotationLayer>(RotationLayer(N2, N2)));
    layers_.push_back(std::make_unique<EntaglementLayer>(EntaglementLayer(N2)));
    layers_.push_back(
        std::make_unique<RotationLayer>(RotationLayer(N2, 2 * N2, true)));
    layers_.push_back(
        std::make_unique<DetanglementLayer>(DetanglementLayer(N2)));
  } else {
    layers_ = std::move(*layers);
  }
  std::normal_distribution<double> dist;
  std::unique_ptr<Encoding> tmp1 =
      std::make_unique<AngleEncoding>(AngleEncoding(N2));
  auto tmp = Network(std::move(layers_), std::move(tmp1), ev, N2);
  tmp.arguments = std::vector<double>();
  for (int i = 0; i < N2 + 1; i++) {
    tmp.arguments.push_back(dist(generator));
  }
  return tmp;
}

Network::ev_strategies create_ev() {
  return Network::ev_strategies{std::make_pair(-1., 1.), 0, 0.4, 0.5, true};
}

double loss(std::vector<std::vector<Qbit>> res,
            std::vector<std::vector<double>> expected) {
  double ret = 0;
  for (unsigned int i = 0; i < res.size(); i++) {
    for (unsigned int j = 0; j < expected[0].size(); j++) {
      ret += std::pow(std::norm(res[i][j].get()(0)) - expected[i][j], 2);
    }
  }
  return ret / res.size();
}

std::vector<unsigned int> create_first_n(unsigned int n) {
  std::vector<unsigned int> out(n);
  for (unsigned int i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

class Generation {
private:
  std::vector<Network> members;
  unsigned int max_iteration;
  unsigned int N;

public:
  Generation(unsigned int max_iteration, unsigned int N) : max_iteration(max_iteration), N(N) {
    std::normal_distribution<double> gaussian_(-1., 1.);
    for (unsigned int i = 0; i < N; i++) {
      auto tmp = Network::ev_strategies{
          std::make_pair(gaussian_(generator), gaussian_(generator)), 0, 0.4,
          0.5, true};
      auto tmp2 = create_network(tmp);
      this->members.push_back(tmp2);
      this->members[i].ev = tmp;
    }
  }
  Generation(Generation &a) : max_iteration(a.max_iteration) {
    for (size_t i = 0; i < a.members.size(); i++) {
      members.push_back(a.members[i]);
    }
  }

  void mutate(Generation &current) {
    for (unsigned int i = 0; i < current.members.size(); i++) {
      mutation_operators[current.members[i].ev.func](current.members[i]);
    }
  }

  void insert(std::vector<Network> &values, std::vector<std::vector<double>> in,
              std::vector<std::vector<double>> expected) {
    std::vector<size_t> indices = this->get_worst(in, expected);
    for(size_t i = 0; i < indices.size(); i++){
      this->members[indices[i]] = values[i];
    }
  }

  std::vector<size_t> get_worst(std::vector<std::vector<double>> in,
                             std::vector<std::vector<double>> expected) {
    std::vector<size_t> indices(this->members.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<double> values(this->members.size());
    std::transform(this->members.begin(), this->members.end(), values.begin(),
                   [&](Network a) { return loss(a.forward(in), expected); });

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return values[i] < values[j]; });
    std::vector<size_t> out;
    for (size_t i = 0; i < N / 3; i++) {
      out.push_back(indices[N - i - 1]);
    }
    cout << "best loss: " << values[indices[0]] << std::endl;
    return out;
  }

  std::vector<Network> get_best(std::vector<std::vector<double>> in,
                                std::vector<std::vector<double>> expected,
                                Generation &ranking) {
    std::vector<size_t> indices(ranking.members.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<double> values(ranking.members.size());
    std::transform(ranking.members.begin(), ranking.members.end(),
                   values.begin(),
                   [&](Network a) { return loss(a.forward(in), expected); });

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return values[i] < values[j]; });
    std::vector<Network> out;
    for (size_t i = 0; i < N / 3; i++) {
      out.push_back(ranking.members[indices[i]]);
    }
    cout << "best loss: " << values[indices[0]] << std::endl;
    return out;
  }

  Network EP(std::vector<std::vector<double>> in,
             std::vector<std::vector<double>> expected) {
    for (unsigned int i = 0; i < this->max_iteration; i++) {
      auto ranking = Generation(*this);
      ranking.mutate(ranking);
      std::vector<Network> best = get_best(in, expected, ranking);
      insert(best, in, expected);
    }
    return this->members[0];
  }
};

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_xor();
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_xor() {
  std::vector<std::vector<double>> inputs = {
      {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<std::vector<double>> outs = {{0}, {1}, {0}, {1}};
  return std::make_pair(inputs, outs);
}

int main() {
  auto xor_ = get_xor();
  Generation result = Generation(1000, 60);
  result.EP(xor_.first, xor_.second);
}
