#include <algorithm>
#include <armadillo>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <qenv/include/Confirm.hh>
#include <qenv/include/Engine.hh>
#include <qenv/include/QuditClass.hh>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

std::random_device rd;
std::minstd_rand generator(rd());

#include <fenv.h>

auto _ = feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);

std::vector<double> &operator+(std::vector<double> &lhs,
                               std::vector<double> &&rhs) {
  for (int i = 0; i < lhs.size(); i++) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

std::vector<double> &operator-(std::vector<double> &lhs,
                               std::vector<double> &rhs) {
  for (int i = 0; i < lhs.size(); i++) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}

std::vector<double> &operator/(std::vector<double> &a, int b) {
  for (int i = 0; i < a.size(); i++) {
    a[i] /= b;
  }
  return a;
}


std::vector<double> &operator*(std::vector<double> &a, int b) {
  for (int i = 0; i < a.size(); i++) {
    a[i] *= b;
  }
  return a;
}

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
    ss << "CNT $a" << 0 << " $a" << 1 << " \%b\n";
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
    this->engine = std::make_optional(this->create_engine(this->buffer));
  }

  void generate_bytecode() {
    this->buffer = "";
    this->buffer = this->encoding->code();
    for (size_t i = 0; i < this->layers.size(); i++) {
      this->buffer += this->layers[i]->code();
    }
  }

  static Engine create_engine(string buffer) { return Engine(buffer, false); }

  size_t size() const { return layers.size(); }

  std::vector<Qudit> forward(const std::vector<std::vector<double>> &inputs) {
    std::vector<Qudit> out;
    for (auto tmp : inputs) {
      tmp.insert(tmp.end(), this->arguments.begin(), this->arguments.end());
      this->engine->exe(tmp);
      out.push_back(this->engine->cache[0]);
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
      net.arguments[randint(net.arguments.size() - 1)] += mut(generator);
    } else {
      choice(net.ev.params) += mut(generator);
      net.ev.ev_target += mut(generator);
      net.ev.mut_chance += mut(generator);
    }
  }
}

std::vector<std::function<void(Network &)>> mutation_operators = {
    [](Network &net) { mutate<std::normal_distribution<>>(net); },
    [](Network &net) { mutate<std::cauchy_distribution<>>(net); }};

Network create_network(
    Network::ev_strategies ev, int N2 = 1,
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
        std::make_unique<RotationLayer>(RotationLayer(N2, 2 * N2 + 1, true)));
  } else {
    layers_ = std::move(*layers);
  }
  std::normal_distribution<double> dist(-1, 1);
  std::unique_ptr<Encoding> tmp1 =
      std::make_unique<AngleEncoding>(AngleEncoding(N2));
  auto tmp = Network(std::move(layers_), std::move(tmp1), ev, N2);
  tmp.arguments = std::vector<double>();
  for (int i = 0; i < N2 + 2; i++) {
    tmp.arguments.push_back(dist(generator));
  }
  return tmp;
}

Network::ev_strategies create_ev() {
  return Network::ev_strategies{std::make_pair(-0.15, 0.15), 1, 0.8, 0.6, true};
}

double out(Qudit &in) {
  return std::norm(in.get()(1)) + std::norm(in.get()(2));
}

double loss(std::vector<Qudit> res, std::vector<std::vector<double>> expected) {
  double ret = 0;
  for (unsigned int i = 0; i < res.size(); i++) {
    for (unsigned int j = 0; j < expected[0].size(); j++) {
      ret += std::pow(out(res[i]) - expected[i][j], 2);
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
  Generation(unsigned int max_iteration, unsigned int N)
      : max_iteration(max_iteration), N(N) {
    std::normal_distribution<double> gaussian_(-1., 1.);
    for (unsigned int i = 0; i < N; i++) {
      auto tmp = create_ev();
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
    for (size_t i = 0; i < indices.size(); i++) {
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
    // std::cout << "best loss: " << values[indices[0]] << std::endl;
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
get_and() {
  std::vector<std::vector<double>> inputs = {
      {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<std::vector<double>> outs = {{1}, {0}, {1}, {0}};
  return std::make_pair(inputs, outs);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_xor() {
  std::vector<std::vector<double>> inputs = {
      {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<std::vector<double>> outs = {{0}, {1}, {0}, {1}};
  return std::make_pair(inputs, outs);
}

class T {
public:
  string buffer = "QAL & 0 $ \"a\"\nSET $a 1 0\nHAD $a\nRZ $a ??0\nRX $a ??1\n";
  Engine engine;
  std::vector<std::vector<double>> members_;
  int max_iteration;
  int n, m;
  std::vector<double> eta, alpha;
  std::vector<double> spread;
  double learning_rate;

  T(int n, int max_iteration, int m = 2)
      : engine(Network::create_engine(this->buffer)),
        max_iteration(max_iteration), n(n), m(m) {
    this->learning_rate = 10;
    for (int i = 0; i < n; i++) {
      std::vector<double> b = {};
      spread.push_back(s(generator));
      eta.push_back(s(generator));
      alpha.push_back(s(generator));
      for (int j = 0; j < this->m * 2; j++) {
        b.push_back(s(generator));
      }
      this->members_.push_back(b);
    }
  }

  double to_z(cx_vec &in) { return std::norm(in(1)); }

  std::vector<double> loss(std::vector<std::vector<double>> &in,
                           std::vector<std::vector<double>> &expected,
                           std::vector<std::vector<double>> members) {
    std::vector<double> losses;
    for (int k = 0; k < members.size(); k++) {
      double loss = 0;
      for (int i = 0; i < in.size(); i++) {
        std::vector<double> params;
        for (int j = 0; j < this->m; j++) {
          params.push_back(in[i][j] * members[k][2 * j] +
                           members[k][2 * j + 1]);
        }
        this->engine.exe(params);
        auto d = this->engine.memory[0][0].get();
        auto tmp2 = to_z(d);
        double diff = tmp2 - expected[i][0];
        loss += diff * diff;
      }
      losses.push_back(loss / in.size());
    }
    return losses;
  }

  double loss(std::vector<std::vector<double>> &in,
              std::vector<std::vector<double>> &expected,
              std::vector<double> members) {
    double loss = 0;
    for (int i = 0; i < in.size(); i++) {
      std::vector<double> params;
      for (int j = 0; j < this->m; j++) {
        params.push_back(in[i][j] * members[2 * j] + members[2 * j + 1]);
      }
      this->engine.exe(params);
      auto d = this->engine.memory[0][0].get();
      auto tmp2 = to_z(d);
      double diff = tmp2 - expected[i][0];
      loss += diff * diff;
    }
    return loss / in.size();
  }

  std::vector<double> lossD(std::vector<std::vector<double>> &in,
                            std::vector<std::vector<double>> &expected,
                            std::vector<double> member) {
    std::vector<double> loss;
    for (int i = 0; i < in.size(); i++) {
      std::vector<double> params;
      for (int j = 0; j < 2; j++) {
        params.push_back(in[i][j] * member[2 * j] + member[2 * j + 1]);
      }
      this->engine.exe(params);
      auto d = this->engine.memory[0][0].get();
      auto tmp2 = to_z(d);
      loss.push_back(tmp2 - expected[i][0]);
    }
    return loss;
  }

  std::vector<std::vector<double>> mutate() {
    std::vector<std::vector<double>> out;
    for (int i = 0; i < this->members_.size(); i++) {
      std::cauchy_distribution<double> a(-spread[i], spread[i]);
      std::vector<double> tmp = this->members_[i];
      for (int j = 0; j < tmp.size(); j++) {
        tmp[j] += a(generator);
      }
      if (choice()) {
        this->eta[i] += a(generator);
        this->alpha[i] += a(generator);
      }
      if (choice()) {
        this->spread[i] += a(generator);
      }
      out.push_back(tmp);
    }
    return out;
  }

  std::vector<std::vector<double>>
  mutate(std::vector<std::vector<double>> mems) {
    std::vector<std::vector<double>> out;
    for (int i = 0; i < mems.size(); i++) {
      std::cauchy_distribution<double> a(-spread[i], spread[i]);
      std::vector<double> tmp = mems[i];
      for (int j = 0; j < tmp.size(); j++) {
        tmp[j] += a(generator);
      }
      if (choice()) {
        this->eta[i] += a(generator);
        this->alpha[i] += a(generator);
      }
      if (choice()) {
        this->spread[i] += a(generator);
      }
      out.push_back(tmp);
    }
    return out;
  }

  void replace(std::vector<int> indices,
               std::vector<std::vector<double>> others) {
    for (int i = 0; i < n / 3; i++) {
      this->members_[indices[i]] = others[indices[i]];
    }
  }

  int EP(std::vector<std::vector<double>> in,
         std::vector<std::vector<double>> expected, double stop = 0.00001) {
    std::vector<double> out;
    double a = 1000;
    int used = 0;
    for (int epoch = 0; (epoch < max_iteration) && a > stop; ++epoch, used++) {
      auto mutated = mutate();
      auto losses_new = loss(in, expected, mutated);
      auto losses_old = loss(in, expected, this->members_);

      for (int i = 0; i < n; i++) {
        if (losses_new[i] < losses_old[i]) {
          this->members_[i] = mutated[i];
        }
      }
      auto m_l = *std::min_element(losses_old.begin(), losses_old.end());
      a = m_l;
      // std::cout << "best loss: " << m_l << std::endl;
      out.push_back(m_l);
    }
    // return out;
    return used;
  }

  std::pair<int, int> smallest(const std::vector<double> &in) {
    double l1 = 1, l2 = 1;
    int i1 = 0, i2 = 0;
    for (int i = 0; i < in.size(); i++) {
      if (in[i] < l1)
        l1 = in[i], i1 = i;
      else if (in[i] < l2)
        l2 = in[i], i2 = i;
    }
    return std::make_pair(i1, i2);
  }

  std::vector<std::vector<double>> blx_alpha(std::pair<double, double> sm) {
    auto p1 = this->members_[sm.first], p2 = this->members_[sm.second];
    std::vector<std::vector<double>> out;
    for (int j = 0; j < n; j++) {
      std::vector<double> current;
      for (int i = 0; i < p1.size(); i++) {
        std::uniform_real_distribution<> blx(0, 1);
        if (p1[i] < p2[i]) {
          double d = p2[i] - p1[i];
          blx = std::uniform_real_distribution<>(p1[i] - alpha[j] * d,
                                                 p2[i] + alpha[j] * d);
        } else {

          double d = p1[i] - p2[i];
          blx = std::uniform_real_distribution<>(p2[i] - alpha[j] * d,
                                                 p1[i] + alpha[j] * d);
        }
        current.push_back(blx(generator));
      }
      out.push_back(current);
    }
    return out;
  }

  std::vector<std::vector<double>> crossover_blx(std::vector<double> losses) {
    auto parents = this->smallest(losses);
    std::vector<std::vector<double>> out = blx_alpha(parents);
    return out;
  }

  std::vector<std::vector<double>> sbx(std::pair<double, double> sm) {
    auto p1 = this->members_[sm.first], p2 = this->members_[sm.second];
    std::vector<std::vector<double>> out;
    for (int i = 0; i < n / 2; i++) {
      std::vector<double> c1, c2;
      for (int i = 0; i < p1.size(); i++) {
        double u = randu();
        double beta = (u <= 0.5)
                          ? std::pow(2 * u, 1 / (eta[sm.first] + 1))
                          : std::pow(1 / 2 * (1 - u), 1 / (eta[sm.first] + 1));
        c1.push_back((1 / 2) * ((1 + beta) * p1[i] + (1 - beta) * p2[i]));
        c2.push_back((1 / 2) * ((1 - beta) * p1[i] + (1 + beta) * p2[i]));
      }
      out.push_back(c1);
      out.push_back(c2);
    }
    return out;
  }

  std::vector<std::vector<double>> crossover_sbx(std::vector<double> losses) {
    auto parents = this->smallest(losses);
    std::vector<std::vector<double>> out = sbx(parents);
    return out;
  }

  double f(std::vector<double> thetas) {
    std::vector<double> params = thetas;
    this->engine.exe(params);
    auto tmp = this->engine.memory[0][0].get();
    return to_z(tmp);
  }

  int PS(std::vector<std::vector<double>> in,
         std::vector<std::vector<double>> expected, double stop = 0.00001) {
    // Parameter-shift, only modifies first vector
    std::vector<double> losses;
    double a = 100;
    int used = 0;
    for (int epoch = 0; (epoch < max_iteration) && a > stop; epoch++, used++) {
      std::vector<double> lossD = this->lossD(in, expected, this->members_[0]);
      std::vector<double> gradients(this->m * 2, 0.);
      for (int i = 0; i < in.size(); i++) {
        std::vector<double> cs;
        for (int j = 0; j < this->m; j++) {
          cs.push_back(this->members_[0][2 * j] * in[i][(j + 2) % 2] +
                       this->members_[0][2 * j + 1]);
        }
        std::vector<double> partials;
        for (int j = 0; j < this->m; j++) {
          auto thetas1 = cs, thetas2 = cs;
          thetas1[j] += std::numbers::pi / 2;
          thetas2[j] -= std::numbers::pi / 2;
          partials.push_back((in[i][(j + 2) % 2] / 2) *
                             (f(thetas1) - f(thetas2)));
          partials.push_back((1. / 2) * (f(thetas1) - f(thetas2)));
        }
        for (int j = 0; j < this->m * 2; j++) {
          gradients[j] += partials[j] * lossD[i];
        }
      }
      gradients = gradients / in.size();
      this->members_[0] = this->members_[0] - gradients * learning_rate;
      auto tmp = loss(in, expected, members_[0]);
      losses.push_back(tmp);
      a = tmp;
    }
    // return losses;
    return used;
  }

  int GA(std::vector<std::vector<double>> in,
         std::vector<std::vector<double>> expected, double stop = 0.00001) {
    std::vector<double> out;
    double a = MAXFLOAT;
    int used = 0;
    for (int epoch = 0; (epoch < max_iteration) && a > stop; ++epoch, used++) {
      auto losses_old = loss(in, expected, this->members_);
      auto next = this->crossover_blx(losses_old);
      auto mutated = mutate(next);
      auto losses_new = loss(in, expected, mutated);

      for (int i = 0; i < n; i++) {
        if (losses_new[i] < losses_old[i]) {
          this->members_[i] = mutated[i];
        }
      }

      a = *std::min_element(losses_old.begin(), losses_old.end());

      out.push_back(a);
      // std::cout << "best loss: " << a__ << std::endl;
    }
    // return out;
    return used;
  }
};
/*
std::vector<double> median_run_1(std::pair<std::vector<std::vector<double>>,
                                           std::vector<std::vector<double>>>
                                     in,
                                 int runs, int psize, int max_it) {
  std::vector<double> out;
  T net(psize, max_it);
  out = net.EP(in.first, in.second);
  for (int i = 0; i < runs; i++) {
    net = T(psize, max_it);
    out = out + net.EP(in.first, in.second);
    cout << "run: " << i << std::endl;
  }
  return out / runs;
}

std::vector<double> median_run_2(std::pair<std::vector<std::vector<double>>,
                                           std::vector<std::vector<double>>>
                                     in,
                                 int runs, int psize, int max_it) {
  std::vector<double> out;
  T net(psize, max_it);
  out = net.GA(in.first, in.second);
  for (int i = 0; i < runs; i++) {
    net = T(psize, max_it);
    out = out + net.GA(in.first, in.second);
    cout << "run: " << i << std::endl;
  }
  return out / runs;
}

std::vector<double> median_run_3(std::pair<std::vector<std::vector<double>>,
                                           std::vector<std::vector<double>>>
                                     in,
                                 int runs, int psize, int max_it) {
  std::vector<double> out;
  T net(psize, max_it);
  out = net.PS(in.first, in.second);
  for (int i = 0; i < runs; i++) {
    net = T(psize, max_it);
    out = out + net.PS(in.first, in.second);
    cout << "run: " << i << std::endl;
  }
  return out / runs;
}

*/

string scale_net(int thetas) {
  string setup = "QAL & 0 $ \"a\"\nSET $a 1 0\nHAD $a\n";
  string repeat1 = "RZ $a ??";
  string repeat2 = "RX $a ??";
  for (int i = 0; i < thetas; i += 2) {
    setup += repeat1 + std::to_string(i) + "\n";
    setup += repeat2 + std::to_string(i+1) + "\n";
  }
  return setup;
}

using clock_ = std::chrono::high_resolution_clock;

void scale_test(std::pair<std::vector<std::vector<double>>,
                          std::vector<std::vector<double>>>
                    in,
                int runs, int psize, int max_it, int max_scale = 20,
                double stop = 0.001) {
  /*
  std::cout << "GA:" << std::endl;
  for (int ns = 2; ns <= max_scale; ns += 2) {
    T net(psize, max_it, ns);
    net.buffer = scale_net(ns);
    auto start = clock_::now();
    for (int i = 0; i < runs; i++) {
      auto used = net.GA(in.first, in.second, stop);
      std::cout << used << std::endl;
      net = T(psize, max_it, ns);
    }
    auto end = clock_::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << ns << " " << elapsed.count() << std::endl;
  }
  std::cerr << "DONE1";
  std::cout << "EP:" << std::endl;
  for (int ns = 2; ns <= max_scale; ns += 2) {
    T net(psize, max_it, ns);
    net.buffer = scale_net(ns);
    auto start = clock_::now();
    for (int i = 0; i < runs; i++) {
      auto used = net.EP(in.first, in.second, stop);
      std::cout << used << std::endl;
      net = T(psize, max_it, ns);
    }
    auto end = clock_::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << ns << " " << elapsed.count() << std::endl;
  }
  std::cerr << "DONE2";
  */
  std::cout << "PS:" << std::endl;
  for (int ns = 2; ns <= max_scale; ns += 2) {
    T net(psize, max_it, ns);
    net.buffer = scale_net(ns);
    auto start = clock_::now();
    for (int i = 0; i < runs; i++) {
      auto used = net.PS(in.first, in.second, stop);
      std::cout << used << std::endl;
      net = T(psize, max_it, ns);
    }
    auto end = clock_::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << ns << " " << elapsed.count() << std::endl;
  }
}

int main() {
  auto xor_ = get_xor();
  scale_test(xor_, 20, 50, 100, 200);
}
