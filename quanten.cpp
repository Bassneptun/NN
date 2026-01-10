#include <algorithm>
#include <armadillo>
#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
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
  RotationLayer(int n, int arg_pos, bool entangled = false, string gate = "RZ")
      : gate(gate), n(n), entangled(entangled), arg_pos(arg_pos) {}
  string code() override {
    std::stringstream ss;
    // s ist ein Rotationsgatter => RX/Y/Z
    assert(gate == "RX" || gate == "RY" || gate == "RZ");
    if (entangled) {
      ss << "D" << this->gate << " \%b ??" << arg_pos << "\n";
    } else {
      for (int i = 0; i < this->n; i++) {
        ss << this->gate << " $a" << i << " ??" << arg_pos << "\n";
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
      ss << "RX $a" << i << " ??" << i << "\n";
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

  std::vector<Qbit> forward(const std::vector<double> &inputs) {
    std::vector<double> tmp = inputs;
    tmp.insert(tmp.end(), this->arguments.begin(), this->arguments.end());
    this->engine->exe(tmp);
    return this->engine->memory[0];
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
    net.arguments[static_cast<size_t>(randint(
        static_cast<size_t>(net.arguments.size() - 1)))] += mut(generator);
  } else {
    choice(net.ev.params) += mut(generator);
    net.ev.ev_target += mut(generator);
    net.ev.mut_chance += mut(generator);
  }
}

std::vector<std::function<void(Network &)>> mutation_operators = {
    [](Network &net) { mutate<std::normal_distribution<>>(net); },
    [](Network &net) { mutate<std::cauchy_distribution<>>(net); }};

template <int N2>
Network create_network(
    Network::ev_strategies ev,
    std::optional<std::vector<std::unique_ptr<Layer>>> layers = std::nullopt) {
  // standard version: AngleEncoding -> RotationLayer -> Entanglement Layer ->
  // RotationLayer -> DetanglementLayer
  std::vector<std::unique_ptr<Layer>> layers_;
  if (!layers) {
    layers_.push_back(std::make_unique<RotationLayer>(RotationLayer(N2, N2)));
    layers_.push_back(std::make_unique<EntaglementLayer>(EntaglementLayer(N2)));
    layers_.push_back(
        std::make_unique<RotationLayer>(RotationLayer(N2, 2 * N2 - 1, true)));
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
  for (int i = 0; i < N2; i++) {
    tmp.arguments.push_back(dist(generator));
  }
  return tmp;
}

Network::ev_strategies create_ev() {
  return Network::ev_strategies{std::make_pair(-1., 1.), 0, 0.4, 0.5};
}

template <typename data2, typename out, typename data>
constexpr out loss(data res, data2 expected) {
  return (res - expected) * (res - expected);
}

template <typename T, typename U>
concept Subtractable = requires(T a, U b) {
  {
    loss<U, double>(create_network<1>(create_ev()).forward(a), b)
  } -> std::convertible_to<double>;
};

std::vector<unsigned int> create_first_n(unsigned int n) {
  std::vector<unsigned int> out(n);
  for (unsigned int i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

template <int N, int N2, typename T, typename U>
  requires Subtractable<T, U>
class Generation {
private:
  std::vector<Network> members;
  int max_iteration;

public:
  Generation(int max_iteration) : max_iteration(max_iteration) {
    std::normal_distribution<double> gaussian_(-1., 1.);
    for (unsigned int i = 0; i < N; i++) {
      auto tmp = Network::ev_strategies{
          std::make_pair(gaussian_(generator), gaussian_(generator)), 0, 0.4,
          0.5};
      auto tmp2 = create_network<N2>(tmp);
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

  void insert(std::vector<Network> &values) {
    auto indices = create_first_n(N);
    for (unsigned int i = 0; i < static_cast<int>(N / 3); i++) {
      unsigned int index = randint(N - i);
      auto val = indices[index];
      indices.erase(indices.begin() + index);
      this->members[val] = values[i];
    }
  }

  std::vector<Network> get_best(std::vector<T> in, std::vector<U> expected, Generation &ranking) {
    std::vector<size_t> indices(ranking.members.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<double> values(ranking.members.size());
    std::transform(ranking.members.begin(), ranking.members.end(),
                   values.begin(), [&](Network a) {
                     return loss<std::vector<double>, double>(
                         (std::vector<Qbit>){a.forward(in)[0]}, expected);
                   });

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return values[i] < values[j]; });
    std::vector<Network> out;
    for (size_t i = 0; i < N / 3; i++) {
      out.push_back(ranking.members[indices[i]]);
    }
    cout << "best loss: " << values[indices[0]] << std::endl;
    return out;
  }

  Network EP(std::vector<T> in, std::vector<U> expected) {
    for (int i = 0; i < this->max_iteration; i++) {
      std::cout << i << std::endl;
      auto ranking = Generation(*this);
      ranking.mutate(ranking);
      std::vector<Network> best = get_best(in, expected, ranking);
      insert(best);
    }
    return this->members[0];
  }
};

double operator-(std::vector<double> lhs, std::vector<double> &rhs) {
  return std::transform_reduce(lhs.begin(), lhs.end(), rhs.begin(), 0.,
                               std::plus<>(), std::minus<>());
}

std::vector<double> chances(std::vector<Qbit> qbits) {
  std::vector<double> out;
  std::transform(qbits.begin(), qbits.end(), std::back_inserter(out),
                 [](Qbit &in) { return std::norm(in.get().at(0)); });
  return out;
}

double operator-(std::vector<Qbit> &lhs, std::vector<double> &rhs) {
  return chances(lhs) - rhs;
}

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
  Generation result =
      Generation<30, 2, std::vector<double>, std::vector<double>>(10);
  result.EP(xor_.first, xor_.second);
}
