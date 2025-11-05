#include <armadillo>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "rapidcsv.h"

#define WIDTH 28
#define BREATH 28

#define GEN_SIZE 30
#define IN PIXELS
#define OUT 10

#define EV_STRATEGIE_CHANCE 0.25
#define CHANCE 0.90
#define RESIZE_CHANCE 0.05
#define DOUBLE_MUT 1

#define MIN 2 // kleinste layer-größe
#define MAX 5 // größte beim initialisieren
// #define IN_LENGTH 5 // initiale hidden layers

#define MAX_EXPANSION_SIZE 10000
#define MAX_C_EXPANSION_SIZE 100 // max kernels

#define KERNEL_SIZE 3

constinit int PIXELS = (WIDTH - KERNEL_SIZE + 1) * (BREATH - KERNEL_SIZE + 1);

#define MAX_ITERATIONS 10000

#define LEARNING_RATE 0.24

std::random_device rd;
std::minstd_rand generator(rd());

/*
std::vector<std::function<void(arma::vec &)>> activations;
std::vector<std::function<void(arma::vec &)>> activationsD;
*/

typedef std::vector<arma::Mat<double>> tensor;
using batch = std::vector<arma::vec>;

bool r_choice() {
  std::uniform_real_distribution<double> dist;
  return dist(generator) > .5;
}
bool r_choice(double chance) {
  std::uniform_real_distribution<double> dist;
  return dist(generator) < chance;
}

template <typename T> T r_choice(std::vector<T> a) {
  std::uniform_int_distribution<int> dist(0, a.size() - 1);
  return a[dist(generator)];
}

int randint(int min, int max) {
  assert(min <= max);
  std::uniform_int_distribution<int> dist(min, max);
  return dist(generator);
}

double mse(batch a, batch b) {
  double out;
  for (int i = 0; i < a.size(); i++) {
    out += arma::accu(arma::pow(static_cast<arma::vec>(a[i] - b[i]), 2)) /
           a.size();
  }
  return out;
}

double cross_entropy(batch output, batch expected) {
  double out;
  for (int i = 0; i < output.size(); i++) {
    out -= arma::accu(expected[i] % arma::log(output[i]));
  }
  return out / output.size();
}

arma::vec mseD(batch a, batch b) {
  arma::vec out = arma::zeros(OUT);
  for (int i = 0; i < a.size(); i++) {
    out = out + (a[i] - b[i]);
  }
  return out / a.size();
}

arma::vec mseD(arma::vec a, arma::vec b) { return a - b; }

void sigmoid(arma::vec &in) { in = 1 / (1 + arma::exp(-in)); }

void softmax(arma::mat &in) {
  in = arma::exp(in) * (1 / (arma::sum(arma::exp(in)))).eval()(0, 0);
}

void tanh_(arma::mat &in) { in = arma::tanh(in); }

void sigmoidD(arma::vec &in) {
  in = arma::exp(-in) / (arma::pow(1 + arma::exp(-in), 2));
}

void relu(arma::vec &in) { in = (in + arma::abs(in)) / 2; }
void reluD(arma::vec &in) {
  in = in.transform([](double a) { return static_cast<double>(a >= 0); });
}

arma::cube to_cube2(arma::vec input) {
  arma::cube out = arma::cube(input.n_elem, 1, 1);
  out.slice(0).col(0) = input;
  return out;
}

arma::cube to_cube(arma::mat input) {
  arma::cube out = arma::cube(input.n_rows, input.n_cols, 1);
  out.slice(0) = input;
  return out;
}

class Layer {
public:
  virtual arma::cube &forward(arma::cube &input) = 0;
  virtual std::complex<int>
  resize() = 0; // resize and tell the next layer you resized
  virtual std::complex<int>
      resize(std::complex<int>) = 0; // respond to last layer resizing
  virtual double *get_element() = 0;
  virtual std::unique_ptr<Layer> clone() const = 0;
  virtual ~Layer() {}
};

class FCLayer : public Layer { // fully connected
private:
  void (*activation)(arma::mat &);
  arma::mat weights;
  arma::vec biases;

public:
  FCLayer(void (*activation)(arma::mat &), arma::cube weights, arma::vec biases)
      : activation(activation), weights(weights), biases(biases) {}
  FCLayer(void (*activation_)(arma::mat &), std::pair<int, int> sizes) {
    this->activation = activation_;
    this->biases = arma::randu<arma::vec>(sizes.first);
    this->weights = arma::randu<arma::mat>(sizes.first, sizes.second);
  }
  arma::cube &forward(arma::cube &input) override {
    input = to_cube2(weights * input.slice(0).col(0) + biases);
    activation(input.slice(0));
    return input;
  }
  std::complex<int> resize() override {
    if (this->biases.n_elem >= MAX_EXPANSION_SIZE)
      return 0;
    int n = r_choice() ? 1 : -1;
    if (this->biases.n_elem <= 10)
      n = 1;
    weights.resize(weights.n_rows + n, weights.n_cols);
    biases.resize(biases.n_elem + n);
    if (n == 1) {
      weights.row(weights.n_rows - 1).randu();
      biases(biases.n_elem - 1) = arma::randu();
      return std::complex<int>(n, 1);
    }
    return std::complex<int>(n, 1);
  }

  std::complex<int> resize(std::complex<int> change) override {
    if (change.imag() == 0) {
      weights.resize(weights.n_rows + change.real(), weights.n_cols);
      biases.resize(biases.n_elem + change.real());
      if (change.real() == std::abs(change.real())) {
        for (int i = weights.n_rows - change.real() - 1; i < weights.n_rows;
             i++) {
          weights.row(i).randu();
          biases(i) = arma::randu();
        }
      }
    } else {
      weights.resize(weights.n_rows, weights.n_cols + change.real());
      if (change.real() == std::abs(change.real())) {
        for (int i = weights.n_cols - change.real() - 1; i < weights.n_cols;
             i++) {
          weights.col(i).randu();
        }
      }
      return 0;
    }
    return change;
  }

  double *get_element() override {
    if (r_choice())
      return &weights(randint(0, weights.n_rows - 1),
                      randint(0, weights.n_cols - 1));
    else
      return &biases(randint(0, biases.n_elem - 1));
  }

  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<FCLayer>(*this);
  }

  ~FCLayer() override = default;
};

class Flatten : public Layer {
public:
  Flatten() {}

  arma::cube &forward(arma::cube &input) override {
    input = to_cube2(arma::vectorise(input));
    return input;
  }

  std::complex<int> resize() override { return 0; }
  std::complex<int> resize(std::complex<int> change) override { return change; }
  double *get_element() override { return nullptr; }
  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<Flatten>(*this);
  }
  ~Flatten() override = default;
};

inline arma::mat conv2_valid(const arma::mat &A, const arma::mat &B) {
  arma::mat full = arma::conv2(A, B, "full");

  if (A.n_rows < B.n_rows || A.n_cols < B.n_cols)
    return arma::mat();

  arma::uword start_row = B.n_rows - 1;
  arma::uword start_col = B.n_cols - 1;

  arma::uword end_row = start_row + (A.n_rows - B.n_rows);
  arma::uword end_col = start_col + (A.n_cols - B.n_cols);

  return full.submat(start_row, start_col, end_row, end_col);
}
class CLayer : public Layer { // Convolution-layer
private:
  std::vector<arma::mat> kernels;
  arma::vec biases;
  void (*activation)(arma::mat &);

public:
  CLayer(void (*activation_)(arma::mat &), std::pair<int, int> size) {
    this->activation = activation_;
    this->kernels = std::vector<arma::mat>(
        size.first, arma::randu<arma::mat>(KERNEL_SIZE, KERNEL_SIZE));
    this->biases = arma::randu<arma::vec>(size.first);
  }
  arma::cube &forward(arma::cube &input) override {
    auto tmp = arma::cube(input.n_rows - kernels[0].n_rows + 1,
                          input.n_cols - kernels[0].n_cols + 1, kernels.size());
    for (int i = 0; i < kernels.size(); i++) {
      tmp.slice(i) = conv2_valid(input.slice(0), kernels[i]) + biases(i);
      this->activation(tmp.slice(i));
    }
    input = tmp;
    return input;
  }
  std::complex<int> resize() override {
    if (this->kernels.size() >= MAX_C_EXPANSION_SIZE)
      return 0;
    int n = r_choice() ? 1 : -1;
    if (this->kernels.size() <= 4)
      n = 1;
    if (n == 1) {
      this->kernels.push_back(arma::randu<arma::mat>(KERNEL_SIZE, KERNEL_SIZE));
      this->biases.resize(biases.n_elem + 1);
      this->biases(biases.n_elem - 1) = arma::randu();
    } else {
      this->kernels.erase(this->kernels.end());
      this->biases.resize(biases.n_elem - 1);
    }
    return std::complex<int>(n, 1);
  }
  std::complex<int> resize(std::complex<int> change) override { return change; }
  double *get_element() override {
    int k = randint(0, kernels.size() - 1);
    if (r_choice(0.80)) {
      return &this->kernels[k](
          randint(kernels[k].n_rows - 1, kernels[k].n_cols - 1));
    }
    return &this->biases(k);
  }

  std::unique_ptr<Layer> clone() const override {
    return std::make_unique<CLayer>(*this);
  }

  ~CLayer() override = default;
};

class GNetwork {
private:
  std::vector<std::unique_ptr<Layer>> layers;

public:
  int size() const { return layers.size(); }

  GNetwork() = default;

  // GNetwork(const GNetwork&) = delete;
  // GNetwork& operator=(const GNetwork&) = delete;
  GNetwork(GNetwork &&) noexcept = default;
  GNetwork &operator=(GNetwork &&) noexcept = default;

  // deep copy constructor
  GNetwork(const GNetwork &other) {
    layers.reserve(other.layers.size());
    for (const auto &l : other.layers)
      layers.push_back(l->clone()); // requires Layer::clone()
  }

  GNetwork &operator=(const GNetwork &other) {
    if (this != &other) {
      layers.clear();
      layers.reserve(other.layers.size());
      for (const auto &l : other.layers)
        layers.push_back(l->clone());
    }
    return *this;
  }

  GNetwork(std::vector<std::unique_ptr<Layer>> &&layers_) {
    this->layers = std::move(layers_);
  }
  batch forward(std::vector<arma::mat> input) {
    batch out;
    int start = randint(0, input.size() - 302);
    for (int i = start; i < start + 300; i++) {
      auto in = input[i];
      auto tmp_input = to_cube(in);
      arma::cube tmp = layers[0]->forward(tmp_input);
      for (int j = 1; j < layers.size(); j++) {
        tmp = layers[j]->forward(tmp);
      }
      out.push_back(tmp);
    }
    return out;
  }

  double loss(batch output, batch expected) {
    return cross_entropy(output, expected);
  }

  void resize(int layer) {
    std::complex<int> response = this->layers[layer]->resize();
    while (response != 0 && layer < layers.size() - 1) {
      layer++;
      response = this->layers[layer]->resize(response);
    }
  }

  double *get_element() {
    int layer = randint(0, layers.size() - 1);
    auto tmp = this->layers[layer]->get_element();
    if (tmp) {
      return tmp;
    }
    return this->get_element();
  }

  void append(std::unique_ptr<Layer> &&item) {
    this->layers.push_back(std::move(item));
  }
};

class Chromosome {
public:
  Chromosome() {}
  Chromosome(GNetwork &net, std::pair<double, double> str_ev)
      : network(std::move(net)), ev_strategie(str_ev) {}
  GNetwork network;
  std::pair<double, double> ev_strategie;
};

std::function<void(Chromosome &)> current_mutation;

enum LType {
  FULLY_CONNECTED,
  CONVOLUTION,
  FLATTEN,
};

std::string ltypetostring(LType t) {
  switch (t) {
  case FULLY_CONNECTED:
    return "FULLY_CONNECTED";
  case CONVOLUTION:
    return "CONVOLUTION";
  case FLATTEN:
    return "FLATTEN";
  }
}

struct LayerType {
  enum LType type;
  std::optional<std::pair<int, int>> info;
  void (*activation)(arma::mat &);
};

std::vector<Chromosome> gen(const std::vector<LayerType> &layers, int size) {
  std::vector<Chromosome> out;
  for (int i = 0; i < size; i++) {
    GNetwork net;
    for (int j = 0; j < layers.size(); j++) {
      switch (layers[j].type) {
      case FULLY_CONNECTED:
        net.append(std::move(std::unique_ptr<FCLayer>(
            new FCLayer(layers[j].activation, *layers[j].info))));
        break;
      case FLATTEN:
        net.append(std::move(std::unique_ptr<Flatten>(new Flatten())));
        break;
      case CONVOLUTION:
        net.append(std::move(std::unique_ptr<CLayer>(
            new CLayer(layers[j].activation, *layers[j].info))));
        break;
      }
    }
    out.push_back(Chromosome{net, std::make_pair(0, 1)});
  }
  return out;
}

std::vector<int> extract_size(tensor x) {
  std::vector<int> out;
  for (auto a : x) {
    out.push_back(a.n_cols);
  }
  return out;
}

void gaussian(Chromosome &ch) {
  std::normal_distribution<double> gaussian(ch.ev_strategie.first,
                                            ch.ev_strategie.second);
  if (r_choice(EV_STRATEGIE_CHANCE)) {
    if (r_choice()) {
      ch.ev_strategie.first += gaussian(generator);
    } else {
      ch.ev_strategie.second += gaussian(generator);
    }
  }
  *ch.network.get_element() += gaussian(generator);
}

void cauchy(Chromosome &ch) {
  std::cauchy_distribution<double> cauchy_(ch.ev_strategie.first,
                                           ch.ev_strategie.second);
  if (r_choice(EV_STRATEGIE_CHANCE)) {
    if (r_choice()) {
      ch.ev_strategie.first += cauchy_(generator);
    } else {
      ch.ev_strategie.second += cauchy_(generator);
    }
  }
  *ch.network.get_element() += cauchy_(generator);
}

void resize(Chromosome &ch) {
  int layer = randint(1, ch.network.size() -
                             2); // IN und OUT Layer müssen die Größe einhalten
  ch.network.resize(layer);
}

std::vector<std::function<void(Chromosome &)>> mutations = {cauchy, gaussian};

void mutate(Chromosome &ch, double chance, bool double_mut) {
  do {
    if (r_choice(chance)) {
      r_choice(mutations)(ch);
    } else
      return;
  } while (double_mut);
}

Chromosome mutate2(Chromosome &ch, double chance, bool double_mut) {
  Chromosome next = ch;
  do {
    if (r_choice(chance)) {
      // r_choice(mutations)(ch);
      if (r_choice(RESIZE_CHANCE)) {
        resize(ch);
      } else {
        current_mutation(ch);
      }
    } else
      return next;
  } while (double_mut);
  return next;
}

void next_gen(std::vector<Chromosome> &a) {
  for (int i = 0; i < a.size(); i++) {
    mutate(a[i], CHANCE, DOUBLE_MUT);
  }
}

std::vector<Chromosome> next_gen2(std::vector<Chromosome> a) {
  std::vector<Chromosome> ret;
  ret.reserve(2 * GEN_SIZE);
  ret.insert(ret.end(), std::make_move_iterator(a.begin()),
             std::make_move_iterator(a.end())); // filled to GEN_SIZE - 1
  for (int i = 0; i < GEN_SIZE; i++) {
    ret.push_back(mutate2(ret[i], CHANCE, DOUBLE_MUT));
  }
  return ret;
}

std::vector<int> concat(std::vector<std::vector<int>> a) {
  std::vector<int> out;
  for (auto b : a) {
    out.insert(out.end(), b.begin(), b.end());
  }
  return out;
}

std::vector<int> rand_int(int size) {
  std::vector<int> out;
  for (int i = 0; i < size; i++) {
    out.push_back(randint(MIN, MAX));
  }
  return out;
}

double loss(Chromosome network, batch output, batch expected) {
  return network.network.loss(output, expected);
}

batch forward(Chromosome network, std::vector<arma::mat> tbatch) {
  return network.network.forward(tbatch);
}

std::vector<Chromosome> get_generation_EP(
    std::optional<std::vector<Chromosome>> generation = std::nullopt,
    std::vector<arma::mat> tbatch = std::vector<arma::mat>(),
    batch expected = batch()) {
  std::vector<Chromosome> gen_;
  gen_.reserve(GEN_SIZE);
  if (!generation) {
    int folds = randint(3, 8), n1 = randint(30, 300);
    gen_ = std::move(
        gen({LayerType{CONVOLUTION,
                       std::make_optional(std::make_pair(folds, 0)), &tanh_},
             LayerType{FLATTEN, std::nullopt, nullptr},
             LayerType{FULLY_CONNECTED,
                       std::make_optional(std::make_pair(n1, folds * PIXELS)),
                       &tanh_},
             LayerType{FULLY_CONNECTED,
                       std::make_optional(std::make_pair(OUT, n1)), &softmax}},
            GEN_SIZE));
  } else {
    auto ranking = next_gen2(*generation);
    std::vector<size_t> indices(ranking.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<double> values(ranking.size());
    std::transform(ranking.begin(), ranking.end(), values.begin(), [&](auto a) {
      return loss(a, forward(a, tbatch), expected);
    });

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return values[i] < values[j]; });
    std::cout << values[indices[0]] << std::endl;
    for (int i = 0; i < GEN_SIZE; i++) {
      gen_.push_back(ranking[indices[i]]);
    }
  }
  return gen_;
}

std::pair<std::vector<arma::vec>, std::vector<arma::vec>> get_xor() {
  std::vector<arma::vec> inputs = {{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<arma::vec> outs = {{0}, {1}, {0}, {1}};
  return std::make_pair(inputs, outs);
}

using generation = std::vector<Chromosome>;

double findEP(std::vector<arma::mat> tbatch, batch expected) {
  generation tmp = get_generation_EP();
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    tmp = get_generation_EP(std::make_optional(tmp), tbatch, expected);
    if (i > MAX_ITERATIONS / 4)
      current_mutation = gaussian;
  }
  auto res = tmp[0].network.forward(tbatch);
  for (int x = 0; x < 4; x++) {
    std::cout << "results from the best network: " << tbatch[x][0] << ", "
              << tbatch[x][1] << " -> " << res[x][0] << std::endl;
  }
  return loss(tmp[0], res, expected);
}

std::pair<rapidcsv::Document, rapidcsv::Document> // (train, test)
get_mnist(std::string pathA = "/home/bassneptun/NN/mnist_train.csv",
          std::string pathB = "/home/bassneptun/NN/mnist_test.csv") {
  rapidcsv::Document mnist_train(pathA);
  rapidcsv::Document mnist_test(pathB);
  return std::make_pair(mnist_train, mnist_test);
}

class labeled_data {
public:
  std::vector<arma::mat> in;
  std::vector<arma::vec> out;

  labeled_data() {}

  labeled_data(std::vector<std::vector<int>> in_) {
    in.resize(in_.size());
    out.resize(in_.size());
    for (int i = 0; i < in_.size(); i++) {
      arma::vec a = arma::zeros(10);
      a(in_[i][0]) = 1.;
      this->out[i] = a;
      in_[i].erase(in_[i].begin());
      std::vector<double> v = std::vector<double>(in_[i].begin(), in_[i].end());
      arma::mat b_mat = arma::reshape<arma::mat>(v, 28, 28);
      b_mat /= 255.;
      this->in[i] = b_mat;
    }
  }
};

labeled_data convert(rapidcsv::Document doc) {
  labeled_data out;
  std::vector<std::vector<int>> acc;
  for (int i = 0; i < doc.GetRowCount(); i++) {
    auto tmp = doc.GetRow<int>(i);
    acc.push_back(tmp);
  }
  out = labeled_data(acc);
  return out;
}

int main() {
  std::srand(std::time(NULL));
  current_mutation = cauchy;
  auto test = get_xor();
  /*
  activations =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoid);
  activationsD =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoidD);
  */

  auto a = get_mnist();
  labeled_data train_data = convert(a.first);
  double xor_loss = findEP(train_data.in, train_data.out);
  // labeled_data test_data = convert(a.second);
  std::cout << "MNIST: " << a.first.GetRowCount() << std::endl;
}
