#include <armadillo>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "rapidcsv.h"

#define GEN_SIZE 30
#define IN 2
#define OUT 1

#define EV_STRATEGIE_CHANCE 0.25
#define CHANCE 0.90
#define RESIZE_CHANCE 0.05
#define DOUBLE_MUT 1

#define MIN 2       // kleinste layer-größe
#define MAX 5       // größte beim initialisieren
#define IN_LENGTH 5 // initiale hidden layers

#define MAX_EXPANSION_SIZE 1000
#define MAX_C_EXPANSION_SIZE 10 // max kernels

#define KERNEL_SIZE 3

#define MAX_ITERATIONS 10000

#define LEARNING_RATE 0.24

std::random_device rd;
std::minstd_rand generator(rd());

std::vector<std::function<void(arma::vec &)>> activations;
std::vector<std::function<void(arma::vec &)>> activationsD;

typedef std::vector<arma::Mat<double>> tensor;
using batch = std::vector<arma::vec>;

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
    out -= arma::accu(expected[i] * arma::log(output[i]));
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

struct NN {
  tensor weights;
  std::vector<arma::vec> biases;
};

class Chromosome {
public:
  tensor weights;
  std::vector<arma::vec> biases;
  std::pair<double, double> ev_strategie;
};

std::function<void(Chromosome &)> current_mutation;

std::vector<Chromosome> gen(std::vector<int> sizes, int size) {
  std::vector<Chromosome> out;
  for (int i = 0; i < size; i++) {
    tensor weights;
    std::vector<arma::vec> biases;
    for (int j = 1; j < sizes.size(); j++) {
      weights.push_back(arma::mat(sizes[j], sizes[j - 1], arma::fill::randu));
      biases.push_back(arma::vec(sizes[j], arma::fill::randu));
    }
    out.push_back(Chromosome{weights, biases, std::make_pair(0, 1)});
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

Chromosome mv(tensor weights, std::vector<arma::vec> biases,
              std::pair<double, double> ev_strategie) {
  return Chromosome{std::move(weights), std::move(biases),
                    std::move(ev_strategie)};
}

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

// Option 1
Chromosome crossover(Chromosome &a, Chromosome &b) {
  return mv(r_choice() ? a.weights : b.weights,
            r_choice() ? a.biases : b.biases,
            r_choice() ? a.ev_strategie : b.ev_strategie);
}

// Option 2
Chromosome crossover2(Chromosome &a, Chromosome &b) {
  Chromosome out;
  for (int i = 0; i < a.biases.size(); i++) {
    bool decision = r_choice();
    out.biases.push_back(std::move(decision ? a.biases[i] : b.biases[i]));
    out.weights.push_back(std::move(decision ? a.weights[i] : b.weights[i]));
  }
  out.ev_strategie = std::move(r_choice() ? a.ev_strategie : b.ev_strategie);
  return out;
}

void random_change_weighted(Chromosome &ch) {
  if (r_choice()) {
    int layer = randint(0, IN_LENGTH);
    std::pair<int, int> pos =
        std::make_pair(randint(0, ch.weights[layer].n_rows - 1),
                       randint(0, ch.weights[layer].n_cols - 1));
    int n = randint(0, 64);
    ch.weights[layer](pos.first, pos.second) =
        std::bit_cast<double>(std::bit_cast<unsigned long long>(
                                  ch.weights[layer](pos.first, pos.second)) ^
                              (1ULL << n));
  } else {
    int layer = randint(0, IN_LENGTH);
    int pos = randint(0, ch.biases[layer].n_elem - 1);
    int n = randint(0, 64);
    ch.biases[layer](pos) = std::bit_cast<double>(
        std::bit_cast<unsigned long long>(ch.biases[layer](pos)) ^ (1ULL << n));
  }
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
  if (r_choice()) {
    int layer = randint(0, IN_LENGTH);
    std::pair<int, int> pos =
        std::make_pair(randint(0, ch.weights[layer].n_rows - 1),
                       randint(0, ch.weights[layer].n_cols - 1));
    ch.weights[layer](pos.first, pos.second) += gaussian(generator);
  } else {
    int layer = randint(0, IN_LENGTH);
    int pos = randint(0, ch.biases[layer].n_elem - 1);
    ch.biases[layer](pos) += gaussian(generator);
  }
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
  if (r_choice()) {
    int layer = randint(0, IN_LENGTH);
    // std::cout << "weigths_info: n_elem{" << ch.weights[layer].n_elem
    //           << "} n_rows{" << ch.weights[layer].n_rows << "} n_cols {"
    //           << ch.weights[layer].n_cols << "}" << std::endl;
    std::pair<int, int> pos =
        std::make_pair(randint(0, ch.weights[layer].n_rows - 1),
                       randint(0, ch.weights[layer].n_cols - 1));
    ch.weights[layer](pos.first, pos.second) += cauchy_(generator);
  } else {
    int layer = randint(0, IN_LENGTH);
    // std::cout << "biases_info: n_elem{" << ch.biases[layer].n_elem << "}"
    //           << std::endl;
    int pos = randint(0, ch.biases[layer].n_elem - 1);
    ch.biases[layer](pos) += cauchy_(generator);
  }
}

void resize(Chromosome &ch) {
  int layer = randint(1, ch.biases.size() -
                             2); // IN und OUT Layer müssen die Größe einhalten
  if (ch.biases[layer].n_elem > MAX_EXPANSION_SIZE)
    return;
  else if (ch.biases[layer].n_elem < MIN + 1)
    return;
  int n = r_choice() ? 1 : -1;
  ch.weights[layer].resize(ch.weights[layer].n_rows + n,
                           ch.weights[layer].n_cols);
  ch.weights[layer + 1].resize(ch.weights[layer + 1].n_rows,
                               ch.weights[layer + 1].n_cols + n);
  if (n == 1) {
    ch.weights[layer].row(ch.weights[layer].n_rows - 1).randu();
    ch.weights[layer + 1].col(ch.weights[layer + 1].n_cols - 1).randu();
  }
  // biases sind nicht verstrickt
  ch.biases[layer].resize(ch.biases[layer].n_elem + n);
  if (n == 1)
    ch.biases[layer](ch.biases[layer].n_elem - 1) = arma::randu();
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

std::vector<Chromosome> get_generation_GA(
    std::optional<std::vector<Chromosome>> generation = std::nullopt,
    std::optional<std::pair<int, int>> best = std::nullopt) {
  std::vector<Chromosome> gen_;
  if (!generation && !best) {
    gen_ = gen(concat({{IN}, rand_int(IN_LENGTH), {OUT}}), GEN_SIZE);
  } else {
    std::vector<Chromosome> c(GEN_SIZE,
                              Chromosome{tensor(), std::vector<arma::vec>()});
    for (int i = 0; i < c.size(); i++) {
      c[i] = crossover((*generation)[best->first], (*generation)[best->second]);
    }
    gen_ = c;
    next_gen(gen_);
  }
  return gen_;
}

class Layer {
public:
  virtual arma::cube &forward(arma::cube &input);
  virtual std::complex<int>
  resize(); // resize and tell the next layer you resized
  virtual std::complex<int>
      resize(std::complex<int>); // respond to last layer resizing
  virtual double* get_element();
};

class FCLayer : public Layer { // fully connected
private:
  std::function<void(arma::mat &)> activation;
  arma::mat weights;
  arma::vec biases;

public:
  FCLayer(std::function<void(arma::mat &)> activation, arma::cube weights,
          arma::vec biases)
      : activation(activation), weights(weights), biases(biases) {}
  arma::cube &forward(arma::cube &input) override {
    input.slice(0).col(0) = input.slice(0).col(0) * weights + biases;
    activation(input.slice(0));
    return input;
  }
  std::complex<int> resize() override {
    if (this->biases.n_elem <= MAX_EXPANSION_SIZE)
      return 0;
    int n = r_choice() ? 1 : -1;
    weights.resize(weights.n_rows + n, weights.n_cols);
    biases.resize(biases.n_elem + n);
    if (n == 1) {
      weights.row(weights.n_rows - 1).randu();
      biases(biases.n_elem-1) = arma::randu();
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

  double* get_element() override {
    if(r_choice()) return &weights(randint(0, weights.n_rows-1), randint(0, weights.n_cols-1));
    else return &biases(randint(0, biases.n_elem - 1));
  }
};

class Flatten : public Layer {
public:
  arma::cube &forward(arma::cube &input) override {
    input.slice(0).col(0) = arma::vectorise(input);
    return input;
  }

  std::complex<int> resize() override { return 0; }
  std::complex<int> resize(std::complex<int> change) override { return change; }
  double* get_element() override { return nullptr; }
};

class CLayer : public Layer { // Convolution-layer
private:
  std::vector<arma::mat> kernels;
  arma::vec biases;
  std::function<void(arma::mat &)> activation;

public:
  CLayer(std::vector<arma::mat> kernels, arma::vec biases,
         std::function<void(arma::mat &)> activation)
      : kernels(kernels), biases(biases), activation(activation) {}
  arma::cube &forward(arma::cube &input) override {
    auto tmp = arma::cube(input.n_rows - kernels[0].n_rows + 1,
                          input.n_cols - kernels[0].n_cols + 1, kernels.size());
    for (int i = 0; i < kernels.size(); i++) {
      tmp.slice(i) =
          arma::conv2(input.slice(0), kernels[i], "valid") + biases(i);
      activation(tmp.slice(i));
    }
    input = tmp;
    return input;
  }
  std::complex<int> resize() override {
    if (this->kernels.size() >= MAX_C_EXPANSION_SIZE)
      return 0;
    int n = r_choice() ? 1 : -1;
    if (n == 1){
      this->kernels.push_back(arma::randu<arma::mat>(KERNEL_SIZE, KERNEL_SIZE));
      this->biases.resize(biases.n_elem + 1);
      this->biases(biases.n_elem - 1) = arma::randu();
    }
    else{
      this->kernels.erase(this->kernels.end());
      this->biases.resize(biases.n_elem - 1);
    }
    return std::complex<int>(n, 1);
  }
  std::complex<int> resize(std::complex<int> change) override { return change; }
  double* get_element() override {

  }
};

arma::cube to_cube(arma::vec input) {
  auto out = arma::cube(input.n_elem, 1, 1);
  out.slice(0).col(0) = input;
  return out;
}

class GNetwork {
public:
  std::vector<std::unique_ptr<Layer>> layers;
  GNetwork(std::vector<std::unique_ptr<Layer>> layers_) {
    this->layers = std::move(layers_);
  }
  batch forward(batch input) {
    batch out;
    for (auto in : input) {
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
    while (response != 0 && layer < layers.size()-1) {
      layer++;
      response = this->layers[layer]->resize(response);
    }
  }
};

class FFNetwork {
public:
  tensor weights;
  std::vector<arma::vec> biases;

  FFNetwork(std::vector<int> sizes) {
    for (int i = 0; i < sizes.size(); i++) {
      this->weights.push_back(
          arma::mat(sizes[i], sizes[i - 1], arma::fill::randu));
      this->biases.push_back(arma::vec(sizes[i], arma::fill::randu));
    }
  }

  FFNetwork(NN Network) {
    this->weights = Network.weights;
    this->biases = Network.biases;
  }

  std::vector<arma::vec> forward(batch in) {
    batch output;
    for (int i = 0; i < in.size(); i++) {
      auto tmp = in[i];
      for (int j = 0; j < this->biases.size(); j++) {
        tmp = weights[j] * tmp;
        tmp = tmp + biases[j];
        activations[j](tmp);
      }
      output.push_back(tmp);
    }
    return output;
  }

  std::vector<std::vector<arma::vec>> forwardL(arma::vec in) {
    std::vector<arma::vec> preactivations;
    std::vector<arma::vec> activations_;
    auto tmp = in;
    for (int j = 0; j < this->biases.size(); j++) {
      activations_.push_back(tmp);
      tmp = weights[j] * tmp;
      tmp = tmp + biases[j];
      preactivations.push_back(tmp);
      activations[j](tmp);
    }
    activations_.push_back(tmp);
    std::vector<arma::vec> output = {tmp};
    return {output, activations_, preactivations};
  }

  std::vector<std::vector<std::vector<arma::vec>>> forwardB(batch in) {
    std::vector<std::vector<std::vector<arma::vec>>> out;
    for (auto &a : in) {
      out.push_back(this->forwardL(a));
    }
    return out;
  }

  void backward(batch in, batch expected) {
    auto tmp2 = forwardB(in);
    std::vector<arma::mat> gradients(weights.size());
    std::vector<arma::vec> bias_gradients(weights.size());
    std::vector<arma::vec> errors(weights.size());

    for (int i = 0; i < weights.size(); i++) {
      gradients[i].zeros(weights[i].n_rows, weights[i].n_cols);
      bias_gradients[i].zeros(biases[i].n_elem);
    }

    for (int s = 0; s < in.size(); ++s) {
      auto &tmp = tmp2[s];
      auto &output = tmp[0];
      auto &activations_ = tmp[1];
      auto &preactivations = tmp[2];

      for (int i = 0; i < preactivations.size(); i++) {
        activationsD[i](preactivations[i]);
      }

      arma::vec loss_grad = this->lossD(output[0], expected[s]);
      errors.back() = loss_grad % preactivations.back();

      for (int i = weights.size() - 2; i >= 0; i--) {
        errors[i] = (weights[i + 1].t() * errors[i + 1]) % preactivations[i];
      }

      for (int i = 0; i < errors.size(); i++) {
        gradients[i] += errors[i] * activations_[i].t();
        bias_gradients[i] += errors[i];
      }
    }

    for (int i = 0; i < weights.size(); i++) {
      this->weights[i] -= LEARNING_RATE * (gradients[i] / in.size());
      this->biases[i] -= LEARNING_RATE * (bias_gradients[i] / in.size());
    }
  }
  double loss(batch a, batch b) { return mse(a, b); }
  arma::vec lossD(batch a, batch b) { return mseD(a, b); }
  arma::vec lossD(arma::vec a, arma::vec b) { return mseD(a, b); }
};

double loss(NN network, batch output, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(network).loss(output, expected);
}

double loss(Chromosome network, batch output, batch expected) {
  NN network2 = NN{network.weights, network.biases};
  return FFNetwork(network2).loss(output, expected);
}

batch forward(NN network, batch tbatch,
              std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(network).forward(tbatch);
}

batch forward(Chromosome network, batch tbatch) {
  NN network2 = NN{network.weights, network.biases};
  return FFNetwork(network2).forward(tbatch);
}

std::vector<Chromosome> get_generation_EP(
    std::optional<std::vector<Chromosome>> generation = std::nullopt,
    batch tbatch = batch(), batch expected = batch()) {
  std::vector<Chromosome> gen_;
  gen_.reserve(GEN_SIZE);
  if (!generation) {
    gen_ = gen(concat({{IN}, {rand_int(IN_LENGTH)}, {OUT}}), GEN_SIZE);
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

NN findM(batch tbatch, batch expected,
         std::vector<std::function<void(arma::vec &)>> activations) {
  std::pair<int, int> indices;
  std::pair<double, double> best_losses = {std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
  generation tmp = get_generation_GA();
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    for (int j = 0; j < GEN_SIZE; j++) {
      auto output =
          forward(NN{tmp[j].weights, tmp[j].biases}, tbatch, activations);
      auto tmp2 = loss(NN{tmp[j].weights, tmp[j].biases}, output, expected,
                       activations);
      if (tmp2 < best_losses.second) {
        if (tmp2 < best_losses.first) {
          indices.first = j;
          best_losses.second = best_losses.first;
          best_losses.first = tmp2;
        } else if (tmp2 != best_losses.second) {
          indices.second = j;
          best_losses.second = tmp2;
        }
      }
    }

    /*
    if (i % 10 == 0)
      std::cout << "best losses: " << best_losses.first << " "
                << best_losses.second << std::endl;
    */

    tmp =
        get_generation_GA(std::make_optional(tmp), std::make_optional(indices));
    best_losses = {std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  }
  auto res = forward(NN{tmp[indices.first].weights, tmp[indices.first].biases},
                     tbatch, activations);
  for (int x = 0; x < 4; x++) {
    std::cout << "results from the best network: " << tbatch[x][0] << ", "
              << tbatch[x][1] << " -> " << res[x][0] << std::endl;
  }
  return NN{tmp[0].weights, tmp[0].biases};
}

double findGA(batch tbatch, batch expected) {
  std::pair<int, int> indices;
  std::pair<double, double> best_losses = {std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
  generation tmp = get_generation_GA();
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    for (int j = 0; j < GEN_SIZE; j++) {
      auto output =
          forward(NN{tmp[j].weights, tmp[j].biases}, tbatch, activations);
      auto tmp2 = loss(NN{tmp[j].weights, tmp[j].biases}, output, expected,
                       activations);
      if (tmp2 < best_losses.second) {
        if (tmp2 < best_losses.first) {
          indices.first = j;
          best_losses.second = best_losses.first;
          best_losses.first = tmp2;
        } else if (tmp2 != best_losses.second) {
          indices.second = j;
          best_losses.second = tmp2;
        }
      }
    }

    /*
    if (i % 10 == 0)
      std::cout << "best losses: " << best_losses.first << " "
                << best_losses.second << std::endl;
    */
    tmp =
        get_generation_GA(std::make_optional(tmp), std::make_optional(indices));
    best_losses = {std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  }
  auto res = forward(NN{tmp[indices.first].weights, tmp[indices.first].biases},
                     tbatch, activations);
  for (int x = 0; x < 4; x++) {
    std::cout << "results from the best network: " << tbatch[x][0] << ", "
              << tbatch[x][1] << " -> " << res[x][0] << std::endl;
  }
  return best_losses.first;
}

double findEP(batch tbatch, batch expected) {
  generation tmp = get_generation_EP();
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    tmp = get_generation_EP(std::make_optional(tmp), tbatch, expected);
    if (i > MAX_ITERATIONS / 4)
      current_mutation = gaussian;
  }
  auto res = forward(NN{tmp[0].weights, tmp[0].biases}, tbatch, activations);
  for (int x = 0; x < 4; x++) {
    std::cout << "results from the best network: " << tbatch[x][0] << ", "
              << tbatch[x][1] << " -> " << res[x][0] << std::endl;
  }
  return loss(tmp[0], res, expected);
}

void sigmoid(arma::vec &in) { in = 1 / (1 + arma::exp(-in)); }

void softmax(arma::vec &in) { in = arma::exp(in) / (arma::sum(arma::exp(in))); }

void tanh(arma::mat &in) { in = arma::tanh(in); }

void sigmoidD(arma::vec &in) {
  in = arma::exp(-in) / (arma::pow(1 + arma::exp(-in), 2));
}

void relu(arma::vec &in) { in = (in + arma::abs(in)) / 2; }
void reluD(arma::vec &in) {
  in = in.transform([](double a) { return static_cast<double>(a >= 0); });
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
  std::vector<arma::vec> in;
  std::vector<arma::vec> out;

  labeled_data() {}

  labeled_data(std::vector<std::vector<int>> in_) {
    in.resize(in_.size());
    out.resize(in_.size());
    for (int i = 0; i < in_.size(); i++) {
      arma::vec a = arma::zeros(10);
      a(in_[i][0]) = 1.;
      this->in[i] = a;
      arma::vec b(std::vector<double>(in_[i].begin() + 1, in_[i].end()));
      b /= 255.;
      this->out[i] = b;
    }
  }
};

labeled_data convert(rapidcsv::Document doc) {
  labeled_data out;
  std::vector<std::vector<int>> acc;
  for (int i = 0; i < doc.GetColumnCount(); i++) {
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
  activations =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoid);
  activationsD =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoidD);

  // double xor_loss = findEP(test.first, test.second);
  auto a = get_mnist();
  labeled_data train_data = convert(a.first);
  labeled_data test_data = convert(a.second);
  std::cout << "MNIST: " << a.first.GetRowCount() << std::endl;
  // double xor_loss2 = findGA(test.first, test.second);
  //  std::cout << xor_loss << std::endl;
  /*
  // genetischer Algorithmus
  auto tmp2 = get_generation_EP()[0];
  FFNetwork network(NN{tmp2.weights, tmp2.biases});
  double loss = std::numeric_limits<double>::max();
  int it = 0;
  for (int i = 0; i < 200; i++) {
    // while (loss > 0.001) {
    loss = network.loss(network.forward(test.first), test.second);
    // if (it % 20 == 0)
    std::cout << "Loss: "
              << network.loss(network.forward(test.first), test.second)
              << std::endl;
    network.backward(test.first, test.second);
    it++;
  }
  for (int i = 0; i < test.first.size(); i++) {
    std::cout << test.first[i] << ": " << network.forward(test.first)[i]
              << std::endl;
  }
  */
}
