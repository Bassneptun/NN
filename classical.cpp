#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rapidcsv.h"

#define GEN_SIZE 30
#define IN 2
#define OUT 1

#define CHANCE 0.90
#define DOUBLE_MUT 1

#define MIN 2
#define MAX 4
#define IN_LENGTH 2

#define MAX_ITERATIONS 10000

#define LEARNING_RATE 0.24

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
};

class Chromosome2 {
public:
  tensor weights;
  std::vector<arma::vec> biases;
};

std::vector<Chromosome> gen(std::vector<int> sizes, int size) {
  for (auto a : sizes) {
  }
  std::vector<Chromosome> out;
  for (int i = 0; i < size; i++) {
    tensor weights;
    std::vector<arma::vec> biases;
    for (int j = 1; j < sizes.size(); j++) {
      weights.push_back(arma::mat(sizes[j], sizes[j - 1], arma::fill::randu));
      biases.push_back(arma::vec(sizes[j], arma::fill::randu));
    }
    out.push_back(Chromosome{weights, biases});
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

Chromosome mv(tensor weights, std::vector<arma::vec> biases) {
  return Chromosome{std::move(weights), std::move(biases)};
}

bool r_choice() { return ((double)std::rand() / (double)RAND_MAX) > .5; }
bool r_choice(double chance) {
  return ((double)std::rand() / (double)RAND_MAX) < chance;
}

template <typename T> T r_choice(std::vector<T> a) {
  return a[((double)std::rand() / (double)RAND_MAX) * a.size()];
}

// Option 1
Chromosome crossover(Chromosome &a, Chromosome &b) {
  return mv(r_choice() ? a.weights : b.weights,
            r_choice() ? a.biases : b.biases);
}

// Option 2
Chromosome crossover2(Chromosome &a, Chromosome &b) {
  Chromosome out;
  for (int i = 0; i < a.biases.size(); i++) {
    bool decision = r_choice();
    out.biases.push_back(std::move(decision ? a.biases[i] : b.biases[i]));
    out.weights.push_back(std::move(decision ? a.weights[i] : b.weights[i]));
  }
  return out;
}

void random_change_weighted(Chromosome &ch) {
  if (r_choice()) {
    int layer =
        std::floor(((double)std::rand() / (double)RAND_MAX) * (IN_LENGTH + 1));
    std::pair<int, int> pos = std::make_pair(
        ((double)std::rand() / (double)RAND_MAX) * ch.weights[layer].n_rows,
        ((double)std::rand() / (double)RAND_MAX) * ch.weights[layer].n_cols);
    int n = floor(((double)std::rand() / (double)RAND_MAX) *
                  65); // bitte sei nicht 65...
    ch.weights[layer](pos.first, pos.second) =
        std::bit_cast<double>(std::bit_cast<unsigned long long>(
                                  ch.weights[layer](pos.first, pos.second)) ^
                              (1ULL << n));
  } else {
    int layer =
        std::floor(((double)std::rand() / (double)RAND_MAX) * (IN_LENGTH + 1));
    int pos =
        ((double)std::rand() / (double)RAND_MAX) * ch.biases[layer].n_elem;
    int n = floor(((double)std::rand() / (double)RAND_MAX) *
                  64); // bitte sei nicht 65...
    ch.biases[layer](pos) = std::bit_cast<double>(
        std::bit_cast<unsigned long long>(ch.biases[layer](pos)) ^ (1ULL << n));
  }
}

void gaussian(Chromosome &ch) {
  if (r_choice()) {
    int layer =
        std::floor(((double)std::rand() / (double)RAND_MAX) * (IN_LENGTH + 1));
    std::pair<int, int> pos = std::make_pair(
        ((double)std::rand() / (double)RAND_MAX) * ch.weights[layer].n_rows,
        ((double)std::rand() / (double)RAND_MAX) * ch.weights[layer].n_cols);
    int n = floor(((double)std::rand() / (double)RAND_MAX) *
                  65); // bitte sei nicht 65...
    ch.weights[layer](pos.first, pos.second) += arma::randn<double>();
  } else {
    int layer =
        std::floor(((double)std::rand() / (double)RAND_MAX) * (IN_LENGTH + 1));
    int pos =
        ((double)std::rand() / (double)RAND_MAX) * ch.biases[layer].n_elem;
    int n = floor(((double)std::rand() / (double)RAND_MAX) *
                  64); // bitte sei nicht 65...
    ch.biases[layer](pos) += arma::randn<double>();
  }
}

std::vector<std::function<void(Chromosome &)>> mutations = {
    random_change_weighted};

void mutate(Chromosome &ch, double chance, bool double_mut) {
  do {
    if (r_choice(chance)) {
      r_choice(mutations)(ch);
    } else
      return;
  } while (double_mut);
}

void next_gen(std::vector<Chromosome> &a) {
  for (int i = 0; i < a.size(); i++) {
    mutate(a[i], CHANCE, DOUBLE_MUT);
  }
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
    out.push_back(std::round(
        ((double)std::rand() / (double)RAND_MAX) * (MAX - MIN) + MIN));
  }
  return out;
}

std::vector<Chromosome>
get_generation(std::optional<std::vector<Chromosome>> generation = std::nullopt,
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

batch forward(NN network, batch tbatch,
              std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(network).forward(tbatch);
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
  generation tmp = get_generation();
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

    tmp = get_generation(std::make_optional(tmp), std::make_optional(indices));
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

double findGA(batch tbatch, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  std::pair<int, int> indices;
  std::pair<double, double> best_losses = {std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
  generation tmp = get_generation();
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

    tmp = get_generation(std::make_optional(tmp), std::make_optional(indices));
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

double findEP(batch tbatch, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  std::pair<int, int> indices;
  std::pair<double, double> best_losses = {std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
  generation tmp = get_generation();
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

    tmp = get_generation(std::make_optional(tmp), std::make_optional(indices));
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



void sigmoid(arma::vec &in) { in = 1 / (1 + arma::exp(-in)); }

void sigmoidD(arma::vec &in) {
  in = arma::exp(-in) / (arma::pow(1 + arma::exp(-in), 2));
}

void relu(arma::vec &in) { in = (in + arma::abs(in)) / 2; }
void reluD(arma::vec &in) {
  in = in.transform([](double a) { return static_cast<double>(a >= 0); });
}

std::pair<rapidcsv::Document, rapidcsv::Document> // (train, test)
get_mnist(std::string pathA = "~/NN/mnist_train.csv",
          std::string pathB = "~/NN/mnist_test.csv") {
  rapidcsv::Document mnist_train(pathA);
  rapidcsv::Document mnist_test(pathB);
  return std::make_pair(mnist_train, mnist_test);
}



int main() {
  std::srand(std::time(NULL));
  auto test = get_xor();
  activations =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoid);
  activationsD =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoidD);
  //
  // double xor_loss = find(test.first, test.second, activations);
  // genetischer Algorithmus
  auto tmp2 = get_generation()[0];
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
}
