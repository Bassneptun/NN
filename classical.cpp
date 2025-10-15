#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <ios>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#define GEN_SIZE 10
#define IN 2
#define OUT 1

#define CHANCE 0.33
#define DOUBLE_MUT 1

#define MIN 2
#define MAX 4
#define IN_LENGTH 2

#define MAX_ITERATIONS 1000

typedef std::vector<arma::Mat<double>> tensor;
using batch = std::vector<arma::vec>;

double mse(batch a, batch b) {
  double out;
  std::cout << "1" << std::endl;
  for (int i = 0; i < a.size(); i++) {
    a[i].print();
    b[i].print();
    out += arma::accu(arma::pow(static_cast<arma::vec>(a[i] - b[i]), 2)) /
           a.size();
  }
  return out;
}

struct NN {
  tensor weights;
  std::vector<arma::vec> biases;
};

class Chromosome {
public:
  std::vector<int> size;
  tensor weights;
  std::vector<arma::vec> biases;
};

std::vector<std::function<void(Chromosome &)>> mutations;

std::vector<Chromosome> gen(std::vector<int> sizes, int size) {
  for (auto a : sizes) {
    std::cout << a << ", ";
  }
  std::cout << size << std::endl;
  std::vector<Chromosome> out;
  for (int i = 0; i < size; i++) {
    tensor weights;
    std::vector<arma::vec> biases;
    for (int j = 1; j < sizes.size(); j++) {
      weights.push_back(arma::mat(sizes[j], sizes[j - 1], arma::fill::randu));
      biases.push_back(arma::vec(sizes[j], arma::fill::randu));
    }
    out.push_back(Chromosome{sizes, weights, biases});
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

Chromosome mv(std::vector<int> size, tensor weights,
              std::vector<arma::vec> biases) {
  return Chromosome{std::move(size), std::move(weights), std::move(biases)};
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
  return mv(r_choice() ? a.size : b.size, r_choice() ? a.weights : b.weights,
            r_choice() ? a.biases : b.biases);
}

void mutate(Chromosome &ch, double chance, bool double_mut) {
  do {
    if (r_choice(chance))
      r_choice(mutations)(ch);
    else
      return;
  } while (double_mut);
}

void random_change_weighted(Chromosome &ch) {
  if (r_choice()) {
    int layer = std::floor(((double)std::rand() / (double)RAND_MAX) *
                           ch.weights.size());
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
    int layer = std::floor(((double)std::rand() / (double)RAND_MAX) *
                           ch.weights.size());
    int pos =
        ((double)std::rand() / (double)RAND_MAX) * ch.biases[layer].n_elem;
    int n = floor(((double)std::rand() / (double)RAND_MAX) *
                  65); // bitte sei nicht 65...
    ch.weights[layer](pos) = std::bit_cast<double>(
        std::bit_cast<unsigned long long>(ch.weights[layer](pos)) ^
        (1ULL << n));
  }
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
  std::cout << std::boolalpha << (bool)generation << " " << (bool)best
            << std::endl;
  std::cout << std::boolalpha << (!(bool)generation && !(bool)best) << " "
            << std::endl;

  if (!generation && !best) {
    std::cout << "1" << std::endl;
    gen_ = gen(concat({{IN}, rand_int(IN_LENGTH), {OUT}}), GEN_SIZE);
  } else {
    std::vector<Chromosome> c(
        IN_LENGTH,
        Chromosome{std::vector<int>(), tensor(), std::vector<arma::vec>()});
    std::for_each(c.begin(), c.end(), [&](Chromosome &d) {
      return crossover((*generation)[best->first], (*generation)[best->second]);
    });
    gen_ = c;
  }
  return gen_;
}

class FFNetwork {
public:
  std::vector<std::function<void(arma::vec &)>> activations;
  tensor weights;
  std::vector<arma::vec> biases;

  FFNetwork(std::vector<std::function<void(arma::vec &)>> activations,
            NN Network) {
    this->activations = activations;
    this->weights = Network.weights;
    this->biases = Network.biases;
  }

  std::vector<arma::vec> forward(batch in) {
    std::cout << "arrived" << std::endl;
    batch output;
    for (int i = 0; i < in.size(); i++) {
      std::cout << 4 << std::endl;
      auto tmp = in[i];
      for (int j = 0; j < this->biases.size(); j++) {
        std::cout << 5 << std::endl;
        tmp = weights[j] * tmp;
        std::cout << tmp.n_elem << " " << biases[j].n_elem << std::endl;
        tmp = tmp + biases[j];
        this->activations[j](tmp);
      }
      output.push_back(tmp);
    }
    return output;
  }

  double loss(batch a, batch b) { return mse(a, b); }
};

double loss(NN network, batch output, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(activations, network).loss(output, expected);
}

batch forward(NN network, batch tbatch,
              std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(activations, network).forward(tbatch);
}

std::pair<std::vector<arma::vec>, std::vector<arma::vec>> get_xor() {
  std::vector<arma::vec> inputs = {{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<arma::vec> outs = {{0}, {1}, {0}, {1}};
  return std::make_pair(inputs, outs);
}

using generation = std::vector<Chromosome>;

double find(batch tbatch, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  std::pair<int, int> indices;
  std::pair<double, double> best_losses = {std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
  std::cout << "1" << std::endl;
  generation tmp = get_generation();
  std::cout << "1" << std::endl;
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    for (int j = 0; j < GEN_SIZE; i++) {
      std::cout << "2" << std::endl;
      auto output =
          forward(NN{tmp[i].weights, tmp[i].biases}, tbatch, activations);
      std::cout << "3" << std::endl;
      auto tmp2 = loss(NN{tmp[i].weights, tmp[i].biases}, output, expected,
                       activations);
      std::cout << tmp2 << std::endl;
      if (tmp2 < best_losses.second) {
        if (tmp2 < best_losses.first) {
          indices.first = i;
          best_losses.second = best_losses.first;
          best_losses.first = tmp2;
        } else {
          indices.second = i;
          best_losses.second = tmp2;
        }
      }
    }

    if (i % 10 == 0)
      std::cout << best_losses.first << " " << best_losses.second << std::endl;

    tmp = get_generation(std::make_optional(tmp), std::make_optional(indices));
    best_losses = {std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  }
  return best_losses.first;
}

void sigmoid(arma::vec &in) { in = 1 / (1 + arma::exp(-in)); }

int main() {
  std::srand(std::time(NULL));
  auto test = get_xor();
  auto activations =
      std::vector<std::function<void(arma::vec &)>>(2 + IN_LENGTH, sigmoid);
  double xor_loss = find(test.first, test.second, activations);
}
