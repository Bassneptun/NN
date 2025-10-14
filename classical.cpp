#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <functional>
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

struct NN {
  tensor weights;
  std::vector<arma::vec> biases;
};

NN seperate(tensor a) {
  tensor weights_;
  std::vector<arma::vec> biases;
  for (int i = 0; i < a.size(); i++) {
    arma::vec tmp;
    tmp = a[i].col(a[i].n_cols - 1);
    biases.push_back(tmp);
    a[i].shed_col(a[i].n_cols - 1);
    weights_.push_back(a[i]);
  }
  return NN{weights_, biases};
}

class Chromosome {
public:
  std::vector<int> size;
  tensor weights;
  std::vector<arma::vec> biases;
};

std::vector<std::function<void(Chromosome &)>> mutations;

std::vector<double> temperatures(int high, int low, int num) {
  std::vector<double> out;
  for (int i = high; i > low; i -= (high - low) / num) {
    out.push_back(i);
  }
  return out;
}

std::vector<Chromosome> gen(std::vector<int> sizes, int size) {
  std::vector<Chromosome> out;
  for (int i = 0; i < size; i++) {
    tensor tmp;
    for (int j = 1; j < sizes.size(); j++) {
      arma::mat new_mat =
          arma::mat(sizes[j - 1] + 1, sizes[i], arma::fill::randu);
      tmp.push_back(new_mat);
    }
    out.push_back(
        Chromosome{sizes, seperate(tmp).weights, seperate(tmp).biases});
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

Chromosome to_chromosome(tensor arr) {
  auto a = seperate(arr);
  return Chromosome{extract_size(arr), a.weights, a.biases};
}

Chromosome mv(std::vector<int> size, tensor weights,
              std::vector<arma::vec> biases) {
  return Chromosome{std::move(size), std::move(weights), std::move(biases)};
}

bool r_choice() { return std::rand() > .5; }

template <typename T> T r_choice(std::vector<T> a) {
  return a[rand() * a.size()];
}

// Option 1
Chromosome crossover(Chromosome &a, Chromosome &b) {
  return mv(r_choice() ? a.size : b.size, r_choice() ? a.weights : b.weights,
            r_choice() ? a.biases : b.biases);
}

void mutate(Chromosome &ch, double chance, bool double_mut) {
  while (1) {
    if (r_choice())
      r_choice(mutations)(ch);
    else
      return;
  }
}

void random_change_weighted(Chromosome &ch) {
  if (r_choice()) {
    int layer = std::floor(rand() * ch.weights.size());
    std::pair<int, int> pos = std::make_pair(rand() * ch.weights[layer].n_rows,
                                             rand() * ch.weights[layer].n_cols);
    int n = floor(rand() * 65); // bitte sei nicht 65...
    ch.weights[layer](pos.first, pos.second) =
        std::bit_cast<double>(std::bit_cast<unsigned long long>(
                                  ch.weights[layer](pos.first, pos.second)) ^
                              (1ULL << n));
  } else {
    int layer = std::floor(rand() * ch.weights.size());
    int pos = rand() * ch.biases[layer].n_elem;
    int n = floor(rand() * 65); // bitte sei nicht 65...
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

constexpr std::vector<int> rand_int(int size) {
  std::vector<int> out;
  for (int i = 0; i < size; i++) {
    out.push_back(std::round(rand() * (MAX - MIN) + MIN));
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
    batch output;
    for (int i = 0; i < in.size(); i++) {
      auto tmp = in[i];
      for (int j = 0; j < this->biases.size(); j++) {
        tmp = tmp * weights[j];
        tmp += biases[j];
        this->activations[j](tmp);
      }
      output.push_back(tmp);
    }
    return output;
  }

  double loss(batch a, batch b) { return mse(a, b); }
};

double loss(NN network, batch tbatch, batch expected,
            std::vector<std::function<void(arma::vec &)>> activations) {
  return FFNetwork(activations, network).loss(tbatch, expected);
}

std::pair<std::vector<arma::vec>, std::vector<arma::vec>> get_xor() {
  std::vector<arma::vec> inputs = {{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  std::vector<arma::vec> outs = {{0}, {1}, {0}, {1}};
  return std::make_pair(inputs, outs);
}

double find()

    int main() {}
