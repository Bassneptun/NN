#include <armadillo>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
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
#define IN_LENGTH 4

typedef std::vector<arma::Mat<double>> tensor;

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

std::vector<tensor> gen(std::vector<int> sizes, int size) {
  std::vector<tensor> out;
  for (int i = 0; i < size; i++) {
    tensor tmp;
    for (int j = 1; j < sizes.size(); j++) {
      arma::mat new_mat =
          arma::mat(sizes[j - 1] + 1, sizes[i], arma::fill::randu);
      tmp.push_back(new_mat);
    }
    out.push_back(tmp);
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

std::vector<int>& operator+(std::vector<int>& a, std::vector<int>& b){
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

std::vector<int> rand_int(int size){
  std::vector<int> out;
  for(int i = 0; i < size; i++){
    out.push_back(std::round(rand()*(MAX-MIN)+MIN));
  }
  return out;
}

std::vector<Chromosome>
get_generation(std::optional<std::vector<Chromosome>> generation = std::nullopt,
               std::optional<std::pair<int, int>> best = std::nullopt) {
  std::vector<Chromosome> gen_;
  if(!generation&&!best){
    gen_ = gen({IN} + {})
  }
}
