#include <armadillo>
#include <functional>
#include <utility>
#include <vector>

#define GEN_SIZE 10
#define IN 2
#define OUT 1

typedef std::vector<arma::Mat<double>> tensor;

class Chromosome {
public:
  std::vector<int> size;
  tensor weights;
  arma::mat biases;
};

std::vector<std::function<Chromosome(Chromosome)>> mutations;

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

std::pair<tensor, arma::mat> seperate(tensor a) {
  tensor weights;
  arma::mat biases;
  for (int i = 0; i < a.size(); i++) {
    arma::vec tmp;
    tmp = a[i]
  }
}

Chromosome to_chromosome(tensor arr) {
  return Chromosome { extract_size(arr), }
}
