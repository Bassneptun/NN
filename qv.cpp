#include <algorithm>
#include <armadillo>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using arma::vec, arma::mat, arma::cx_vec, arma::cx_cube, arma::cx_mat;
using std::string;
using namespace std::complex_literals;

double eps = 1e-10;

arma::cx_mat gellmann(std::size_t n, std::size_t d) {
  arma::cx_mat lambda = arma::zeros<arma::cx_mat>(d, d);

  std::size_t count = 0;

  for (std::size_t j = 0; j < d; ++j)
    for (std::size_t k = j + 1; k < d; ++k) {
      if (count == n) {
        lambda(j, k) = 1.0;
        lambda(k, j) = 1.0;
        return lambda;
      }
      ++count;
    }

  for (std::size_t j = 0; j < d; ++j)
    for (std::size_t k = j + 1; k < d; ++k) {
      if (count == n) {
        lambda(j, k) = -1.0i;
        lambda(k, j) = 1.0i;
        return lambda;
      }
      ++count;
    }

  for (std::size_t l = 1; l < d; ++l) {
    if (count == n) {
      double norm = std::sqrt(2.0 / (l * (l + 1)));
      for (std::size_t m = 0; m < l; ++m)
        lambda(m, m) = norm;
      lambda(l, l) = -l * norm;
      return lambda;
    }
    ++count;
  }

  throw std::runtime_error("Gell-Mann index out of range");
}

cx_cube gell_mann_generator(int d) {
  int gens = std::pow(d, 2) - 1;
  cx_cube out(d, d, gens);
  for (int i = 0; i < gens; i++)
    out.slice(i) = gellmann(i, d);
  return out;
}

cx_cube pauli_generator() {
  cx_mat S0 = arma::eye<cx_mat>(2, 2);
  cx_mat SX = {{0i, 1}, {1, 0}};
  cx_mat SY = {{0, -1i}, {1i, 0}};
  cx_mat SZ = {{1, 0i}, {0i, -1}};
  cx_cube out(S0.n_rows, S0.n_cols, 4);
  out.slice(0) = S0;
  out.slice(1) = SX;
  out.slice(2) = SY;
  out.slice(3) = SZ;
  return out;
}

class QV {
public:
  int max_it, m, n, components;
  double learning_rate;
  // cx_cube gell_manns;
  cx_cube paulis;

  QV(int max_it, int m, int components, double learning_rate = 0.1)
      : max_it(max_it), m(m), n(std::pow(2, m)), components(components),
        learning_rate(learning_rate) {
    paulis = pauli_generator();
  }

  double to_z(cx_vec &in) { return std::norm(in(1)); }

  cx_mat U(int i, double y) {
    // std::cout << "y: " << y << std::endl;
    if (i == 0) {
      /*
      cx_mat I = arma::kron(paulis.slice(3), paulis.slice(3));
      return arma::expmat(-1i * std::fmod(y, 360) * .5 * I);
      */

      return {{1, 0, 0 , 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}};

    } else if (i < 4) {
      // qb1
      cx_mat U1 = arma::expmat(-1i * std::fmod(y, 360) * .5 * paulis.slice(i));
      return arma::kron(U1, paulis.slice(0));
    } else {
      //qb2
      cx_mat U1 = arma::expmat(-1i * std::fmod(y, 360) * .5 *
                               paulis.slice((i % 4) + 1));
      return arma::kron(paulis.slice(0), U1);
    }
  }

  double theta(vec params, vec inputs, double bias, double ps = 0) {
    return arma::dot(params, inputs) + bias + ps;
  }

  cx_vec Us(mat &parameters, vec &biases, vec input, int ps_i = -1,
            double ps = 0) {
    cx_vec O = arma::zeros<cx_vec>(this->n);
    O(0) = 1;
    for (size_t i = 0; i < 7; i++) {
      if (ps_i == i)
        O = U(i, theta(parameters.col(i), input, biases(i), ps)) * O;
      else
        O = U(i, theta(parameters.col(i), input, biases(i))) * O;
    }
    return O;
  }

  vec f(mat &parameters, vec &biases, vec inputs) {
    auto tmp = Us(parameters, biases, inputs);
    vec probs = arma::abs(arma::pow(tmp, 2));
    vec means(this->m);

    for (int i = 0; i < this->m; i++) {
      long mask = 1 << i;
      double mean = 0;
      int used = 0;
      for (int j = 0; j < 7; j++) {
        if ((j & mask) != 0)
          mean += probs(j), used++;
      }
      means(i) = mean / used;
    }
    return means;
  }

  double f1(mat &parameters, vec &biases, vec inputs) {
    auto tmp = Us(parameters, biases, inputs);
    return to_z(tmp);
  }

  vec f2(mat &parameters, vec &biases, vec inputs, int ps_i = -1,
         double ps = 0) {
    auto tmp = Us(parameters, biases, inputs, ps_i, ps);
    return arma::abs(arma::pow(tmp, 2));
  }

  double cel(vec out, vec expected) {
    vec tmp = arma::log(out + arma::ones(out.n_elem) * eps);
    double tmp2 = -1 * sum(expected % tmp);
    return tmp2;
  }

  vec cel_d(vec out, vec expected) { return (-1) * expected / (out + eps); }

  vec class_to_vec(int class_label) {
    vec out = arma::zeros<vec>(this->n);
    out(class_label) = 1;
    return out;
  }

  vec to_real(cx_vec c1) { return arma::abs(arma::pow(c1, 2)); }

  double loss(mat &parameters, vec biases, mat &in, vec &expected) {
    double loss = 0;
    for (size_t i = 0; i < in.n_cols; i++) {
      vec output = f2(parameters, biases, in.col(i));
      double loss_val = cel(output, class_to_vec(expected(i)));
      loss += loss_val;
    }
    return loss / in.n_cols;
  }

  /*

  mat lossD(mat in, vec &expected) {
    mat loss(this->n, in.n_cols);
    for (size_t i = 0; i < in.n_cols; i++) {
      vec output = f2(in.col(i));
      vec loss_val = cel_d(output, class_to_vec(expected(i)));
      loss.col(i) = loss_val;
    }
    return loss;
  }

  mat partials(arma::subview_col<double> in) {
    mat out(this->n, this->parameters.n_cols);
    for (size_t i = 0; i < out.n_cols; i++) {
      auto tmp1 = f2(in, i, std::numbers::pi/2);
      auto tmp2 = f2(in, i, -std::numbers::pi/2);
      out.col(i) = 1./2*(tmp1 - tmp2);
    }
    return out;
  }

  mat to_mat(arma::subview_col<double> inputs){
    mat out(inputs.n_elem, std::pow(this->n, 2)-1);
    for(int i = 0; i < inputs.n_elem; i++){
      out.col(i) = inputs;
    }
    return out;
  }

  void gradient_descent(mat in, vec expected, double stop_loss = -1000) {
    double last_loss = 100;
    for (int epoch = 0; (epoch < this->max_it) && last_loss > stop_loss;
         epoch++) {
      mat gradients_weights = arma::zeros(size(this->parameters));
      vec gradients_bias = arma::zeros(this->biases.n_elem);
      mat lossD = this->lossD(in, expected);
      for (size_t i = 0; i < in.n_cols; i++) {
        auto tmp = partials(in.col(i));
        vec partials_ = (lossD.col(i).t()*tmp).t();
        gradients_bias += partials_;
        for(int j = 0; j < gradients_weights.n_cols; j++){
          gradients_weights.col(j) += partials_(j) * in.col(i);
        }
      }
      gradients_weights = (1. / in.n_cols) * gradients_weights;
      gradients_bias = (1. / in.n_cols) * gradients_bias;
      this->parameters -= gradients_weights * this->learning_rate;
      this->biases -= gradients_bias * this->learning_rate;
      last_loss = this->loss(in, expected);
      std::cout << "loss at epoch " << epoch << ": " << last_loss << std::endl;
    }
  }
  */
};

std::random_device rd;
std::minstd_rand generator(rd());
std::uniform_real_distribution<double> s;

bool choice(double chance = 0.5) { return chance > s(generator); }

template <typename T> T choice(std::vector<T> container) {
  return container[randint(container.size())];
}

template <typename T> T &choice(std::pair<T, T> &container) {
  return choice() ? container.first : container.second;
}

class EP {
public:
  arma::cube members_;
  mat biases;
  QV qv;
  vec spread;

  EP(int max_it, int m, int components, int population,
     double learning_rate = 0.1)
      : members_(components, 7, population, arma::fill::randu),
        biases(7, population, arma::fill::randu),
        qv(max_it, m, components, learning_rate),
        spread(population, arma::fill::randu) {}

  std::tuple<arma::cube, arma::mat, vec> mutate() {
    arma::cube out = members_;
    mat bias = this->biases;
    vec spread_ = spread;
    for (size_t i = 0; i < out.n_slices; i++) {
      std::cauchy_distribution<double> a(-spread[i], spread[i]);
      for (size_t j = 0; j < out.n_rows; j++) {
        for (size_t k = 0; k < out.n_cols; k++) {
          out(j, k, i) += a(generator);
          bias(k, i) += a(generator);
        }
      }
      if (choice()) {
        spread_(i) += a(generator);
      }
    }
    return std::make_tuple(out, bias, spread_);
  }

  vec loss(mat &in, vec &expected, arma::cube &params, mat &bias) {
    vec losses(this->members_.n_slices);
    for (size_t i = 0; i < this->members_.n_slices; i++) {
      losses(i) = this->qv.loss(params.slice(i), bias.col(i), in, expected);
    }
    return losses;
  }

  int EP_(mat in, vec expected, double stop = 0.00001) {
    double a = 1000;
    int used = 0;
    for (int epoch = 0; (epoch < this->qv.max_it) && a > stop;
         ++epoch, used++) {
      auto mutated = mutate();
      vec losses_new =
          loss(in, expected, std::get<0>(mutated), std::get<1>(mutated));
      vec losses_old = loss(in, expected, this->members_, this->biases);

      for (size_t i = 0; i < losses_new.n_elem; i++) {
        if (losses_new[i] < losses_old[i]) {
          this->members_.slice(i) = std::get<0>(mutated).slice(i);
          this->biases.col(i) = std::get<1>(mutated).col(i);
          this->spread(i) = std::get<2>(mutated)(i);
        }
      }
      auto m_l = *std::min_element(losses_old.begin(), losses_old.end());
      a = m_l;
      std::cout << "best loss: " << m_l << std::endl;
    }
    // return out;
    return used;
  }
};

vec parse_py_vec(string line) {
  std::vector<double> out;
  string current;
  for (size_t i = 0; i < line.size(); i++) {
    char c = line[i];
    if ((c == ' ' || c == ']') && current != "")
      out.push_back(std::stod(current)), current.clear();
    else if (c == '[' || c == ' ' || c == ']')
      continue;
    else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' ||
             c == '+')
      current.push_back(c);
    else
      throw std::runtime_error("invalid data");
  }
  return out;
}

std::pair<mat, vec> data(std::string path = "pca_data10", int components = 10) {
  std::ifstream file(path);
  std::vector<vec> in;
  std::vector<double> out;
  while (file.good()) {
    string classifier, data, waste;
    std::getline(file, classifier);
    std::getline(file, data);
    std::getline(file, waste);
    if (classifier == "")
      break;
    out.push_back(std::stoi(classifier));
    in.push_back(parse_py_vec(data));
  }

  mat in2(components, out.size());

  for (size_t j = 0; j < out.size(); j++) {
    in2.col(j) = in[j];
  }

  return std::make_pair(in2, vec(out));
}

int main() {
  int components = 10, population = 50, max_it = 1000, classes = 2;
  auto pca_data = data("pca_data" + std::to_string(components), components);

  EP alg(max_it, classes, components, population);
  alg.EP_(pca_data.first, pca_data.second);
}
