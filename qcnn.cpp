#include <armadillo>
#include <fstream>
#include <optional>
#include <qenv/include/Engine.hh>
#include <string>
#include <utility>
#include <vector>

std::pair<arma::vec, int> parse_line(std::string line) {
  std::string current;
  arma::vec out;
  int class_;
  for (unsigned int i = 0; i < line.size(); i++) {
    switch (line[i]) {
    case ' ':
      current += line[i];
      break;
    case ',':
      out += std::stod(current);
      current.clear();
      break;
    case '.':
      class_ = std::stoi(current);
      current.clear();
      break;
    default:
      break;
    }
  }
  return std::make_pair(out, class_);
}

std::pair<std::vector<arma::vec>, std::vector<int>>
load_data(std::string path) {
  // loads n-component pca image data
  std::ifstream file(path);
  std::vector<arma::vec> components;
  std::vector<int> classes;
  while (file.good()) {
    std::string line;
    std::getline(file, line);
    auto parsed = parse_line(line);
    classes.push_back(parsed.second);
    components.push_back(parsed.first);
  }
  return std::make_pair(components, classes);
}

class ImageClassifier {
private:
  std::string bytecode;
  Engine engine;
  std::pair<int, int> dims; // in(number of principal components), out(number of
                            // classes) dimensions. in < number of layers, out =
                            // log2(states of system) must be given
public:
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
};
