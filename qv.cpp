#include <armadillo>
#include <qenv/include/Engine.hh>
#include <vector>

struct evolution_parameter{};
struct mutation_parameters{};
struct evolution_settings{};

class QV{
public:
  Engine engine;
  mat parameters;
  std::vector<evolution_parameter> ev_parameters;
  int max_it;
  double learning_rate;

  void mutate(mutation_parameters  params){}

  
  
  std::vector<double> EV(mat in, vec expected, evolution_settings ev){}

  std::vector<double> gradient_descent(mat in, vec expected){}
};
