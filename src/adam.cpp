#include "optimizer.hpp"
#include <cmath>

class Adam : public Optimizer {
    double lr = 0.001;
public:
    void step(std::vector<double>& w, const std::vector<double>& g) override {
        for(size_t i=0; i<w.size(); ++i) w[i] -= lr * g[i];
    }
};
