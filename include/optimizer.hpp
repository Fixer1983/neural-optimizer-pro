#pragma once
#include <vector>

class Optimizer {
public:
    virtual void step(std::vector<double>& weights, const std::vector<double>& grads) = 0;
};
