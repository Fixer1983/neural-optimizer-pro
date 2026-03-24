
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

class NeuralOptimizer {
public:
    NeuralOptimizer(double learning_rate = 0.001) : lr(learning_rate) {}

    void optimize_weights(std::vector<double>& weights, const std::vector<double>& gradients) {
        if (weights.size() != gradients.size()) {
            throw std::invalid_argument("Size mismatch between weights and gradients");
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * gradients[i];
        }
    }

    void apply_momentum(std::vector<double>& weights, const std::vector<double>& gradients, std::vector<double>& velocity, double beta = 0.9) {
        for (size_t i = 0; i < weights.size(); ++i) {
            velocity[i] = beta * velocity[i] + (1 - beta) * gradients[i];
            weights[i] -= lr * velocity[i];
        }
    }

private:
    double lr;
};

int main() {
    NeuralOptimizer optimizer(0.01);
    std::vector<double> weights = {0.5, -0.1, 0.8};
    std::vector<double> gradients = {0.01, 0.05, -0.02};
    std::vector<double> velocity = {0.0, 0.0, 0.0};

    std::cout << "Optimizing weights..." << std::endl;
    optimizer.apply_momentum(weights, gradients, velocity);

    for (double w : weights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;

    return 0;
}
