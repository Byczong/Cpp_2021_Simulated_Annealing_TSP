#ifndef SIMULATED_ANNEALING_ANNEALING_H
#define SIMULATED_ANNEALING_ANNEALING_H

/**
 * @file annealing.h
 *
 * @brief Header allowing using class SimAnnealTSP to find an approximated solution
 * to Traveling Salesman Problem using simulated annealing method.
 *
 * @author Piotr Zawislan
 * BookID: 400427
 * Email: piotrzwsln@student.agh.edu.pl
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>



using namespace std;



class RandomDoubleGenerator {
private:
    mt19937 gen;
    uniform_real_distribution<double> uniformDistribution;
    normal_distribution<double> normalDistribution;

public:
    RandomDoubleGenerator(double from, double to, double mean, double sd);

    double getRandomUniform();

    double getRandomNormal();
};


class RandomIntGenerator {
private:
    mt19937 gen;
    uniform_int_distribution<int> uniformIntDistribution;
public:
    RandomIntGenerator(int from, int to);

    int getRandomUniform();
};


class Point {
private:
    double x;
    double y;
    int label;
    static int count;

public:
    Point(double x, double y): x{x}, y{y}, label{++count} {}

    [[nodiscard]] double getX() const { return x; }

    [[nodiscard]] double getY() const { return y; }

    [[nodiscard]] int getLabel() const { return label; }

    [[nodiscard]] double getDistanceTo(const Point& other) const;

    static double getDistanceBetween(const Point& pA, const Point& pB);\

    [[nodiscard]] string toString() const { return '(' + to_string(x) + ", "  + to_string(y) + ')'; }
};


class PointGraph {
private:
    vector<Point> points;
    size_t _size;
    RandomIntGenerator randIndexGen;
public:
    PointGraph():

            points{vector<Point>()},
            _size{0},
            randIndexGen{RandomIntGenerator(0, 0)}
    {}

    explicit PointGraph(const vector<Point>& vec):

            points{vec},
            _size{vec.size()},
            randIndexGen{RandomIntGenerator(0, (int) vec.size() - 1)}
    {}

    PointGraph(const PointGraph& other):

            _size{other._size},
            randIndexGen{RandomIntGenerator(0, (int) other.size() - 1)},
            points{other.points}  // Deep copy as points vector consists of Point objects, not Point* pointers.
    {}

    PointGraph(PointGraph&& other) noexcept:
            _size{other._size},
            randIndexGen{RandomIntGenerator(0, (int) other.size() - 1)},
            points{move(other.points)} { other._size = 0; }

    ~PointGraph() = default;

    PointGraph& operator=(const PointGraph& other);

    PointGraph& operator=(PointGraph&& other) noexcept;

    void initGraphUniform(RandomDoubleGenerator& randGenX, RandomDoubleGenerator& randGenY, size_t size);

    void initGraphNormal(RandomDoubleGenerator& randGenX, RandomDoubleGenerator& randGenY, size_t size);

    [[nodiscard]] size_t size() const {return _size; }

    double getTotalDistance();

    void consecutiveSwap();

    void arbitrarySwap();

    friend ostream& operator<<(ostream& out, const PointGraph& graph) {
        string pointStr{};
        for(auto p: graph.points) {
            pointStr += p.toString();
            pointStr += '\n';
        }
        return out << "---PointGraph---\nsize: " << graph._size << "\npoints:\n" << pointStr;
    }

    const vector<Point> &getPoints() const { return points; }
};



enum class Temperature { Linear, PowerSlow, PowerFast };

enum class NextState { Consecutive, Arbitrary, Mixed };



class SimulatedAnnealingTSP {
private:

    // Constants / initial parameters
    constexpr static double initialT = 1000.;  // Initial temperature
    constexpr static int mixedAttemptsNumber = 10;  // Number of attempts to arbitrarily find next state in mixed choice
    const int kStop;  // Desired number of iterations
    const shared_ptr<PointGraph> initialState;  // Initial state (input graph)
    const Temperature temperatureChoice;  // Defines which method to use when calculating temperature
    const NextState nextStateChoice;  // Defines which method to use when finding next state
    const int maxHigherEnergyIterations;  // Defines the number of higher energy iterations to reset to best state
    const int maxHillDescendingIterations;  // Defines the number of iterations to be made after reaching T = 0
    RandomDoubleGenerator randDoubleGen;  // Used to get random double from 0. to 1.

    // Variables describing current situation
    int k;     // Current iteration
    double T;  // Current temperature
    double E;  // Current energy
    shared_ptr<PointGraph> currentState;  // Current state (graph)

    // History variables
    vector<double> energyHistory;  // Vector containing history of energy change
    vector<double> temperatureHistory;  // Vector containing history of temperature change
    double bestE;  // Lowest energy so far
    shared_ptr<PointGraph> bestState;  // State which had lowest energy so far
    int iterationsSinceBest;  // Number of iterations since being in best state

    double getTemperature();

    [[nodiscard]] double getTemperatureLinear() const;

    [[nodiscard]] double getTemperaturePowerSlow() const;

    [[nodiscard]] double getTemperaturePowerFast() const;

    [[nodiscard]] shared_ptr<PointGraph> getNextState();

    void attemptAccepting(shared_ptr<PointGraph>& candidate);

    [[nodiscard]] double acceptanceProbability(double candidateE) const;

    void updateBest();

    double getRandomProbability();

    static double getEnergy(const shared_ptr<PointGraph>& state) { return state->getTotalDistance(); }

    void annealAll();

public:
    SimulatedAnnealingTSP(const shared_ptr<PointGraph>& pointGraph,
                          int numberOfIterations,
                          int maxHigherEnergyIterations,
                          int maxHillDescendingIterations,
                          Temperature temperatureChoice=Temperature::Linear,
                          NextState nextStateChoice=NextState::Consecutive
    ):

            initialState{make_shared<PointGraph>(*pointGraph)},
            kStop{numberOfIterations},
            maxHigherEnergyIterations{maxHigherEnergyIterations},
            maxHillDescendingIterations{maxHillDescendingIterations},
            temperatureChoice{temperatureChoice},
            nextStateChoice{nextStateChoice},
            randDoubleGen{RandomDoubleGenerator(0., 1., 0.5, 0.)},

            k{0},
            T{SimulatedAnnealingTSP::initialT},
            E{getEnergy(pointGraph)},
            currentState{make_shared<PointGraph>(*pointGraph)},

            energyHistory{vector<double>(1, getEnergy(pointGraph))},
            temperatureHistory{vector<double>(1, SimulatedAnnealingTSP::initialT)},
            bestE{getEnergy(pointGraph)},
            bestState{make_shared<PointGraph>(*pointGraph)},
            iterationsSinceBest{0}
    {
        // annealAll();
    }

    bool makeStep();

    [[nodiscard]] const vector<double> &getEnergyHistory() const;

    [[nodiscard]] const vector<double> &getTemperatureHistory() const;

    [[nodiscard]] double getE() const;

    [[nodiscard]] const shared_ptr<PointGraph> &getCurrentState() const;

    [[nodiscard]] double getBestE() const;

    [[nodiscard]] const shared_ptr<PointGraph> &getBestState() const;

    int getKStop() const;

    int getMaxHillDescendingIterations() const;
};

#endif //SIMULATED_ANNEALING_ANNEALING_H
