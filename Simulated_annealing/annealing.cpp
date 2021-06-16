/**
 * @file annealing.cpp
 *
 * @author Piotr Zawislan
 * BookID: 400427
 * Email: piotrzwsln@student.agh.edu.pl
 */

#include "annealing.h"


// RandomDoubleGenerator

RandomDoubleGenerator::RandomDoubleGenerator(double from, double to, double mean, double sd) {
    random_device rd;
    mt19937::result_type seed = rd() ^ (
            (mt19937::result_type) chrono::duration_cast<chrono::seconds>(
                    chrono::system_clock::now().time_since_epoch()).count()
            +
            (mt19937::result_type) chrono::duration_cast<chrono::microseconds>(
                    chrono::high_resolution_clock::now().time_since_epoch()).count()
    );

    mt19937 engine(seed);
    gen = engine;
    uniformDistribution = uniform_real_distribution(from, to);
    normalDistribution = normal_distribution(mean, sd);
}

double RandomDoubleGenerator::getRandomUniform() {
    return uniformDistribution(gen);
}

double RandomDoubleGenerator::getRandomNormal() {
    return normalDistribution(gen);
}


// RandomIntGenerator

RandomIntGenerator::RandomIntGenerator(int from, int to) {
    random_device rd;
    mt19937::result_type seed = rd() ^ (
            (mt19937::result_type) chrono::duration_cast<chrono::seconds>(
                    chrono::system_clock::now().time_since_epoch()).count()
            +
            (mt19937::result_type) chrono::duration_cast<chrono::microseconds>(
                    chrono::high_resolution_clock::now().time_since_epoch()).count()
    );

    mt19937 engine(seed);
    gen = engine;
    uniformIntDistribution = uniform_int_distribution(from, to);
}

int RandomIntGenerator::getRandomUniform() {
    return uniformIntDistribution(gen);
}


// Point

int Point::count = 0;

double Point::getDistanceTo(const Point &other) const {
    return sqrt((this->getX() - other.getX())*(this->getX() - other.getX()) +
                (this->getY() - other.getY())*(this->getY() - other.getY()));
}

double Point::getDistanceBetween(const Point &pA, const Point &pB) {
    return sqrt((pA.getX() - pB.getX())*(pA.getX() - pB.getX()) +
                (pA.getY() - pB.getY())*(pA.getY() - pB.getY()));
}


// PointGraph

void PointGraph::initGraphUniform(RandomDoubleGenerator& randGenX, RandomDoubleGenerator& randGenY, size_t size) {
    points.clear();
    _size = 0;
    for(int i = 0; i < size; i++) {
        points.emplace_back(randGenX.getRandomUniform(), randGenY.getRandomUniform());
        _size++;
    }
    randIndexGen = RandomIntGenerator(0, (int) _size - 1);
}

void PointGraph::initGraphNormal(RandomDoubleGenerator &randGenX, RandomDoubleGenerator &randGenY, size_t size) {
    points.clear();
    _size = 0;
    for(int i = 0; i < size; i++) {
        points.emplace_back(randGenX.getRandomNormal(), randGenY.getRandomNormal());
        _size++;
    }
    randIndexGen = RandomIntGenerator(0, (int) _size - 1);
}

double PointGraph::getTotalDistance() {
    if(_size == 0 || _size == 1)
        return 0.;
    else if(_size == 2)
        return points.front().getDistanceTo(points.back());

    double acc = 0.;
    auto prevP = points.back();
    for(auto p: points) {
        acc += p.getDistanceTo(prevP);
        prevP = p;
    }
    return acc;
}

void PointGraph::consecutiveSwap() {
    int idxA = randIndexGen.getRandomUniform();
    int idxB = idxA == _size - 1 ? 0 : idxA + 1;
    swap(points[idxA], points[idxB]);
}

void PointGraph::arbitrarySwap() {
    int idxA = randIndexGen.getRandomUniform();
    int idxB = randIndexGen.getRandomUniform();
    while(idxA == idxB)
        idxB = randIndexGen.getRandomUniform();

    swap(points[idxA], points[idxB]);
}

PointGraph &PointGraph::operator=(const PointGraph &other) {
    if(this == &other)
        return *this;
    _size = other._size;
    randIndexGen = RandomIntGenerator(0, (int) other.size() - 1);
    points = other.points;
    return *this;
}

PointGraph &PointGraph::operator=(PointGraph&& other) noexcept {
    if(this == &other)
        return *this;
    _size = other._size;
    randIndexGen = RandomIntGenerator(0, (int) other.size() - 1);
    points = move(other.points);
    other._size = 0;
    return *this;
}


// SimulatedAnnealingTSP

double SimulatedAnnealingTSP::getTemperature() {
    switch(temperatureChoice) {
        case Temperature::Linear:
            return getTemperatureLinear();
        case Temperature::PowerSlow:
            return getTemperaturePowerSlow();
        case Temperature::PowerFast:
            return getTemperaturePowerFast();
    }
    return 0.;
}

double SimulatedAnnealingTSP::getTemperatureLinear() const {
    return SimulatedAnnealingTSP::initialT * ( (double) (kStop - k) / (double) kStop);
}

double SimulatedAnnealingTSP::getTemperaturePowerSlow() const {
    return -(SimulatedAnnealingTSP::initialT / ( (double) kStop * kStop)) * ((double) k * k) +
           SimulatedAnnealingTSP::initialT;
}

double SimulatedAnnealingTSP::getTemperaturePowerFast() const {
    return (SimulatedAnnealingTSP::initialT / ((double) kStop * kStop)) * ( (double) k * k) +
           (-2 * SimulatedAnnealingTSP::initialT / ((double) kStop)) * ((double) k) + SimulatedAnnealingTSP::initialT;
}

shared_ptr<PointGraph> SimulatedAnnealingTSP::getNextState() {
    switch(nextStateChoice) {
        case NextState::Consecutive: {
            shared_ptr<PointGraph> nextState = make_shared<PointGraph>(*currentState);
            nextState->consecutiveSwap();
            return nextState;
        }
        case NextState::Arbitrary: {
            shared_ptr<PointGraph> nextState = make_shared<PointGraph>(*currentState);
            nextState->arbitrarySwap();
            return nextState;
        }
        case NextState::Mixed: {
            shared_ptr<PointGraph> nextState = make_shared<PointGraph>(*currentState);
            for(int i = 0; i < SimulatedAnnealingTSP::mixedAttemptsNumber; i++) {
                nextState->arbitrarySwap();
                if(getEnergy(nextState) < E)
                    return nextState;
            }
            nextState = make_shared<PointGraph>(*currentState);
            nextState->consecutiveSwap();
            return nextState;
        }
    }
    return make_shared<PointGraph>(*currentState);
}

void SimulatedAnnealingTSP::attemptAccepting(shared_ptr<PointGraph> &candidate) {
    double candidateE = getEnergy(candidate);
    if(candidateE < E) {
        E = candidateE;
        currentState = move(candidate);
        updateBest();
    }
    else if(T > 0.) {
        if(getRandomProbability() < acceptanceProbability(candidateE)) {
            E = candidateE;
            currentState = move(candidate);
        }
    }
}

double SimulatedAnnealingTSP::acceptanceProbability(double candidateE) const {
    return (T == 0.) ? 0. : exp(-abs(candidateE - E) / T);
}

void SimulatedAnnealingTSP::updateBest() {
    if(E < bestE) {
        bestE = E;
        bestState = make_shared<PointGraph>(*currentState);
        iterationsSinceBest = 0;
    }
}

double SimulatedAnnealingTSP::getRandomProbability() {
    return randDoubleGen.getRandomUniform();
}

void SimulatedAnnealingTSP::annealAll() {

    for(int i = 0; i < kStop; i++) {
        if(i % (kStop / 10) == 0) {
            cout << "Iteration " << i << endl;
            cout << "Temperature " << T << endl;
            cout << "Energy " << E << endl << endl;
        }
        k = i;
        iterationsSinceBest++;
        shared_ptr<PointGraph> candidate = getNextState();
        attemptAccepting(candidate);
        T = getTemperature();

        if(iterationsSinceBest > maxHigherEnergyIterations) {
            currentState = make_shared<PointGraph>(*bestState);
            E = getEnergy(currentState);
            iterationsSinceBest = 0;
        }

        energyHistory.push_back(E);
        temperatureHistory.push_back(T);
    }
    T = 0.;

    currentState = make_shared<PointGraph>(*bestState);
    E = getEnergy(currentState);

    for(int i = 0; i < maxHillDescendingIterations; i++) {
        shared_ptr<PointGraph> candidate = getNextState();
        attemptAccepting(candidate);
        energyHistory.push_back(E);
        temperatureHistory.push_back(T);
    }
    updateBest();
}

bool SimulatedAnnealingTSP::makeStep() {
    if(k < kStop) {
        if(k % (kStop / 10) == 0) {
            cout << "Iteration " << k << ": " << endl;
            cout << "Temperature " << T << endl;
            cout << "Energy " << E << endl << endl;
        }
        k++;
        iterationsSinceBest++;
        shared_ptr<PointGraph> candidate = getNextState();
        attemptAccepting(candidate);
        T = getTemperature();

        if(iterationsSinceBest > maxHigherEnergyIterations) {
            currentState = make_shared<PointGraph>(*bestState);
            E = getEnergy(currentState);
            iterationsSinceBest = 0;
        }

        energyHistory.push_back(E);
        temperatureHistory.push_back(T);
        return true;
    }
    else if(k >= kStop && k < kStop + maxHillDescendingIterations) {
        if(k == kStop) {
            T = 0.;
            currentState = make_shared<PointGraph>(*bestState);
            E = getEnergy(currentState);
            cout << "---Ending annealing---" << endl << endl;
            cout << "---Starting hill-descending---" << endl;
            cout << "Temperature " << T << endl;
            cout << "Energy " << E << endl << endl;
        }
        if(k % ((kStop + maxHillDescendingIterations) / 10) == 0) {
            cout << "Iteration " << k << ": " << endl;
            cout << "Temperature " << T << endl;
            cout << "Energy " << E << endl << endl;
        }
        k++;
        shared_ptr<PointGraph> candidate = getNextState();
        attemptAccepting(candidate);
        energyHistory.push_back(E);
        temperatureHistory.push_back(T);
        updateBest();
        return true;
    }

    cout << "---Ending hill-descending---" << endl;
    cout << "Temperature " << T << endl;
    cout << "Energy " << E << endl << endl;

    return false;
}

const vector<double> &SimulatedAnnealingTSP::getEnergyHistory() const {
    return energyHistory;
}

const vector<double> &SimulatedAnnealingTSP::getTemperatureHistory() const {
    return temperatureHistory;
}

double SimulatedAnnealingTSP::getE() const {
    return E;
}

const shared_ptr<PointGraph> &SimulatedAnnealingTSP::getCurrentState() const {
    return currentState;
}

double SimulatedAnnealingTSP::getBestE() const {
    return bestE;
}

const shared_ptr<PointGraph> &SimulatedAnnealingTSP::getBestState() const {
    return bestState;
}

int SimulatedAnnealingTSP::getKStop() const {
    return kStop;
}

int SimulatedAnnealingTSP::getMaxHillDescendingIterations() const {
    return maxHillDescendingIterations;
}
