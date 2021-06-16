/**
 * @file main.cpp
 *
 * @author Piotr Zawislan
 * BookID: 400427
 * Email: piotrzwsln@student.agh.edu.pl
 */

#include "application.h"
#include "annealing.h"

using namespace std;

int WINDOW_WIDTH = 720;
int WINDOW_HEIGHT = 720;
int PADDING = 20;

int main() {

    // Initialize annealing object with random PointGraph
    auto randGenX = RandomDoubleGenerator(PADDING, WINDOW_WIDTH - PADDING, 0., 0.);
    auto randGenY = RandomDoubleGenerator(PADDING, WINDOW_HEIGHT - PADDING, 0., 0.);

    shared_ptr<PointGraph> pointGraph = make_shared<PointGraph>();
    pointGraph->initGraphUniform(randGenX, randGenY, 30);

    cout << *pointGraph << endl;
    cout << "Total distance: " << pointGraph->getTotalDistance() << endl;

    auto TSPAnnealing = SimulatedAnnealingTSP(pointGraph,
                                              100000,
                                              20000,
                                              10000,
                                              Temperature::PowerFast,
                                              NextState::Mixed);


    Application app = Application(WINDOW_WIDTH, WINDOW_HEIGHT);

    app.initApp(TSPAnnealing);

    return 0;
}
