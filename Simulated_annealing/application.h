
#ifndef SIMULATED_ANNEALING_APPLICATION_H
#define SIMULATED_ANNEALING_APPLICATION_H

#include "SFML/include/SFML/Graphics.hpp"
#include "SFML/include/SFML/Window.hpp"

#include "annealing.h"

#include <unistd.h>


using namespace std;
using namespace sf;

class Application {
    unsigned int dimX;
    unsigned int dimY;
public:
    Application(int dimX, int dimY): dimX(dimX), dimY(dimY) {}

    void initApp(SimulatedAnnealingTSP annealingTsp) const;

    void drawPointGraph(sf::RenderWindow &window, SimulatedAnnealingTSP &annealingTsp) const;
};


#endif //SIMULATED_ANNEALING_APPLICATION_H
