#include "application.h"


void Application::initApp(SimulatedAnnealingTSP annealingTsp) const {
    RenderWindow window(VideoMode(dimX, dimY), "Simulated Annealing TSP");

    // window.setVerticalSyncEnabled(true);
    bool continueSteps = true;
    while(window.isOpen()) {
        sf::Event event{};
        while(window.pollEvent(event)) {
            if(event.type == Event::Closed)
                window.close();
        }
        window.clear();

        drawPointGraph(window, annealingTsp);

        if(continueSteps)
            continueSteps = annealingTsp.makeStep();

        window.display();
    }
}


void Application::drawPointGraph(sf::RenderWindow &window, SimulatedAnnealingTSP &annealingTsp) const {

    Color colorLines = Color::Cyan;
    Color colorPoints = Color(209, 62, 252);

    int pointsNumber = (int) annealingTsp.getCurrentState()->size();

    Vector2f pos;
    const shared_ptr<PointGraph>& pointGraph = annealingTsp.getCurrentState();
    auto points = pointGraph->getPoints();

    VertexArray lines(LineStrip, pointsNumber + 1);

    pos.x = (float) points[0].getX();
    pos.y = (float) points[0].getY();
    lines[pointsNumber].position = pos;

    for(int i = 0; i < pointsNumber; i++) {
        pos.x = (float) points[i].getX();
        pos.y = (float) points[i].getY();
        lines[i].position = pos;
        lines[i].color = colorLines;

        sf::CircleShape circle;
        circle.setRadius(1);
        circle.setOutlineColor(colorPoints);
        circle.setOutlineThickness(5);
        circle.setPosition((float) points[i].getX(), (float) points[i].getY());
        window.draw(circle);
    }
    window.draw(lines);
}
