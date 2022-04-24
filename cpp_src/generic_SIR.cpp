// SIR model re-written in cpp
#include <iostream>
#include <string>
#include "SIR.h"
#include "common.h"


// todo spike json parse method to load sim data in....
using namespace std; // use all std object names etc. 

// Class holding one public function
class Simulation{
    // one pubic function to execute a simulation
    public: int Execute(char* a){
            LoadJson(a);
            // todo -> load in json parameters...
            std:: cout << a << std::endl;;
            return 0;
        }
};


extern "C" {
    Simulation* newSimOjb(){ return new Simulation(); }
    int execute(Simulation* simulation, char* a){ return simulation->Execute(a); }
    }
