// SIR model re-written in cpp
#include <iostream>
#include <string>
#include "SIR.h"


// todo spike json parse method to load sim data in....

using namespace std; // use all std object names etc. 

// Class holding one public function
class Simulation{
    // one pubic function to execute a simulation
    public: int ExecuteRun(int a){
            inHeaderFileFunct("input text from main");
            // 1 - setup data structures 
            // 2 - exedcute t step for loop
            // 3 - evolve t step by one unit
            return 0;
        }
};


extern "C" {
    Simulation* newSimOjb(){ return new Simulation(); }
    int execute(Simulation* simulation, int a){ return simulation->ExecuteRun(a); }
    }
