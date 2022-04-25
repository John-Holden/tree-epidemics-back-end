// SIR model re-written in cpp
#include <iostream>
#include <string>
#include "SIR.h"
#include "common.h"
#include <fstream>
#include <jsoncpp/json/json.h>

// todo spike json parse method to load sim data in....
using namespace std; // use all std object names etc. 

// Class holding one public function
class Simulation{
    // one pubic function to execute a simulation
    public: int Execute(char* a){ 
            Json::Value ParamRoot {LoadJson(a)};

            print(ParamRoot.toStyledString());
            // TODO: load in csv for SIR from python BE?? 
            SetSIR(ParamRoot["domain"], ParamRoot["initial_conditions"], ParamRoot["infectious_lt"]);
            // TODO pretty much go straight into main foor loop... 
            return 0;
        }
};


extern "C" {
    Simulation* newSimOjb(){ return new Simulation(); }
    int execute(Simulation* simulation, char* a){ return simulation->Execute(a); }
    }
