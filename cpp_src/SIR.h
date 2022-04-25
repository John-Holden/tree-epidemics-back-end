#include <string>
#include <iostream>
#include <jsoncpp/json/json.h>
// #include "common.h"

using namespace std; // use all std object names etc. 

int JsonToInt(Json::Value value) {return std:: stoi(value.toStyledString());}
float JsonToFloat(Json::Value value) {return std:: stof(value.toStyledString());}
double JsonToDouble(Json::Value value) {return std:: stod(value.toStyledString());}

int SetI(string DistType, int InitInfected, int PatchSz[2], int InfectedLT) 
{
    std:: cout << DistType << std :: endl;
    std:: cout << InitInfected << std :: endl;
    std:: cout << PatchSz << std :: endl;
    std:: cout << InfectedLT << std :: endl;
};

int SetSIR(Json::Value Domain, Json:: Value InitialConditions, Json:: Value InfectiousLT )
{   
    int InfLT = JsonToInt(InfectiousLT);
    float TreeDensity = JsonToFloat(Domain["tree_density"]);
    string PatchSize = Domain["patch_size"].toStyledString();
    string InitInfDist = InitialConditions["distribution"].toStyledString();
    
    return 0;
}