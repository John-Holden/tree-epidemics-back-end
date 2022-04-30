#include <string>
#include <iostream>
#include <jsoncpp/json/json.h>
// #include "common.h"

using namespace std; // use all std object names etc. 
using std :: vector;
// Convert Json struct values to output of desired type, e.g. call JsonToInt(json_obj["o"]
int JsonToInt(Json::Value value) {return std:: stoi(value.toStyledString());}
float JsonToFloat(Json::Value value) {return std:: stof(value.toStyledString());}
double JsonToDouble(Json::Value value) {return std:: stod(value.toStyledString());}

int SetI(string DistType, int InitInfected, int PatchSz[2], int InfectedLT) 
{
    std:: cout << DistType << std :: endl;
    std:: cout << InitInfected << std :: endl;
    std:: cout << PatchSz << std :: endl;
    std:: cout << InfectedLT << std :: endl;
    return 0;
};

vector<vector<int>> LoadS(string SimLocation)
{   
    vector<vector<int>> S = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    return S;
}

vector<vector<int>> LoadI(string SimLocation)
{   
    // TODO: load SIR fields from input param csv
    vector<vector<int>> I = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    return I;
}


vector<vector<int>> LoadR(string SimLocation)
{   
    vector<vector<int>> R = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    return R;
}