#include <string>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>

using namespace std; // use all std object names etc. 

Json::Value LoadJson(string sim_loc)

{ 
    ifstream file_input(sim_loc);
    Json::Reader reader;
    Json::Value root;
    reader.parse(file_input, root);
    return root;
}

void print(string char_seq) {std:: cout << char_seq << std :: endl;}