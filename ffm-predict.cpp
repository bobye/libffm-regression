#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include "ffm.h"

using namespace std;
using namespace ffm;

struct Option
{
    string test_path, model_path, output_path;
};

string predict_help()
{
    return string(
"usage: ffm-predict test_file model_file output_file\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 4)
        throw invalid_argument("cannot parse argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.output_path = string(args[3]);

    return option;
}

void predict(string test_path, string model_path, string output_path)
{
    int const kMaxLineSize = 1000000;

    FILE *f_in = fopen(test_path.c_str(), "r");
    ofstream f_out(output_path);
    char line[kMaxLineSize];

    ffm_model *model = ffm_load_model(model_path.c_str());

    ffm_double loss = 0;
    vector<ffm_node> x;
    ffm_int i = 0;

    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++)
    {
        x.clear();
        char *y_char = strtok(line, " \t");
        ffm_float y = atof(y_char);

        while(true)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            x.push_back(N);
        }

        ffm_float t = ffm_predict(x.data(), x.data()+x.size(), model, true);

        ffm_float e = y - t;

        loss += e * e;

        f_out << t << "\n";
    }

    loss = sqrt(loss / i);

    cerr << "rmse = " << fixed << setprecision(5) << loss << endl;

    ffm_destroy_model(&model);

}

int main(int argc, char **argv)
{
    cerr << "This is a branch of LIBFFM for regression problems" << endl;

    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument const &e)
    {
        cerr << e.what() << endl;
        return 1;
    }

    predict(option.test_path, option.model_path, option.output_path);
}
