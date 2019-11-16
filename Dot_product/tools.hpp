#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <random>
#include <sys/time.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
using std::abs;

    #define print_variable(X) cout << #X": " << X << endl;

    double cpuSecond() {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
    }

    /*
        Function to fill an array with random values between lower and upper
    */
    template <class T>
    void random_array(T *arr, const size_t arr_size, T lower, T upper) {
        /* INITIALISE RNG */
            std::random_device rd;
            std::mt19937 engine(rd());
            std::uniform_real_distribution<T> dist(lower,upper);

        for(int i = 0; i < arr_size; i++) {
            arr[i] = dist(engine);
        }
    }

    int str2int(char* str) {
        int res;
        try {
            res = std::stoi(str);
        } catch(std::exception e) {
            throw std::invalid_argument("Bad command line arguments. Read the code for instructions.");
        }
        return res;
    }

    bool float_equal(const float a, const float b, float tol = 1e-6) {
        return std::abs(a-b) < tol;
    }

    void compare_float_arrays(const float *a, const float *b, const size_t arr_size) {
        for(int i = 0; i < arr_size; i++) {
            if(!float_equal(a[i],b[i])) {
                cout << "Arrays are not the same!" << endl;
                return;
            }
        }
        cout << "Arrays are the same!" << endl;
    }

    void detailed_print_float(float x) {
        cout << std::setprecision(10) << x << endl;
    }

    inline void checkCuda(cudaError_t result) {
        if(result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
    }
#endif