//
//  main.cpp
//  OpenCL.AddValue
//
//  Created by Peter.Li on 2015/7/1.
//  Copyright (c) 2015年 Peter.Li. All rights reserved.
//

#include <iostream>
#include <iomanip>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

typedef VECTOR_CLASS<cl_float> DATA_VEC_TYPE;

// OpenCL kernel function string
const cl::STRING_CLASS cl_source =
" \
__kernel void add_float(__global const float* a, __global const float* b, __global float* result) \
{ \
    int idx = get_global_id(0); \
    result[idx] = a[idx] + b[idx]; \
} \
__kernel void multiple_float(__global const float* a, __global const float* b, __global float* result) { \
int idx = get_global_id(0); \
result[idx] = a[idx] * b[idx]; \
} \
";

using namespace std;

int main(int argc, const char * argv[]) {
    
    cout << "OpenCL Add and Multiple example" << endl;
    
    try {
        
        cl_int err;
        auto context = cl::Context::getDefault(&err);
        if (CL_SUCCESS != err) {
            throw cl::Error(err, "Default OpenCL Context doesn't exist");
        }
        
        const size_t DATA_SIZE = 8;
        const size_t data_bytes = sizeof(DATA_VEC_TYPE::value_type) * DATA_SIZE;
    
        DATA_VEC_TYPE val_a(DATA_SIZE), val_b(DATA_SIZE), add_res(DATA_SIZE), mul_res(DATA_SIZE, 0);
        
        // Inital random number
        srand((int)time(nullptr));
        
        // Initial data
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            val_a[i] = rand() % 10;
            val_b[i] = rand() % 5;
            add_res[i] = 0;
            mul_res[i] = 0;
        }
        
        // Create data buffer
        cl::Buffer buf_a(val_a.begin(), val_a.end(), true, true);
        cl::Buffer buf_b(val_b.begin(), val_b.end(), true, true);
        cl::Buffer buf_add_res(CL_MEM_WRITE_ONLY, data_bytes);
        cl::Buffer buf_mul_res(CL_MEM_WRITE_ONLY, data_bytes);
        
        cl::Program program(cl_source, true);
        
        // Create add_float kernel
        cl::Kernel add_ker(program, "add_float");
        add_ker.setArg(0, buf_a);
        add_ker.setArg(1, buf_b);
        add_ker.setArg(2, buf_add_res);
        
        // Create multiple_float kernel
        cl::Kernel mul_ker(program, "multiple_float");
        mul_ker.setArg(0, buf_a);
        mul_ker.setArg(1, buf_b);
        mul_ker.setArg(2, buf_mul_res);
        
        
        cout << "Add Kernel to Command Queue" << endl;
        cl::CommandQueue queue(context);
        queue.enqueueNDRangeKernel(add_ker, cl::NullRange, cl::NDRange(data_bytes));
        queue.enqueueNDRangeKernel(mul_ker, cl::NullRange, cl::NDRange(data_bytes));
        
        // Read data from buffer to vector
        queue.enqueueReadBuffer(buf_add_res, CL_TRUE, 0, data_bytes, add_res.data());
        queue.enqueueReadBuffer(buf_mul_res, CL_TRUE, 0, data_bytes, mul_res.data());
        
        // show data and result
        cout << "Execute result : " << endl;
        cout << boolalpha << endl;
        for (size_t i = 0; i < DATA_SIZE; ++i) {
            cout << "a[" << i << "] = " << val_a[i] << ", ";
            cout << "b[" << i << "] = " << val_b[i] << ", ";
            cout << "Add : " << left << setw(4) << add_res[i] << ", ";
            cout << "Multiple : " << mul_res[i] << endl;
        }
        
        cout << endl << "Run example successfully!" << endl;
    } catch (cl::Error err) {
        cout << "cl::Error:" << endl << err.what() << endl;
    } catch (std::exception e) {
        cout << "std::exception:" << endl << e.what() << endl;
    }
    
    
    return 0;
}
