# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:29:09 2018

@author: Aditya Prakash
"""
import random
import numpy as np
def matrix_creation(Ni, Lr, Lc, No, Fr, Fc, Mr, Mc, ros_type):
    input_feature_maps, filter_coefficients = data_generation(Ni, Lr, Lc, No, Fr, Fc, ros_type)
    output_feature_maps = [[[0 for x in range(Mc)] for y in range(Mr)] for z in range(No)]
    print("\nInput Feature Matrix Initial Shape: ")
    print(np.shape(input_feature_maps))
    #print(input_feature_maps)
    print("\nFilter Coefficients Initial Shape: ")
    print(np.shape(filter_coefficients))
    #print(filter_coefficients)
    print("\nOutput Matrix Initial Shape: ")
    print(np.shape(output_feature_maps))
    #print(output_feature_maps)
    return input_feature_maps, filter_coefficients, output_feature_maps
    
def data_generation(Ni, Lr, Lc, No, Fr, Fc, ros_type):
    if ros_type == "sequential":
        ifm = [[[((x)+5*y+25*z) for x in range(Lc)] for y in range(Lr)] for z in range(Ni)]
        fc = [[[[((x)+3*y+9*z + 18*xx) for x in range(Fc)] for y in range(Fr)] for z in range(Ni)] for xx in range(No)]
        return ifm, fc
    elif ros_type == "random":
        ifm = [[[random.random() for x in range(Lc)] for y in range(Lr)] for z in range(Ni)]
        fc = [[[[random.random() for x in range(Fc)] for y in range(Fr)] for z in range(Ni)] for xx in range(No)]
        return ifm, fc
    
def pre_processing(ifm, fc, Ur, Uc):
    # Padding
    ifmp = np.zeros((np.shape(ifm)[0],np.shape(ifm)[1]+2,np.shape(ifm)[2]+2))
    for x in range(np.shape(ifm)[0]):
        ifmt = np.pad(ifm[x], 1, mode = 'constant')
        ifmp[x] = ifmt
    # Upsampling input
    ifmu = np.zeros((np.shape(ifmp)[0],np.shape(ifmp)[1]*Ur - 3,np.shape(ifmp)[2]*Uc - 3))
    for x in range(np.shape(ifmp)[0]):
        for y in range(np.shape(ifmp)[1]-1):
            for z in range(np.shape(ifmp)[2]-1):
                ifmu[x,y*Ur-1,z*Uc-1] = ifmp[x,y,z]
    # Upsampling filter
    fcu = np.zeros((np.shape(fc)[0], np.shape(fc)[1], np.shape(fc)[2]*Ur, np.shape(fc)[3]*Uc))
    for x in range(np.shape(fc)[0]):
        t = fc[x]
        for y in range(np.shape(fc)[1]):
            t1 = t[y]
            for z in range(np.shape(fc)[2]):
                t2 = t1[z]
                for zz in range(np.shape(fc)[3]):
                    fcu[x,y,z*Ur,zz*Uc] = t2[zz]    
    # Padding filter not reqd
    '''fcp = np.zeros((np.shape(fc)[0],np.shape(fc)[1],np.shape(fc)[2]+2,np.shape(fc)[3]+2))
    for x in range(np.shape(fc)[0]):
        t1 = fc[x]
        for y in range(np.shape(fc)[1]):
            t2 = np.pad(t1[y], 1, mode = 'constant')
            fcp[x,y] = t2'''
    return ifmu, fcu

def matrix_multiplication(ifm, fc):
    output = np.zeros((np.shape(fc)[0], np.shape(fc)[2], np.shape(fc)[3]))
    for x in range(np.shape(output)[0]):
        f1 = fc[x]
        for y in range(np.shape(output)[1]):
            for z in range(np.shape(output)[2]):
                # In loop for output
                for xx in range(np.shape(ifm)[0]):
                    fi = f1[xx]
                    for yy in range(np.shape(output)[1]):
                        for zz in range(np.shape(output)[2]):
                            output[x][y][z] = output[x][y][z] + fi[yy][zz] * ifm[xx][y+yy][z+zz]
    return output
    
def post_processing(output):
    ofm = np.zeros((np.shape(output)[0], np.shape(fc)[2], np.shape(fc)[3]))
    c=0
    for x in range(np.shape(output)[0]):
        a=0
        for y in range(np.shape(output)[1]):
            b=0
            for z in range(np.shape(output)[2]):
                if(y%2==1 and z%2==1):
                    ofm[c][a][b] = output[x][y][z]
                    b=b + 1
            if(y%2 == 1):
                a = a + 1
        if(y%2==1):
            c = c + 1
    return ofm

def visualize(ifm, fc, ofm):
    print("\nInput Feature Matrix with Zero Padding and Upsampled\n")
    print("Shape: ")
    print(np.shape(ifm))
    print("\n")
    print(ifm)
    print("\nFilter Coefficients Upsampled\n")
    print("Shape: ")
    print(np.shape(fc))
    print("\n")
    print(fc)
    print("\nOutput Matrix Downsampled\n")
    print("Shape: ")
    print(np.shape(ofm))
    print("\n")
    print(ofm)

ifm, fc, ofm = matrix_creation(2, 5, 5, 3, 3, 3, 3, 3, "sequential")
ifmu, fcu = pre_processing(ifm,fc, 2, 2)
output = matrix_multiplication(ifmu, fcu)
ofm = post_processing(output)
visualize(ifmu, fcu, ofm)