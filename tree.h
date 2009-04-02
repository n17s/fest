/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declarations of tree related structures and functions       *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/


#ifndef TREE_H
#define TREE_H

#include "dataset.h"

#define BAGGING      1
#define BOOSTING     2
#define RANDOMFOREST 3


typedef struct node_t{
    struct node_t* left;
    struct node_t* right;
    int split;
    float threshold;
    float pos;
    float neg;
} node_t;

typedef struct tree_t{
    node_t* root;
    float* pred; /* prediction of tree for i-th example */
    int* feats; /* Just a permutation of the features */
    int* valid; /* Is the ith example valid for consideration? */
    int* used; /* Is the ith feature used? */
    int fpn; /* Features to consider per node */ 
    int maxdepth; /* maximum depth the tree is allowed to reach */
    int committee; /* committee type */ 
} tree_t;

typedef struct split_t{
    int feature;
    float threshold;
    float posleft;
    float negleft;
    float posright;
    float negright;
    float gain;
} split_t;


void freeTree(node_t* t);
void grow(tree_t* t, dataset_t* d);
void classifyTrainingData(tree_t* t, node_t* root, dataset_t* d);
void classifyOOBData(tree_t* t, node_t* root, dataset_t* d);
float classifyBag(node_t* t, float* example);
float classifyBoost(node_t* t, float* example);
void writeTree(FILE* fp, node_t* t);
void readTree(FILE* fp, node_t** t);
#endif
