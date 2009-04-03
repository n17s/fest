/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declaration of forest structure (ie. ensemble of trees)    *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#ifndef FOREST_H
#define FOREST_H

#include "tree.h"

typedef struct forest_t{
    node_t** tree;
    int ngrown;
    int ntrees;
    int committee;
    int oob;
    int nfeat;    /* number of features in the training set */
    int maxdepth; /* maximum depth the tree is allowed to reach */
    float factor; /* random forest only; how many features to consider */
    float wneg;   /* relative weight of the negative class */
} forest_t;

void initForest(forest_t* f,int committee, int maxdepth, float param, int trees, float w, int oob);
void freeForest(forest_t* f);
float classifyForest(forest_t* f, float* example);
void growForest(forest_t* f, dataset_t* d);
void readForest(forest_t* f, const char* fname);
void writeForest(forest_t* f, const char* fname);
#endif /* FOREST_H */
