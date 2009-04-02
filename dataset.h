/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Declarations of data structures for storing the input      *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>

/* Example-Value pair. Similar to feature value pair
 * when indexing by example
 */
typedef struct evpair_t{
    int example; /* id of example */
    float value; /* value of feature for this example */
}evpair_t;

typedef struct dataset_t{
    evpair_t** feature; /* array of arrays of example value pairs */
    int* size; /* size[i]=number of examples with non-zero feature i */
    /* Would it be better if these were short/char?*/
    int* cont;  /* Is the ith feature continuous? */
    int* target; /* Target values */
    float* weight; /* Weight of the ith example */
    int nfeat; /* number of features */
    int nex; /* number of examples */
    int* oobvotes;
}dataset_t;

void loadData(const char* name, dataset_t* d);
int getDimensions(FILE* fp, int* examples, int* features);
int readExample(FILE* fp, int maxline, float* example, int nfeat, int* target);
void freeData(dataset_t* d);

#endif /* DATASET_H */
