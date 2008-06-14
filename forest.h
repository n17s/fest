#ifndef FOREST_H
#define FOREST_H

typedef struct forest_t{
    node_t** tree;
    int ntrees;
    int maxdepth; /* maximum depth the tree is allowed to reach */
    float factor; /* random forest only; how many features to consider */
} forest_t;

void rfrelease(forest_t* f);
float rfclassify(forest_t* f, float* example);

void randomForest(forest_t* f, dataset_t* d);




#endif
