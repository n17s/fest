#ifndef TREE_H
#define TREE_H

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
    int* feats; /* Just a permutation of the features */
	int* valid; /* Is the ith example valid for consideration? */
    int* used; /* Is the ith feature used? */
    int fpn; /* Features to consider per node */
    int maxdepth; /* maximum depth the tree is allowed to reach */
} tree_t;

typedef struct forest_t{
    node_t** tree;
    int ntrees;
    int maxdepth; /* maximum depth the tree is allowed to reach */
    float factor;
} forest_t;

typedef struct split_t{
    int feature;
    float threshold;
    float posleft;
    float negleft;
    float posright;
    float negright;
    float gain;
} split_t;

#endif
