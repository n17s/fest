#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <math.h>

void rfrelease(forest_t* f){
    int i;
    for(i=0; i<f->ntrees; i++)
        freeTree(f->tree[i]);
    free(f->tree);
}

float rfclassify(forest_t* f, float* example){
    int i;
    float sum = 0;
    for(i=0; i<f->ntrees; i++)
        sum += classify(f->tree[i], example);
    return sum/f->ntrees;
}

void randomForest(forest_t* f, dataset_t* d){
    int i,t,r;
    tree_t tree;

    /* Hard code these for now */
    f->factor = 1.0f;
    f->maxdepth = 1000;
    f->ntrees = 500;
    f->tree = malloc(f->ntrees*sizeof(node_t*));

    tree.valid = malloc(d->nex*sizeof(int));
    tree.used = calloc(d->nfeat,sizeof(int));
    tree.feats = malloc(d->nfeat*sizeof(int));
    for(i=0; i<d->nfeat; i++)
        tree.feats[i]=i;
    tree.fpn=(int)(f->factor*sqrt(d->nfeat));
    tree.maxdepth = f->maxdepth;

    for(t=0; t<f->ntrees; t++){
        for(i=0; i<d->nex; i++){
            tree.valid[i]=0;
            d->weight[i]=0;
        }
        /* Bootstrap sampling */
        for(i=0; i<d->nex; i++){
            r = rand()%d->nex;
            tree.valid[r] = 1;
            d->weight[r] += 1.0f/d->nex;
        }
        grow(&tree, d);
        f->tree[t]=tree.root;
    }
    free(tree.valid);
    free(tree.used);
    free(tree.feats);
}

