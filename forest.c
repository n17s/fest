/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions to handle forests (ie. ensembles of trees)       *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <math.h>

void initForest(forest_t* f, int committee, int maxdepth, float param, int trees){
    f->committee = committee;
    f->maxdepth = maxdepth;
    f->factor = param;
    f->ntrees = trees;
    f->ngrown = 0;
}

void freeForest(forest_t* f){
    int i;
    for(i=0; i<f->ngrown; i++)
        freeTree(f->tree[i]);
    free(f->tree);
}


void growForest(forest_t* f, dataset_t* d){
    int i,t,r;
    tree_t tree;
    float sum;

    f->nfeat = d->nfeat;
    f->tree = malloc(f->ntrees*sizeof(node_t*));
    tree.valid = malloc(d->nex*sizeof(int));
    tree.used = calloc(d->nfeat,sizeof(int));
    tree.feats = malloc(d->nfeat*sizeof(int));
    for(i=0; i<d->nfeat; i++)
        tree.feats[i]=i;
    tree.maxdepth = f->maxdepth;
    tree.committee = f->committee;
    if (f->committee == BOOSTING){
        tree.pred = malloc(d->nex*sizeof(float));
        for(i=0; i<d->nex; i++){
            tree.valid[i]=1;
            d->weight[i]=1.0f/d->nex;
        }
    }
    if(f->committee == RANDOMFOREST)
        tree.fpn=(int)(f->factor*sqrt(d->nfeat));
    else
        tree.fpn = d->nfeat;
    for(t=0; t<f->ntrees; t++){
       // printf("growing tree %d\n",t);
        if (f->committee == BOOSTING){
            grow(&tree, d);
            classifyTrainingData(&tree, tree.root, d);
            sum=0.0f;
            for(i=0; i<d->nex; i++){
                d->weight[i]*=exp(-d->target[i]*tree.pred[i]);
                sum+=d->weight[i];
            }
            for(i=0; i<d->nex; i++)
                d->weight[i]/=sum;
        }
        else{
            /* Bootstrap sampling */ 
            for(i=0; i<d->nex; i++){
                tree.valid[i]=0;
                d->weight[i]=0;
            }
            for(i=0; i<d->nex; i++){
                r = rand()%d->nex;
                tree.valid[r] = 1;
                d->weight[r] += 1.0f/d->nex;
            }
            grow(&tree, d);
        }
        f->tree[t] = tree.root;
        f->ngrown += 1;
    }
    if (f->committee == BOOSTING){
        free(tree.pred);
    }
    free(tree.valid);
    free(tree.used);
    free(tree.feats);
}

float classifyForest(forest_t* f, float* example){
    int i;
    float sum = 0;
    if(f->committee == BOOSTING){
        for(i=0; i<f->ngrown; i++){
            sum += classifyBoost(f->tree[i], example);
        }
    }
    else{
        for(i=0; i<f->ngrown; i++){
            sum += classifyBag(f->tree[i], example);
        }
    }
    return sum/f->ngrown;
}

void writeForest(forest_t* f, const char* fname){
    int i;
    char* committeename[8];
    FILE* fp = fopen(fname,"w");
    if(fp == NULL){
        fprintf(stderr,"could not write to output file: %s\n",fname);
        return;
    }
    committeename[BAGGING]="Bagging";
    committeename[BOOSTING]="Boosting";
    committeename[RANDOMFOREST]="RandomForest";
    
    fprintf(fp, "committee: %d (%s)\n",f->committee, committeename[f->committee]);
    fprintf(fp, "trees: %d\n", f->ngrown);
    fprintf(fp, "features: %d\n", f->nfeat);
    fprintf(fp, "maxdepth: %d\n", f->maxdepth);
    fprintf(fp, "fpnfactor: %g\n", f->factor);
    for(i=0; i<f->ngrown; i++){
        writeTree(fp,f->tree[i]);
    }
    fclose(fp);
}

void readForest(forest_t* f, const char* fname){
    int i;
    FILE* fp = fopen(fname,"r");
    if(fp == NULL){
        fprintf(stderr,"could not read input file: %s\n",fname);
        exit(1);
    }
    fscanf(fp, "%*s%d%*s",&f->committee);
    fscanf(fp, "%*s%d", &f->ngrown);
    fscanf(fp, "%*s%d", &f->nfeat);
    fscanf(fp, "%*s%d", &f->maxdepth);
    if(fscanf(fp, "%*s%g", &f->factor)==EOF) {
        fprintf(stderr,"corrupt input file: %s\n",fname);
        exit(1);
    }
    f->tree = malloc(sizeof(node_t*)*f->ngrown);
    for(i=0; i<f->ngrown; i++){
        readTree(fp,&(f->tree[i]));
    }
    if(fscanf(fp, "%*s")!=EOF){
        fprintf(stderr,"garbage at the end of input file: %s\n",fname);
    }
    fclose(fp);
}

