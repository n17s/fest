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

void initForest(forest_t* f, int committee, int maxdepth, float param, int trees, float wneg, int oob){
    f->committee = committee;
    f->maxdepth = maxdepth;
    f->factor = param;
    f->ntrees = trees;
    f->ngrown = 0;
    f->wneg = wneg;
    f->oob = oob;
}

void freeForest(forest_t* f){
    int i;
    for(i=0; i<f->ngrown; i++)
        freeTree(f->tree[i]);
    free(f->tree);
}

void tabulateOOBVotes(tree_t* tree, dataset_t* d) {
    // classify the out-of-bag examples and record the votes
    classifyOOBData(tree,tree->root,d);
    int i;
    for(i=0; i < d->nex; i++) {
        if(d->weight[i] == 0) {
            if (tree->pred[i] > 0.5)
                d->oobvotes[i]+=1;
            else
                d->oobvotes[i]-=1;
        }
    }
}

void reportOOBError(dataset_t* d, int iter) {
    float tp,fp,tn,fn;
    int confusion[2][2]={{0,0},{0,0}};
    int i;
    for (i=0; i < d->nex; i++) {
        if(d->oobvotes[i] == 0)
            continue;
        int margin = d->oobvotes[i] > 0 ? 1 : 0;
        confusion[d->target[i]][margin]+=1;
    }
    tp=confusion[1][1];
    fn=confusion[1][0];
    fp=confusion[0][1];
    tn=confusion[0][0];

    float acc = (tp+tn) / (tp+tn+fp+fn);
    float sens = tp / (tp+fn);  // acc on pos examples = recall
    float spec = tn / (tn+fp);  // acc on neg examples
    printf("%5d  %5.2f%%  %5.2f%%  %5.2f%%\n", iter+1, 100*(1-acc), 100*(1-spec), 100*(1-sens));
}

void reportOOBHeader() {
    printf("Error rate (1-acc), on neg examples (1-spec), and on pos examples (1-sens)\n");
    printf("%5s  %6s  %6s  %6s\n","tree","err","negerr","poserr");
}

void growForest(forest_t* f, dataset_t* d){
    int i,t,r;
    tree_t tree;
    float sum,c[2],w[2];

    f->nfeat = d->nfeat;
    f->tree = malloc(f->ntrees*sizeof(node_t*));
    tree.valid = malloc(d->nex*sizeof(int));
    tree.used = calloc(d->nfeat,sizeof(int));
    tree.feats = malloc(d->nfeat*sizeof(int));
    for(i=0; i<d->nfeat; i++)
        tree.feats[i]=i;
    tree.maxdepth = f->maxdepth;
    tree.committee = f->committee;
    tree.pred = malloc(d->nex*sizeof(float));

    c[0]=c[1]=0;
    for(i=0; i<d->nex; i++){
        c[d->target[i]]+=1;
    }
    w[0]=f->wneg/(f->wneg*c[0]+c[1]);
    w[1]=1.0/(f->wneg*c[0]+c[1]);

    if (f->committee == BOOSTING){
        for(i=0; i<d->nex; i++){
            tree.valid[i]=1;
            d->weight[i]=w[d->target[i]];
        }
    }
    if(f->oob)
        reportOOBHeader();
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
                d->weight[i]*=exp(-(2*d->target[i]-1)*tree.pred[i]);
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
                d->weight[r] += w[d->target[r]];
            }
            grow(&tree, d);
            if(f->oob){
                for(i=0; i<d->nex; i++){
                    tree.valid[i] = 1;
                }
                tabulateOOBVotes(&tree, d);
                reportOOBError(d, t);
            }
        }
        f->tree[t] = tree.root;
        f->ngrown += 1;
    }
    free(tree.pred);
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

