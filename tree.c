/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions that grow and operate on a decision tree         *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#include "tree.h"
#include "dataset.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>

#define EPS 1e-6 /* Smoothing constant */

/* generate random subset of k elements that are not used */
void randomSubset(int* ss, int n, int k, int* used){
    int selected=0;
    int r,t,i,sum=0;
    for(i=0; i<n; i++)
        sum+=used[i];
    if(sum>n-k)
        return;
    do{
        r = selected + rand() % (n-selected);
        if(used[r])
            continue;
        t = ss[r];
        ss[r] = ss[selected];
        ss[selected] = t;
        selected += 1;
    }while(selected<k);
}

/* These functions are declared static in the hope that
 * the compiler will do some inlining. GCC doesn't seem 
 * to pick up such hints unless it can also inline 
 * non static functions. A macro such as 
 * #define max(a,b) ((a)>(b)?(a):(b))
 * provides a small gain in performance.
 */
static float max(float a, float b){
    return a > b ? a : b;
}

static float min(float a, float b){
    return a < b ? a : b;
}

static float entropy(float p){
    return -p*logf(p)-(1.0f-p)*logf(1.0f-p);
}

/* Approximates binary entropy. Some compromise
 * between Gini index and info gain that is
 * significantly faster than entropy. However 
 * the resulting growing procedure is slower 
 * ie. the features that are picked don't 
 * contribute as much in finishing the growing.
 */
/*
static float apxentropy(float p){
    float q=p*(1.0f-p);
    return q*(4.0f-5.0f*q);
}
*/

/* Update the best split if necessary */ 
static void updateSplit(int feature, float threshold, float posleft, float negleft, node_t* node, split_t* split){
    float posright = max(FLT_EPSILON, node->pos - posleft);
    float negright = max(FLT_EPSILON, node->neg - negleft);
    float sizeleft = posleft+negleft;
    float sizeright = posright+negright;
    float total = node->pos+node->neg;
    float gain = -(sizeleft/total*entropy(posleft/sizeleft)+sizeright/total*entropy(posright/sizeright));
    if (gain > split->gain){
        split->gain = gain;
        split->feature = feature;
        split->threshold = threshold;
        split->posleft = posleft;
        split->negleft = negleft;
        split->posright = posright;
        split->negright = negright;
    }
}

/* Find the best split for node root along with other relevant information */
split_t bestSplit(tree_t* t, node_t* root, dataset_t* d){
    split_t ret;
    int ii,i,j,ex,prev,prevex;
    float posleft,negleft,poszero,negzero,posnonzero,negnonzero;
    float threshold;
    float total = root->pos+root->neg;
    evpair_t* fi;

    ret.feature = -1;
    /* First compute the entropy of the parent */
    ret.gain = -entropy(root->pos/total);
    /* Select random subset of features */
    if(t->committee == RANDOMFOREST)
        randomSubset(t->feats, d->nfeat, t->fpn, t->used);
    for(ii=0; ii<t->fpn; ii++){
        i=t->feats[ii];
        if(t->used[i])
            continue;
        fi=d->feature[i];
        if(d->cont[i]){ /* If the feature is continuous */
            /* Find the first valid example */
            prevex = -1;
            for(j=0; j<d->size[i]; j++){
                ex = fi[j].example;
                if(t->valid[ex]>0){
                    prevex = ex;
                    break;
                }
            }
            if (prevex<0)
                continue;
            prev = j;

            /* Calculate the mass allocated to the zero value */
            /* We start with the mass allocated to the nonzero values */
            posnonzero = FLT_EPSILON;
            negnonzero = FLT_EPSILON;
            for(j=prev; j<d->size[i]; j++){
                ex = fi[j].example;
                if(t->valid[ex]<=0)
                    continue;
                if(d->target[ex])
                    posnonzero += d->weight[ex];
                else
                    negnonzero += d->weight[ex];
            }
            /* The mass allocated to the zero value is the rest */
            poszero = max(FLT_EPSILON, root->pos - posnonzero);
            negzero = max(FLT_EPSILON, root->neg - negnonzero);

            /* Initialize counts */
            posleft = FLT_EPSILON;
            negleft = FLT_EPSILON;
            /* Add the mass allocated to zero if the first valid example is > 0 */
            if (fi[prev].value > 0){
                posleft += poszero;
                negleft += negzero;
                /*Also check the split between 0 and value */
                threshold = 0.5*(0 + fi[prev].value);
                updateSplit(i,threshold,posleft,negleft,root,&ret);
            }
            for(j=prev+1; j<d->size[i]; j++){
                ex = fi[j].example;
                if(t->valid[ex]<=0)
                    continue;
                if(d->target[prevex]){
                    posleft += d->weight[prevex];
                }
                else{
                    negleft += d->weight[prevex];
                }
                if (fi[prev].value < 0 &&  0 < fi[j].value){
                    threshold = 0.5*(fi[prev].value + 0);
                    /* First check the split between previous value and 0 */
                    updateSplit(i,threshold,posleft,negleft,root,&ret);
                    posleft += poszero;
                    negleft += negzero;
                    /* Now check the split between 0 and current value */
                    threshold = 0.5*(0 + fi[j].value);
                    updateSplit(i,threshold,posleft,negleft,root,&ret);
                }
                /* Check the split between the two values if they are different */
                /* The extra condition d->target[ex] != d->target[prevex] is not used because
                 * it's not correct if the examples don't take unique values */
                if(fi[j].value != fi[prev].value){
                    threshold = 0.5*(fi[j].value + fi[prev].value);
                    updateSplit(i,threshold,posleft,negleft,root,&ret);
                }
                prev = j; 
                prevex = ex;
            }
        }
        else{ /* The feature is binary */
            /* These values are not used in the computation of entropy
             * so they don't need to be smoothed */
            float posright = 0;
            float negright = 0;
            /* Count the number of positive and negative examples that will go to the right */
            for(j=0; j<d->size[i]; j++){
                ex = fi[j].example;
                if(t->valid[ex]<=0)
                    continue;
                if(d->target[ex])
                    posright += d->weight[ex];
                else
                    negright += d->weight[ex];
            }
            /* The ones that will go to the left are the rest */
            posleft = max(FLT_EPSILON, root->pos - posright);
            negleft = max(FLT_EPSILON, root->neg - negright);
            updateSplit(i,0.5,posleft,negleft,root,&ret);
        }
    }
    return ret;
}


void growrec(tree_t* t, node_t* root, dataset_t* d, int depth){
    split_t best;
    int i,k,l,u;
    node_t* first;
    node_t* second;
    evpair_t* b;

    /* Stop if max depth is reached or node is pure */
    if(depth>=t->maxdepth || root->pos <= FLT_EPSILON || root->neg <= FLT_EPSILON){
        root->split=-1;
        return;
    }

    /* Find the best split */
    best = bestSplit(t,root,d);

    /* Stop if no good split is left or the counts in one of the children are very small */
    if (best.feature < 0 || 
            (best.posleft <= FLT_EPSILON && best.negleft <= FLT_EPSILON) || 
            (best.posright <= FLT_EPSILON && best.negright <= FLT_EPSILON)){
        root->split=-1;
        return;
    }

    /* Install the split */
    root->split=best.feature;
    root->threshold=best.threshold;
    root->left=malloc(sizeof(node_t));
    root->left->pos=best.posleft;
    root->left->neg=best.negleft;
    root->right=malloc(sizeof(node_t));
    root->right->pos=best.posright;
    root->right->neg=best.negright;

    /* Mark the feature as used */
    if(!d->cont[best.feature])
        t->used[best.feature]=1;
    b = d->feature[best.feature];
    /* Find the first example whose value exceeds the threshold */
    k = 0;
    u = d->size[best.feature];
    while (k < u) {
        i = (k + u)/2;
        if (b[i].value > best.threshold)
            u = i;
        else
            k = i + 1;
    }
    if (best.threshold > 0){
        l=k;
        u=d->size[best.feature];
        first = root->left;
        second = root->right;
    }
    else{
        l=0;
        u=k;
        first = root->right;
        second = root->left;
    }
    /* Here's how this works when threshold > 0. The case where threshold < 0 is analogous:
     * Let X be the set of all examples whose feature best.feature has value > threshold 
     * For every x in X decrease valid[x] by 1.
     * This leads to valid[x] > 0 iff x was previously valid and has value < threshold
     *
     * build left subtree (using the valid examples) 
     *
     * For every x in X increase valid[x] by 2.
     * For every example x decrease valid[x] by 1.
     * This leads to valid[x] > 0 iff x was previously valid and has value > threshold
     *
     * build right subtree (using the valid examples)
     *
     * Finally restore: 
     * For every x in X decrease valid[x] by 1.
     * For every example x increase valid[x] by 1.
     * This makes valid obtain its original state 
     * (One can verify this by adding up all the transformations)
     */
    for(i=l; i<u; i++)
        t->valid[b[i].example]-=1;
    growrec(t, first, d, depth+1);
    for(i=l; i<u; i++)
        t->valid[b[i].example]+=2;
    for(i=0; i<d->nex; i++)
        t->valid[i]-=1;
    growrec(t, second, d, depth+1);
    for(i=l; i<u; i++)
        t->valid[b[i].example]-=1;
    for(i=0; i<d->nex; i++)
        t->valid[i]+=1;
    /* Unmark the feature */
    if(!d->cont[best.feature])
        t->used[best.feature]=0;
}

void grow(tree_t* t, dataset_t* d){
    int i;

    /* Initialize root fields */
    t->root = malloc(sizeof(node_t));
    t->root->pos = FLT_EPSILON;
    t->root->neg = FLT_EPSILON;
    for(i=0; i<d->nex; i++){
        if(t->valid[i]<=0)
            continue;
        if(d->target[i])
            t->root->pos += d->weight[i];
        else
            t->root->neg += d->weight[i];
    }
    t->root->pos = min(1-FLT_EPSILON, t->root->pos);
    t->root->neg = min(1-FLT_EPSILON, t->root->neg);
    /* Recursively grow tree */
    growrec(t, t->root, d, 0);
}

float classifyBag(node_t* t, float* example){
    if(t->split < 0){
        if(t->pos <= FLT_EPSILON)
            return 0;
        if(t->neg <= FLT_EPSILON)
            return 1;
        return t->pos/(t->pos+t->neg);
    }
    else{
        if (example[t->split] <= t->threshold)
            return classifyBag(t->left, example);
        else
            return classifyBag(t->right, example);
    }
}

/* This is suggested by Schapire and Singer in their paper
"Improved boosting algorithms using confidence-rated predictions"
Machine Learning Journal 1999 
*/
float classifyBoost(node_t* t, float* example){
    if(t->split < 0){
        return 0.5*logf((t->pos+EPS)/(t->neg+EPS));
    }
    else{
        if (example[t->split] <= t->threshold)
            return classifyBoost(t->left, example);
        else
            return classifyBoost(t->right, example);
    }
}

void classifyTrainingData(tree_t* t, node_t* root, dataset_t* d){
    int i,k,l,u;
    node_t* first;
    node_t* second;
    evpair_t* b;

    /* Classify all valid points here */
    if ( root->split < 0 ){
        float pred=0.5f*logf((root->pos+EPS)/(root->neg+EPS));
        for(i=0; i<d->nex; i++){
            if(t->valid[i]<=0)
                continue;
            t->pred[i] = pred;
        }
        return;
    }

    /*The rest is similar to the recursive tree growing procedure 
    See the comments there for an explanation.
    */

    b = d->feature[root->split];
    k = 0;
    u = d->size[root->split];
    while (k < u) {
        i = (k + u)/2;
        if (b[i].value > root->threshold)
            u = i;
        else
            k = i + 1;
    }
    if ( root->threshold > 0 ){
        l=k;
        u=d->size[root->split];
        first = root->left;
        second = root->right;
    }
    else{
        l=0;
        u=k;
        first = root->right;
        second = root->left;
    }
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]-=1;
    classifyTrainingData ( t, first, d );
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]+=2;
    for ( i=0; i<d->nex; i++ )
        t->valid[i]-=1;
    classifyTrainingData ( t, second, d );
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]-=1;
    for ( i=0; i<d->nex; i++ )
        t->valid[i]+=1;
}

void classifyOOBData(tree_t* t, node_t* root, dataset_t* d){
    int i,k,l,u;
    node_t* first;
    node_t* second;
    evpair_t* b;

    /* Classify all out of bag points here */
    if ( root->split < 0 ){
        float pred=root->pos/(root->pos+root->neg);
        for(i=0; i<d->nex; i++){
            if(t->valid[i] > 0 && d->weight[i]<=0)
                t->pred[i] = pred;
        }
        return;
    }

    /*The rest is similar to the recursive tree growing procedure 
    See the comments there for an explanation.
    */

    b = d->feature[root->split];
    k = 0;
    u = d->size[root->split];
    while (k < u) {
        i = (k + u)/2;
        if (b[i].value > root->threshold)
            u = i;
        else
            k = i + 1;
    }
    if ( root->threshold > 0 ){
        l=k;
        u=d->size[root->split];
        first = root->left;
        second = root->right;
    }
    else{
        l=0;
        u=k;
        first = root->right;
        second = root->left;
    }
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]-=1;
    classifyOOBData ( t, first, d );
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]+=2;
    for ( i=0; i<d->nex; i++ )
        t->valid[i]-=1;
    classifyOOBData ( t, second, d );
    for ( i=l; i<u; i++ )
        t->valid[b[i].example]-=1;
    for ( i=0; i<d->nex; i++ )
        t->valid[i]+=1;
}


void freeTree(node_t* t){
    if(t->split < 0){
        free(t);
    }
    else{
        freeTree(t->left);
        freeTree(t->right);
        free(t);
    }
}

void writerec(FILE* fp, node_t* root){
    if(root->split >= 0){
        fprintf(fp,"%d %g ",root->split, root->threshold);
        writerec(fp,root->left);
        writerec(fp,root->right);
    }
    else{
        fprintf(fp,"%d %g %g ",root->split, root->pos, root->neg);
    }
}

void writeTree(FILE* fp, node_t* t){
    writerec(fp, t);
    fprintf(fp,"\n");
}


node_t* readrec(FILE* fp){
    node_t* root = malloc(sizeof(node_t));
    if(fscanf(fp,"%d",&root->split)==EOF){
        fprintf(stderr,"corrupt input file\n");
        exit(1);
    }
    if(root->split >= 0){
        fscanf(fp,"%g",&(root->threshold));
        root->left=readrec(fp);
        root->right=readrec(fp);
    }
    else{
        fscanf(fp,"%g%g",&root->pos, &root->neg);
        root->left=root->right=NULL;
    }
    return root;
}

void readTree(FILE* fp, node_t** t){
    *t=readrec(fp);
}
