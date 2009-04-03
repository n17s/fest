/***************************************************************************
 * Author: Nikos Karampatziakis <nk@cs.cornell.edu>, Copyright (C) 2008    *
 *                                                                         *
 * Description: Functions to read a dataset into memory.                   *
 *                                                                         *
 * License: See LICENSE file that comes with this distribution             *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "dataset.h"

#define BUFSZ 4096

void isort(evpair_t* a, int* f, int n){
    int i,j;
    float tv;
    int te;
    for(i=1; i<n; i++){
        for(j=i; j>0 && (f[j-1] > f[j] || (f[j-1] == f[j] && a[j-1].value > a[j].value)); j--){
            te=f[j];         f[j]=f[j-1];                 f[j-1]=te;
            te=a[j].example; a[j].example=a[j-1].example; a[j-1].example=te;
            tv=a[j].value;   a[j].value=a[j-1].value;     a[j-1].value=tv;
        }
    }
}

void qsortlazy(evpair_t* a, int* f, int l, int u){
    int i,j,r;
    float sv,tv;
    int se,te;
    if (u-l<7)
        return;
    r=l+rand()%(u-l);
    te=a[r].example; a[r].example=a[l].example; a[l].example=te;
    tv=a[r].value;   a[r].value=a[l].value;     a[l].value=tv;
    te=f[r];         f[r]=f[l];                 f[l]=te;
    i=l;
    j=u+1;
    while(1){
        do i++; while (i<=u && (f[i] < te || (f[i]==te && a[i].value < tv)));
        do j--; while (f[j] > te || (f[j]==te && a[j].value > tv));
        if (i>j)
            break;
        se=f[i];         f[i]=f[j];                 f[j]=se;
        se=a[i].example; a[i].example=a[j].example; a[j].example=se;
        sv=a[i].value;   a[i].value=a[j].value;     a[j].value=sv;
    }
    te=a[l].example; a[l].example=a[j].example; a[j].example=te;
    tv=a[l].value;   a[l].value=a[j].value;     a[j].value=tv;
    te=f[l];         f[l]=f[j];                 f[j]=te;
    qsortlazy(a,f,l,j-1);
    qsortlazy(a,f,j+1,u);
}

void sort(evpair_t* a, int* f, int len){
    qsortlazy(a,f,0,len-1);
    isort(a,f,len);
}

int getDimensions(FILE* fp, int* examples, int* totalfeatures){
    char buf[BUFSZ];
    int i,buflen,previous,total,max,len,example,inside;

    previous=-1;
    total=0;
    max=0;
    example=0;
    inside=0;
    rewind(fp);

    *examples = 0;
    *totalfeatures = 0;

    while((buflen=fread(buf,sizeof(char),BUFSZ,fp))!=0){
        for(i=0; i<buflen; i++,total++){
            switch(buf[i]){
                case ':':
                    if(inside)
                        continue;
                    *totalfeatures += 1;
                    break;
                case '#':
                    inside=1;
                    break;
                case '\n':
                    if(example) *examples+=1;
                    inside=0;
                    example=0;
                    len=total-previous;
                    previous=total;
                    if(max<len) max=len;
                    break;
                case '0':
                case '1':
                    if(inside)
                        continue;
                    example=1;
                    break;
                default:
                    break;
            }
        }
    }

    rewind(fp);
    return max+4; /* Just in case I was sloppy */
}

int readExamples(FILE* fp, int maxline, evpair_t* em, int* fm, int* trg){
    int i,target,example;
    float val;
    char* line;
    char* c;
    char* v;
    char* p;

    line=malloc(maxline*sizeof(char));
    rewind(fp);
    example=0;
    i=0;
    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        c=strchr(line,'#');
        if(c!=NULL)
            *c = '\0';
        c = strtok(line," \t");
        if(c==NULL)
            /* The line was a comment */
            continue;
        target = strtol(c,&p,10);
        trg[example] = target <=0 ? 0 : 1;
        while((c = strtok(NULL,":"))){
            v = strtok(NULL," \t");
            if(!v)
                break;
            val = strtod(v,&p);
            /* We don't want to store any zeros even if they appear explicitly in the input. 
             * Storing zero values will bite us later becaus of counting tricks etc. */
            if(val==0)
                continue;
            fm[i] = strtol(c,&p,10);
            em[i].example=example;
            em[i].value=val;
            i+=1;
        }
        example+=1;
    }
    free(line);
    return i;
}

int readExample(FILE* fp, int maxline, float* example, int nfeat, int* target){
    int offset,feat,len;
    float val;
    char* line;
    char* comment;

    line=malloc(maxline*sizeof(char));
    memset(example,0,nfeat*sizeof(float));

    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",target,&len)==EOF)
            /* The line was a comment */
            continue;
        *target = *target <=0 ? 0 : 1;
        for(offset=len; sscanf(line+offset,"%d:%f%n",&feat,&val,&len)>=2; offset+=len){
            /* Throw away features that do not exist in the tree */
            if (feat < nfeat)
                example[feat]=val;
        }
        free(line);
        return 1;
    }
    free(line);
    return 0;
}

void loadData(const char* name, dataset_t* d){
    FILE* fp;
    int total,i,maxline,sum;
    evpair_t* em;
    int* fm;

    fp=fopen(name,"r");
    if(fp==NULL){
        printf("Could not open file %s\n",name);
        exit(1);
    }
    maxline=getDimensions(fp, &d->nex, &total);

    d->target=malloc(d->nex*sizeof(int));
    d->oobvotes=calloc(d->nex,sizeof(int));
    d->weight=malloc(d->nex*sizeof(float));

    em=malloc(total*sizeof(evpair_t));
    fm=malloc(total*sizeof(int));

    total=readExamples(fp, maxline, em, fm, d->target);
    sort(em,fm,total);

    /* This is because the array of features is starting from 0 */
    d->nfeat=fm[total-1]+1;

    d->size=calloc(d->nfeat,sizeof(int));
    d->cont=calloc(d->nfeat,sizeof(int));

    for(i=0; i<total; i++){
        d->size[fm[i]]+=1;
        if(em[i].value!=1)
            d->cont[fm[i]]=1;
    }
    free(fm);
    
    d->feature=malloc(d->nfeat*sizeof(evpair_t*));
    d->feature[0]=em;

    sum=0;
    for(i=1; i<d->nfeat; i++){
        sum+=d->size[i-1];
        d->feature[i]=d->feature[0]+sum;
    }
    fclose(fp);
}

void freeData(dataset_t* d){  
    free(d->size);
    free(d->cont);
    free(d->target);
    free(d->oobvotes);
    free(d->weight);
    free(d->feature[0]);
    free(d->feature);
}
