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
#include "dataset.h"

void isort(evpair_t* a, int n){
    int i,j;
    float tv;
    int te;
    for(i=1; i<n; i++){
        for(j=i; j>0 && a[(j-1)].value > a[j].value; j--){
            te=a[j].example; a[j].example=a[j-1].example; a[j-1].example=te;
            tv=a[j].value;   a[j].value=a[j-1].value;     a[j-1].value=tv;
        }
    }
}

void qsortlazy(evpair_t* a, int l, int u){
    int i,j,r;
    float sv,tv;
    int se,te;
    if (u-l<7)
        return;
    r=l+rand()%(u-l);
    te=a[r].example; a[r].example=a[l].example; a[l].example=te;
    tv=a[r].value;   a[r].value=a[l].value;     a[l].value=tv;
    i=l;
    j=u+1;
    while(1){
        do i++; while (i<=u && a[i].value < tv);
        do j--; while (a[j].value > tv);
        if (i>j)
            break;
        se=a[i].example; a[i].example=a[j].example; a[j].example=se;
        sv=a[i].value;   a[i].value=a[j].value;     a[j].value=sv;
    }
    te=a[l].example; a[l].example=a[j].example; a[j].example=te;
    tv=a[l].value;   a[l].value=a[j].value;     a[j].value=tv;
    qsortlazy(a,l,j-1);
    qsortlazy(a,j+1,u);
}

void sort(evpair_t* a, int len){
    qsortlazy(a,0,len-1);
    isort(a,len);
}

int getDimensions(FILE* fp, int* examples, int* features){
    char buf[4096];
    int i,buflen,previous,total,max,target,len,lastfeature;
    char* line;
    char *comment,*colon,*space,*tab;

    previous=-1;
    total=0;
    max=0;
    /* find maximum line length */
    rewind(fp);
    while((buflen=fread(buf,sizeof(char),4096,fp))!=0){
        for(i=0; i<buflen; i++,total++){
            if(buf[i]=='\n'){
                len=total-previous;
                previous=total;
                if(max<len) max=len;
            }
        }
    }

    max+=4; /* Just in case I was sloppy */
    line=malloc(max*sizeof(char));

    rewind(fp);
    *examples = 0;
    *features = 0;
    while(fgets(line,max,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d",&target)==EOF)
            /* The line was a comment */
            continue;
        *examples += 1;
        colon=strrchr(line,':');
        if(colon==NULL) /* This can happen when the zero vector is in the data */
            lastfeature=1;
        else{
            *colon = '\0';
            space = strrchr(line,' ');
            space = space == NULL ? line : space;
            tab = strrchr(space,'\t');
            tab = tab == NULL ? space : tab;
            sscanf(tab,"%d",&lastfeature);
            if(*features<lastfeature)
                *features=lastfeature;
        }
    }
    rewind(fp);
    /* This is because the array of features is starting from 0 */
    *features += 1;
    free(line);
    return max;
}

int getSizes(FILE* fp, int maxline, int* size){
    int target,offset,feat,len,total;
    float val;
    char* line;
    char* comment;

    total=0;
    line=malloc(maxline*sizeof(char));
    rewind(fp);

    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",&target,&len)==EOF)
            /* The line was a comment */
            continue;
        for(offset=len; sscanf(line+offset,"%d:%f%n",&feat,&val,&len)>=2; offset+=len){
            /* First we don't want to spend any space for zeros even if they appear
             * explicitly in the input. Storing zero values will bite us later because
             * of counting tricks etc. */
            if(val==0)
                continue;
            size[feat]+=1;
            total+=1;
        }
    }

    free(line);
    return total;
}

void readExamples(FILE* fp, int maxline, dataset_t* d){
    int target,offset,feat,len,example;
    int* cur;
    float val;
    char* line;
    char* comment;

    line=malloc(maxline*sizeof(char));
    cur=calloc(d->nfeat,sizeof(int));
    rewind(fp);
    example=0;
    while(fgets(line,maxline,fp)!=NULL){
        /* remove comments */
        comment=strchr(line,'#');
        if(comment!=NULL)
            *comment = '\0';
        if(sscanf(line,"%d%n",&target,&len)==EOF)
            /* The line was a comment */
            continue;
        d->target[example] = target <=0 ? 0 : 1;
        for(offset=len; sscanf(line+offset,"%d:%f%n",&feat,&val,&len)>=2; offset+=len){
            if(val==0)
                continue;
            d->feature[feat][cur[feat]].example=example;
            d->feature[feat][cur[feat]].value=val;
            cur[feat]+=1;
            if(val!=1 && val!=0)
                d->cont[feat]=1;
        }
        example+=1;
    }
    free(cur);
    free(line);
}

void sortContinuous(dataset_t* d){
    int i;
    for(i=0; i<d->nfeat; i++){
        if(d->cont[i]){
            sort(d->feature[i],d->size[i]);
        }
    }
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

    fp=fopen(name,"r");
    if(fp==NULL){
        printf("Could not open file %s\n",name);
        exit(1);
    }
    maxline=getDimensions(fp, &d->nex, &d->nfeat);
    d->size=calloc(d->nfeat,sizeof(int));
    d->cont=calloc(d->nfeat,sizeof(int));
    d->target=malloc(d->nex*sizeof(int));
    d->oobvotes=calloc(d->nex,sizeof(int));
    d->weight=malloc(d->nex*sizeof(float));
    total=getSizes(fp, maxline, d->size);
    d->feature=malloc(d->nfeat*sizeof(evpair_t*));
    d->feature[0]=malloc(total*sizeof(evpair_t));

    sum=0;
    for(i=1; i<d->nfeat; i++){
        sum+=d->size[i-1];
        d->feature[i]=d->feature[0]+sum;
    }
    /* Finally memory has been set up and we can read the data */
    readExamples(fp, maxline, d);
    fclose(fp);
    sortContinuous(d);
}

void freeData(dataset_t* d){  
    free(d->size);
    free(d->cont);
    free(d->target);
    free(d->weight);
    free(d->feature[0]);
    free(d->feature);
}
