#include "dataset.h"
#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    float* example;
    int maxline,target,nf,nex;
    float p;
    dataset_t d;
    forest_t f;
    FILE* fp;

    srand(43);
    loadData("tis.train",&d);
    randomForest(&f, &d);

    fp = fopen("tis.test","r");
    maxline = getDimensions(fp,&nex,&nf);
    rewind(fp);
    example=malloc(d.nfeat*sizeof(float));
    while(readExample(fp, maxline, example, d.nfeat, &target)){
        p=rfclassify(&f,example);
        printf("%d %f\n",(target+1)/2,p);
    }
    fclose(fp);
    rfrelease(&f);
    freeData(&d);
    free(example);

    return 0;
}
