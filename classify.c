#include "dataset.h"
#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    float* example;
    int maxline,target,nf,nex;
    float p;
    forest_t f;
    FILE* fp;

    readForest(&f, "model.out");
    fp = fopen("tis.test","r");
    maxline = getDimensions(fp,&nex,&nf);
    rewind(fp);
    example=malloc(f.nfeat*sizeof(float));
    while(readExample(fp, maxline, example, f.nfeat, &target)){
        p=classifyForest(&f,example);
        printf("%d %f\n",(target+1)/2,p);
    }
    free(example);
    fclose(fp);
    freeForest(&f);
    return 0;
}
