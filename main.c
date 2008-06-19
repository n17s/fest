#include "dataset.h"
#include "tree.h"
#include "forest.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char* argv[]){
    dataset_t d;
    forest_t f;

    srand(43);

    loadData("tis.train",&d);
    initForest(&f);
    growForest(&f, &d);
    writeForest(&f, "model.out");

    freeForest(&f);
    freeData(&d);
    return 0;
}
