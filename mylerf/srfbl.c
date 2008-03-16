#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
  CONFIGURATION
 **/
// maximum depth of the tree
#define MAX_DEPTH 500
// maximum number of allowed features
#define MAX_NUM_FEAT 1000000
// negative class label (default is -1 for SVM-Light formatted files)
#define NEGATIVE -1
// either use log2 (exact) or approxLn2 (approximate)
#define LOG2 log

/**
  STRUCTURES
 **/
struct test_node {
	int nexmpls;
	int *exmpls;
	int npos;
	int nneg;

	int test;
	int nchildren;
	struct test_node *children;
};
typedef struct test_node test;

#define error(msg) { printf("%s at line %d.\n", msg, __LINE__); exit(-1); }
#define TRUE 1
#define FALSE 0

// variables to be used throughout
char *train_fn, *test_fn;
int num_trees = 500;
int num_threads = 0;
int num_feat = -1, num_feat_tpn = -1;
int num_train = 0;
int *feat_not_zero_c;
int *targets;
int *raw_data;
int **data;
int temp;
int offset_trees=-1;

float *predictions;

// function prototypes
void *RF();
void sort(int *a, int na);
int int_union(int *c, int *a, int na, int *b, int nb);
void setCounts(test *t);
void generateTree(test *t);
void _generateTree(test *t, int *used_feat, int depth, int high);
int randomSubset(int min, int max, int *ss, int ss_size, int *exclude, int nexclude);
void doSplit(test *t);
float calcScore(test *t);
float entropy(int classes, int *counts, int sum);
void printAndFreeTree(test *t, int doPrint, int doFree);
void _printAndFreeTree(test *t, int indent, int doPrint, int doFree);
float runTree(test *t, int *features);
void saveRecursive(test* root, FILE* fp);
void saveModel(test* root, const char* name);


/**
  LOG APPROXIMATION ( http://www.musicdsp.org/showone.php?id=91 )
 **/
//float approxLn2( float f ) {
//  assert( f > 0. );
//  assert( sizeof(f) == sizeof(int) );
//  assert( sizeof(f) == 4 );
//  int i = (*(int *)&f);
//  return (((i&0x7f800000)>>23)-0x7f)+(i&0x007fffff)/(float)0x800000;
//}

/**
  MAIN PROGRAM
 **/
int main(int argc, char **argv)
{
	int c;
	while ((c = getopt(argc, argv, "r:t:f:"))!=-1){
		switch (c){
			case 't':
				num_trees = atoi(optarg);
				break;
			case 'f':
				num_feat_tpn = atoi(optarg);
				break;
			case 'r':
				offset_trees = atoi(optarg);
				break;
			case '?':
				break;
			default:
				error("usage: sparse_RF [-t trees=500] [-f=sqrt(numfeat)] [train]\n");
		}
	}
	//printf("%d %d %s\n",optind,argc,argv[optind]);
	if  (optind >= argc)
		error("usage: sparse_RF [-t trees=500] [-f=sqrt(numfeat)] [train]\n");

	train_fn = argv[optind];

	/*********************/
	int i, j, *ptr;
	int feature, value, read_class, last_feature;

	// when reading training file, keep following tallies
	feat_not_zero_c = (int *)calloc(MAX_NUM_FEAT, sizeof(int)); // = num times feature is non-zero
	if (feat_not_zero_c == NULL) error("Unable to allocate memory at line %i.\n");
	int sum_feat_not_zero_c = 0; // = sum{i}(feat_not_zero_c)

	// training file
	FILE *train_fp = fopen(train_fn, "r");
	if (train_fp == NULL) error("Error reading training data at line %i.\n");

	// calculate sizes for training set
	read_class = TRUE;
	last_feature = -1;
	for (i = 1; !feof(train_fp); ) {
		char str[2];

		fscanf(train_fp, "%*[ \t]");

		if (fscanf(train_fp, "%1[\n\r]", &str[0]) == 1) {
			read_class = TRUE;
			++i;

			// newline
		} else if (read_class && fscanf(train_fp, "%i", &value) == 1) {
			read_class = FALSE;
			last_feature = -1;

			// read in class (into "value")
			++num_train;
		} else if (fscanf(train_fp, "%1[#]", &str[0]) == 1)  {
			while (fscanf(train_fp, "%1[^\n\r]", &str[0]) == 1); // read up until end-of-line
		} else if (fscanf(train_fp, "%i:%i", &feature, &value) == 2 && --feature > last_feature) {
			last_feature = feature;

			// read in feature:value pair
			if (feature > num_feat) // keep count of largest feature seen
				num_feat = feature;

			++feat_not_zero_c[feature];
			++sum_feat_not_zero_c;
		} else if (!feof(train_fp)) {
			char e[80];
			sprintf(e, "Error reading line %i of training data.\n", i);
			error(e);
		}
	}
	++num_feat;

	if (num_train == 0)
		error("Training set has no examples.\n");

	// default number of features to try per node is sqrt(# features)
	if (num_feat_tpn == -1)
		num_feat_tpn = (int)ceil(sqrt(num_feat));

	if (num_feat_tpn > num_feat) {
		char e[80];
		sprintf(e, "Features per node (%i) exceeds the number of features (%i).\n", num_feat_tpn, num_feat);
		error(e);
	}

	// allocate targets (int) array
	targets = (int *)malloc(num_train * sizeof(int *));
	if (targets == NULL) error("Unable to allocate memory at line %i.\n");

	// allocate raw data (int) array of size sum{i}(feat_not_zero_c[i])
	raw_data = (int *)calloc(sum_feat_not_zero_c, sizeof(int));
	if (raw_data == NULL) error("Unable to allocate memory at line %i.\n");

	// establish feature boundaries in memory
	data = (int **)malloc(num_feat * sizeof(int *));
	if (data == NULL) error("Unable to allocate memory at line %i.\n");
	ptr = raw_data;
	for (i = 0;	i < num_feat; ++i) {
		data[i] = ptr;
		ptr += feat_not_zero_c[i];
	}

	// load data into above (data) array
	rewind(train_fp);
	int *indices = (int *)calloc(num_feat, sizeof(int));
	if (indices == NULL) error("Unable to allocate memory at line %i.\n");
	read_class = TRUE;
	last_feature = -1;
	for (i = 1, j = -1; !feof(train_fp); ) {
		char str[2];

		fscanf(train_fp, "%*[ \t]");

		if (fscanf(train_fp, "%1[\n\r]", &str[0]) == 1) {
			read_class = TRUE;
			++i;

			// newline
		} else if (read_class && fscanf(train_fp, "%i", &value) == 1) {
			read_class = FALSE;
			last_feature = -1;

			// read in class (into "value")
			targets[++j] = value;
		} else if (fscanf(train_fp, "%1[#]", &str[0]) == 1)  {
			// read comment
			while (fscanf(train_fp, "%1[^\n\r]", &str[0]) == 1); // read up until end-of-line
		} else if (fscanf(train_fp, "%i:%i", &feature, &value) == 2 && --feature > last_feature) {
			last_feature = feature;

			// read in feature:value pair
			data[feature][indices[feature]++] = j; // index by feature, and record ex # (i)
		} else if (!feof(train_fp)) {
			char e[80];
			sprintf(e, "Error reading line %i of training data.\n", i);
			error(e);
		}
	}
	RF();

	free(indices);
	free(data);
	free(raw_data);
	free(targets);
	fclose(train_fp);
	free(feat_not_zero_c);
	return 0;
}

void *RF()
{
	int i;
	// setup variables used in all trees
	test *root;
	int *features = (int *)malloc(num_feat * sizeof(int));
	if (features == NULL) error("Unable to allocate memory at line %i.\n");

	int tree;
	// initialize random seed
	srand(time(NULL));

	for (tree = offset_trees+1; tree < num_trees; ++tree) {

		// initialize first (root) test node
		root = (test *)malloc(sizeof(test)); // root test
		if (root == NULL) error("Unable to allocate memory at line %i.\n");
		root->nexmpls = num_train;
		root->exmpls = (int *)calloc(num_train, sizeof(int));
		if (root->exmpls == NULL) error("Unable to allocate memory at line %i.\n");
		for (i = 0; i < num_train; ++i) {
			root->exmpls[i] = rand() % num_train;
		}
		sort(root->exmpls, num_train);
		setCounts(root);

		// generate tree
		generateTree(root);

		char buf[128];
		sprintf(buf,"tree.%d",tree);
		saveModel(root, buf);
		printf("%s written\n",buf);
		printAndFreeTree(root, 0, 1); // don't print, just free
		free(root);
	}

	free(features);

	return NULL;
}

// perform merge sort on int array a of size na
void sort(int *a, int na)
{
	if (na < 2)
		return;

	int *c = (int *)calloc(na * 2, sizeof(int));
	int mid, i;

	mid = na / 2;
	sort(a, mid);
	sort(a + mid, na - mid);
	int_union(c, a, mid, a + mid, na - mid);
	for (i = 0; i < na; ++i) {
		a[i] = c[i];
	}

	free(c);
}

// calculate the sorted union of the two sorted lists a and b of sizes na and nb
int int_union(int *c, int *a, int na, int *b, int nb)
{
	int ai = 0, bi = 0, ci = 0;
	while (ai < na && bi < nb) {
		if (a[ai] < b[bi])
			c[ci++] = a[ai++];
		else
			c[ci++] = b[bi++];
	}
	while (ai < na)
		c[ci++] = a[ai++];
	while (bi < nb)
		c[ci++] = b[bi++];

	return ci;
}

// sets the number of positive/negative cases for test t
void setCounts(test *t)
{
	int npos = 0, nneg = 0;
	int i;
	for (i = 0; i < t->nexmpls; ++i) {
		if (targets[t->exmpls[i]] == NEGATIVE)
			nneg++;
		else
			npos++;
	}
	t->npos = npos;
	t->nneg = nneg;
}

// recursively generate tree
void generateTree(test *t)
{
	int *used_feat = (int *)calloc(num_feat, sizeof(int));
	if (used_feat == NULL) error("Unable to allocate memory at line %i.\n");

	// don't use features if they have never been non-default
	int i, j;
	for (i = 0, j = 0; i < num_feat; ++i) {
		if (feat_not_zero_c[i] == 0)
			used_feat[j++] = i;
	}

	_generateTree(t, used_feat, j, 0);

	free(used_feat);
}
void _generateTree(test *t, int *used_feat, int depth, int high)
{
	int i, j;

	int best_split = -1;
	float best_score = INT_MIN;
	int *feats,feats_size;
	
	
	// don't split if too deep
        if (high >= MAX_DEPTH) {
                t->test = -1;
                t->nchildren = 0;
                return;
        }

	feats = (int *)calloc(num_feat_tpn, sizeof(int));
	if (feats == NULL) error("Unable to allocate memory at line %i.\n");
	feats_size = randomSubset(0, num_feat - 1, feats, num_feat_tpn, used_feat, depth);
	for (i = 0; i < feats_size; ++i) {
		t->test = feats[i];
		doSplit(t);

		// calculate score for feature
		float score = calcScore(t);
		if (score >= best_score) {
			best_score = score;
			best_split = feats[i];
		}

		// free split from memory
		for (j = 0; j < t->nchildren; ++j) {
			printAndFreeTree(&(t->children[j]), 0, 1);
		}
		free(t->children);
		t->children=0;
	}
	free(feats);

	// don't split on scores <= 0
	if (best_score <= 0) {
		t->test = -1;
		t->nchildren = 0;
		return;
	}

	// now that we've found the best split, use it
	t->test = best_split;
	doSplit(t);
	used_feat[depth] = best_split;
	for (i = 0; i < t->nchildren; ++i) {
		if (t->children[i].npos == 0 || t->children[i].nneg == 0)
			continue;

		_generateTree(&(t->children[i]), used_feat, depth + 1, high + 1);
	}
	used_feat[depth] = 0;
}

// generate random subset of size "size" from the range min, max
int randomSubset(int min, int max, int *ss, int ss_size, int *exclude, int nexclude)
{
	int set_size = (max - min) + 1;

	if (min > max || ss_size > set_size)
		return 0;

	// setup i(ncluded)set, set, and subset arrays
	int *set = (int *)malloc(set_size * sizeof(int));
	if (set == NULL) error("Unable to allocate memory at line %i.\n");

	// generate set with everything included
	int i;
	for (i = 0; i < set_size; ++i) {
		set[i] = min + i;
	}

	// remove excluded values

	// pick elements from set and put them in ss
	for (i = 0; i < ss_size; ++i) {
		int r = rand() % set_size;

		ss[i] = set[r];
		set[r] = set[set_size - 1];
		--set_size;
	}

	free(set);

	return ss_size;
}

// perform split on test t, assuming counts already set for t
void doSplit(test *t)
{
	t->nchildren = 2;
	t->children = (test *)calloc(t->nchildren, sizeof(test));
	if (t->children == NULL) error("Unable to allocate memory at line %i.\n");

	// by default assume all children are leaf nodes
	t->children[0].test = -1;
	t->children[1].test = -1;

	// initialize example lists for children nodes
	int sz = feat_not_zero_c[t->test];
	t->children[0].exmpls = (int *)calloc(t->nexmpls, sizeof(int));
	if (t->children[0].exmpls == NULL) error("Unable to allocate memory at line %i.\n");
	t->children[1].exmpls = (int *)calloc(t->nexmpls, sizeof(int));
	if (t->children[1].exmpls == NULL) error("Unable to allocate memory at line %i.\n");

	// split examples among children nodes
	int i = 0, j = 0;
	while (i < t->nexmpls && j < sz) {
		if (t->exmpls[i] == data[t->test][j]) { // found it
			t->children[1].exmpls[t->children[1].nexmpls++] = t->exmpls[i++];
			++j;
		} else if (t->exmpls[i] < data[t->test][j]) { // it's not there
			t->children[0].exmpls[t->children[0].nexmpls++] = t->exmpls[i++];
		} else
			++j;
	}
	while (i < t->nexmpls)
		t->children[0].exmpls[t->children[0].nexmpls++] = t->exmpls[i++];

	assert(t->children[0].nexmpls == (t->nexmpls - t->children[1].nexmpls));

	// calculate positive/negative counts for children nodes
	setCounts(&(t->children[0]));
	setCounts(&(t->children[1]));
}

// calculate and return IG/GR on t.npos/nneg + t.children[i].npos/nneg
float calcScore(test *t)
{
	int classes = 2;
	int counts[] = {t->npos, t->nneg};

	int sum = t->nexmpls;

	assert(sum == t->npos + t->nneg);

	float e = entropy(classes, counts, sum);
	int i;
	for (i = 0; i < t->nchildren; ++i) {
		counts[0] = t->children[i].npos;
		counts[1] = t->children[i].nneg;

		if ((sum = t->children[i].nexmpls) == 0)
			continue;

		e -= sum * entropy(classes, counts, sum) / t->nexmpls;
	}

	return e;
}

// calculate entropy for "classes" number of classes with counts "counts" that add to "sum"
float entropy(int classes, int *counts, int sum)
{
	float e = 0.0, p;
	int i;
	for (i = 0; i < classes; ++i) {
		if ((p = counts[i] / (float)sum) == 0)
			continue;

		e -= p * LOG2(p);
	}

	return e;
}

// recursively prints and frees the tree from memory
void printAndFreeTree(test *t, int doPrint, int doFree)
{
	_printAndFreeTree(t, 0, doPrint, doFree);
}
void _printAndFreeTree(test *t, int indent, int doPrint, int doFree)
{
	if (doPrint) {
		int i;
		for (i = 0; i < indent; ++i)
			printf("\t");

		printf("%i/%i/%i\n", t->test, t->npos, t->nneg);
	}

	if (doFree){
		free(t->exmpls);
		t->exmpls=0;
	}

	if (t->nchildren == 0)
		return;

	_printAndFreeTree(&(t->children[0]), indent + 1, doPrint, doFree);
	_printAndFreeTree(&(t->children[1]), indent + 1, doPrint, doFree);

	if (doFree){
		free(t->children);
		t->children=0;
	}
}

// returns the prediction of tree with root t on the case corresponding to features
float runTree(test *t, int *features)
{
	if (t->test == -1) {
		if (t->npos == 0)
			return 0;
		else if (t->nneg == 0)
			return 1;
		else
			return (t->npos / (float)(t->npos + t->nneg));
	}

	return runTree(&(t->children[features[t->test]]), features);
}

void saveRecursive(test* root, FILE* fp){
	int i;
	if(root->test>=0){
		fprintf(fp,"%d %d\n",root->test,root->nchildren);
		for(i=0; i<root->nchildren; i++)
			saveRecursive(&(root->children[i]), fp);
	}
	else
		fprintf(fp,"%d %d %d %d\n",root->test,root->nchildren,root->npos,root->nneg);
}

void saveModel(test* root, const char* name){
	FILE *fp=fopen(name,"w");
	fprintf(fp,"%d\n",num_feat);
	saveRecursive(root,fp);
	fclose(fp);
}
