#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/**
 CONFIGURATION
 **/

// maximum depth of the tree 
#define MAX_DEPTH 500
// maximum number of allowed features
#define MAX_NUM_FEAT 1000000
// default value for sparse matrices (default is 0)
#define DEFAULT 0
// negative/positive class labels (default is -1 for SVM-Light formatted files)
// note: might not work for anything other than -1 for negatives and 1 for positives
#define NEGATIVE -1
#define POSITIVE  1
// either use log2 (exact) or approxLn2 (approximate)
#define LOG2 log

/**
 STRUCTURES
 **/

// change to something like http://vergil.chemistry.gatech.edu/resources/programming/c-tutorial/structs.html
typedef struct {
	int i;
	float v;
} exmpl;
struct test_node {
	int nexmpls;
	int *exmpls;
	int npos;
	int nneg;

	int test;
	float threshold; // if less than or equal then left else right
	int nchildren;
	struct test_node *children;
};
typedef struct test_node test;
typedef struct {
	float v;
	int npos;
	int nneg;
} threshold;

#define error(msg) { printf("%s at line %d.\n", msg, __LINE__); exit(-1); }
#define TRUE 1
#define FALSE 0

// variables to be used throughout
char *train_fn, *test_fn;
int num_trees = 500;
int num_threads = 0;
int num_feat = -1, num_feat_tpn = -1;
int num_train = 0;
int *feat_not_default_c;
int *targets;
exmpl *raw_data;
exmpl **data;
int temp;
int offset_trees=-1;
// predictions storage
float *predictions;

// function prototypes
void *RF();
void sorti(int *a, int na);
int unioni(int *c, int *a, int na, int *b, int nb);
void setCounts(test *t);
void generateTree(test *t,int d);
int randomSubset(int min, int max, int *ss, int ss_size);
float doSplit(test *t);
void sortt(threshold *a, int na);
int uniont(threshold *c, threshold *a, int na, threshold *b, int nb);
float calcScore(test *t, float entropy);
float entropy2(int class1, int class2, int sum);
void printAndFreeTree(test *t, int doPrint, int doFree);
void _printAndFreeTree(test *t, int indent, int doPrint, int doFree);
float runTree(test *t, float *features);
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
    printf("%d %d %s\n",optind,argc,argv[optind]);
    if  (optind >= argc)
        error("usage: sparse_RF [-t trees=500] [-f=sqrt(numfeat)] [train]\n");

	train_fn = argv[optind];

	int i, j;
	int feature, read_class, last_feature;
	float value;

	// when reading training file, keep following tallies
	feat_not_default_c = calloc(MAX_NUM_FEAT, sizeof *feat_not_default_c); // = num times feature is non-zero
	if (feat_not_default_c == NULL) error("Unable to allocate memory at line %i.\n");
	int sum_feat_not_default_c = 0; // = sum{i}(feat_not_default_c)

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
		} else if (read_class && fscanf(train_fp, "%i", &feature) == 1) {
			read_class = FALSE;
			last_feature = -1;

			// read in class (into "feature")
			++num_train;
		} else if (fscanf(train_fp, "%1[#]", &str[0]) == 1)  {
			while (fscanf(train_fp, "%1[^\n\r]", &str[0]) == 1); // read up until end-of-line
		} else if (fscanf(train_fp, "%i:%f", &feature, &value) == 2 && --feature > last_feature) {
			last_feature = feature;

			// read in feature:value pair
			if (feature > num_feat) // keep count of largest feature seen
				num_feat = feature;

			++feat_not_default_c[feature];
			++sum_feat_not_default_c;
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
	targets = malloc(num_train * sizeof *targets);
	if (targets == NULL) error("Unable to allocate memory at line %i.\n");

	// allocate raw data (exmpl) array of size sum{i}(feat_not_default_c[i])
	raw_data = malloc(sum_feat_not_default_c * sizeof *raw_data);
	if (raw_data == NULL) error("Unable to allocate memory at line %i.\n");

	// establish feature boundaries in memory
	data = malloc(num_feat * sizeof *data);
	if (data == NULL) error("Unable to allocate memory at line %i.\n");
	exmpl *boundary = raw_data;
	for (i = 0;	i < num_feat; ++i) {
		data[i] = boundary;
		boundary += feat_not_default_c[i];
	}

	// load data into above (data) array
	rewind(train_fp);
	int *indices = calloc(num_feat, sizeof *indices);
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
		} else if (read_class && fscanf(train_fp, "%i", &feature) == 1) {
			read_class = FALSE;
			last_feature = -1;

			// read in class (into "feature")
			targets[++j] = feature;
		} else if (fscanf(train_fp, "%1[#]", &str[0]) == 1)  {
			// read comment
			while (fscanf(train_fp, "%1[^\n\r]", &str[0]) == 1); // read up until end-of-line
		} else if (fscanf(train_fp, "%i:%f", &feature, &value) == 2 && --feature > last_feature) {
			last_feature = feature;

			// read in feature:value pair
			data[feature][indices[feature]].i = j; // index by feature, and record ex # (j)
			data[feature][indices[feature]++].v = value;
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
	free(feat_not_default_c);
    return 0;
}

void *RF()
{
	int i;
	// holds data for example currently being tested
	float *example = malloc(num_feat * sizeof *example);
	if (example == NULL) error("Unable to allocate memory at line %i.\n");

	// initialize root test node (similar accross all trees)
	test *root = malloc(sizeof *root);
	if (root == NULL) error("Unable to allocate memory at line %i.\n");
	root->exmpls = malloc(num_train * sizeof *root->exmpls);
	if (root->exmpls == NULL) error("Unable to allocate memory at line %i.\n");
	root->nexmpls = num_train;

	// initialize random seed
	srand(time(NULL));

	int tree;
	for (tree = offset_trees+1; tree < num_trees; ++tree) {
		// generate bootstrap training set
		for (i = 0; i < num_train; ++i) {
			root->exmpls[i] = rand() % num_train;
		}
		sorti(root->exmpls, num_train);
		setCounts(root);

		// generate tree
		generateTree(root,0);

		char buf[128];
		sprintf(buf,"tree.%d",tree);
		saveModel(root, buf);
		printf("%s written\n",buf);
		printAndFreeTree(root, 0, 1); // don't print, just free
		// don't free root->exmpls
	}

	free(root->exmpls);
	free(root);
	free(example);

	return NULL;
}

// perform merge sort on int array a of size na
void sorti(int *a, int na)
{
	if (na < 2)
		return;

	int i, mid, *c = malloc(na * 2 * sizeof *c);
	if (c == NULL) error("Unable to allocate memory at line %i.\n");

	mid = na / 2;
	sorti(a, mid);
	sorti(a + mid, na - mid);
	unioni(c, a, mid, a + mid, na - mid);
	for (i = 0; i < na; ++i) {
		a[i] = c[i];
	}

	free(c);
}

// calculate the sorted union of the two sorted int arrays a and b of sizes na and nb
int unioni(int *c, int *a, int na, int *b, int nb)
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
			++nneg;
		else
			++npos;
	}
	t->npos = npos;
	t->nneg = nneg;
}

// recursively generate tree
void generateTree(test *t, int depth)
{
	int i, j;

	int *feats = malloc(num_feat_tpn * sizeof *feats);
	if (feats == NULL) error("Unable to allocate memory at line %i.\n");

	// don't split if too deep 
        if (depth >= MAX_DEPTH ) {
                t->test = -1;
                t->nchildren = 0;
                return;
        }

	float best_score = INT_MIN;
	int best_test = -1;
	float best_threshold = 0;
	int feats_size = randomSubset(0, num_feat - 1, feats, num_feat_tpn);

	for (i = 0; i < feats_size; ++i) {
		// split using test and find a good threshold
		t->test = feats[i];
		t->threshold = INT_MAX;

		// calculate score
		float score = doSplit(t);
		if (score >= best_score) {
			best_score = score;
			best_test = t->test;
			best_threshold = t->threshold;
		}

		// free test from memory
		for (j = 0; j < t->nchildren; ++j) {
			free(t->children[j].exmpls);
			t->children[j].exmpls=0;
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

	// now that we've found the best test and threshold, use them
	t->test = best_test;
	t->threshold = best_threshold;
	doSplit(t);
	for (i = 0; i < t->nchildren; ++i) {
		if (t->children[i].npos == 0 || t->children[i].nneg == 0)
			continue;

		generateTree(&(t->children[i]), depth+1);
	}
}

// generate random subset of size "size" from the range min, max
int randomSubset(int min, int max, int *ss, int ss_size)
{
	int set_size = (max - min) + 1;

	if (min > max || ss_size > set_size) {
		error("randomSubset error");
		return 0;
	}

	// setup set array
	int *set = malloc(set_size * sizeof *set);
	if (set == NULL) error("Unable to allocate memory at line %i.\n");

	// generate set with everything included
	int i;
	for (i = 0; i < set_size; ++i) {
		set[i] = min + i;
	}

	if (ss_size > set_size)
		ss_size = set_size;

	// pick elements from set and put them in ss
	for (i = 0; i < ss_size; ++i) {
		int r = rand() % set_size;

		ss[i] = set[r];
		set[r] = set[set_size-- - 1];
	}

	free(set);

	return ss_size;
}

// perform split on test t (with threshold), assuming counts already set
// if threshold is INT_MAX, a better threshold is found, and the score is returned
// otherwise, the split is done and INT_MIN is returned
// note: split is [ <= threshold ] and [ > threshold ]
float doSplit(test *t)
{
	assert(t->nexmpls > 0);

	int i = 0, j = 0;

	t->nchildren = 2;
	t->children = calloc(t->nchildren, sizeof(test));
	if (t->children == NULL) error("Unable to allocate memory at line %i.\n");

	// by default assume all children are leaf nodes
	t->children[0].test = -1;
	t->children[1].test = -1;

	// initialize example lists for children nodes
	t->children[0].exmpls = malloc(t->nexmpls * sizeof *t->children[0].exmpls);
	if (t->children[0].exmpls == NULL) error("Unable to allocate memory at line %i.\n");
	t->children[1].exmpls = malloc(t->nexmpls * sizeof *t->children[1].exmpls);
	if (t->children[1].exmpls == NULL) error("Unable to allocate memory at line %i.\n");

	float best_score = INT_MIN;

	if (t->threshold == INT_MAX) { // need to find a better threshold
		if (feat_not_default_c[t->test] == 0) // if only one (default) value, split has score 0
			t->threshold = DEFAULT;
		else { // at least one non-default value, so at least one good threshold
			threshold *thresholds = malloc((t->nexmpls + 1) * sizeof *thresholds); // + 1 for DEFAULT
			if (thresholds == NULL) error("Unable to allocate memory at line %i.\n");

			int nthresholds = 0;
			while (i < t->nexmpls && j < feat_not_default_c[t->test]) {
				if (t->exmpls[i] > data[t->test][j].i) { // don't know anything, keep searching
					++j;
					continue;
				}
				// else we know if it's there or if it's not there

				int npos = 0, nneg = 0;
				if (targets[t->exmpls[i]] == NEGATIVE)
					nneg = 1;
				else
					npos = 1;

				float v = DEFAULT;
				if (t->exmpls[i] == data[t->test][j].i) // found it
					v = data[t->test][j++].v;

				++i;

				// do a sorted insert/update of (v, npos, nneg)
				int k = 0;
				while (k < nthresholds && v > thresholds[k].v) ++k; // find k : v <= thresholds[k].v
				if (k != nthresholds && v == thresholds[k].v) { // found it
					thresholds[k].npos += npos;
					thresholds[k].nneg += nneg;
				} else { // v needs to be inserted
					if (k != nthresholds) { // insert into the middle
						// shift everything past here right 1
						memmove(&thresholds[k] + 1, &thresholds[k], (nthresholds - k) * sizeof *thresholds);
					}
					// else insert at the end

					// insert at position k
					++nthresholds;
					thresholds[k].v = v;
					thresholds[k].npos = npos;
					thresholds[k].nneg = nneg;
				}
			}

			// find or create DEFAULT threshold
			int k = 0;
			while (k < nthresholds && DEFAULT > thresholds[k].v) ++k; // find k : DEFAULT <= thresholds[k].v
			if (k != nthresholds && DEFAULT != thresholds[k].v) { // DEFAULT doesn't exist, create it
				if (k != nthresholds) { // insert into the middle
					// shift everything past here right 1
					memmove(&thresholds[k] + 1, &thresholds[k], (nthresholds - k) * sizeof *thresholds);
				}
				// else insert at the end

				// insert at position k
				++nthresholds;
				thresholds[k].v = DEFAULT;
				thresholds[k].npos = 0;
				thresholds[k].nneg = 0;
			}
			// deal with any remaining (DEFAULT) examples
			for (; i < t->nexmpls; ++i) {
				if (targets[t->exmpls[i]] == NEGATIVE)
					++thresholds[k].nneg;
				else
					++thresholds[k].npos;
			}

			--nthresholds; // the last threshold is meaningless

			t->threshold = thresholds[0].v;
			best_score = 0;

			if (nthresholds == 0) { // if only one value, make it the only threshold
				t->threshold = thresholds[0].v;
				best_score = 0;
			}
			else { // start looking for a good threshold
				t->children[0].npos = 0;
				t->children[0].nneg = 0;
				t->children[0].nexmpls = 0;
				t->children[1].npos = t->npos;
				t->children[1].nneg = t->nneg;
				t->children[1].nexmpls = t->nexmpls;

				float e = entropy2(t->npos, t->nneg, t->nexmpls);

				for (i = 0; i < nthresholds; ++i) {
					t->children[0].npos += thresholds[i].npos;
					t->children[0].nneg += thresholds[i].nneg;
					t->children[0].nexmpls += thresholds[i].npos + thresholds[i].nneg;
					t->children[1].npos -= thresholds[i].npos;
					t->children[1].nneg -= thresholds[i].nneg;
					t->children[1].nexmpls = t->children[1].nexmpls - thresholds[i].npos - thresholds[i].nneg;

					float score = calcScore(t, e);
					if (score > best_score) {
						best_score = score;
						// ***REMOVE***
						if (i < nthresholds - 1)
							t->threshold = 0.5*(thresholds[i].v + thresholds[i+1].v);
					}
				}

				t->children[0].nexmpls = 0;
				t->children[1].nexmpls = 0;
			}

			free(thresholds);
		}
	}

	// split examples among children nodes
	for (i = 0, j = 0; i < t->nexmpls && j < feat_not_default_c[t->test]; ) {
		if (t->exmpls[i] == data[t->test][j].i) { // found it
			if (data[t->test][j].v <= t->threshold) // "it" is less than or equal to threshold, go left
				t->children[0].exmpls[t->children[0].nexmpls++] = t->exmpls[i++];
			else // "it" is greater than threshold, go right
				t->children[1].exmpls[t->children[1].nexmpls++] = t->exmpls[i++];
			++j;
		} else if (t->exmpls[i] < data[t->test][j].i) { // it's not there
			if (DEFAULT <= t->threshold) // DEFAULT is less than or equal to threshold, go left
				t->children[0].exmpls[t->children[0].nexmpls++] = t->exmpls[i++];
			else
				t->children[1].exmpls[t->children[1].nexmpls++] = t->exmpls[i++];
		}
		else
			++j;
	}
	while (i < t->nexmpls) {
		if (DEFAULT <= t->threshold) // DEFAULT is less than or equal to threshold, go left
			t->children[0].exmpls[t->children[0].nexmpls++] = t->exmpls[i++];
		else
			t->children[1].exmpls[t->children[1].nexmpls++] = t->exmpls[i++];
	}

	assert(t->children[0].nexmpls == (t->nexmpls - t->children[1].nexmpls));

	// calculate positive/negative counts for children nodes
	setCounts(&(t->children[0]));
	setCounts(&(t->children[1]));

	return best_score;
}

// perform merge sort on threshold array a of size na
void sortt(threshold *a, int na)
{
	if (na < 2)
		return;

	threshold *c = malloc(na * 2 * sizeof *c);
	if (c == NULL) error("Unable to allocate memory at line %i.\n");

	int mid, i;

	mid = na / 2;
	sortt(a, mid);
	sortt(a + mid, na - mid);
	uniont(c, a, mid, a + mid, na - mid);
	for (i = 0; i < na; ++i)
		memcpy(&a[i], &c[i], sizeof *a);

	free(c);
}

// calculate the sorted union of the two sorted threshold arrays a and b of sizes na and nb
int uniont(threshold *c, threshold *a, int na, threshold *b, int nb)
{
	int ai = 0, bi = 0, ci = 0;
	while (ai < na && bi < nb) {
		if (a[ai].v < b[bi].v)
			memcpy(&c[ci++], &a[ai++], sizeof *c);
		else
			memcpy(&c[ci++], &b[bi++], sizeof *c);
	}
	while (ai < na)
		memcpy(&c[ci++], &a[ai++], sizeof *c);
	while (bi < nb)
		memcpy(&c[ci++], &b[bi++], sizeof *c);

	return ci;
}

// calculate and return IG/GR on t.npos/nneg + t.children[i].npos/nneg
float calcScore(test *t, float e)
{
	int i;
	for (i = 0; i < t->nchildren; ++i) {
		if (t->children[i].nexmpls == 0)
			continue;
		e -= (t->children[i].nexmpls / (float)t->nexmpls) * entropy2(t->children[i].npos, t->children[i].nneg, t->children[i].nexmpls);
	}
	return e;
}

// calculate entropy for "classes" number of classes with counts "counts" that add to "sum"
float entropy2(int class1, int class2, int sum)
{
	if (sum == 0)
		return 0;

	float p1 = (class1 == 0) ? 1 : (class1 / (float)sum);
	float p2 = (class2 == 0) ? 1 : (class2 / (float)sum);

	return - (p1 * LOG2(p1)) - (p2 * LOG2(p2));
}

// recursively prints and frees the tree from memory
void printAndFreeTree(test *t, int doPrint, int doFree)
{
	_printAndFreeTree(t, 0, doPrint, doFree);
}
void _printAndFreeTree(test *t, int indent, int doPrint, int doFree)
{
	int i;

	if (doPrint) {
		for (i = 0; i < indent; ++i)
			printf("\t");

		printf("[%i <= %f]/%i/%i\n", t->test, t->threshold, t->npos, t->nneg);
	}

	if (t->nchildren == 0)
		return;

	for (i = 0; i < t->nchildren; ++i) {
		if (doFree){
			free(t->children[i].exmpls);
			t->children[i].exmpls=0;
		}

		_printAndFreeTree(&(t->children[i]), indent + 1, doPrint, doFree);
	}
	
	if (doFree){
		free(t->children);
		t->children=0;
	}
}

// returns the prediction of tree with root t on the case corresponding to features
float runTree(test *t, float *features)
{
	if (t->test == -1) {
		if (t->npos == 0)
			return 0;
		else if (t->nneg == 0)
			return 1;
		else
			return (t->npos / (float)(t->npos + t->nneg));
	}

	return runTree(&(t->children[features[t->test] > t->threshold]), features);
}

void saveRecursive(test* root, FILE* fp){
    int i;
    if(root->test>=0){
        fprintf(fp,"%d %d %f\n",root->test,root->nchildren, root->threshold);
        for(i=0; i<root->nchildren; i++)
            saveRecursive(&(root->children[i]), fp);
    }
    else
        fprintf(fp,"%d %d %f %d %d\n",root->test,root->nchildren, root->threshold, root->npos,root->nneg);
}

void saveModel(test* root, const char* name){
    FILE *fp=fopen(name,"w");
	fprintf(fp,"%d\n",num_feat);
    saveRecursive(root,fp);
	fclose(fp);
}
