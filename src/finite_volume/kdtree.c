/*
 
 K-d tree
 Copyright (C) 2012 RosettaCode User:Ledrug.
 Permission is granted to copy, distribute and/or modify this document
 under the terms of the GNU Free Documentation License, Version 1.2
 or any later version published by the Free Software Foundation;
 with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
 A copy of the license is included in the section entitled "GNU
 Free Documentation License".
 
 C source from:
 https://rosettacode.org/wiki/K-d_tree
 
 Page revision information:
 14:20, 8 April 2017 Trizen (Talk | contribs) m . . (73,308 bytes) (0) . . (->{{header|Sidef}}:  updated code) (undo)
 
 Notes regarding usage can be found under "C Entry":
 https://rosettacode.org/wiki/Talk:K-d_tree#C_Entry
 
 Caution / implementation limitation:
 - The method does not behave correctly if two input nodes (different pointer but identical x[] values)
   are placed within the kdtree.

 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
 Changes made:
 + Re-ordered members of kd_node_t.
 
 + find_median() and swap() accepts dim as arg. Enables optimal memcpy() size to be used.
 
 + Added static keyword to all functions
 
 + Renamed 
     struct kd_node_t 
  to 
     struct _p_kd_node_t
 
 + Added typedef struct _p_kd_node_t* kd_node (and updated functions accordingly)
 
 + Added the member 
     int index 
   into the definition of struct kd_node_t

 + Added the "namespace" straing kdtr_ to the original methods
     inline double dist()  --> double kdtr_dist()
     inline void swap()    --> kdtr_swap()
     kd_node find_median() --> kdtr_find_median()
     kd_node make_tree()   --> kdtr_make_tree()
     void nearest()        --> kdtr_nearest()
 
 + Renamed the global variable
     int visited
   to 
    int kdtr_visited
 
 + Added the object KDTree and helpers to generalize the implementation
 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <kdtree.h>

/* private prototypes */
static inline double kdtr_dist(kd_node a, kd_node b, int dim);
static inline void   kdtr_swap(kd_node x, kd_node y, int dim);
static kd_node       kdtr_find_median(kd_node start, kd_node end, int idx, int dim);
static kd_node       kdtr_make_tree(kd_node t, int len, int i, int dim);
static void          kdtr_nearest(kd_node root, kd_node nd, int i, int dim, kd_node *best, double *best_dist);

/* global variable, so sue me */
int kdtr_visited;

static inline double kdtr_dist(kd_node a, kd_node b, int dim)
{
  double t, d = 0;
  while (dim--) {
    t = a->x[dim] - b->x[dim];
    d += t * t;
  }
  return d;
}

static inline void kdtr_swap(kd_node x, kd_node y, int dim)
{
  double tmp[] = {0, 0, 0};
  int    tmp_i;
  memcpy(tmp,  x->x, sizeof(double)*dim);
  memcpy(x->x, y->x, sizeof(double)*dim);
  memcpy(y->x, tmp,  sizeof(double)*dim);
  tmp_i = x->index;
  x->index = y->index;
  y->index = tmp_i;
}


/* see quickselect method */
static kd_node kdtr_find_median(kd_node start, kd_node end, int idx, int dim)
{
  kd_node p,store,md ;
  double  pivot;
  
  if (end <= start) return NULL;
  if (end == start + 1)
  return start;
  
  md = start + (end - start) / 2;
  
  while (1) {
    pivot = md->x[idx];
    
    kdtr_swap(md, end - 1, dim);
    for (store = p = start; p < end; p++) {
      if (p->x[idx] < pivot) {
        if (p != store)
        kdtr_swap(p, store, dim);
        store++;
      }
    }
    kdtr_swap(store, end - 1, dim);
    
    /* median has duplicate values */
    if (store->x[idx] == md->x[idx])
    return md;
    
    if (store > md) end = store;
    else            start = store;
  }
}

static kd_node kdtr_make_tree(kd_node t, int len, int i, int dim)
{
  kd_node n;
  
  if (!len) return 0;
  
  if ((n = kdtr_find_median(t, t + len, i, dim))) {
    i = (i + 1) % dim;
    n->left  = kdtr_make_tree(t, n - t, i, dim);
    n->right = kdtr_make_tree(n + 1, t + len - (n + 1), i, dim);
  }
  return n;
}


static void kdtr_nearest(kd_node root, kd_node nd, int i, int dim,
                  kd_node *best, double *best_dist)
{
  double d, dx, dx2;
  
  if (!root) return;
  d = kdtr_dist(root, nd, dim);
  dx = root->x[i] - nd->x[i];
  dx2 = dx * dx;
  
  kdtr_visited ++;
  
  if (!*best || d < *best_dist) {
    *best_dist = d;
    *best = root;
  }
  
  /* if chance of exact match is high */
  if (!*best_dist) return;
  
  if (++i >= dim) i = 0;
  
  kdtr_nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist);
  if (dx2 >= *best_dist) return;
  kdtr_nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist);
}

/* Extensions */
void kdtr_node_init(kd_node n)
{
  memset(n->x,0,sizeof(double)*KDTR_MAX_DIM);
  n->left  = NULL;
  n->right = NULL;
  n->index = 0;
}

void KDTreeCreate(int dim,KDTree *_k)
{
  KDTree kt;

  if (dim > KDTR_MAX_DIM) {
    printf("[kdtree error] KDTree cannot be created. dim must be <= %d\n",KDTR_MAX_DIM);
    *_k = NULL;
  }
  kt = (KDTree)malloc(sizeof(struct _p_KDTree));
  memset(kt,0,sizeof(struct _p_KDTree));
  kt->root = NULL;
  kt->point = NULL;
  kt->npoints = 0;
  kt->dim = dim;
  kt->visited = 0;
  kt->setup = 0;
  kt->point =(kd_node)calloc(1, sizeof(struct _p_kd_node_t));
  kt->cnt = 0;
  *_k = kt;
}

void KDTreeDestroy(KDTree *_k)
{
  KDTree k;
  
  if (!*_k) return;
  k = *_k;
  if (!k) return;
  if (k->point) {
    free(k->point);
  }
  k->point = NULL;
  k->root = NULL;
  free(k);
  *_k = NULL;
}

void KDTreeReset(KDTree kt)
{
  kt->npoints = 0;
  kt->cnt = 0;
  kt->root = NULL;
  kt->setup = 0;
}

void KDTreeView(KDTree k)
{
  double size;
  printf("KDTree\n");
  printf(" npoints: %d\n",k->npoints);
  printf(" dim:     %d\n",k->dim);
  size = ((double)sizeof(struct _p_kd_node_t)) * ((double)k->npoints);
  printf(" memory:  %1.4e (MB)\n",size/1.0e6);
}

void KDTreeSetPoints(KDTree k,int np)
{
  int p;
  if (k->setup == 1) {
    printf("[kdtree error] KDTree already setup. Cannot call KDTreeSetPoints() after KDTreeSetup() has been called.\n");
    return;
  }
  if (np != k->npoints) {
    kd_node tmp;
    
    k->npoints = np;
    tmp =(kd_node)realloc(k->point,k->npoints * sizeof(struct _p_kd_node_t));
    k->point = tmp;
  }
  memset(k->point,0,k->npoints * sizeof(struct _p_kd_node_t));
  for (p=0; p<k->npoints; k++) {
    k->point[p].index = p;
  }
  k->cnt = 0;
}

void KDTreeGetPoints(KDTree k,int *n,kd_node *nodes)
{
  if (n) { *n = k->npoints; }
  if (nodes) { *nodes = k->point; }
}

void KDTreeInsertPoint(KDTree k,double coor[])
{
  if (k->setup == 1) {
    printf("[kdtree error] KDTree already setup. Cannot call KDTreeInsertPoint() after KDTreeSetup() has been called.\n");
    return;
  }
  if (k->cnt >= k->npoints) {
    printf("[kdtree error] Cannot insert into slot %d. Max. index = %d\n",k->cnt,k->npoints);
    return;
  }
  memcpy(k->point[k->cnt].x, coor, sizeof(double)*k->dim);
  k->cnt++;
}

void KDTreeSetup(KDTree kt)
{
  if (kt->setup == 1) return;
  kt->root = NULL;
  // make_tree
  kt->root = kdtr_make_tree(&kt->point[0], kt->npoints, 0, kt->dim);
  kt->setup = 1;
}

void KDTreeFindNearest(KDTree k,double coor[],kd_node *nearest,double *sep)
{
  struct _p_kd_node_t test_node;
  kd_node             found = NULL;
  double              best_dist = 1.0e32;
  
  if (k->setup == 0) {
    printf("[kdtree error] KDTree not setup. Must call KDTreeSetup() before KDTreeFindNearest().\n");
    *nearest = NULL;
    if (sep) { *sep = 1.0e32; }
    return;
  }
  
  kdtr_node_init(&test_node);
  memcpy(test_node.x, coor, sizeof(double)*k->dim);
  
  kdtr_visited = 0;
  kdtr_nearest(k->root, &test_node, 0, k->dim, &found, &best_dist);
  k->visited = kdtr_visited;
  
  *nearest = found;
  if (sep) { *sep = sqrt(best_dist); }
}

#if 0

#define N 1000000
#define rand1() (rand() / (double)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }

int ex1(void)
{
  int                 i;
  struct _p_kd_node_t wp[] = {
    {{2, 3},NULL,NULL,0}, {{5, 4},NULL,NULL,1}, {{9, 6},NULL,NULL,2}, {{4, 7},NULL,NULL,3}, {{8, 1},NULL,NULL,4}, {{7, 2},NULL,NULL,5}
  };
  struct _p_kd_node_t test_node = {{9, 2},NULL,NULL,0};
  kd_node             root, found, million;
  double              best_dist;
  
  root = kdtr_make_tree(wp, sizeof(wp) / sizeof(wp[1]), 0, 2);
  
  kdtr_visited = 0;
  found = 0;
  kdtr_nearest(root, &test_node, 0, 2, &found, &best_dist);
  
  printf(">> WP tree\nsearching for (%g, %g)\n"
         "found [%d](%g, %g) dist %g\nseen %d nodes\n\n",
         test_node.x[0], test_node.x[1],
         found->index,found->x[0], found->x[1], sqrt(best_dist), kdtr_visited);
  
  million =(kd_node) calloc(N, sizeof(struct _p_kd_node_t));

  srand(0);
  for (i = 0; i < N; i++) {
    million[i].index = i;
    rand_pt(million[i]);
  }
  
  root = kdtr_make_tree(million, N, 0, 3);
  rand_pt(test_node);
  
  kdtr_visited = 0;
  found = 0;
  kdtr_nearest(root, &test_node, 0, 3, &found, &best_dist);
  
  printf(">> Million tree\nsearching for (%g, %g, %g)\n"
         "found [%d](%g, %g, %g) dist %g\nseen %d nodes\n",
         test_node.x[0], test_node.x[1], test_node.x[2],
         found->index,found->x[0], found->x[1], found->x[2],
         sqrt(best_dist), kdtr_visited);
  
  /* search many random points in million tree to see average behavior.
   tree size vs avg nodes visited:
   10      ~  7
   100     ~ 16.5
   1000        ~ 25.5
   10000       ~ 32.8
   100000      ~ 38.3
   1000000     ~ 42.6
   10000000    ~ 46.7              */
  int sum = 0, test_runs = 100000;
  for (i = 0; i < test_runs; i++) {
    found = 0;
    kdtr_visited = 0;
    rand_pt(test_node);
    kdtr_nearest(root, &test_node, 0, 3, &found, &best_dist);
    sum += kdtr_visited;
  }
  printf("\n>> Million tree\n"
         "visited %d nodes for %d random findings (%f per lookup)\n",
         sum, test_runs, sum/(double)test_runs);
  
  free(million);
  
  return 0;
}

void ex2(void)
{
  
  int     npoints,dim,i;
  KDTree  k;
  kd_node nearest;
  struct _p_kd_node_t wp[] = {
    {{2, 3},NULL,NULL,0}, {{5, 4},NULL,NULL,1}, {{9, 6},NULL,NULL,2}, {{4, 7},NULL,NULL,3}, {{8, 1},NULL,NULL,4}, {{7, 2},NULL,NULL,4}
  };
  struct _p_kd_node_t test_node = {{9, 2},NULL,NULL,0};
  double sep;
  
  
  dim = 2;
  npoints = sizeof(wp) / sizeof(wp[1]);
  
  KDTreeCreate(dim,&k);
  KDTreeSetPoints(k,npoints);
  for (i=0; i<npoints; i++) {
    KDTreeInsertPoint(k,wp[i].x);
  }
  KDTreeSetup(k);

  KDTreeFindNearest(k,test_node.x,&nearest,&sep);
  
  printf(">> KDTree \nsearching for (%g, %g)\n"
         "found [%d](%g, %g) dist %g\nseen %d nodes\n\n",
         test_node.x[0], test_node.x[1],
         nearest->index,nearest->x[0], nearest->x[1], sep, kdtr_visited);
  
  KDTreeDestroy(&k);
}

void ex3(void)
{
  
  KDTree  k;
  int     npoints,dim;
  int     i;
  kd_node nearest;
  struct _p_kd_node_t wp;
  struct _p_kd_node_t test_node = {{9, 2},0};
  double              sep;
  
  
  dim = 3;
  npoints = N;
  
  KDTreeCreate(dim,&k);
  KDTreeSetPoints(k,npoints);

  srand(0);
  for (i=0; i<npoints; i++) {
    rand_pt(wp);
    KDTreeInsertPoint(k,wp.x);
  }
  KDTreeSetup(k);
  
  rand_pt(test_node);
  KDTreeFindNearest(k,test_node.x,&nearest,&sep);

  printf(">> Million tree\nsearching for (%g, %g, %g)\n"
         "found [%d](%g, %g, %g) dist %g\nseen %d nodes\n",
         test_node.x[0], test_node.x[1], test_node.x[2],
         nearest->index,nearest->x[0], nearest->x[1], nearest->x[2],
         sep, kdtr_visited);
  KDTreeView(k);

  
  npoints = npoints / 2;
  
  KDTreeReset(k);
  KDTreeSetPoints(k,npoints);
  
  srand(0);
  for (i=0; i<npoints; i++) {
    rand_pt(wp);
    KDTreeInsertPoint(k,wp.x);
  }
  KDTreeSetup(k);
  
  rand_pt(test_node);
  KDTreeFindNearest(k,test_node.x,&nearest,&sep);
  
  printf(">> Million/2 tree\nsearching for (%g, %g, %g)\n"
         "found [%d](%g, %g, %g) dist %g\nseen %d nodes\n",
         test_node.x[0], test_node.x[1], test_node.x[2],
         nearest->index,nearest->x[0], nearest->x[1], nearest->x[2],
         sep, kdtr_visited);
  
  KDTreeView(k);

  KDTreeDestroy(&k);
}

void ex_petgs(void)
{
  
  KDTree  k;
  int     npoints,dim;
  kd_node nearest;
  struct _p_kd_node_t wp;
  struct _p_kd_node_t test_node;
  double  sep;
  
  
  dim = 2;
  npoints = 4;
  
  KDTreeCreate(dim,&k);
  KDTreeSetPoints(k,npoints);

  wp.x[0] = 1.1; wp.x[1] = 0.0;
  KDTreeInsertPoint(k,wp.x);

  wp.x[0] = 1.1; wp.x[1] = -1.0;
  KDTreeInsertPoint(k,wp.x);

  wp.x[0] = 0.0; wp.x[1] = 0.0;
  KDTreeInsertPoint(k,wp.x);

  wp.x[0] = 0.0; wp.x[1] = -1.0;
  KDTreeInsertPoint(k,wp.x);

  for (int i=0; i<npoints; i++) {
    printf("point[%d] %+1.4e %+1.4e [index %d]\n",i,k->point[i].x[0],k->point[i].x[1],k->point[i].index);
  }

  KDTreeSetup(k);

  for (int i=0; i<npoints; i++) {
    printf("point[%d] %+1.4e %+1.4e [index %d]\n",i,k->point[i].x[0],k->point[i].x[1],k->point[i].index);
  }

  test_node.index = 0;
  test_node.x[0] =  2.0168e-03;
  test_node.x[1] = -0.9967e-03;
  //test_node[2] = 0.0;
  KDTreeFindNearest(k,test_node.x,&nearest,&sep);
  
  printf(">> 4 tree / searching for (%g, %g)\n",test_node.x[0], test_node.x[1]);
  printf(">> found [init-point %d](%g, %g)\n",nearest->index,nearest->x[0],nearest->x[1]);
  printf(">> distance %g\n",sep);
  printf(">> seen %d nodes\n",kdtr_visited);
  
  for (int i=0; i<npoints; i++) {
    double d = 0.0;
    d += (k->point[i].x[0] - test_node.x[0])*(k->point[i].x[0] - test_node.x[0]);
    d += (k->point[i].x[1] - test_node.x[1])*(k->point[i].x[1] - test_node.x[1]);
    printf("point[%d] %+1.4e %+1.4e [index %d]\n",i,k->point[i].x[0],k->point[i].x[1],k->point[i].index);
    printf("point[%d] sep %1.4e\n",i,sqrt(d));
  }
  
  
  KDTreeDestroy(&k);
}

void ex_petgs_3d(int nsub,int npoints)
{
  
  KDTree  k;
  int     p,dim;
  kd_node nearest;
  double  sep,*xi;
  int     ii,jj,kk,l;
  
  xi = malloc(sizeof(double)*nsub);
  for (ii=0; ii<nsub; ii++) {
    double dxi = 2.0/(double)nsub;
    xi[ii] = 0.5 * dxi + ii * dxi;
  }
  
  dim = 3;
  KDTreeCreate(dim,&k);
  KDTreeSetPoints(k,npoints);
  {
    int     np;
    kd_node nodes;
    
    KDTreeGetPoints(k,&np,&nodes);
    srand(0);
    for (p=0; p<np; p++) {
      nodes[p].index = p;
      //rand_pt(nodes[p]);
      nodes[p].x[0] = p/4.0;
      nodes[p].x[1] = 0.0;
      nodes[p].x[2] = p/3.0;
    }
  }
  
  KDTreeSetup(k);
  

  double dd = 0.0;
  //for (l=0; l<32*32*32; l++) {
  
  for (kk=0; kk<nsub; kk++) {
    for (jj=0; jj<nsub; jj++) {
      for (ii=0; ii<nsub; ii++) {
        double test[] = { xi[ii], xi[jj], xi[kk] };
        
        KDTreeFindNearest(k,test,&nearest,&sep);
        dd += sep;
      }
    }
    //}
    
  }

  KDTreeDestroy(&k);
}

int main(int nargs,char *args[])
{
  //ex1();
  //ex2();
  //for (int k=0; k<100000; k++) {
  
  //ex3();
  
  //}
  //ex_petgs();
  
#if 1
  {
    int nsub = 3;
    int npoints = 30;
    if (nargs == 3) {
      nsub = atoi(args[1]);
      npoints = atoi(args[2]);
    }
    printf("nsub-divisions %d\n",nsub);
    printf("npoints        %d\n",npoints);
    
    //ex_petgs_3d(nsub,npoints);
    
    for (int l=0; l<32*32*32; l++) {
      //for (int l=0; l<24*24*24; l++) {
      ex_petgs_3d(nsub,npoints);
    }
  }
#endif
  
  return(0);
}

#endif
