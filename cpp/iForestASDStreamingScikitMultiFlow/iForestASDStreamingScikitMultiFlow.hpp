/* Standard headers */
#include <cstddef>
#include <cstdint>
#include <vector>
using std::size_t;

/*  The library has overloaded functions supporting different input types.
    Note that, while 'float' type is supported, it will
    be slower to fit models to them as the models internally use
    'double' and 'long double', and it's not recommended to use.

    In order to use the library with different types than the ones
    suggested here, add something like this before including the
    library header:
      #define real_t float
      #define sparse_ix int
      #include "iForestASDStreamingScikitMultiFlow.hpp"
    The header may be included multiple times if required. */
#ifndef real_t
    #define real_t double     /* supported: float, double */
#endif
#ifndef sparse_ix
    #define sparse_ix int  /* supported: int, int64_t, size_t */
#endif

#ifndef ISOLATIONFORESTSTREAM_H
#define ISOLATIONFORESTSTREAM_H

#ifdef _WIN32
    #define ISOLATIONFORESTSTREAM_EXPORTED __declspec(dllimport)
#else
    #define ISOLATIONFORESTSTREAM_EXPORTED 
#endif


/* Types used through the package - zero is the suggested value (when appropriate) */
typedef enum  NewCategAction {Weighted=0,  Smallest=11,    Random=12}  NewCategAction; /* Weighted means Impute in the extended model */
typedef enum  MissingAction  {Divide=21,   Impute=22,      Fail=0}     MissingAction;  /* Divide is only for non-extended model */
typedef enum  ColType        {Numeric=31,  Categorical=32, NotUsed=0}  ColType;
typedef enum  CategSplit     {SubSet=0,    SingleCateg=41}             CategSplit;
typedef enum  CoefType       {Uniform=61,  Normal=0}                   CoefType;       /* For extended model */
typedef enum  UseDepthImp    {Lower=71,    Higher=0,       Same=72}    UseDepthImp;    /* For NA imputation */
typedef enum  WeighImpRows   {Inverse=0,   Prop=81,        Flat=82}    WeighImpRows;   /* For NA imputation */
typedef enum  ScoringMetric  {Depth=0,     Density=92,     BoxedDensity=94, BoxedDensity2=96, BoxedRatio=95,
                              AdjDepth=91, AdjDensity=93}              ScoringMetric;


/* Structs that are output (modified) from the main function */
typedef struct IsolationForestStream {
    ColType  col_type = NotUsed;
    size_t   col_num;
    double   num_split;
    std::vector<char> cat_split;
    int      chosen_cat;
    size_t   tree_left;
    size_t   tree_right;
    double   pct_tree_left;
    double   score;        /* will not be integer when there are weights or early stop */
    double   range_low  = -HUGE_VAL;
    double   range_high =  HUGE_VAL;
    double   remainder; /* only used for distance/similarity */


     int n_estimators;
     int window_size;
     double random_state;
     double anomaly_threshold;
     double drift_threshold;
     int sample_size;
     float anomaly_rate;
     int cpt;

    IsolationForestStream() = default;

} IsolationForestStream;


typedef struct ensemble {
     int X;
     int window_size;
} ensemble;

typedef struct window  {
     int X;
     int y;
     int window (window* int window_size, int n_features);
} window;

typedef struct prec_window {
     int X;
     int window_size;
} prec_window;

typedef struct samples_seen {
     int X;
     int window_size;
} samples_seen;

typedef struct n_tree {
     int X;
     int window_size;
} n_tree;

typedef struct depth {
     int X;
     int sample_size;
} depth;




typedef struct IsoHPlane {
    std::vector<size_t>   col_num;
    std::vector<ColType>  col_type;
    std::vector<double>   coef;
    std::vector<double>   mean;
    std::vector<std::vector<double>> cat_coef;
    std::vector<int>      chosen_cat;
    std::vector<double>   fill_val;
    std::vector<double>   fill_new;

    double   split_point;
    size_t   hplane_left;
    size_t   hplane_right;
    double   score;        /* will not be integer when there are weights or early stop */
    double   range_low  = -HUGE_VAL;
    double   range_high =  HUGE_VAL;
    double   remainder; /* only used for distance/similarity */

    IsoHPlane() = default;
} IsoHPlane;

typedef struct IsoForest {
    std::vector< std::vector<IsolationForestStream> > trees;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    double            exp_avg_depth;
    double            exp_avg_sep;
    size_t            orig_sample_size;
    bool              has_range_penalty;
    IsoForest() = default;
} IsoForest;

typedef struct ExtIsoForest {
    std::vector< std::vector<IsoHPlane> > hplanes;
    NewCategAction    new_cat_action;
    CategSplit        cat_split_type;
    MissingAction     missing_action;
    double            exp_avg_depth;
    double            exp_avg_sep;
    size_t            orig_sample_size;
    bool              has_range_penalty;
    ExtIsoForest() = default;
} ExtIsoForest;

typedef struct ImputeNode {
    std::vector<double>  num_sum;
    std::vector<double>  num_weight;
    std::vector<std::vector<double>>  cat_sum;
    std::vector<double>  cat_weight;
    size_t               parent;
    ImputeNode() = default;
} ImputeNode; /* this is for each tree node */

typedef struct Imputer {
    size_t               ncols_numeric;
    size_t               ncols_categ;
    std::vector<int>     ncat;
    std::vector<std::vector<ImputeNode>> imputer_tree;
    std::vector<double>  col_means;
    std::vector<int>     col_modes;
    Imputer() = default;
} Imputer;

typedef struct SingleTreeIndex {
    std::vector<size_t> terminal_node_mappings;
    std::vector<double> node_distances;
    std::vector<double> node_depths;
    std::vector<size_t> reference_points;
    std::vector<size_t> reference_indptr;
    std::vector<size_t> reference_mapping;
    size_t n_terminal;
} TreeNodeIndex;

typedef struct TreesIndexer {
    std::vector<SingleTreeIndex> indices;
    TreesIndexer() = default;
} TreesIndexer;




#endif /* ISOLATIONFORESTSTREAM_H */

/*  Fit Isolation Forest model, or variant of it such as SCiForest
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
int fit_iforest(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth,   size_t ncols_per_tree,
                bool   limit_depth, bool penalize_range, bool standardize_data,
                ScoringMetric scoring_metric, bool fast_bratio,
                bool   standardize_dist, double tmat[],
                double output_depths[], bool standardize_depth,
                real_t col_weights[], bool weigh_by_kurt,
                double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                double prob_pick_by_full_gain, double prob_pick_by_dens,
                double prob_pick_col_by_range, double prob_pick_col_by_var,
                double prob_pick_col_by_kurt,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, Imputer *imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, bool use_long_double, int nthreads,
		int window_size, int n_estimators, double random_state);



/* Add additional trees to already-fitted isolation forest model
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
int add_tree(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
             real_t numeric_data[],  size_t ncols_numeric,
             int    categ_data[],    size_t ncols_categ,    int ncat[],
             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
             size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
             real_t sample_weights[], size_t nrows,
             size_t max_depth,     size_t ncols_per_tree,
             bool   limit_depth,   bool penalize_range, bool standardize_data,
             bool   fast_bratio,
             real_t col_weights[], bool weigh_by_kurt,
             double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
             double prob_pick_by_full_gain, double prob_pick_by_dens,
             double prob_pick_col_by_range, double prob_pick_col_by_var,
             double prob_pick_col_by_kurt,
             double min_gain, MissingAction missing_action,
             CategSplit cat_split_type, NewCategAction new_cat_action,
             UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
             bool   all_perm, Imputer *imputer, size_t min_imp_obs,
             TreesIndexer *indexer,
             real_t ref_numeric_data[], int ref_categ_data[],
             bool ref_is_col_major, size_t ref_ld_numeric, size_t ref_ld_categ,
             real_t ref_Xc[], sparse_ix ref_Xc_ind[], sparse_ix ref_Xc_indptr[],
             uint64_t random_seed, bool use_long_double);


/* Predict outlier score, average depth, or terminal node numbers
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void predict_iforest(real_t numeric_data[], int categ_data[],
                     bool is_col_major, size_t ld_numeric, size_t ld_categ,
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                     size_t nrows, int nthreads, bool standardize,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double output_depths[],   sparse_ix tree_num[],
                     double per_tree_depths[],
                     TreesIndexer *indexer);



/* Get the number of nodes present in a given model, per tree
*/
ISOLATIONFORESTSTREAM_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;



/* Calculate distance or similarity or kernel/proximity between data points
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void calc_similarity(real_t numeric_data[], int categ_data[],
                     real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     size_t nrows, bool use_long_double, int nthreads,
                     bool assume_full_distr, bool standardize_dist, bool as_kernel,
                     IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                     double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                     TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ);

/* Impute missing values in new data
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                           real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                           size_t nrows, bool use_long_double, int nthreads,
                           IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                           Imputer &imputer);


/* Append trees from one model into another
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void merge_models(IsoForest*     model,      IsoForest*     other,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                  Imputer*       imputer,    Imputer*       iother,
                  TreesIndexer*  indexer,    TreesIndexer*  ind_other);

/* Create a model containing a sub-set of the trees from another model
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void subset_model(IsoForest*     model,      IsoForest*     model_new,
                  ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                  Imputer*       imputer,    Imputer*       imputer_new,
                  TreesIndexer*  indexer,    TreesIndexer*  indexer_new,
                  size_t *trees_take, size_t ntrees_take);

/* Build indexer for faster terminal node predictions and/or distance calculations
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void build_tree_indices(TreesIndexer &indexer, const IsoForest &model, int nthreads, const bool with_distances);
ISOLATIONFORESTSTREAM_EXPORTED
void build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model, int nthreads, const bool with_distances);
ISOLATIONFORESTSTREAM_EXPORTED
void build_tree_indices
(
    TreesIndexer *indexer,
    const IsoForest *model_outputs,
    const ExtIsoForest *model_outputs_ext,
    int nthreads,
    const bool with_distances
);
/* Gets the number of reference points stored in an indexer object */
ISOLATIONFORESTSTREAM_EXPORTED
size_t get_number_of_reference_points(const TreesIndexer &indexer) noexcept;


/* Functions to inspect serialized objects
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
void inspect_serialized_object
(
    const char *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_Indexer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void inspect_serialized_object
(
    FILE *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_Indexer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void inspect_serialized_object
(
    std::istream &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_Indexer,
    bool &has_metadata,
    size_t &size_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void inspect_serialized_object
(
    const std::string &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_Indexer,
    bool &has_metadata,
    size_t &size_metadata
);

/* Serialization and de-serialization functions (individual objects)
*
*/
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size(const IsoForest &model) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size(const ExtIsoForest &model) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size(const Imputer &model) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size(const TreesIndexer &model) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_IsoForest(const IsoForest &model, char *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_IsoForest(const IsoForest &model, FILE *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_IsoForest(const IsoForest &model, std::ostream &out);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_IsoForest(const IsoForest &model);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_IsoForest(IsoForest &model, const char *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_IsoForest(IsoForest &model, FILE *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_IsoForest(IsoForest &model, std::istream &in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_IsoForest(IsoForest &model, const std::string &in);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, char *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_ExtIsoForest(const ExtIsoForest &model);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Imputer(const Imputer &model, char *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Imputer(const Imputer &model, FILE *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Imputer(const Imputer &model, std::ostream &out);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_Imputer(const Imputer &model);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Imputer(Imputer &model, const char *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Imputer(Imputer &model, FILE *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Imputer(Imputer &model, std::istream &in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Imputer(Imputer &model, const std::string &in);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Indexer(const TreesIndexer &model, char *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Indexer(const TreesIndexer &model, FILE *out);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_Indexer(const TreesIndexer &model, std::ostream &out);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_Indexer(const TreesIndexer &model);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const char *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Indexer(TreesIndexer &model, FILE *in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Indexer(TreesIndexer &model, std::istream &in);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_Indexer(TreesIndexer &model, const std::string &in);


/* Serialization and de-serialization functions (combined objects)
*
*/
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
    const size_t size_optional_metadata
) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
    const size_t size_optional_metadata
) noexcept;
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    char *out
);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const TreesIndexer *indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
);
ISOLATIONFORESTSTREAM_EXPORTED
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
);
ISOLATIONFORESTSTREAM_EXPORTED
std::string serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *serialized_indexer,
    const char *optional_metadata,
    const size_t size_optional_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_combined
(
    const char* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_combined
(
    FILE* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_combined
(
    std::istream &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);
ISOLATIONFORESTSTREAM_EXPORTED
void deserialize_combined
(
    const std::string &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    TreesIndexer *indexer,
    char *optional_metadata
);


/* Serialize additional trees into previous serialized bytes
*
*/
ISOLATIONFORESTSTREAM_EXPORTED
bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes);
ISOLATIONFORESTSTREAM_EXPORTED
bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes);
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees);
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees);
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees);
ISOLATIONFORESTSTREAM_EXPORTED
size_t determine_serialized_size_additional_trees(const TreesIndexer &model, size_t old_ntrees);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, char *old_bytes_reallocated);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);
ISOLATIONFORESTSTREAM_EXPORTED
void incremental_serialize_Indexer(const TreesIndexer &model, std::string &old_bytes);


/* Translate isolation forest model into a single SQL select statement
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                          std::string &table_from, std::string &select_as,
                                          std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                          std::vector<std::vector<std::string>> &categ_levels,
                                          bool index1, int nthreads);


/* Translate model trees into SQL select statements
* 
*/
ISOLATIONFORESTSTREAM_EXPORTED
std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                      std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                      std::vector<std::vector<std::string>> &categ_levels,
                                      bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                      int nthreads);


ISOLATIONFORESTSTREAM_EXPORTED
void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                          const bool with_distances,
                          real_t *numeric_data, int *categ_data,
                          bool is_col_major, size_t ld_numeric, size_t ld_categ,
                          real_t *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                          real_t *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                          size_t nrows, int nthreads);
