
# Détecter une valeur aberrante évidente

Référence https://github.com/david-cortes/isotree

Isolation Forest est un algorithme développé à l'origine pour la détection de valeurs aberrantes qui consiste à diviser au hasard des sous-échantillons de données en fonction d'attributs/caractéristiques/colonnes. L'idée est que, plus l'observation est rare, plus il est probable qu'une répartition uniforme aléatoire sur une caractéristique placerait des valeurs aberrantes seules dans une branche, et moins il faudra de divisions pour isoler une observation aberrante comme celle-ci. Le concept est étendu à la division des hyperplans dans le modèle étendu (c'est-à-dire la division par plus d'une colonne à la fois) et aux divisions guidées (pas entièrement aléatoires) dans le modèle SCiForest qui visent à isoler plus rapidement les valeurs aberrantes et à trouver des valeurs aberrantes groupées.

# L'algorithme en Langage C++

Description de l'algorithme

1. Données aléatoires d'une distribution normale standard (100 points générés aléatoirement, plus 1 valeur aberrante ajoutée manuellement) La bibliothèque suppose qu'il est passé en tant que pointeur unidimensionnel, suivant l'ordre des colonnes majeures (comme Fortran)

2. Ajoutez maintenant un point aberrant évident (3,3)

3. Ajuster un petit modèle de forêt d'isolement

4. Vérifiez quelle ligne a le score de valeur aberrante le plus élevé

5. Les modèles peuvent être sérialisés et désérialisés de manière très idiomatique

*isotree_cpp_oop_ex.cpp*

        # /home/mamadou/big-data/cerin24022022/cpp-isotree/example/isotree_cpp_oop_ex.cpp

        #include <random>
        #include <algorithm>
        #include <iostream>
        #include <sstream>
        #include "isotree_oop.hpp"

        /*  To compile this example, first build the package through the cmake system:
              mkdir build
              cd build
              cmake ..
              make
              sudo make install
              sudo ldconfig
           Then compile this single file and link to the shared library:
             g++ -o test isotree_cpp_oop_ex.cpp -lisotree -std=c++11

           Or to link against it without a system install, assuming the cmake system
           has already built the library under ./build and this command is called from
           the root folder:
             g++ -o test example/isotree_cpp_oop_ex.cpp -std=c++11 -I./include -l:libisotree.so -L./build -Wl,-rpath,./build

           Then run with './test'
        */

        int which_max(std::vector<double> &v)
        {
            auto loc_max_el = std::max_element(v.begin(), v.end());
            return std::distance(v.begin(), loc_max_el);
        }


        int main()
        {
            /* Random data from a standard normal distribution
               (100 points generated randomly, plus 1 outlier added manually)
               Library assumes it is passed as a single-dimensional pointer,
               following column-major order (like Fortran) */
            int nrow = 101;
            int ncol = 2;
            std::vector<double> X( nrow * ncol );
            std::default_random_engine rng(1);
            std::normal_distribution<double> rnorm(0, 1);
            #define get_ix(row, col) (row + col*nrow)
            for (int col = 0; col < ncol; col++)
                for (int row = 0; row < 100; row++)
                    X[get_ix(row, col)] = rnorm(rng);

            /* Now add obvious outlier point (3,3) */
            X[get_ix(100, 0)] = 3.;
            X[get_ix(100, 1)] = 3.;

            /* Fit a small isolation forest model
               (see 'fit_model.cpp' for the documentation) */
            isotree::IsolationForest iso = isotree::IsolationForest();
            iso.fit(X.data(), nrow, ncol);

            /* Check which row has the highest outlier score
               (see file 'predict.cpp' for the documentation) */
            std::vector<double> outlier_scores = iso.predict(X.data(), nrow, true);

            int row_highest = which_max(outlier_scores);
            std::cout << "Point with highest outlier score: [";
            std::cout << X[get_ix(row_highest, 0)] << ", ";
            std::cout << X[get_ix(row_highest, 1)] << "]" << std::endl;

            /* Models can be serialized and de-serialized very idiomatically */
            std::stringstream ss;
            ss << iso; /* <- serialize */
            ss >> iso; /* <- deserialize */

            return EXIT_SUCCESS;
        }




*isotree_oop.hpp*

Forêts isolées et variantes de celles-ci, avec ajustements pour incorporation

    * des variables catégorielles et des valeurs manquantes.
    
    * Écrit pour la norme C++11 et destiné à être utilisé dans R et Python.

Interface POO IsoTree : Ceci est fourni comme une interface alternative plus facile à utiliser pour cette bibliothèque
qui suit les méthodes de style scikit-learn avec une seule classe C++. C'est un
wrapper sur l'en-tête non-POO 'isotree.hpp', fournissant la même fonctionnalité
dans une structure peut-être plus compréhensible, tout en offrant un accès direct
aux objets sous-jacents afin de permettre l'utilisation des fonctions de 'isotree.hpp'.

C'est une interface plus limitée car elle n'implémente pas toutes les fonctionnalités
pour la sérialisation, la prédiction de distance, la production de prédictions dans le même appel
que le modèle est ajusté, ou ajustement/prédiction sur des données avec des types autres que
'double' et 'int'.
Les descriptions ici ne contiennent pas la documentation complète, mais uniquement
quelques conseils afin de les rendre plus compréhensibles, visant à produire la fonction
des signatures auto-descriptives à la place (si vous êtes familier avec le
bibliothèque scikit-learn pour Python).
Pour une documentation détaillée, voir les méthodes identiques ou similaires dans le
l'en-tête 'isotree.hpp' à la place.

Sachez que de nombreuses combinaisons de paramètres ne sont pas valides.
Cette fonction ne fera aucune validation des entrées qu'elle reçoit.
Appeler 'fit' avec une combinaison de paramètres invalides *peut* lancer un
exception d'exécution, mais il ne pourra pas détecter toutes les
des combinaisons de paramètres non valides et pourraient potentiellement conduire à un silence
des erreurs comme des modèles statistiquement incorrects ou des prédictions qui ne
faire sens. Voir la documentation de l'en-tête non-POO ou du R
et les interfaces Python pour plus de détails sur les paramètres et les
combinaisons valides et invalides de paramètres.

        ## /home/mamadou/big-data/cerin24022022/cpp-isotree/example/isotree_oop.hpp

        /*    Isolation forests and variations thereof, with adjustments for incorporation
        *     of categorical variables and missing values.
        *     Writen for C++11 standard and aimed at being used in R and Python.

        */


        /***********************************************************************************
            ---------------------
            IsoTree OOP interface
            ---------------------

            This is provided as an alternative easier-to-use interface for this library
            which follows scikit-learn-style methods with a single C++ class. It is a
            wrapper over the non-OOP header 'isotree.hpp', providing the same functionality
            in a perhaps more comprehensible structure, while still offering direct access
            to the underlying objects so as to allow using the functions from 'isotree.hpp'.

            It is a more limited interface as it does not implement all the functionality
            for serialization, distance prediction, oproducing predictions in the same call
            as the model is fit, or fitting/predicting on data with types other than
            'double' and 'int'.

            The descriptions here do not contain the full documentation, but rather only
            some hints so as to make them more comprehensible, aiming at producing function
            signatures that are self-descriptive instead (if you are familiar with the
            scikit-learn library for Python).

            For detailed documentation see the same or similar-looking methods in the
            'isotree.hpp' header instead.

        ***********************************************************************************/
        #ifndef ISOTREE_OOP_H
        #define ISOTREE_OOP_H

        #include "isotree.hpp"

        namespace isotree {

        class ISOTREE_EXPORTED IsolationForest
        {
        public:
            /*  Note: if passing nthreads<0, will reset it to 'max_threads + nthreads + 1',
                so passing -1 means using all available threads.  */
            int nthreads = -1; /* <- May be manually changed at any time */

            uint64_t random_seed = 1;

            /*  General tree construction parameters  */
            size_t ndim = 3;
            size_t ntry = 1;
            CoefType coef_type = Uniform; /* only for ndim>1 */
            bool   with_replacement = false;
            bool   weight_as_sample = true;
            size_t sample_size = 0;
            size_t ntrees = 500;
            size_t max_depth = 0;
            size_t ncols_per_tree = 0;
            bool   limit_depth = true; /* if 'true', then 'max_depth' is ignored */
            bool   penalize_range = false;
            bool   standardize_data = true; /* only for ndim==1 */
            ScoringMetric scoring_metric = Depth;
            bool   fast_bratio = true; /* only for scoring_metric with 'Boxed' */
            bool   weigh_by_kurt = false;
            double prob_pick_by_gain_pl = 0.;
            double prob_pick_by_gain_avg = 0.;
            double prob_pick_by_full_gain = 0.;
            double prob_pick_by_dens = 0.;
            double prob_pick_col_by_range = 0.;
            double prob_pick_col_by_var = 0.;
            double prob_pick_col_by_kurt = 0.;
            double min_gain = 0.;
            MissingAction missing_action = Impute;

            /*  For categorical variables  */
            CategSplit cat_split_type = SubSet;
            NewCategAction new_cat_action = Weighted;
            bool   coef_by_prop = false;
            bool   all_perm = false;

            /*  For imputation methods (when using 'build_imputer=true' and calling 'impute')  */
            bool   build_imputer = false;
            size_t min_imp_obs = 3;
            UseDepthImp depth_imp = Higher;
            WeighImpRows weigh_imp_rows = Inverse;

            /*  Internal objects which can be used with the non-OOP interface  */
            IsoForest model;
            ExtIsoForest model_ext;
            Imputer imputer;
            TreesIndexer indexer;

            IsolationForest() = default;

            ~IsolationForest() = default;

            /*  Be aware that many combinations of parameters are invalid.
                This function will not do any validation of the inputs it receives.

                Calling 'fit' with a combination of invalid parameters *may* throw a
                runtime exception, but it will not be able to detect all the possible
                invalid parameter combinations and could potentially lead to silent
                errors like statistically incorrect models or predictions that do not
                make sense. See the documentation of the non-OOP header or of the R
                and Python interfaces for more details about the parameters and the
                valid and invalid combinations of parameters.  */
            IsolationForest
            (
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                bool with_replacement, bool weight_as_sample,
                size_t sample_size, size_t ntrees,
                size_t max_depth, size_t ncols_per_tree, bool   limit_depth,
                bool penalize_range, bool standardize_data,
                ScoringMetric scoring_metric, bool fast_bratio, bool weigh_by_kurt,
                double prob_pick_by_gain_pl, double prob_pick_by_gain_avg,
                double prob_pick_col_by_range, double prob_pick_col_by_var,
                double prob_pick_col_by_kurt,
                double min_gain, MissingAction missing_action,
                CategSplit cat_split_type, NewCategAction new_cat_action,
                bool   all_perm, bool build_imputer, size_t min_imp_obs,
                UseDepthImp depth_imp, WeighImpRows weigh_imp_rows,
                uint64_t random_seed, int nthreads
            );

            /*  'X' must be in column-major order (like Fortran).  */
            void fit(double X[], size_t nrows, size_t ncols);

            /*  Model can also be fit to categorical data (must also be column-major).
                Categorical data should be passed as integers starting at zero, with
                negative values denoting missing, and must pass also the number of
                categories to expect in each column.

                Can also pass row and column weights (see the documentation for options
                on how to interpret the row weights).  */
            void fit(double numeric_data[],   size_t ncols_numeric,  size_t nrows,
                     int    categ_data[],     size_t ncols_categ,    int ncat[],
                     double sample_weights[], double col_weights[]);

            /*  Numeric data may also be supplied as a sparse matrix, in which case it
                must be CSC format (colum-major). Categorical data is not supported in
                sparse format.  */
            void fit(double Xc[], int Xc_ind[], int Xc_indptr[],
                     size_t ncols_numeric,      size_t nrows,
                     int    categ_data[],       size_t ncols_categ,   int ncat[],
                     double sample_weights[],   double col_weights[]);

            /*  'predict' will return a vector with the standardized outlier scores
                (output length is the same as the number of rows in the data), in
                which higher values mean more outlierness.

                The data must again be in column-major format.

                This function will run multi-threaded if there is more than one row and
                the object has number of threads set to more than 1.  */
            std::vector<double> predict(double X[], size_t nrows, bool standardize);

            /*  Can optionally write to a non-owned array, or obtain the non-standardized
                isolation depth instead of the standardized score (also on a per-tree basis
                if desired), or get the terminal node numbers/indices for each tree. Note
                that 'tree_num' and 'per_tree_depths' are optional (pass NULL if not desired),
                while 'output_depths' should always be passed. Be aware that the outputs of
                'tree_num' will be filled in column-major order ([nrows, ntrees]), while the
                outputs of 'per_tree_depths' will be in row-major order.

                Note: 'tree_num' and 'per_tree_depths' will not be calculable when using
                'ndim==1' plus either 'missing_action==Divide' or 'new_cat_action==Weighted'.
                These can be checked through 'check_can_predict_per_tree'.

                Here, the data might be passed as either column-major or row-major (getting
                predictions in row-major order will be faster). If the data is in row-major
                order, must also provide the leading dimension of the array (typically this
                corresponds to the number of columns, but might be larger if using a subset
                of a larger array).  */
            void predict(double numeric_data[], int categ_data[], bool is_col_major,
                         size_t nrows, size_t ld_numeric, size_t ld_categ, bool standardize,
                         double output_depths[], int tree_num[], double per_tree_depths[]);

            /*  Numeric data may also be provided in sparse format, which can be either
                CSC (column-major) or CSR (row-major). If the number of rows is large,
                predictions in CSC format will be faster than in CSR (assuming that
                categorical data is either missing or column-major). Note that for CSC,
                parallelization is done by trees instead of by rows, and outputs are
                subject to numerical rounding error between runs.  */
            void predict(double X_sparse[], int X_ind[], int X_indptr[], bool is_csc,
                         int categ_data[], bool is_col_major, size_t ld_categ, size_t nrows, bool standardize,
                         double output_depths[], int tree_num[], double per_tree_depths[]);

            /*  Distances between observations will be returned either as a triangular matrix
                representing an upper diagonal (length is nrows*(nrows-1)/2), or as a full
                square matrix (length is nrows^2).  */
            std::vector<double> predict_distance(double X[], size_t nrows,
                                                 bool as_kernel,
                                                 bool assume_full_distr, bool standardize,
                                                 bool triangular);

            void predict_distance(double numeric_data[], int categ_data[],
                                  size_t nrows,
                                  bool as_kernel,
                                  bool assume_full_distr, bool standardize,
                                  bool triangular,
                                  double dist_matrix[]);

            /*  Sparse data is only supported in CSC format.  */
            void predict_distance(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                                  size_t nrows,
                                  bool as_kernel,
                                  bool assume_full_distr, bool standardize,
                                  bool triangular,
                                  double dist_matrix[]);

            /*  This will impute missing values in-place. Data here must be in column-major order.   */
            void impute(double X[], size_t nrows);

            /*  This variation will accept data in either row-major or column-major order.
                The leading dimension must match with the number of columns for row major,
                or with the number of rows for column-major (custom leading dimensions are
                not supported).  */
            void impute(double numeric_data[], int categ_data[], bool is_col_major, size_t nrows);

            /*  Numeric data may be passed in sparse CSR format. Note however that it will
                impute the values that are NAN, not the values that are ommited from the
                sparse format.  */
            void impute(double Xr[], int Xr_ind[], int Xr_indptr[],
                        int categ_data[], bool is_col_major, size_t nrows);

            void build_indexer(const bool with_distances);

            /*  Sets points as reference to later calculate distances or kernel from arbitrary points
                to these ones, without having to save these reference points's original features.  */
            void set_as_reference_points(double numeric_data[], int categ_data[], bool is_col_major,
                                         size_t nrows, size_t ld_numeric, size_t ld_categ,
                                         const bool with_distances);

            void set_as_reference_points(double Xc[], int Xc_ind[], int Xc_indptr[], int categ_data[],
                                         size_t nrows, const bool with_distances);

            size_t get_num_reference_points() const noexcept;

            /*  Must call 'set_as_reference_points' to make this method available.

                Here 'dist_matrix' should have dimension [nrows, n_references],
                and will be filled in row-major order.

                This will always take 'assume_full_distr=true'.  */
            void predict_distance_to_ref_points(double numeric_data[], int categ_data[],
                                                double Xc[], int Xc_ind[], int Xc_indptr[],
                                                size_t nrows, bool is_col_major, size_t ld_numeric, size_t ld_categ,
                                                bool as_kernel, bool standardize,
                                                double dist_matrix[]);

            /*  Serialize (save) the model to a file. See 'isotree.hpp' for compatibility
                details. Note that this does not save all the details of the object, but
                rather only those that are necessary for prediction.

                The file must be opened in binary write mode ('wb').

                Note that models serialized through this interface are not importable in
                the R and Python wrappers around this library.  */
            void serialize(FILE *out) const;

            /*  The stream must be opened in binary mode.  */
            void serialize(std::ostream &out) const;

            /*  The number of threads here does not mean 'how many threads to use while
                deserializing', but rather, 'how many threads will be set for the prediction
                functions of the resulting object'.

                The input file must be opened in binary read more ('rb').

                Note that not all the members of an 'IsolationForest' object are saved
                when serializing, so if you access members such as 'prob_pick_by_gain_avg',
                they will all be at their default values.

                These functions can de-serialize models saved from the R and Python interfaces,
                but models that are serialized from this C++ interface are not importable in
                those R and Python versions.  */
            static IsolationForest deserialize(FILE *inp, int nthreads);

            /*  The stream must be opened in binary mode.  */
            static IsolationForest deserialize(std::istream &inp, int nthreads);

            /*  To serialize and deserialize in a more idiomatic way
                ('stream << model' and 'stream >> model').
                Note that 'ist >> model' will set 'nthreads=-1', which you might
                want to modify afterwards. */
            friend std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);

            friend std::istream& operator>>(std::istream &ist, IsolationForest &model);

            /*  These functions allow getting the underlying objects to use with the more
                featureful non-OOP interface.

                Note that it is also possible to use the C-interface functions with this
                object by passing a pointer to the 'IsolationForest' object instead.  */
            IsoForest& get_model();

            ExtIsoForest& get_model_ext();

            Imputer& get_imputer();

            TreesIndexer& get_indexer();

            /*  This converts from a negative 'nthreads' to the actual number (provided it
                was compiled with OpenMP support), and will set to 1 if the number is invalid.
                If the library was compiled without multi-threading and it requests more than
                one thread, will write a message to 'stderr'.  */
            void check_nthreads();

            /*  This will return the number of trees in the object. If it is not fitted, will
                throw an error instead.  */
            size_t get_ntrees() const;

            /*  This checks whether 'predict' can output 'tree_num' and 'per_tree_depths'.  */
            bool check_can_predict_per_tree() const;

        private:
            bool is_fitted = false;

            void override_previous_fit();
            void check_params();
            void check_is_fitted() const;
            IsolationForest(int nthreads, size_t ndim, size_t ntrees, bool build_imputer);
            template <class otype>
            void serialize_template(otype &out) const;
            template <class itype>
            static IsolationForest deserialize_template(itype &inp, int nthreads);

        };

        ISOTREE_EXPORTED
        std::ostream& operator<<(std::ostream &ost, const IsolationForest &model);
        ISOTREE_EXPORTED
        std::istream& operator>>(std::istream &ist, IsolationForest &model);

        }

        #endif /* ifndef ISOTREE_OOP_H */

*isotree.hpp*

        /* mamadou@port-lipn12:~/big-data/cerin24022022/cpp-isotree$ cat example/isotree.hpp

        */

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
              #include "isotree.hpp"
            The header may be included multiple times if required. */
        #ifndef real_t
            #define real_t double     /* supported: float, double */
        #endif
        #ifndef sparse_ix
            #define sparse_ix int  /* supported: int, int64_t, size_t */
        #endif

        #ifndef ISOTREE_H
        #define ISOTREE_H

        #ifdef _WIN32
            #define ISOTREE_EXPORTED __declspec(dllimport)
        #else
            #define ISOTREE_EXPORTED 
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
        typedef struct IsoTree {
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

            IsoTree() = default;

        } IsoTree;

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
            std::vector< std::vector<IsoTree> > trees;
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

        #endif /* ISOTREE_H */

        /*  Fit Isolation Forest model, or variant of it such as SCiForest
        */
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        * 

        */
        ISOTREE_EXPORTED void get_num_nodes(IsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;
        ISOTREE_EXPORTED void get_num_nodes(ExtIsoForest &model_outputs, sparse_ix *n_nodes, sparse_ix *n_terminal, int nthreads) noexcept;



        /* Calculate distance or similarity or kernel/proximity between data points
        * 

        */
        ISOTREE_EXPORTED
        void calc_similarity(real_t numeric_data[], int categ_data[],
                             real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                             size_t nrows, bool use_long_double, int nthreads,
                             bool assume_full_distr, bool standardize_dist, bool as_kernel,
                             IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                             double tmat[], double rmat[], size_t n_from, bool use_indexed_references,
                             TreesIndexer *indexer, bool is_col_major, size_t ld_numeric, size_t ld_categ);

        /* Impute missing values in new data

        */
        ISOTREE_EXPORTED
        void impute_missing_values(real_t numeric_data[], int categ_data[], bool is_col_major,
                                   real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
                                   size_t nrows, bool use_long_double, int nthreads,
                                   IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                   Imputer &imputer);


        /* Append trees from one model into another
        * 

        */
        ISOTREE_EXPORTED
        void merge_models(IsoForest*     model,      IsoForest*     other,
                          ExtIsoForest*  ext_model,  ExtIsoForest*  ext_other,
                          Imputer*       imputer,    Imputer*       iother,
                          TreesIndexer*  indexer,    TreesIndexer*  ind_other);

        /* Create a model containing a sub-set of the trees from another model
        * 

        */
        ISOTREE_EXPORTED
        void subset_model(IsoForest*     model,      IsoForest*     model_new,
                          ExtIsoForest*  ext_model,  ExtIsoForest*  ext_model_new,
                          Imputer*       imputer,    Imputer*       imputer_new,
                          TreesIndexer*  indexer,    TreesIndexer*  indexer_new,
                          size_t *trees_take, size_t ntrees_take);

        /* Build indexer for faster terminal node predictions and/or distance calculations
        * 

        */
        ISOTREE_EXPORTED
        void build_tree_indices(TreesIndexer &indexer, const IsoForest &model, int nthreads, const bool with_distances);
        ISOTREE_EXPORTED
        void build_tree_indices(TreesIndexer &indexer, const ExtIsoForest &model, int nthreads, const bool with_distances);
        ISOTREE_EXPORTED
        void build_tree_indices
        (
            TreesIndexer *indexer,
            const IsoForest *model_outputs,
            const ExtIsoForest *model_outputs_ext,
            int nthreads,
            const bool with_distances
        );
        /* Gets the number of reference points stored in an indexer object */
        ISOTREE_EXPORTED
        size_t get_number_of_reference_points(const TreesIndexer &indexer) noexcept;


        /* Functions to inspect serialized objects
        * 

        */
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
        size_t determine_serialized_size(const IsoForest &model) noexcept;
        ISOTREE_EXPORTED
        size_t determine_serialized_size(const ExtIsoForest &model) noexcept;
        ISOTREE_EXPORTED
        size_t determine_serialized_size(const Imputer &model) noexcept;
        ISOTREE_EXPORTED
        size_t determine_serialized_size(const TreesIndexer &model) noexcept;
        ISOTREE_EXPORTED
        void serialize_IsoForest(const IsoForest &model, char *out);
        ISOTREE_EXPORTED
        void serialize_IsoForest(const IsoForest &model, FILE *out);
        ISOTREE_EXPORTED
        void serialize_IsoForest(const IsoForest &model, std::ostream &out);
        ISOTREE_EXPORTED
        std::string serialize_IsoForest(const IsoForest &model);
        ISOTREE_EXPORTED
        void deserialize_IsoForest(IsoForest &model, const char *in);
        ISOTREE_EXPORTED
        void deserialize_IsoForest(IsoForest &model, FILE *in);
        ISOTREE_EXPORTED
        void deserialize_IsoForest(IsoForest &model, std::istream &in);
        ISOTREE_EXPORTED
        void deserialize_IsoForest(IsoForest &model, const std::string &in);
        ISOTREE_EXPORTED
        void serialize_ExtIsoForest(const ExtIsoForest &model, char *out);
        ISOTREE_EXPORTED
        void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out);
        ISOTREE_EXPORTED
        void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
        ISOTREE_EXPORTED
        std::string serialize_ExtIsoForest(const ExtIsoForest &model);
        ISOTREE_EXPORTED
        void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
        ISOTREE_EXPORTED
        void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in);
        ISOTREE_EXPORTED
        void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
        ISOTREE_EXPORTED
        void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
        ISOTREE_EXPORTED
        void serialize_Imputer(const Imputer &model, char *out);
        ISOTREE_EXPORTED
        void serialize_Imputer(const Imputer &model, FILE *out);
        ISOTREE_EXPORTED
        void serialize_Imputer(const Imputer &model, std::ostream &out);
        ISOTREE_EXPORTED
        std::string serialize_Imputer(const Imputer &model);
        ISOTREE_EXPORTED
        void deserialize_Imputer(Imputer &model, const char *in);
        ISOTREE_EXPORTED
        void deserialize_Imputer(Imputer &model, FILE *in);
        ISOTREE_EXPORTED
        void deserialize_Imputer(Imputer &model, std::istream &in);
        ISOTREE_EXPORTED
        void deserialize_Imputer(Imputer &model, const std::string &in);
        ISOTREE_EXPORTED
        void serialize_Indexer(const TreesIndexer &model, char *out);
        ISOTREE_EXPORTED
        void serialize_Indexer(const TreesIndexer &model, FILE *out);
        ISOTREE_EXPORTED
        void serialize_Indexer(const TreesIndexer &model, std::ostream &out);
        ISOTREE_EXPORTED
        std::string serialize_Indexer(const TreesIndexer &model);
        ISOTREE_EXPORTED
        void deserialize_Indexer(TreesIndexer &model, const char *in);
        ISOTREE_EXPORTED
        void deserialize_Indexer(TreesIndexer &model, FILE *in);
        ISOTREE_EXPORTED
        void deserialize_Indexer(TreesIndexer &model, std::istream &in);
        ISOTREE_EXPORTED
        void deserialize_Indexer(TreesIndexer &model, const std::string &in);


        /* Serialization and de-serialization functions (combined objects)
        *

        */
        ISOTREE_EXPORTED
        size_t determine_serialized_size_combined
        (
            const IsoForest *model,
            const ExtIsoForest *model_ext,
            const Imputer *imputer,
            const TreesIndexer *indexer,
            const size_t size_optional_metadata
        ) noexcept;
        ISOTREE_EXPORTED
        size_t determine_serialized_size_combined
        (
            const char *serialized_model,
            const char *serialized_model_ext,
            const char *serialized_imputer,
            const char *serialized_indexer,
            const size_t size_optional_metadata
        ) noexcept;
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
        std::string serialize_combined
        (
            const IsoForest *model,
            const ExtIsoForest *model_ext,
            const Imputer *imputer,
            const TreesIndexer *indexer,
            const char *optional_metadata,
            const size_t size_optional_metadata
        );
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
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
        ISOTREE_EXPORTED
        std::string serialize_combined
        (
            const char *serialized_model,
            const char *serialized_model_ext,
            const char *serialized_imputer,
            const char *serialized_indexer,
            const char *optional_metadata,
            const size_t size_optional_metadata
        );
        ISOTREE_EXPORTED
        void deserialize_combined
        (
            const char* in,
            IsoForest *model,
            ExtIsoForest *model_ext,
            Imputer *imputer,
            TreesIndexer *indexer,
            char *optional_metadata
        );
        ISOTREE_EXPORTED
        void deserialize_combined
        (
            FILE* in,
            IsoForest *model,
            ExtIsoForest *model_ext,
            Imputer *imputer,
            TreesIndexer *indexer,
            char *optional_metadata
        );
        ISOTREE_EXPORTED
        void deserialize_combined
        (
            std::istream &in,
            IsoForest *model,
            ExtIsoForest *model_ext,
            Imputer *imputer,
            TreesIndexer *indexer,
            char *optional_metadata
        );
        ISOTREE_EXPORTED
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
        * 
        */
        ISOTREE_EXPORTED
        bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes);
        ISOTREE_EXPORTED
        bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes);
        ISOTREE_EXPORTED
        size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees);
        ISOTREE_EXPORTED
        size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees);
        ISOTREE_EXPORTED
        size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees);
        ISOTREE_EXPORTED
        size_t determine_serialized_size_additional_trees(const TreesIndexer &model, size_t old_ntrees);
        ISOTREE_EXPORTED
        void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated);
        ISOTREE_EXPORTED
        void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated);
        ISOTREE_EXPORTED
        void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated);
        ISOTREE_EXPORTED
        void incremental_serialize_Indexer(const TreesIndexer &model, char *old_bytes_reallocated);
        ISOTREE_EXPORTED
        void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes);
        ISOTREE_EXPORTED
        void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes);
        ISOTREE_EXPORTED
        void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes);
        ISOTREE_EXPORTED
        void incremental_serialize_Indexer(const TreesIndexer &model, std::string &old_bytes);


        /* Translate isolation forest model into a single SQL select statement
        * 
        */
        ISOTREE_EXPORTED
        std::string generate_sql_with_select_from(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                                  std::string &table_from, std::string &select_as,
                                                  std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                                  std::vector<std::vector<std::string>> &categ_levels,
                                                  bool index1, int nthreads);


        /* Translate model trees into SQL select statements
        * 

        */
        ISOTREE_EXPORTED
        std::vector<std::string> generate_sql(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                                              std::vector<std::string> &numeric_colnames, std::vector<std::string> &categ_colnames,
                                              std::vector<std::vector<std::string>> &categ_levels,
                                              bool output_tree_num, bool index1, bool single_tree, size_t tree_num,
                                              int nthreads);


        ISOTREE_EXPORTED
        void set_reference_points(IsoForest *model_outputs, ExtIsoForest *model_outputs_ext, TreesIndexer *indexer,
                                  const bool with_distances,
                                  real_t *numeric_data, int *categ_data,
                                  bool is_col_major, size_t ld_numeric, size_t ld_categ,
                                  real_t *Xc, sparse_ix *Xc_ind, sparse_ix *Xc_indptr,
                                  real_t *Xr, sparse_ix *Xr_ind, sparse_ix *Xr_indptr,
                                  size_t nrows, int nthreads);

**Exécution**

        mamadou@port-lipn12:~/big-data/cerin24022022/cpp-isotree/example$./isotree_cpp_oop_ex 
        Point with highest outlier score: [3, 3]
