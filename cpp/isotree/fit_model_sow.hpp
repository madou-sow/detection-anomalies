/*    Isolation forests and variations thereof, with adjustments for incorporation
*     of categorical variables and missing values.
*     Writen for C++11 standard and aimed at being used in R and Python.
*     
*     This library is based on the following works:
*     [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation forest."
*         2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
*     [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation-based anomaly detection."
*         ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
*     [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
*         "Extended Isolation Forest."
*         arXiv preprint arXiv:1811.02141 (2018).
*     [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "On detecting clustered anomalies using SCiForest."
*         Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
*     [5] https://sourceforge.net/projects/iforest/
*     [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
*     [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
*     [8] Cortes, David.
*         "Distance approximation using Isolation Forests."
*         arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David.
*         "Imputing missing values with unsupervised random trees."
*         arXiv preprint arXiv:1911.06646 (2019).
*     [10] https://math.stackexchange.com/questions/3333220/expected-average-depth-in-random-binary-tree-constructed-top-to-bottom
*     [11] Cortes, David.
*          "Revisiting randomized choices in isolation forests."
*          arXiv preprint arXiv:2110.13402 (2021).
*     [12] Guha, Sudipto, et al.
*          "Robust random cut forest based anomaly detection on streams."
*          International conference on machine learning. PMLR, 2016.
*     [13] Cortes, David.
*          "Isolation forests: looking beyond tree depth."
*          arXiv preprint arXiv:2111.11639 (2021).
*     [14] Ting, Kai Ming, Yue Zhu, and Zhi-Hua Zhou.
*          "Isolation kernel and its effect on SVM"
*          Proceedings of the 24th ACM SIGKDD
*          International Conference on Knowledge Discovery & Data Mining. 2018.
* 
*     BSD 2-Clause License
*     Copyright (c) 2019-2022, David Cortes
*     All rights reserved.
*     Redistribution and use in source and binary forms, with or without
*     modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this
*       list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice,
*       this list of conditions and the following disclaimer in the documentation
*       and/or other materials provided with the distribution.
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "isotree.hpp"

template <class real_t, class sparse_ix>
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
                uint64_t random_seed, bool use_long_double, int nthreads)
{
    if (use_long_double && !has_long_double()) {
        use_long_double = false;
        fprintf(stderr, "Passed 'use_long_double=true', but library was compiled without long double support.\n");
    }
    #ifndef NO_LONG_DOUBLE
    if (likely(!use_long_double))
    #endif
        return fit_iforest_internal<real_t, sparse_ix, double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, with_replacement, weight_as_sample,
            nrows, sample_size, ntrees,
            max_depth, ncols_per_tree,
            limit_depth, penalize_range, standardize_data,
            scoring_metric, fast_bratio,
            standardize_dist, tmat,
            output_depths, standardize_depth,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            all_perm, imputer, min_imp_obs,
            depth_imp, weigh_imp_rows, impute_at_fit,
            random_seed, nthreads
        );
    #ifndef NO_LONG_DOUBLE
    else
        return fit_iforest_internal<real_t, sparse_ix, long double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, with_replacement, weight_as_sample,
            nrows, sample_size, ntrees,
            max_depth, ncols_per_tree,
            limit_depth, penalize_range, standardize_data,
            scoring_metric, fast_bratio,
            standardize_dist, tmat,
            output_depths, standardize_depth,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            all_perm, imputer, min_imp_obs,
            depth_imp, weigh_imp_rows, impute_at_fit,
            random_seed, nthreads
        );
    #endif
}

template <class real_t, class sparse_ix, class ldouble_safe>
int fit_iforest_internal(
                IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],  size_t ncols_numeric,
                int    categ_data[],    size_t ncols_categ,    int ncat[],
                real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                size_t ndim, size_t ntry, CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[], bool with_replacement, bool weight_as_sample,
                size_t nrows, size_t sample_size, size_t ntrees,
                size_t max_depth, size_t ncols_per_tree,
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
                uint64_t random_seed, int nthreads)
{
    if (
        prob_pick_by_gain_avg  < 0 || prob_pick_by_gain_pl  < 0 ||
        prob_pick_by_full_gain < 0 || prob_pick_by_dens     < 0 ||
        prob_pick_col_by_range < 0 ||
        prob_pick_col_by_var   < 0 || prob_pick_col_by_kurt < 0
    ) {
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    }
    if (prob_pick_col_by_range && ncols_categ)
        throw std::runtime_error("'prob_pick_col_by_range' is not compatible with categorical data.\n");
    if (prob_pick_by_full_gain && ncols_categ)
        throw std::runtime_error("'prob_pick_by_full_gain' is not compatible with categorical data.\n");
    if (prob_pick_col_by_kurt && weigh_by_kurt)
        throw std::runtime_error("'weigh_by_kurt' and 'prob_pick_col_by_kurt' cannot be used together.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");
    if (penalize_range &&
        (scoring_metric == Density ||
         scoring_metric == AdjDensity ||
         is_boxed_metric(scoring_metric))
    )
        throw std::runtime_error("'penalize_range' is incompatible with density scoring.\n");
    if (with_replacement) {
        if (tmat != NULL)
            throw std::runtime_error("Cannot calculate distance while sampling with replacement.\n");
        if (output_depths != NULL)
            throw std::runtime_error("Cannot make predictions at fit time when sampling with replacement.\n");
        if (impute_at_fit)
            throw std::runtime_error("Cannot impute at fit time when sampling with replacement.\n");
    }
    if (sample_size != 0 && sample_size < nrows) {
        if (output_depths != NULL)
            throw std::runtime_error("Cannot produce outlier scores at fit time when using sub-sampling.\n");
        if (tmat != NULL)
            throw std::runtime_error("Cannot calculate distances at fit time when using sub-sampling.\n");
        if (impute_at_fit)
            throw std::runtime_error("Cannot produce missing data imputations at fit time when using sub-sampling.\n");
    }


    /* TODO: this function should also accept the array as a memoryview with a
       leading dimension that might not correspond to the number of columns,
       so as to avoid having to make deep copies of memoryviews in python and to
       allow using pointers to columns of dataframes in R and Python. */

    /* calculate maximum number of categories to use later */
    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    bool calc_dist = tmat != NULL;

    if (sample_size == 0)
        sample_size = nrows;

    if (model_outputs != NULL)
        ntry = std::min(ntry, ncols_numeric + ncols_categ);

    if (ncols_per_tree == 0)
        ncols_per_tree = ncols_numeric + ncols_categ;

    /* put data in structs to shorten function calls */
    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                weight_as_sample, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0, NULL,
                                (double*)NULL, (double*)NULL, (int*)NULL, std::vector<double>(),
                                std::vector<double>(), std::vector<double>(),
                                std::vector<size_t>(), std::vector<size_t>()};
    ModelParams model_params = {with_replacement, sample_size, ntrees, ncols_per_tree,
                                limit_depth? log2ceil(sample_size) : max_depth? max_depth : (sample_size - 1),
                                penalize_range, standardize_data, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, prob_pick_by_gain_pl,
                                prob_pick_by_full_gain, prob_pick_by_dens,
                                prob_pick_col_by_range, prob_pick_col_by_var,
                                prob_pick_col_by_kurt,
                                min_gain, cat_split_type, new_cat_action, missing_action,
                                scoring_metric, fast_bratio, all_perm,
                                (model_outputs != NULL)? 0 : ndim, ntry,
                                coef_type, coef_by_prop, calc_dist, (bool)(output_depths != NULL), impute_at_fit,
                                depth_imp, weigh_imp_rows, min_imp_obs};

    /* if calculating full gain, need to produce copies of the data in row-major order */
    if (prob_pick_by_full_gain)
    {
        if (input_data.Xc_indptr == NULL)
            colmajor_to_rowmajor(input_data.numeric_data, input_data.nrows, input_data.ncols_numeric, input_data.X_row_major);
        else
            colmajor_to_rowmajor(input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                 input_data.nrows, input_data.ncols_numeric,
                                 input_data.Xr, input_data.Xr_ind, input_data.Xr_indptr);
    }

    /* if using weights as sampling probability, build a binary tree for faster sampling */
    if (input_data.weight_as_sample && input_data.sample_weights != NULL)
    {
        build_btree_sampler(input_data.btree_weights_init, input_data.sample_weights,
                            input_data.nrows, input_data.log2_n, input_data.btree_offset);
    }

    /* same for column weights */
    /* TODO: this should also save the kurtoses when using 'prob_pick_col_by_kurt' */
    ColumnSampler<ldouble_safe> base_col_sampler;
    if (
        col_weights != NULL ||
        (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
         (model_params.ncols_per_tree >= input_data.ncols_tot / (model_params.ntrees * 2)))
    )
    {
        bool avoid_col_weights = (model_outputs != NULL && model_params.ntry >= model_params.ncols_per_tree &&
                                  model_params.prob_pick_by_gain_avg  + model_params.prob_pick_by_gain_pl +
                                  model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1)
                                    ||
                                 (model_outputs == NULL && model_params.ndim >= model_params.ncols_per_tree)
                                    ||
                                 (model_params.ncols_per_tree == 1);
        if (!avoid_col_weights)
        {
            if (model_params.weigh_by_kurt && model_params.sample_size == input_data.nrows && !model_params.with_replacement)
            {
                RNG_engine rnd_generator(random_seed);
                std::vector<double> kurt_weights = calc_kurtosis_all_data<InputData<real_t, sparse_ix>, ldouble_safe>(input_data, model_params, rnd_generator);
                if (col_weights != NULL)
                {
                    for (size_t col = 0; col < input_data.ncols_tot; col++)
                    {
                        if (kurt_weights[col] <= 0) continue;
                        kurt_weights[col] *= col_weights[col];
                        kurt_weights[col]  = std::fmax(kurt_weights[col], 1e-100);
                    }
                }
                base_col_sampler.initialize(kurt_weights.data(), input_data.ncols_tot);

                if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
                {
                    input_data.all_kurtoses = std::move(kurt_weights);
                }
            }

            else
            {
                base_col_sampler.initialize(input_data.col_weights, input_data.ncols_tot);
            }

            input_data.preinitialized_col_sampler = &base_col_sampler;
        }
    }

    /* in some cases, all trees will need to calculate variable ranges for all columns */
    /* TODO: the model might use 'leave_m_cols', or have 'prob_pick_col_by_range<1', in which
       case it might not be beneficial to do this beforehand. Find out when the expected gain
       from doing this here is not beneficial. */
    /* TODO: move this to a different file, it doesn't belong here */
    std::vector<double> variable_ranges_low;
    std::vector<double> variable_ranges_high;
    std::vector<int> variable_ncats;
    if (
        model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
        (model_params.ncols_per_tree >= input_data.ncols_numeric) &&
        ((model_params.prob_pick_col_by_range && input_data.ncols_numeric)
            ||
         is_boxed_metric(model_params.scoring_metric))
    )
    {
        variable_ranges_low.resize(input_data.ncols_numeric);
        variable_ranges_high.resize(input_data.ncols_numeric);

        std::unique_ptr<unsigned char[]> buffer_cats;
        size_t adj_col;
        if (is_boxed_metric(model_params.scoring_metric))
        {
            variable_ncats.resize(input_data.ncols_categ);
            buffer_cats = std::unique_ptr<unsigned char[]>(new unsigned char[input_data.max_categ]);
        }

        if (base_col_sampler.col_indices.empty())
            base_col_sampler.initialize(input_data.ncols_tot);

        bool unsplittable;
        size_t n_tried_numeric = 0;
        size_t col;
        base_col_sampler.prepare_full_pass();
        while (base_col_sampler.sample_col(col))
        {
            if (col < input_data.ncols_numeric)
            {
                if (input_data.Xc_indptr == NULL)
                {
                    get_range(input_data.numeric_data + nrows*col,
                              input_data.nrows,
                              model_params.missing_action,
                              variable_ranges_low[col],
                              variable_ranges_high[col],
                              unsplittable);
                }

                else
                {
                    get_range(col, input_data.nrows,
                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                              model_params.missing_action,
                              variable_ranges_low[col],
                              variable_ranges_high[col],
                              unsplittable);
                }

                n_tried_numeric++;

                if (unsplittable)
                {
                    variable_ranges_low[col] = 0;
                    variable_ranges_high[col] = 0;
                    base_col_sampler.drop_col(col);
                }
            }

            else
            {
                if (!is_boxed_metric(model_params.scoring_metric))
                {
                    if (n_tried_numeric >= input_data.ncols_numeric)
                        break;
                    else
                        continue;
                }
                adj_col = col - input_data.ncols_numeric;


                variable_ncats[adj_col] = count_ncateg_in_col(input_data.categ_data + input_data.nrows*adj_col,
                                                              input_data.nrows, input_data.ncat[adj_col],
                                                              buffer_cats.get());
                if (variable_ncats[adj_col] <= 1)
                    base_col_sampler.drop_col(col);
            }
        }

        input_data.preinitialized_col_sampler = &base_col_sampler;
        if (input_data.ncols_numeric) {
            input_data.range_low = variable_ranges_low.data();
            input_data.range_high = variable_ranges_high.data();
        }
        if (input_data.ncols_categ) {
            input_data.ncat_ = variable_ncats.data();
        }
    }

    /* if imputing missing values on-the-fly, need to determine which are missing */
    std::vector<ImputedData<sparse_ix, ldouble_safe>> impute_vec;
    hashed_map<size_t, ImputedData<sparse_ix, ldouble_safe>> impute_map;
    if (model_params.impute_at_fit)
        check_for_missing(input_data, impute_vec, impute_map, nthreads);

    /* store model data */
    if (model_outputs != NULL)
    {
        model_outputs->trees.resize(ntrees);
        model_outputs->trees.shrink_to_fit();
        model_outputs->new_cat_action = new_cat_action;
        model_outputs->cat_split_type = cat_split_type;
        model_outputs->missing_action = missing_action;
        model_outputs->scoring_metric = scoring_metric;
        if (
            model_outputs->scoring_metric != Density &&
            model_outputs->scoring_metric != BoxedDensity &&
            model_outputs->scoring_metric != BoxedDensity2 &&
            model_outputs->scoring_metric != BoxedRatio
        )
            model_outputs->exp_avg_depth  = expected_avg_depth<ldouble_safe>(sample_size);
        else
            model_outputs->exp_avg_depth  = 1;
        model_outputs->exp_avg_sep = expected_separation_depth<ldouble_safe>(model_params.sample_size);
        model_outputs->orig_sample_size = input_data.nrows;
        model_outputs->has_range_penalty = penalize_range;
    }

    else
    {
        model_outputs_ext->hplanes.resize(ntrees);
        model_outputs_ext->hplanes.shrink_to_fit();
        model_outputs_ext->new_cat_action = new_cat_action;
        model_outputs_ext->cat_split_type = cat_split_type;
        model_outputs_ext->missing_action = missing_action;
        model_outputs_ext->scoring_metric = scoring_metric;
        if (
            model_outputs_ext->scoring_metric != Density &&
            model_outputs_ext->scoring_metric != BoxedDensity &&
            model_outputs_ext->scoring_metric != BoxedDensity2 &&
            model_outputs_ext->scoring_metric != BoxedRatio
        )
            model_outputs_ext->exp_avg_depth  = expected_avg_depth<ldouble_safe>(sample_size);
        else
            model_outputs_ext->exp_avg_depth  = 1;
        model_outputs_ext->exp_avg_sep = expected_separation_depth<ldouble_safe>(model_params.sample_size);
        model_outputs_ext->orig_sample_size = input_data.nrows;
        model_outputs_ext->has_range_penalty = penalize_range;
    }

    if (imputer != NULL)
        initialize_imputer<decltype(input_data), ldouble_safe>(
            *imputer, input_data, ntrees, nthreads
        );

    /* initialize thread-private memory */
    if ((size_t)nthreads > ntrees)
        nthreads = (int)ntrees;
    #ifdef _OPENMP
        std::vector<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> worker_memory(nthreads);
    #else
        std::vector<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> worker_memory(1);
    #endif

    /* Global variable that determines if the procedure receives a stop signal */
    SignalSwitcher ss = SignalSwitcher();

    /* For exception handling */
    bool threw_exception = false;
    std::exception_ptr ex = NULL;

    /* grow trees */
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic) shared(model_outputs, model_outputs_ext, worker_memory, input_data, model_params, threw_exception, ex)
    for (size_t_for tree = 0; tree < (decltype(tree))ntrees; tree++)
    {
        if (interrupt_switch || threw_exception)
            continue; /* Cannot break with OpenMP==2.0 (MSVC) */

        try
        {
            if (
                model_params.impute_at_fit &&
                input_data.n_missing &&
                !worker_memory[omp_get_thread_num()].impute_vec.size() &&
                !worker_memory[omp_get_thread_num()].impute_map.size()
                )
            {
                #ifdef _OPENMP
                if (nthreads > 1)
                {
                    worker_memory[omp_get_thread_num()].impute_vec = impute_vec;
                    worker_memory[omp_get_thread_num()].impute_map = impute_map;
                }

                else
                #endif
                {
                    worker_memory[0].impute_vec = std::move(impute_vec);
                    worker_memory[0].impute_map = std::move(impute_map);
                }
            }

            fit_itree<decltype(input_data), typename std::remove_pointer<decltype(worker_memory.data())>::type, ldouble_safe>(
                      (model_outputs != NULL)? &model_outputs->trees[tree] : NULL,
                      (model_outputs_ext != NULL)? &model_outputs_ext->hplanes[tree] : NULL,
                      worker_memory[omp_get_thread_num()],
                      input_data,
                      model_params,
                      (imputer != NULL)? &(imputer->imputer_tree[tree]) : NULL,
                      tree);

            if ((model_outputs != NULL))
                model_outputs->trees[tree].shrink_to_fit();
            else
                model_outputs_ext->hplanes[tree].shrink_to_fit();
        }

        catch (...)
        {
            #pragma omp critical
            {
                if (!threw_exception)
                {
                    threw_exception = true;
                    ex = std::current_exception();
                }
            }
        }
    }

    /* check if the procedure got interrupted */
    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* check if some exception was thrown */
    if (threw_exception)
        std::rethrow_exception(ex);

    if ((model_outputs != NULL))
        model_outputs->trees.shrink_to_fit();
    else
        model_outputs_ext->hplanes.shrink_to_fit();

    /* if calculating similarity/distance, now need to reduce and average */
    if (calc_dist)
        gather_sim_result< PredictionData<real_t, sparse_ix>, InputData<real_t, sparse_ix> >
                         (NULL, &worker_memory,
                          NULL, &input_data,
                          model_outputs, model_outputs_ext,
                          tmat, NULL, 0,
                          model_params.ntrees, false,
                          standardize_dist, false, nthreads);

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* same for depths */
    if (output_depths != NULL)
    {
        #ifdef _OPENMP
        if (nthreads > 1)
        {
            for (auto &w : worker_memory)
            {
                if (w.row_depths.size())
                {
                    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(input_data, output_depths, w, worker_memory)
                    for (size_t_for row = 0; row < (decltype(row))input_data.nrows; row++)
                        output_depths[row] += w.row_depths[row];
                }
            }
        }
        else
        #endif
        {
            std::copy(worker_memory[0].row_depths.begin(), worker_memory[0].row_depths.end(), output_depths);
        }

        if (standardize_depth)
        {
            double depth_divisor = (double)ntrees * ((model_outputs != NULL)?
                                                     model_outputs->exp_avg_depth : model_outputs_ext->exp_avg_depth);
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] = std::exp2( - output_depths[row] / depth_divisor );
        }

        else
        {
            double ntrees_dbl = (double) ntrees;
            for (size_t row = 0; row < nrows; row++)
                output_depths[row] /= ntrees_dbl;
        }
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    /* if imputing missing values, now need to reduce and write final values */
    if (model_params.impute_at_fit)
    {
        #ifdef _OPENMP
        if (nthreads > 1)
        {
            for (auto &w : worker_memory)
                combine_tree_imputations(w, impute_vec, impute_map, input_data.has_missing, nthreads);
        }

        else
        #endif
        {
            impute_vec = std::move(worker_memory[0].impute_vec);
            impute_map = std::move(worker_memory[0].impute_map);
        }

        apply_imputation_results(impute_vec, impute_map, *imputer, input_data, nthreads);
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return EXIT_FAILURE;
    #endif

    return EXIT_SUCCESS;
}


template <class real_t, class sparse_ix>
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
             uint64_t random_seed, bool use_long_double)
{
    if (use_long_double && !has_long_double()) {
        use_long_double = false;
        fprintf(stderr, "Passed 'use_long_double=true', but library was compiled without long double support.\n");
    }
    #ifndef NO_LONG_DOUBLE
    if (likely(!use_long_double))
    #endif
        return add_tree_internal<real_t, sparse_ix, double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, nrows,
            max_depth,     ncols_per_tree,
            limit_depth,   penalize_range, standardize_data,
            fast_bratio,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            depth_imp, weigh_imp_rows,
            all_perm, imputer, min_imp_obs,
            indexer,
            ref_numeric_data, ref_categ_data,
            ref_is_col_major, ref_ld_numeric, ref_ld_categ,
            ref_Xc, ref_Xc_ind, ref_Xc_indptr,
            random_seed
        );
    #ifndef NO_LONG_DOUBLE
    else
        return add_tree_internal<real_t, sparse_ix, long double>(
            model_outputs, model_outputs_ext,
            numeric_data,  ncols_numeric,
            categ_data,    ncols_categ,    ncat,
            Xc, Xc_ind, Xc_indptr,
            ndim, ntry, coef_type, coef_by_prop,
            sample_weights, nrows,
            max_depth,     ncols_per_tree,
            limit_depth,   penalize_range, standardize_data,
            fast_bratio,
            col_weights, weigh_by_kurt,
            prob_pick_by_gain_pl, prob_pick_by_gain_avg,
            prob_pick_by_full_gain, prob_pick_by_dens,
            prob_pick_col_by_range, prob_pick_col_by_var,
            prob_pick_col_by_kurt,
            min_gain, missing_action,
            cat_split_type, new_cat_action,
            depth_imp, weigh_imp_rows,
            all_perm, imputer, min_imp_obs,
            indexer,
            ref_numeric_data, ref_categ_data,
            ref_is_col_major, ref_ld_numeric, ref_ld_categ,
            ref_Xc, ref_Xc_ind, ref_Xc_indptr,
            random_seed
        );
    #endif
}

template <class real_t, class sparse_ix, class ldouble_safe>
int add_tree_internal(
             IsoForest *model_outputs, ExtIsoForest *model_outputs_ext,
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
             uint64_t random_seed)
{
    if (
        prob_pick_by_gain_avg  < 0  || prob_pick_by_gain_pl  < 0 ||
        prob_pick_by_full_gain < 0  || prob_pick_by_dens     < 0 ||
        prob_pick_col_by_range < 0  ||
        prob_pick_col_by_var   < 0  || prob_pick_col_by_kurt < 0
    ) {
        throw std::runtime_error("Cannot pass negative probabilities.\n");
    }
    if (prob_pick_col_by_range && ncols_categ)
        throw std::runtime_error("'prob_pick_col_by_range' is not compatible with categorical data.\n");
    if (prob_pick_by_full_gain && ncols_categ)
        throw std::runtime_error("'prob_pick_by_full_gain' is not compatible with categorical data.\n");
    if (prob_pick_col_by_kurt && weigh_by_kurt)
        throw std::runtime_error("'weigh_by_kurt' and 'prob_pick_col_by_kurt' cannot be used together.\n");
    if (ndim == 0 && model_outputs == NULL)
        throw std::runtime_error("Must pass 'ndim>0' in the extended model.\n");
    if (indexer != NULL && !indexer->indices.empty() && !indexer->indices.front().reference_points.empty()) {
        if (ref_numeric_data == NULL && ref_categ_data == NULL && ref_Xc_indptr == NULL)
            throw std::runtime_error("'indexer' has reference points. Those points must be passed to index them in the new tree to add.\n");
    }

    std::vector<ImputeNode> *impute_nodes = NULL;

    int max_categ = 0;
    for (size_t col = 0; col < ncols_categ; col++)
        max_categ = (ncat[col] > max_categ)? ncat[col] : max_categ;

    if (model_outputs != NULL)
        ntry = std::min(ntry, ncols_numeric + ncols_categ);

    if (ncols_per_tree == 0)
        ncols_per_tree = ncols_numeric + ncols_categ;

    if (indexer != NULL && indexer->indices.empty())
        indexer = NULL;

    InputData<real_t, sparse_ix>
              input_data     = {numeric_data, ncols_numeric, categ_data, ncat, max_categ, ncols_categ,
                                nrows, ncols_numeric + ncols_categ, sample_weights,
                                false, col_weights,
                                Xc, Xc_ind, Xc_indptr,
                                0, 0, std::vector<double>(),
                                std::vector<char>(), 0, NULL,
                                (double*)NULL, (double*)NULL, (int*)NULL, std::vector<double>(),
                                std::vector<double>(), std::vector<double>(),
                                std::vector<size_t>(), std::vector<size_t>()};
    ModelParams model_params = {false, nrows, (size_t)1, ncols_per_tree,
                                max_depth? max_depth : (nrows - 1),
                                penalize_range, standardize_data, random_seed, weigh_by_kurt,
                                prob_pick_by_gain_avg, prob_pick_by_gain_pl,
                                prob_pick_by_full_gain, prob_pick_by_dens,
                                prob_pick_col_by_range, prob_pick_col_by_var,
                                prob_pick_col_by_kurt,
                                min_gain, cat_split_type, new_cat_action, missing_action,
                                (model_outputs != NULL)? model_outputs->scoring_metric : model_outputs_ext->scoring_metric,
                                fast_bratio, all_perm,
                                (model_outputs != NULL)? 0 : ndim, ntry,
                                coef_type, coef_by_prop, false, false, false, depth_imp, weigh_imp_rows, min_imp_obs};

    if (prob_pick_by_full_gain)
    {
        if (input_data.Xc_indptr == NULL)
            colmajor_to_rowmajor(input_data.numeric_data, input_data.nrows, input_data.ncols_numeric, input_data.X_row_major);
        else
            colmajor_to_rowmajor(input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                 input_data.nrows, input_data.ncols_numeric,
                                 input_data.Xr, input_data.Xr_ind, input_data.Xr_indptr);
    }

    std::unique_ptr<WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>> workspace(
        new WorkerMemory<ImputedData<sparse_ix, ldouble_safe>, ldouble_safe, real_t>()
    );

    size_t last_tree;
    bool added_tree = false;
    try
    {
        if (model_outputs != NULL)
        {
            last_tree = model_outputs->trees.size();
            model_outputs->trees.emplace_back();
            added_tree = true;
        }

        else
        {
            last_tree = model_outputs_ext->hplanes.size();
            model_outputs_ext->hplanes.emplace_back();
            added_tree = true;
        }

        if (imputer != NULL)
        {
            imputer->imputer_tree.emplace_back();
            impute_nodes = &(imputer->imputer_tree.back());
        }

        if (indexer != NULL)
        {
            indexer->indices.emplace_back();
        }

        SignalSwitcher ss = SignalSwitcher();
        check_interrupt_switch(ss);

        fit_itree<decltype(input_data), typename std::remove_pointer<decltype(workspace.get())>::type, ldouble_safe>(
                  (model_outputs != NULL)? &model_outputs->trees.back() : NULL,
                  (model_outputs_ext != NULL)? &model_outputs_ext->hplanes.back() : NULL,
                  *workspace,
                  input_data,
                  model_params,
                  impute_nodes,
                  last_tree);

        check_interrupt_switch(ss);

        if (model_outputs != NULL) {
            model_outputs->trees.back().shrink_to_fit();
            model_outputs->has_range_penalty = model_outputs->has_range_penalty || penalize_range;
        }
        else {
            model_outputs_ext->hplanes.back().shrink_to_fit();
            model_outputs_ext->has_range_penalty = model_outputs_ext->has_range_penalty || penalize_range;
        }

        if (imputer != NULL)
            imputer->imputer_tree.back().shrink_to_fit();

        if (indexer != NULL)
        {
            if (model_outputs != NULL)
                build_terminal_node_mappings_single_tree(indexer->indices.back().terminal_node_mappings,
                                                         indexer->indices.back().n_terminal,
                                                         model_outputs->trees.back());
            else
                build_terminal_node_mappings_single_tree(indexer->indices.back().terminal_node_mappings,
                                                         indexer->indices.back().n_terminal,
                                                         model_outputs_ext->hplanes.back());

            check_interrupt_switch(ss);


            if (!indexer->indices.front().node_distances.empty())
            {
                std::vector<size_t> temp;
                temp.reserve(indexer->indices.back().n_terminal);
                if (model_outputs != NULL) {
                    build_dindex(
                        temp,
                        indexer->indices.back().terminal_node_mappings,
                        indexer->indices.back().node_distances,
                        indexer->indices.back().node_depths,
                        indexer->indices.back().n_terminal,
                        model_outputs->trees.back()
                    );
                }
                else {
                    build_dindex(
                        temp,
                        indexer->indices.back().terminal_node_mappings,
                        indexer->indices.back().node_distances,
                        indexer->indices.back().node_depths,
                        indexer->indices.back().n_terminal,
                        model_outputs_ext->hplanes.back()
                    );
                }
            }

            check_interrupt_switch(ss);
            if (!indexer->indices.front().reference_points.empty())
            {
                size_t n_ref = indexer->indices.front().reference_points.size();
                std::vector<sparse_ix> terminal_indices(n_ref);
                std::unique_ptr<double[]> ignored(new double[n_ref]);
                if (model_outputs != NULL)
                {
                    IsoForest single_tree_model;
                    single_tree_model.new_cat_action = model_outputs->new_cat_action;
                    single_tree_model.cat_split_type = model_outputs->cat_split_type;
                    single_tree_model.missing_action = model_outputs->missing_action;
                    single_tree_model.trees.push_back(model_outputs->trees.back());

                    predict_iforest(ref_numeric_data, ref_categ_data,
                                    ref_is_col_major, ref_ld_numeric, ref_ld_categ,
                                    ref_Xc, ref_Xc_ind, ref_Xc_indptr,
                                    (real_t*)NULL, (sparse_ix*)NULL, (sparse_ix*)NULL,
                                    n_ref, 1, false,
                                    &single_tree_model, (ExtIsoForest*)NULL,
                                    ignored.get(), terminal_indices.data(),
                                    (double*)NULL,
                                    indexer);
                }

                else
                {
                    ExtIsoForest single_tree_model;
                    single_tree_model.new_cat_action = model_outputs_ext->new_cat_action;
                    single_tree_model.cat_split_type = model_outputs_ext->cat_split_type;
                    single_tree_model.missing_action = model_outputs_ext->missing_action;
                    single_tree_model.hplanes.push_back(model_outputs_ext->hplanes.back());

                    predict_iforest(ref_numeric_data, ref_categ_data,
                                    ref_is_col_major, ref_ld_numeric, ref_ld_categ,
                                    ref_Xc, ref_Xc_ind, ref_Xc_indptr,
                                    (real_t*)NULL, (sparse_ix*)NULL, (sparse_ix*)NULL,
                                    n_ref, 1, false,
                                    (IsoForest*)NULL, &single_tree_model,
                                    ignored.get(), terminal_indices.data(),
                                    (double*)NULL,
                                    indexer);
                }

                ignored.reset();
                indexer->indices.back().reference_points.assign(terminal_indices.begin(), terminal_indices.end());
                indexer->indices.back().reference_points.shrink_to_fit();
                build_ref_node(indexer->indices.back());
            }

            check_interrupt_switch(ss);
        }
    }

    catch (...)
    {
        if (added_tree)
        {
            if (model_outputs != NULL)
                model_outputs->trees.pop_back();
            else
                model_outputs_ext->hplanes.pop_back();
            if (imputer != NULL) {
                if (model_outputs != NULL)
                    imputer->imputer_tree.resize(model_outputs->trees.size());
                else
                    imputer->imputer_tree.resize(model_outputs_ext->hplanes.size());
            }
            if (indexer != NULL) {
                if (model_outputs != NULL)
                    indexer->indices.resize(model_outputs->trees.size());
                else
                    indexer->indices.resize(model_outputs_ext->hplanes.size());
            }
        }
        throw;
    }

    return EXIT_SUCCESS;
}

template <class InputData, class WorkerMemory, class ldouble_safe>
void fit_itree(std::vector<IsoTree>    *tree_root,
               std::vector<IsoHPlane>  *hplane_root,
               WorkerMemory             &workspace,
               InputData                &input_data,
               ModelParams              &model_params,
               std::vector<ImputeNode> *impute_nodes,
               size_t                   tree_num)
{
    /* initialize array for depths if called for */
    if (workspace.ix_arr.empty() && model_params.calc_depth)
        workspace.row_depths.resize(input_data.nrows, 0);

    /* choose random sample of rows */
    if (workspace.ix_arr.empty()) workspace.ix_arr.resize(model_params.sample_size);
    if (input_data.log2_n > 0)
        workspace.btree_weights.assign(input_data.btree_weights_init.begin(),
                                       input_data.btree_weights_init.end());
    workspace.rnd_generator.seed(model_params.random_seed + tree_num);
    workspace.rbin  = UniformUnitInterval(0, 1);
    sample_random_rows<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                       workspace.ix_arr, input_data.nrows, model_params.with_replacement,
                       workspace.rnd_generator, workspace.ix_all,
                       (input_data.weight_as_sample)? input_data.sample_weights : NULL,
                       workspace.btree_weights, input_data.log2_n, input_data.btree_offset,
                       workspace.is_repeated);
    workspace.st  = 0;
    workspace.end = model_params.sample_size - 1;

    /* in some cases, it's not possible to use column weights even if they are given,
       because every single column will always need to be checked or end up being used. */
    bool avoid_col_weights = (tree_root != NULL && model_params.ntry >= model_params.ncols_per_tree &&
                              model_params.prob_pick_by_gain_avg  + model_params.prob_pick_by_gain_pl +
                              model_params.prob_pick_by_full_gain + model_params.prob_pick_by_dens >= 1)
                                ||
                             (tree_root == NULL && model_params.ndim >= model_params.ncols_per_tree)
                                ||
                             (model_params.ncols_per_tree == 1);
    if (input_data.preinitialized_col_sampler == NULL)
    {
        if (input_data.col_weights != NULL && !avoid_col_weights && !model_params.weigh_by_kurt)
            workspace.col_sampler.initialize(input_data.col_weights, input_data.ncols_tot);
    }


    /* set expected tree size and add root node */
    {
        size_t exp_nodes = mult2(model_params.sample_size);
        if (model_params.sample_size >= div2(SIZE_MAX))
            exp_nodes = SIZE_MAX;
        else if (model_params.max_depth <= (size_t)30)
            exp_nodes = std::min(exp_nodes, pow2(model_params.max_depth));
        if (tree_root != NULL)
        {
            tree_root->reserve(exp_nodes);
            tree_root->emplace_back();
        }
        else
        {
            hplane_root->reserve(exp_nodes);
            hplane_root->emplace_back();
        }
        if (impute_nodes != NULL)
        {
            impute_nodes->reserve(exp_nodes);
            impute_nodes->emplace_back((size_t) 0);
        }
    }

    /* initialize array with candidate categories if not already done */
    if (workspace.categs.empty())
        workspace.categs.resize(input_data.max_categ);

    /* initialize array with per-node column weights if needed */
    if ((model_params.prob_pick_col_by_range ||
         model_params.prob_pick_col_by_var ||
         model_params.prob_pick_col_by_kurt) && workspace.node_col_weights.empty())
    {
        workspace.node_col_weights.resize(input_data.ncols_tot);
        if (tree_root != NULL || model_params.standardize_data || model_params.missing_action != Fail)
        {
            workspace.saved_stat1.resize(input_data.ncols_numeric);
            workspace.saved_stat2.resize(input_data.ncols_numeric);
        }
    }

    /* IMPORTANT!!!!!
       The standard library implementation is likely going to use the Box-Muller method
       for normal sampling, which has some state memory in the **distribution object itself**
       in addition to the state memory from the RNG engine. DO NOT avoid re-generating this
       object on each tree, despite being inefficient, because then it can cause seed
       irreproducibility when the number of splitting dimensions is odd and the number
       of threads is more than 1. This is a very hard issue to debug since everything
       works fine depending on the order in which trees are assigned to threads.
       DO NOT PUT THESE LINES BELOW THE NEXT IF. */
    if (hplane_root != NULL)
    {
        if (input_data.ncols_categ || model_params.coef_type == Normal)
            workspace.coef_norm = StandardNormalDistr(0, 1);
        if (model_params.coef_type == Uniform)
            workspace.coef_unif = UniformMinusOneToOne(-1, 1);
    }

    /* for the extended model, initialize extra vectors and objects */
    if (hplane_root != NULL && workspace.comb_val.empty())
    {
        workspace.comb_val.resize(model_params.sample_size);
        workspace.col_take.resize(model_params.ndim);
        workspace.col_take_type.resize(model_params.ndim);

        if (input_data.ncols_numeric)
        {
            workspace.ext_offset.resize(input_data.ncols_tot);
            workspace.ext_coef.resize(input_data.ncols_tot);
            workspace.ext_mean.resize(input_data.ncols_tot);
        }

        if (input_data.ncols_categ)
        {
            workspace.ext_fill_new.resize(input_data.max_categ);
            switch(model_params.cat_split_type)
            {
                case SingleCateg:
                {
                    workspace.chosen_cat.resize(input_data.max_categ);
                    break;
                }

                case SubSet:
                {
                    workspace.ext_cat_coef.resize(input_data.ncols_tot);
                    for (std::vector<double> &v : workspace.ext_cat_coef)
                        v.resize(input_data.max_categ);
                    break;
                }
            }
        }

        workspace.ext_fill_val.resize(input_data.ncols_tot);

    }

    /* If there are density weights, need to standardize them to sum up to
       the sample size here. Note that weights for missing values with 'Divide'
       are only initialized on-demand later on. */
    workspace.changed_weights = false;
    if (hplane_root == NULL) workspace.weights_map.clear();

    ldouble_safe weight_scaling = 0;
    if (input_data.sample_weights != NULL && !input_data.weight_as_sample)
    {
        workspace.changed_weights = true;

        /* For the extended model, if there is no sub-sampling, these weights will remain
           constant throughout and do not need to be re-generated. */
        if (!(  hplane_root != NULL &&
                (!workspace.weights_map.empty() || !workspace.weights_arr.empty()) &&
                model_params.sample_size == input_data.nrows && !model_params.with_replacement
              )
            )
        {
            workspace.weights_map.clear();

            /* if the sub-sample size is small relative to the full sample size, use a mapping */
            if (input_data.Xc_indptr != NULL && model_params.sample_size < input_data.nrows / 50)
            {
                for (const size_t ix : workspace.ix_arr)
                    weight_scaling += input_data.sample_weights[ix];
                weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                workspace.weights_map.reserve(workspace.ix_arr.size());
                for (const size_t ix : workspace.ix_arr)
                    workspace.weights_map[ix] = input_data.sample_weights[ix] * weight_scaling;
            }

            /* if the sub-sample size is large, fill a full array matching to the sample size */
            else
            {
                if (workspace.weights_arr.empty())
                {
                    workspace.weights_arr.assign(input_data.sample_weights, input_data.sample_weights + input_data.nrows);
                    weight_scaling = std::accumulate(workspace.ix_arr.begin(),
                                                     workspace.ix_arr.end(),
                                                     (ldouble_safe)0,
                                                     [&input_data](const ldouble_safe a, const size_t b){return a + (ldouble_safe)input_data.sample_weights[b];}
                                                     );
                    weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;
                }

                else
                {
                    for (const size_t ix : workspace.ix_arr)
                    {
                        weight_scaling += input_data.sample_weights[ix];
                        workspace.weights_arr[ix] = input_data.sample_weights[ix];
                    }
                    weight_scaling = (ldouble_safe)model_params.sample_size / weight_scaling;
                    for (double &w : workspace.weights_arr)
                        w *= weight_scaling;
                }
            }
        }
    }

    /* if producing distance/similarity, also need to initialize the triangular matrix */
    if (model_params.calc_dist && workspace.tmat_sep.empty())
        workspace.tmat_sep.resize((input_data.nrows * (input_data.nrows - 1)) / 2, 0);

    /* make space for buffers if not already allocated */
    if (
            (model_params.prob_pick_by_gain_avg    > 0  ||
             model_params.prob_pick_by_gain_pl     > 0  ||
             model_params.prob_pick_by_full_gain   > 0  ||
             model_params.prob_pick_by_dens        > 0  ||
             model_params.prob_pick_col_by_range   > 0  ||
             model_params.prob_pick_col_by_var     > 0  ||
             model_params.prob_pick_col_by_kurt    > 0  ||
             model_params.weigh_by_kurt || hplane_root != NULL)
                &&
            (workspace.buffer_dbl.empty() && workspace.buffer_szt.empty() && workspace.buffer_chr.empty())
        )
    {
        size_t min_size_dbl = 0;
        size_t min_size_szt = 0;
        size_t min_size_chr = 0;

        bool gain = model_params.prob_pick_by_gain_avg  > 0 ||
                    model_params.prob_pick_by_gain_pl   > 0 ||
                    model_params.prob_pick_by_full_gain > 0 ||
                    model_params.prob_pick_by_dens      > 0;

        if (input_data.ncols_categ)
        {
            min_size_szt = (size_t)2 * (size_t)input_data.max_categ;
            min_size_dbl = input_data.max_categ + 1;
            if (gain && model_params.cat_split_type == SubSet)
                min_size_chr = input_data.max_categ;
        }

        if (input_data.Xc_indptr != NULL && gain)
        {
            min_size_szt = std::max(min_size_szt, model_params.sample_size);
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
        }

        /* TODO: revisit if this covers all the cases */
        if (model_params.ntry > 1 || gain)
        {
            min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
            if (model_params.ndim < 2 && input_data.Xc_indptr != NULL)
                min_size_dbl = std::max(min_size_dbl, (size_t)2*model_params.sample_size);
        }

        /* for sampled column choices */
        if (model_params.prob_pick_col_by_var)
        {
            if (input_data.ncols_categ) {
                min_size_szt = std::max(min_size_szt, (size_t)input_data.max_categ + 1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ + 1);
            }
        }

        if (model_params.prob_pick_col_by_kurt)
        {
            if (input_data.ncols_categ) {
                min_size_szt = std::max(min_size_szt, (size_t)input_data.max_categ + 1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ);
            }

        }

        /* for the extended model */
        if (hplane_root != NULL)
        {
            min_size_dbl = std::max(min_size_dbl, pow2(log2ceil(input_data.ncols_tot) + 1));
            if (model_params.missing_action != Fail)
            {
                min_size_szt = std::max(min_size_szt, model_params.sample_size);
                min_size_dbl = std::max(min_size_dbl, model_params.sample_size);
            }

            if (input_data.ncols_categ && model_params.cat_split_type == SubSet)
            {
                min_size_szt = std::max(min_size_szt, (size_t)2 * (size_t)input_data.max_categ + (size_t)1);
                min_size_dbl = std::max(min_size_dbl, (size_t)input_data.max_categ);
            }

            if (model_params.weigh_by_kurt)
                min_size_szt = std::max(min_size_szt, input_data.ncols_tot);

            if (gain && (!workspace.weights_arr.empty() || !workspace.weights_map.empty()))
            {
                workspace.sample_weights.resize(model_params.sample_size);
                min_size_szt = std::max(min_size_szt, model_params.sample_size);
            }
        }

        /* now resize */
        if (workspace.buffer_dbl.size() < min_size_dbl)
            workspace.buffer_dbl.resize(min_size_dbl);

        if (workspace.buffer_szt.size() < min_size_szt)
            workspace.buffer_szt.resize(min_size_szt);

        if (workspace.buffer_chr.size() < min_size_chr)
            workspace.buffer_chr.resize(min_size_chr);

        /* for guided column choice, need to also remember the best split so far */
        if (
            model_params.cat_split_type == SubSet &&
            (
                model_params.prob_pick_by_gain_avg  || 
                model_params.prob_pick_by_gain_pl   ||
                model_params.prob_pick_by_full_gain ||
                model_params.prob_pick_by_dens
            )
           )
        {
            workspace.this_split_categ.resize(input_data.max_categ);
        }

    }

    /* Other potentially necessary buffers */
    if (
        tree_root != NULL && model_params.missing_action == Impute &&
        (model_params.prob_pick_by_gain_avg  || model_params.prob_pick_by_gain_pl ||
         model_params.prob_pick_by_full_gain || model_params.prob_pick_by_dens) &&
        input_data.Xc_indptr == NULL && input_data.ncols_numeric && workspace.imputed_x_buffer.empty()
    )
    {
        workspace.imputed_x_buffer.resize(input_data.nrows);
    }

    if (model_params.prob_pick_by_full_gain && workspace.col_indices.empty())
        workspace.col_indices.resize(model_params.ncols_per_tree);

    if (
        (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) &&
        model_params.weigh_by_kurt &&
        model_params.sample_size == input_data.nrows && !model_params.with_replacement &&
        (model_params.ncols_per_tree == input_data.ncols_tot) &&
        !input_data.all_kurtoses.empty()
    ) {
        workspace.tree_kurtoses = input_data.all_kurtoses.data();
    }
    else {
        workspace.tree_kurtoses = NULL;
    }

    /* weigh columns by kurtosis in the sample if required */
    /* TODO: this one could probably be refactored to use the function in the helpers */
    std::vector<double> kurt_weights;
    bool avoid_leave_m_cols = false;
    if (
        model_params.weigh_by_kurt &&
        !avoid_col_weights &&
        (input_data.preinitialized_col_sampler == NULL
            ||
         ((model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var) && workspace.tree_kurtoses == NULL))
    )
    {
        kurt_weights.resize(input_data.ncols_numeric + input_data.ncols_categ, 0.);

        if (model_params.ncols_per_tree >= input_data.ncols_tot)
        {

            if (input_data.Xc_indptr == NULL)
            {

                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                                          input_data.numeric_data + col * input_data.nrows,
                                                          model_params.missing_action);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, decltype(workspace.weights_arr), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                   decltype(workspace.weights_map), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                   input_data.numeric_data + col * input_data.nrows,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            else
            {
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());
                for (size_t col = 0; col < input_data.ncols_numeric; col++)
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                          typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                          ldouble_safe>(
                                                          workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                          input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                          model_params.missing_action);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                   typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                   decltype(workspace.weights_arr), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_arr);
                    else
                        kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                   typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                   decltype(workspace.weights_map), ldouble_safe>(
                                                                   workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                   input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                   model_params.missing_action, workspace.weights_map);
                }
            }

            for (size_t col = 0; col < input_data.ncols_categ; col++)
            {
                if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis<ldouble_safe>(
                                      workspace.ix_arr.data(), workspace.st, workspace.end,
                                      input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                      workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                      model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                else if (!workspace.weights_arr.empty())
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_arr);
                else
                    kurt_weights[col + input_data.ncols_numeric] =
                        calc_kurtosis_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                               workspace.ix_arr.data(), workspace.st, workspace.end,
                                               input_data.categ_data + col * input_data.nrows, input_data.ncat[col],
                                               workspace.buffer_dbl.data(),
                                               model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                               workspace.weights_map);
            }

            for (auto &w : kurt_weights) w = (w == -HUGE_VAL)? 0. : std::fmax(1e-8, -1. + w);
            if (input_data.col_weights != NULL)
            {
                for (size_t col = 0; col < input_data.ncols_tot; col++)
                {
                    if (kurt_weights[col] <= 0) continue;
                    kurt_weights[col] *= input_data.col_weights[col];
                    kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
                }
            }
            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
        }

        

        else
        {
            std::vector<size_t> cols_take(model_params.ncols_per_tree);
            std::vector<size_t> buffer1;
            std::vector<bool> buffer2;
            sample_random_rows<double, double>(
                               cols_take, input_data.ncols_tot, false,
                               workspace.rnd_generator, buffer1,
                               (double*)NULL, kurt_weights, /* <- will not get used */
                               (size_t)0, (size_t)0, buffer2);

            if (
                model_params.sample_size == input_data.nrows &&
                !model_params.with_replacement &&
                !input_data.all_kurtoses.empty()
            )
            {
                for (size_t col : cols_take)
                    kurt_weights[col] = input_data.all_kurtoses[col];
                goto skip_kurt_calculations;
            }

            if (input_data.Xc_indptr != NULL)
                std::sort(workspace.ix_arr.begin(), workspace.ix_arr.end());

            for (size_t col : cols_take)
            {
                if (col < input_data.ncols_numeric)
                {
                    if (input_data.Xc_indptr == NULL)
                    {
                        if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                            kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.numeric_data)>::type, ldouble_safe>(
                                                              workspace.ix_arr.data(), workspace.st, workspace.end,
                                                              input_data.numeric_data + col * input_data.nrows,
                                                              model_params.missing_action);
                        else if (!workspace.weights_arr.empty())
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                       decltype(workspace.weights_arr), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.numeric_data)>::type,
                                                                       decltype(workspace.weights_map), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end,
                                                                       input_data.numeric_data + col * input_data.nrows,
                                                                       model_params.missing_action, workspace.weights_map);
                    }

                    else
                    {
                        if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                            kurt_weights[col] = calc_kurtosis<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                              typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                              ldouble_safe>(
                                                              workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                              input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                              model_params.missing_action);
                        else if (!workspace.weights_arr.empty())
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                       typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                       decltype(workspace.weights_arr), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_arr);
                        else
                            kurt_weights[col] = calc_kurtosis_weighted<typename std::remove_pointer<decltype(input_data.Xc)>::type,
                                                                       typename std::remove_pointer<decltype(input_data.Xc_indptr)>::type,
                                                                       decltype(workspace.weights_map), ldouble_safe>(
                                                                       workspace.ix_arr.data(), workspace.st, workspace.end, col,
                                                                       input_data.Xc, input_data.Xc_ind, input_data.Xc_indptr,
                                                                       model_params.missing_action, workspace.weights_map);
                    }
                }

                else
                {
                    if (workspace.weights_arr.empty() && workspace.weights_map.empty())
                        kurt_weights[col] =
                            calc_kurtosis<ldouble_safe>(
                                          workspace.ix_arr.data(), workspace.st, workspace.end,
                                          input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                          input_data.ncat[col - input_data.ncols_numeric],
                                          workspace.buffer_szt.data(), workspace.buffer_dbl.data(),
                                          model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator);
                    else if (!workspace.weights_arr.empty())
                        kurt_weights[col] =
                            calc_kurtosis_weighted<decltype(workspace.weights_arr), ldouble_safe>(
                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                   input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                   input_data.ncat[col - input_data.ncols_numeric],
                                                   workspace.buffer_dbl.data(),
                                                   model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                   workspace.weights_arr);
                    else
                        kurt_weights[col] =
                            calc_kurtosis_weighted<decltype(workspace.weights_map), ldouble_safe>(
                                                   workspace.ix_arr.data(), workspace.st, workspace.end,
                                                   input_data.categ_data + (col - input_data.ncols_numeric) * input_data.nrows,
                                                   input_data.ncat[col - input_data.ncols_numeric],
                                                   workspace.buffer_dbl.data(),
                                                   model_params.missing_action, model_params.cat_split_type, workspace.rnd_generator,
                                                   workspace.weights_map);
                }

                /* Note to self: don't move this  to outside of the braces, as it needs to assign a weight
                   of zero to the columns that were not selected, thus it should only do this clipping
                   for columns that are chosen. */
                if (kurt_weights[col] == -HUGE_VAL)
                {
                    kurt_weights[col] = 0;
                }

                else
                {
                    kurt_weights[col] = std::fmax(1e-8, -1. + kurt_weights[col]);
                    if (input_data.col_weights != NULL)
                    {
                        kurt_weights[col] *= input_data.col_weights[col];
                        kurt_weights[col] = std::fmax(kurt_weights[col], 1e-100);
                    }
                }
            }

            skip_kurt_calculations:
            workspace.col_sampler.initialize(kurt_weights.data(), kurt_weights.size());
            avoid_leave_m_cols = true;
        }

        if (model_params.prob_pick_col_by_range || model_params.prob_pick_col_by_var)
        {
            workspace.tree_kurtoses = kurt_weights.data();
        }
    }

    bool col_sampler_is_fresh = true;
    if (input_data.preinitialized_col_sampler == NULL) {
        workspace.col_sampler.initialize(input_data.ncols_tot);
    }
    else {
        workspace.col_sampler = *((ColumnSampler<ldouble_safe>*)input_data.preinitialized_col_sampler);
        col_sampler_is_fresh = false;
    }
    /* TODO: this can be done more efficiently when sub-sampling columns */
    if (!avoid_leave_m_cols)
        workspace.col_sampler.leave_m_cols(model_params.ncols_per_tree, workspace.rnd_generator);
    if (model_params.ncols_per_tree < input_data.ncols_tot) col_sampler_is_fresh = false;
    workspace.try_all = false;
    if (hplane_root != NULL && model_params.ndim >= input_data.ncols_tot)
        workspace.try_all = true;

    if (model_params.scoring_metric != Depth && !is_boxed_metric(model_params.scoring_metric))
    {
        workspace.density_calculator.initialize(model_params.max_depth,
                                                input_data.ncols_categ? input_data.max_categ : 0,
                                                tree_root != NULL && input_data.ncols_categ,
                                                model_params.scoring_metric);
    }

    else if (is_boxed_metric(model_params.scoring_metric))
    {
        if (tree_root != NULL)
            workspace.density_calculator.initialize_bdens(input_data,
                                                          model_params,
                                                          workspace.ix_arr,
                                                          workspace.col_sampler);
        else
            workspace.density_calculator.initialize_bdens_ext(input_data,
                                                              model_params,
                                                              workspace.ix_arr,
                                                              workspace.col_sampler,
                                                              col_sampler_is_fresh);
    }

    if (tree_root != NULL)
    {
        split_itree_recursive<InputData, WorkerMemory, ldouble_safe>(
                              *tree_root,
                              workspace,
                              input_data,
                              model_params,
                              impute_nodes,
                              0);
    }

    else
    {
        split_hplane_recursive<InputData, WorkerMemory, ldouble_safe>(
                               *hplane_root,
                               workspace,
                               input_data,
                               model_params,
                               impute_nodes,
                               0);
    }

    /* if producing imputation structs, only need to keep the ones for terminal nodes */
    if (impute_nodes != NULL)
        drop_nonterminal_imp_node(*impute_nodes, tree_root, hplane_root);
}
