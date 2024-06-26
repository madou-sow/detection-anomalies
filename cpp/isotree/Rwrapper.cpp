#ifdef _FOR_R

#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(unwindProtect)]]

/* This is to serialize the model objects */
// [[Rcpp::depends(Rcereal)]]
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <sstream>
#include <string>
#include <limits>

/* This is the package's header */
#include "outlier_tree.hpp"

SEXP alloc_RawVec(void *data)
{
    size_t vec_size = *(size_t*)data;
    if (vec_size > (size_t)std::numeric_limits<R_xlen_t>::max())
        Rcpp::stop("Resulting model is too large for R to handle.");
    return Rcpp::RawVector((R_xlen_t)vec_size);
}

/* for model serialization and re-usage in R */
/* https://stackoverflow.com/questions/18474292/how-to-handle-c-internal-data-structure-in-r-in-order-to-allow-save-load */
/* this extra comment below the link is a workaround for Rcpp issue 675 in GitHub, do not remove it */
Rcpp::RawVector serialize_OutlierTree(ModelOutputs *model_outputs)
{
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
        oarchive(*model_outputs);
    }
    ss.seekg(0, ss.end);
    std::stringstream::pos_type vec_size = ss.tellg();
    if (vec_size <= 0) {
        Rcpp::Rcerr << "Error: model is too big to serialize, resulting object will not be usable.\n" << std::endl;
        return Rcpp::RawVector();
    }
    size_t vec_size_ = (size_t)vec_size;
    Rcpp::RawVector retval = Rcpp::unwindProtect(alloc_RawVec, (void*)&vec_size_);
    if (!retval.size())
        return retval;
    ss.seekg(0, ss.beg);
    ss.read(reinterpret_cast<char*>(RAW(retval)), retval.size());
    return retval;
}

SEXP safe_XPtr(void *model_ptr)
{
    return Rcpp::XPtr<ModelOutputs>((ModelOutputs*)model_ptr, true);
}

void R_delete_model(SEXP R_ptr)
{
    ModelOutputs *model = static_cast<ModelOutputs*>(R_ExternalPtrAddr(R_ptr));
    delete model;
    R_ClearExternalPtr(R_ptr);
}

// [[Rcpp::export(rng = false)]]
SEXP deserialize_OutlierTree(Rcpp::RawVector src, SEXP ptr_obj)
{
    std::stringstream ss;
    ss.write(reinterpret_cast<char*>(RAW(src)), src.size());
    ss.seekg(0, ss.beg);
    std::unique_ptr<ModelOutputs> model_outputs = std::unique_ptr<ModelOutputs>(new ModelOutputs());
    {
        cereal::BinaryInputArchive iarchive(ss);
        iarchive(*model_outputs);
    }
    R_SetExternalPtrAddr(ptr_obj, model_outputs.get());
    R_RegisterCFinalizerEx(ptr_obj, R_delete_model, TRUE);
    model_outputs.release();
    return R_NilValue;
}

SEXP safe_int(void *x)
{
    return Rcpp::wrap(*(int*)x);
}

SEXP safe_bool(void *x)
{
    return Rcpp::wrap(*(bool*)x);
}

// [[Rcpp::export(rng = false)]]
Rcpp::LogicalVector check_null_ptr_model(SEXP ptr_model)
{
    return Rcpp::LogicalVector(R_ExternalPtrAddr(ptr_model) == NULL);
}

double* set_R_nan_as_C_nan(double *restrict x_R, std::vector<double> &x_C, size_t n, int nthreads)
{
    x_C.assign(x_R, x_R + n);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(x_R, x_C, n)
    for (size_t_for i = 0; i < n; i++)
        if (isnan(x_R[i]) || Rcpp::NumericVector::is_na(x_R[i]) || Rcpp::traits::is_nan<REALSXP>(x_R[i]))
            x_C[i] = NAN;
    return x_C.data();
}

double* set_R_nan_as_C_nan(double *restrict x_R, Rcpp::NumericVector &x_C, size_t n, int nthreads)
{
    x_C = Rcpp::NumericVector(x_R, x_R + n);
    #pragma omp parallel for schedule(static) num_threads(nthreads) shared(x_R, x_C, n)
    for (size_t_for i = 0; i < n; i++)
        if (isnan(x_R[i]) || Rcpp::NumericVector::is_na(x_R[i]) || Rcpp::traits::is_nan<REALSXP>(x_R[i]))
            x_C[i] = NAN;
    return REAL(x_C);
}


/* for predicting outliers */
Rcpp::List describe_outliers(ModelOutputs &model_outputs,
                             double *restrict arr_num,
                             int    *restrict arr_cat,
                             int    *restrict arr_ord,
                             Rcpp::ListOf<Rcpp::StringVector> cat_levels,
                             Rcpp::ListOf<Rcpp::StringVector> ord_levels,
                             Rcpp::StringVector colnames_num,
                             Rcpp::StringVector colnames_cat,
                             Rcpp::StringVector colnames_ord,
                             Rcpp::NumericVector min_date,
                             Rcpp::NumericVector min_ts)
{
    size_t nrows         = model_outputs.outlier_scores_final.size();
    size_t ncols_num     = model_outputs.ncols_numeric;
    size_t ncols_cat     = model_outputs.ncols_categ;
    size_t ncols_num_num = model_outputs.ncols_numeric - min_date.size() - min_ts.size();
    size_t ncols_date    = min_date.size();
    size_t ncols_cat_cat = cat_levels.size();
    Rcpp::List outp;
    
    Rcpp::LogicalVector has_na_col       = Rcpp::LogicalVector(nrows, NA_LOGICAL);
    Rcpp::IntegerVector tree_depth       = Rcpp::IntegerVector(nrows, NA_INTEGER);
    Rcpp::NumericVector outlier_score    = Rcpp::NumericVector(nrows, NA_REAL);
    Rcpp::ListOf<Rcpp::List> outlier_val = Rcpp::ListOf<Rcpp::List>(nrows);
    Rcpp::ListOf<Rcpp::List> lst_stats   = Rcpp::ListOf<Rcpp::List>(nrows);
    Rcpp::ListOf<Rcpp::List> lst_cond    = Rcpp::ListOf<Rcpp::List>(nrows);
    
    
    size_t outl_col;
    size_t outl_clust;
    size_t curr_tree;
    size_t parent_tree;
    Rcpp::LogicalVector tmp_bool;
    
    for (size_t row = 0; row < nrows; row++) {
        if (model_outputs.outlier_scores_final[row] < 1) {
            
            outl_col   = model_outputs.outlier_columns_final[row];
            outl_clust = model_outputs.outlier_clusters_final[row];
            
            /* metrics of outlierness - used to rank when choosing which to print */
            outlier_score[row] = model_outputs.outlier_scores_final[row];
            tree_depth[row]    = (int)model_outputs.outlier_depth_final[row];
            has_na_col[row]    = model_outputs.all_clusters[outl_col][outl_clust].has_NA_branch;
            
            /* first determine outlier column and suspected value */
            if (outl_col < ncols_num) {
                if (outl_col < ncols_num_num) {
                    outlier_val[row] = Rcpp::List::create(
                        Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_num[outl_col]),
                        Rcpp::_["value"]  = Rcpp::wrap(arr_num[row + outl_col * nrows]),
                        Rcpp::_["decimals"] = Rcpp::wrap(model_outputs.outlier_decimals_distr[row])
                    );
                } else if (outl_col < (ncols_num_num + ncols_date)) {
                    outlier_val[row] = Rcpp::List::create(
                        Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_num[outl_col]),
                        Rcpp::_["value"]  = Rcpp::Date(arr_num[row + outl_col * nrows] - 1 + min_date[outl_col - ncols_num_num])
                    );
                } else {
                    outlier_val[row] = Rcpp::List::create(
                        Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_num[outl_col]),
                        Rcpp::_["value"]  = Rcpp::Datetime(arr_num[row + outl_col * nrows] - 1 + min_ts[outl_col - ncols_num_num - ncols_date])
                    );
                }
            } else if (outl_col < (ncols_num + ncols_cat)) {
                if (outl_col < (ncols_num + ncols_cat_cat)) {
                    outlier_val[row] = Rcpp::List::create(
                        Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_cat[outl_col - ncols_num]),
                        Rcpp::_["value"]  = Rcpp::CharacterVector(1, cat_levels[outl_col - ncols_num]
                                                                               [arr_cat[row + (outl_col - ncols_num) * nrows]])
                    );
                } else {
                    outlier_val[row] = Rcpp::List::create(
                        Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_cat[outl_col - ncols_num]),
                        Rcpp::_["value"]  = Rcpp::wrap((bool)arr_cat[row + (outl_col - ncols_num) * nrows])
                    );
                }
            } else {
                outlier_val[row] = Rcpp::List::create(
                    Rcpp::_["column"] = Rcpp::CharacterVector(1, colnames_ord[outl_col - ncols_num - ncols_cat]),
                    Rcpp::_["value"]  = Rcpp::CharacterVector(1, ord_levels[outl_col - ncols_num - ncols_cat]
                                                                           [arr_ord[row + (outl_col - ncols_num - ncols_cat) * nrows]])
                );
            }
            
            
            /* info about the normal observations in the cluster */
            if (outl_col < ncols_num) {
                if (outl_col < ncols_num_num) {
                    if (arr_num[row + outl_col * nrows] >= model_outputs.all_clusters[outl_col][outl_clust].upper_lim) {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["upper_thr"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_lim_high),
                            Rcpp::_["pct_below"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_below),
                            Rcpp::_["mean"]      = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_mean),
                            Rcpp::_["sd"]        = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_sd),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    } else {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["lower_thr"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_lim_low),
                            Rcpp::_["pct_above"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_above),
                            Rcpp::_["mean"]      = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_mean),
                            Rcpp::_["sd"]        = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].display_sd),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    }
                } else if (outl_col < (ncols_num_num + ncols_date)) {
                    if (arr_num[row + outl_col * nrows] >= model_outputs.all_clusters[outl_col][outl_clust].upper_lim) {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["upper_thr"] = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].display_lim_high
                                                              - 1 + min_date[outl_col - ncols_num_num]),
                            Rcpp::_["pct_below"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_below),
                            Rcpp::_["mean"]      = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].display_mean - 1 + min_date[outl_col - ncols_num_num]),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    } else {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["lower_thr"] = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].display_lim_low
                                                              - 1 + min_date[outl_col - ncols_num_num]),
                            Rcpp::_["pct_above"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_above),
                            Rcpp::_["mean"]      = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].display_mean - 1 + min_date[outl_col - ncols_num_num]),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    }
                } else {
                    if (arr_num[row + outl_col * nrows] >= model_outputs.all_clusters[outl_col][outl_clust].upper_lim) {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["upper_thr"] = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].display_lim_high
                                                                  - 1 + min_ts[outl_col - ncols_num_num - ncols_date]),
                            Rcpp::_["pct_below"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_below),
                            Rcpp::_["mean"]      = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].display_mean
                                                                  - 1 + min_ts[outl_col - ncols_num_num - ncols_date]),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    } else {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["lower_thr"] = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].display_lim_low
                                                                  - 1 + min_ts[outl_col - ncols_num_num - ncols_date]),
                            Rcpp::_["pct_above"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_above),
                            Rcpp::_["mean"]      = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].display_mean
                                                                  - 1 + min_ts[outl_col - ncols_num_num - ncols_date]),
                            Rcpp::_["n_obs"]     = Rcpp::wrap((int)model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    }
                }
            } else if (outl_col < (ncols_num + ncols_cat)) {
                if (outl_col < (ncols_num + ncols_cat_cat)) {
                    tmp_bool = Rcpp::LogicalVector(model_outputs.all_clusters[outl_col][outl_clust].subset_common.size(), false);
                    for (size_t cat = 0; cat < tmp_bool.size(); cat++) {
                        if (model_outputs.all_clusters[outl_col][outl_clust].subset_common[cat] == 0) {
                            tmp_bool[cat] = true;
                            }
                        }
                    if (model_outputs.all_clusters[outl_col][outl_clust].split_type != Root) {
                        if (model_outputs.all_clusters[outl_col][outl_clust].categ_maj < 0) {
                            lst_stats[row] = Rcpp::List::create(
                                Rcpp::_["categs_common"]      = Rcpp::as<Rcpp::CharacterVector>(cat_levels[outl_col - ncols_num][tmp_bool]),
                                Rcpp::_["pct_common"]         = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                                Rcpp::_["pct_next_most_comm"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm),
                                Rcpp::_["prior_prob"]         = Rcpp::wrap(model_outputs.prop_categ[model_outputs.start_ix_cat_counts[outl_col - ncols_num] +
                                                                           arr_cat[row + (outl_col - ncols_num) * nrows]]),
                                Rcpp::_["n_obs"]              = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                            );
                        } else {
                            lst_stats[row] = Rcpp::List::create(
                                Rcpp::_["categ_maj"]   = Rcpp::as<Rcpp::CharacterVector>(cat_levels[outl_col - ncols_num][
                                                                                                model_outputs.all_clusters[outl_col][outl_clust].categ_maj
                                                                                                ]),
                                Rcpp::_["pct_common"]  = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                                Rcpp::_["prior_prob"]  = Rcpp::wrap(model_outputs.prop_categ[model_outputs.start_ix_cat_counts[outl_col - ncols_num] +
                                                                    arr_cat[row + (outl_col - ncols_num) * nrows]]),
                                Rcpp::_["n_obs"]       = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                            );
                        }
                    } else {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["categs_common"]      = Rcpp::as<Rcpp::CharacterVector>(cat_levels[outl_col - ncols_num][tmp_bool]),
                            Rcpp::_["pct_common"]         = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                            Rcpp::_["pct_next_most_comm"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm),
                            Rcpp::_["n_obs"]              = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    }
                } else {
                    lst_stats[row] = Rcpp::List::create(
                        Rcpp::_["pct_other"]  = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                        Rcpp::_["prior_prob"] = Rcpp::wrap(model_outputs.prop_categ[model_outputs.start_ix_cat_counts[outl_col - ncols_num] +
                                                           arr_cat[row + (outl_col - ncols_num) * nrows]]),
                        Rcpp::_["n_obs"]      = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                    );
                }
            } else {
                tmp_bool = Rcpp::LogicalVector(model_outputs.all_clusters[outl_col][outl_clust].subset_common.size(), false);
                for (size_t cat = 0; cat < tmp_bool.size(); cat++) {
                    if (model_outputs.all_clusters[outl_col][outl_clust].subset_common[cat] == 0) {
                        tmp_bool[cat] = true;
                    }
                }
                if (model_outputs.all_clusters[outl_col][outl_clust].split_type != Root) {
                    if (model_outputs.all_clusters[outl_col][outl_clust].categ_maj < 0) {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["categs_common"]      = Rcpp::as<Rcpp::CharacterVector>(ord_levels[outl_col - ncols_num - ncols_cat][tmp_bool]),
                            Rcpp::_["pct_common"]         = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                            Rcpp::_["pct_next_most_comm"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm),
                            Rcpp::_["prior_prob"]         = Rcpp::wrap(model_outputs.prop_categ[model_outputs.start_ix_cat_counts[outl_col - ncols_num] +
                                                                       arr_ord[row + (outl_col - ncols_num - ncols_cat) * nrows]]),
                            Rcpp::_["n_obs"]              = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    } else {
                        lst_stats[row] = Rcpp::List::create(
                            Rcpp::_["categ_maj"]   = Rcpp::as<Rcpp::CharacterVector>(ord_levels[outl_col - ncols_num - ncols_cat][
                                                                                        model_outputs.all_clusters[outl_col][outl_clust].categ_maj
                                                                                        ]),
                            Rcpp::_["pct_common"]  = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                            Rcpp::_["prior_prob"]  = Rcpp::wrap(model_outputs.prop_categ[model_outputs.start_ix_cat_counts[outl_col - ncols_num] +
                                                                arr_ord[row + (outl_col - ncols_num - ncols_cat) * nrows]]),
                            Rcpp::_["n_obs"]       = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                        );
                    }
                } else {
                    lst_stats[row] = Rcpp::List::create(
                        Rcpp::_["categs_common"]      = Rcpp::as<Rcpp::CharacterVector>(ord_levels[outl_col - ncols_num - ncols_cat][tmp_bool]),
                        Rcpp::_["pct_common"]         = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset),
                        Rcpp::_["pct_next_most_comm"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm),
                        Rcpp::_["n_obs"]              = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].cluster_size)
                    );
                }
            }
            
            
            /* then determine conditions from the cluster */
            Rcpp::List cond_clust;
            if (model_outputs.all_clusters[outl_col][outl_clust].column_type != NoType) {

                /* add the column name and actual value for the row */
                switch(model_outputs.all_clusters[outl_col][outl_clust].column_type) {
                    case Numeric:
                    {
                        cond_clust["column"] = Rcpp::CharacterVector(1, colnames_num[model_outputs.all_clusters[outl_col][outl_clust].col_num]);
                        if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_num_num) {
                            cond_clust["value_this"] = Rcpp::wrap(arr_num[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]);
                            if (model_outputs.all_clusters[outl_col][outl_clust].split_type != IsNa)
                                cond_clust["decimals"] = Rcpp::wrap(model_outputs.min_decimals_col[model_outputs.all_clusters[outl_col][outl_clust].col_num]);
                        } else if (model_outputs.all_clusters[outl_col][outl_clust].col_num < (ncols_num_num + ncols_date)) {
                            cond_clust["value_this"] = Rcpp::Date(arr_num[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]
                                                                  - 1 + min_date[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num]);
                        } else {
                            cond_clust["value_this"] = Rcpp::Datetime(arr_num[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]
                                                                      - 1 + min_ts[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num - ncols_date]);
                        }
                        break;
                    }
                        
                    case Categorical:
                    {
                        cond_clust["column"] = Rcpp::CharacterVector(1, colnames_cat[model_outputs.all_clusters[outl_col][outl_clust].col_num]);
                        if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_cat_cat) {
                            if (arr_cat[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows] >= 0) {
                                cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                              [arr_cat[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]]);
                            } else {
                                cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                            }
                        } else {

                            if (arr_cat[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows] >= 0) {
                                cond_clust["value_this"] = Rcpp::wrap((bool)arr_cat[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]);
                            } else {
                                cond_clust["value_this"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                            }
                        }
                        break;
                    }
                    
                    case Ordinal:
                    {
                        cond_clust["column"] = Rcpp::CharacterVector(1, colnames_ord[model_outputs.all_clusters[outl_col][outl_clust].col_num]);
                        if (arr_ord[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows] >= 0) {
                            cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                          [arr_ord[row + model_outputs.all_clusters[outl_col][outl_clust].col_num * nrows]]);
                        } else {
                            cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                        }
                        break;
                    }

                    default:
                    {
                        assert(0);
                        break;
                    }
                }
                
                /* add the comparison point */
                switch(model_outputs.all_clusters[outl_col][outl_clust].split_type) {
                    
                    case IsNa:
                    {
                        cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                        switch(model_outputs.all_clusters[outl_col][outl_clust].column_type) {
                            case Numeric:
                            {
                                /* http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2012-October/004379.html */
                                /* this comment below will prevent bug with Rcpp comments having forward slashes */
                                cond_clust["value_comp"] = Rcpp::wrap(NA_REAL);
                                break;
                            }

                            case Categorical:
                            {
                                if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_cat_cat) {
                                    cond_clust["value_comp"] = Rcpp::wrap(NA_STRING);
                                } else {
                                    cond_clust["value_comp"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                                }
                                break;
                            }

                            case Ordinal:
                            {
                                cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                break;
                            }

                            default:
                            {
                                unexpected_error();
                            }
                        }
                        break;
                    }
                        
                    case LessOrEqual:
                    {
                        if (model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_num_num) {
                                cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].split_point);
                            } else if (model_outputs.all_clusters[outl_col][outl_clust].col_num < (ncols_num_num + ncols_date)) {
                                cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].split_point
                                                                      - 1 + min_date[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num]);
                            } else {
                                cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].split_point
                                                                          - 1 + min_ts[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num - ncols_date]);
                            }
                        } else {
                            tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num].size(), false);
                            for (int cat = 0; cat <= model_outputs.all_clusters[outl_col][outl_clust].split_lev; cat++) tmp_bool[cat] = true;
                            cond_clust["comparison"] = Rcpp::CharacterVector("in");
                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num][tmp_bool]);
                        }
                        break;
                    }
                        
                    case Greater:
                    {
                        if (model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_num_num) {
                                cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_clusters[outl_col][outl_clust].split_point);
                            } else if (model_outputs.all_clusters[outl_col][outl_clust].col_num < (ncols_num_num + ncols_date)) {
                                cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_clusters[outl_col][outl_clust].split_point
                                                                      - 1 + min_date[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num]);
                            } else {
                                cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_clusters[outl_col][outl_clust].split_point
                                                                          - 1 + min_ts[model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_num_num - ncols_date]);
                            }
                        } else {
                            tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num].size(), true);
                            for (int cat = 0; cat <= model_outputs.all_clusters[outl_col][outl_clust].split_lev; cat++) tmp_bool[cat] = false;
                            cond_clust["comparison"] = Rcpp::CharacterVector("in");
                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num][tmp_bool]);
                        }
                        break;
                    }
                        
                    case InSubset:
                    {
                        tmp_bool = Rcpp::LogicalVector(model_outputs.all_clusters[outl_col][outl_clust].split_subset.size(), false);
                        for (size_t cat = 0; cat < model_outputs.all_clusters[outl_col][outl_clust].split_subset.size(); cat++) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].split_subset[cat] > 0) {
                                tmp_bool[cat] = true;
                            }
                        }
                        cond_clust["comparison"] = Rcpp::CharacterVector("in");
                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num][tmp_bool]);
                        break;
                    }
                        
                    case NotInSubset:
                    {
                        tmp_bool = Rcpp::LogicalVector(model_outputs.all_clusters[outl_col][outl_clust].split_subset.size(), false);
                        for (size_t cat = 0; cat < model_outputs.all_clusters[outl_col][outl_clust].split_subset.size(); cat++) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].split_subset[cat] == 0) {
                                tmp_bool[cat] = true;
                            }
                        }
                        cond_clust["comparison"] = Rcpp::CharacterVector("in");
                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num][tmp_bool]);
                        break;
                    }
                        
                    case Equal:
                    {
                        if (model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_cat_cat) {
                                cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                              [model_outputs.all_clusters[outl_col][outl_clust].split_lev]);
                            } else {
                                cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_clusters[outl_col][outl_clust].split_lev);
                            }
                        } else {
                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                            cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                          [model_outputs.all_clusters[outl_col][outl_clust].split_lev]);
                        }
                        break;
                    }
                        
                    case NotEqual:
                    {
                        if (model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical) {
                            if (model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_cat_cat) {
                                cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                              [model_outputs.all_clusters[outl_col][outl_clust].split_lev]);
                            } else {
                                cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                cond_clust["value_comp"] = Rcpp::wrap(!((bool)model_outputs.all_clusters[outl_col][outl_clust].split_lev));
                            }
                        } else {
                            cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                            cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                                                          [model_outputs.all_clusters[outl_col][outl_clust].split_lev]);
                        }
                        break;
                    }

                    default:
                    {
                        assert(0);
                        break;
                    }
                    
                }
                lst_cond[row] = Rcpp::List::create(Rcpp::clone(cond_clust));

                /* finally, add conditions from branches that lead to the cluster */
                curr_tree = model_outputs.outlier_trees_final[row];
                Rcpp::List temp_list;
                while (true) {
                    if (curr_tree == 0 || model_outputs.all_trees[outl_col][curr_tree].parent_branch == SubTrees) {
                        break;
                    }
                    parent_tree = model_outputs.all_trees[outl_col][curr_tree].parent;
                    cond_clust = Rcpp::List();

                    /* when using 'follow_all' */
                    if (model_outputs.all_trees[outl_col][parent_tree].all_branches.size() > 0) {

                        /* add column name and value */
                        switch(model_outputs.all_trees[outl_col][curr_tree].column_type) {
                            case Numeric:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_num[model_outputs.all_trees[outl_col][curr_tree].col_num]);
                                break;
                            }

                            case Categorical:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_cat[model_outputs.all_trees[outl_col][curr_tree].col_num]);
                                break;
                            }

                            case Ordinal:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_ord[model_outputs.all_trees[outl_col][curr_tree].col_num]);
                                break;
                            }

                            default:
                            {
                                assert(0);
                                break;
                            }
                        }

                        /* add conditions from tree */
                        switch(model_outputs.all_trees[outl_col][curr_tree].column_type) {

                            case Numeric:
                            {
                                /* add decimals if appropriate */
                                if (
                                    model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_num_num &&
                                    model_outputs.all_trees[outl_col][curr_tree].split_this_branch != IsNa
                                    )
                                {
                                    cond_clust["decimals"] = Rcpp::wrap(model_outputs.min_decimals_col[model_outputs.all_trees[outl_col][curr_tree].col_num]);
                                }

                                /* then conditions */
                                switch(model_outputs.all_trees[outl_col][curr_tree].split_this_branch) {

                                    case IsNa:
                                    {
                                        cond_clust["value_this"] = Rcpp::wrap(NA_REAL);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                        cond_clust["value_comp"] = Rcpp::wrap(NA_REAL);
                                        break;
                                    }

                                    case LessOrEqual:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_num_num) {
                                            cond_clust["value_this"] = Rcpp::wrap(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                            cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_trees[outl_col][curr_tree].split_point);
                                        } else if (model_outputs.all_trees[outl_col][curr_tree].col_num < (ncols_num_num + ncols_date)) {
                                            cond_clust["value_this"] = Rcpp::Date(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]
                                                                                  - 1 + min_date[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                            cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_trees[outl_col][curr_tree].split_point
                                                                                  - 1 + min_date[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::Datetime(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]
                                                                                      - 1 + min_ts[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num - ncols_date]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                            cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_trees[outl_col][curr_tree].split_point
                                                                                      - 1 + min_ts[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num - ncols_date]);
                                        }
                                        break;
                                    }

                                    case Greater:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_num_num) {
                                            cond_clust["value_this"] = Rcpp::wrap(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                            cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_trees[outl_col][curr_tree].split_point);
                                        } else if (model_outputs.all_trees[outl_col][curr_tree].col_num < (ncols_num_num + ncols_date)) {
                                            cond_clust["value_this"] = Rcpp::Date(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]
                                                                                  - 1 + min_date[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                            cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_trees[outl_col][curr_tree].split_point
                                                                                  - 1 + min_date[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::Datetime(arr_num[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]
                                                                                      - 1 + min_ts[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num - ncols_date]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                            cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_trees[outl_col][curr_tree].split_point
                                                                                      - 1 + min_ts[model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_num_num - ncols_date]);
                                        }
                                        break;
                                    }

                                    default: {}
                                }
                                break;
                            }

                            case Categorical:
                            {
                                switch(model_outputs.all_trees[outl_col][curr_tree].split_this_branch) {

                                    case IsNa:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_cat_cat) {
                                            cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                            cond_clust["value_comp"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                                        }
                                        break;
                                    }

                                    case InSubset:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_cat_cat) {
                                            tmp_bool = Rcpp::LogicalVector(model_outputs.all_trees[outl_col][curr_tree].split_subset.size(), false);
                                            for (size_t cat = 0; cat < model_outputs.all_trees[outl_col][curr_tree].split_subset.size(); cat++) {
                                                if (model_outputs.all_trees[outl_col][curr_tree].split_subset[cat] > 0) {
                                                    tmp_bool[cat] = true;
                                                }
                                            }
                                            cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num][tmp_bool]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                            cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][curr_tree].split_subset[1]);
                                        }
                                        break;
                                    }

                                    case NotInSubset:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_cat_cat) {
                                            tmp_bool = Rcpp::LogicalVector(model_outputs.all_trees[outl_col][curr_tree].split_subset.size(), true);
                                            for (size_t cat = 0; cat < model_outputs.all_trees[outl_col][curr_tree].split_subset.size(); cat++) {
                                                if (model_outputs.all_trees[outl_col][curr_tree].split_subset[cat] > 0) {
                                                    tmp_bool[cat] = false;
                                                }
                                            }
                                            cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num][tmp_bool]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                            cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][curr_tree].split_subset[0]);
                                        }
                                        break;
                                    }

                                    case Equal:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_cat_cat) {
                                            cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                            cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [model_outputs.all_trees[outl_col][curr_tree].split_lev]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                            cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][curr_tree].split_lev);
                                        }
                                        break;
                                    }

                                    case NotEqual:
                                    {
                                        if (model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_cat_cat) {
                                            cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                            cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                          [model_outputs.all_trees[outl_col][curr_tree].split_lev]);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                            cond_clust["value_comp"] = Rcpp::wrap((bool) !model_outputs.all_trees[outl_col][curr_tree].split_lev);
                                            /* note: booleans should always get converted to Equals, this code is redundant */
                                        }
                                        break;
                                    }

                                    default: {}
                                }
                                break;
                            }

                            case Ordinal:
                            {
                                switch(model_outputs.all_trees[outl_col][curr_tree].split_this_branch) {

                                    case IsNa:
                                    {
                                        cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        break;
                                    }

                                    case LessOrEqual:
                                    {
                                        tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num].size(), false);
                                        for (int cat = 0; cat <= model_outputs.all_trees[outl_col][curr_tree].split_lev; cat++) {
                                            tmp_bool[cat] = true;
                                        }
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [arr_ord[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num][tmp_bool]);
                                        break;
                                    }

                                    case Greater:
                                    {
                                        tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num].size(), true);
                                        for (int cat = 0; cat <= model_outputs.all_trees[outl_col][curr_tree].split_lev; cat++) {
                                            tmp_bool[cat] = false;
                                        }
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [arr_ord[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num][tmp_bool]);
                                        break;
                                    }

                                    case Equal:
                                    {
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [arr_ord[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                        cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [model_outputs.all_trees[outl_col][curr_tree].split_lev]);
                                        break;
                                    }

                                    case NotEqual:
                                    {
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [arr_ord[row + model_outputs.all_trees[outl_col][curr_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                        cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][curr_tree].col_num]
                                                                                                      [model_outputs.all_trees[outl_col][curr_tree].split_lev]);
                                        break;
                                    }

                                    default: {}
                                }
                                break;
                            }

                            default:
                            {
                                assert(0);
                                break;
                            }
                        }
                    }

                    /* regular case (no 'follow_all') */
                    else
                    {

                        /* add column name and value */
                        switch(model_outputs.all_trees[outl_col][parent_tree].column_type) {
                            case Numeric:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_num[model_outputs.all_trees[outl_col][parent_tree].col_num]);
                                /* add decimals if appropriate */
                                if (
                                    model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_num_num &&
                                    model_outputs.all_trees[outl_col][curr_tree].parent_branch != IsNa
                                    )
                                {
                                    cond_clust["decimals"] = Rcpp::wrap(model_outputs.min_decimals_col[model_outputs.all_trees[outl_col][parent_tree].col_num]);
                                }
                                break;
                            }

                            case Categorical:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_cat[model_outputs.all_trees[outl_col][parent_tree].col_num]);
                                break;
                            }

                            case Ordinal:
                            {
                                cond_clust["column"] = Rcpp::as<Rcpp::CharacterVector>(colnames_ord[model_outputs.all_trees[outl_col][parent_tree].col_num]);
                                break;
                            }

                            default:
                            {
                                assert(0);
                                break;
                            }
                        }


                        /* add conditions from tree */
                        switch(model_outputs.all_trees[outl_col][curr_tree].parent_branch) {


                            case IsNa:
                            {
                                switch(model_outputs.all_trees[outl_col][parent_tree].column_type) {
                                    case Numeric:
                                    {
                                        cond_clust["value_this"] = Rcpp::wrap(NA_REAL);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                        cond_clust["value_comp"] = Rcpp::wrap(NA_REAL);
                                        break;
                                    }

                                    case Categorical:
                                    {
                                        if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                            cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                            cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        } else {
                                            cond_clust["value_this"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                                            cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                            cond_clust["value_comp"] = Rcpp::LogicalVector(1, NA_LOGICAL);
                                        }
                                        break;
                                    }

                                    case Ordinal:
                                    {
                                        cond_clust["value_this"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("is NA");
                                        cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(NA_STRING);
                                        break;
                                    }

                                    default:
                                    {
                                        unexpected_error();
                                    }
                                }
                                break;
                            }

                            case LessOrEqual:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric) {
                                    if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_num_num) {
                                        cond_clust["value_this"] = Rcpp::wrap(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                        cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_trees[outl_col][parent_tree].split_point);
                                    } else if (model_outputs.all_trees[outl_col][parent_tree].col_num < (ncols_num_num + ncols_date)) {
                                        cond_clust["value_this"] = Rcpp::Date(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]
                                                                              - 1 + min_date[model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_num_num]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                        cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_trees[outl_col][parent_tree].split_point
                                                                              - 1 + min_date[model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_num_num]);
                                    } else {
                                        cond_clust["value_this"] = Rcpp::Datetime(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]
                                                                                  - 1 + min_ts[model_outputs.all_trees[outl_col][parent_tree].col_num
                                                                                               - ncols_num_num - ncols_date]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("<=");
                                        cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_trees[outl_col][parent_tree].split_point
                                                                                  - 1 + min_ts[model_outputs.all_trees[outl_col][parent_tree].col_num
                                                                                               - ncols_num_num - ncols_date]);
                                    }
                                } else {
                                    tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num].size(), false);
                                    for (int cat = 0; cat <= model_outputs.all_trees[outl_col][parent_tree].split_lev; cat++) tmp_bool[cat] = true;
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_ord[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                    cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num][tmp_bool]);
                                }
                                break;
                            }

                            case Greater:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric) {
                                    if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_num_num) {
                                        cond_clust["value_this"] = Rcpp::wrap(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                        cond_clust["value_comp"] = Rcpp::wrap(model_outputs.all_trees[outl_col][parent_tree].split_point);
                                    } else if (model_outputs.all_trees[outl_col][parent_tree].col_num < (ncols_num_num + ncols_date)) {
                                        cond_clust["value_this"] = Rcpp::Date(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]
                                                                              - 1 + min_date[model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_num_num]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                        cond_clust["value_comp"] = Rcpp::Date(model_outputs.all_trees[outl_col][parent_tree].split_point
                                                                              - 1 + min_date[model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_num_num]);
                                    } else {
                                        cond_clust["value_this"] = Rcpp::Datetime(arr_num[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]
                                                                                  - 1 + min_ts[model_outputs.all_trees[outl_col][parent_tree].col_num
                                                                                                 - ncols_num_num - ncols_date]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector(">");
                                        cond_clust["value_comp"] = Rcpp::Datetime(model_outputs.all_trees[outl_col][parent_tree].split_point
                                                                                  - 1 + min_ts[model_outputs.all_trees[outl_col][parent_tree].col_num
                                                                                                 - ncols_num_num - ncols_date]);
                                    }
                                } else {
                                    tmp_bool = Rcpp::LogicalVector(ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num].size(), true);
                                    for (int cat = 0; cat <= model_outputs.all_trees[outl_col][parent_tree].split_lev; cat++) tmp_bool[cat] = false;
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_ord[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                    cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num][tmp_bool]);
                                }
                                break;
                            }

                            case InSubset:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                    tmp_bool = Rcpp::LogicalVector(cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num].size(), false);
                                    for (size_t cat = 0; cat < model_outputs.all_trees[outl_col][parent_tree].split_subset.size(); cat++) {
                                        if (model_outputs.all_trees[outl_col][parent_tree].split_subset[cat] > 0) {
                                            tmp_bool[cat] = true;
                                        }
                                    }
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                    cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num][tmp_bool]);
                                } else {
                                    cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                    cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][parent_tree].split_subset[1]);
                                }
                                break;
                            }

                            case NotInSubset:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                    tmp_bool = Rcpp::LogicalVector(cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num].size(), false);
                                    for (size_t cat = 0; cat < model_outputs.all_trees[outl_col][parent_tree].split_subset.size(); cat++) {
                                        if (model_outputs.all_trees[outl_col][parent_tree].split_subset[cat] == 0) {
                                            tmp_bool[cat] = true;
                                        }
                                    }
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("in");
                                    cond_clust["value_comp"] = Rcpp::as<Rcpp::CharacterVector>(cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num][tmp_bool]);
                                } else {
                                    cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                    cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][parent_tree].split_subset[0]);
                                }
                                break;
                            }

                            case Equal:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].column_type == Categorical) {
                                    if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                      [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                        cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                      [model_outputs.all_trees[outl_col][parent_tree].split_lev]);
                                    } else {
                                        cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                        cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][parent_tree].split_subset[1]);
                                    }
                                } else {
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_ord[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                    cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [model_outputs.all_trees[outl_col][parent_tree].split_lev]);
                                }
                                break;
                            }

                            case NotEqual:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].column_type == Categorical) {
                                    if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                        cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                      [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                        cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                      [model_outputs.all_trees[outl_col][parent_tree].split_lev]);
                                    } else {
                                        cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                        cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                        cond_clust["value_comp"] = Rcpp::wrap((bool) model_outputs.all_trees[outl_col][parent_tree].split_subset[0]);
                                    }
                                } else {
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_ord[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("!=");
                                    cond_clust["value_comp"] = Rcpp::CharacterVector(1, ord_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [model_outputs.all_trees[outl_col][parent_tree].split_lev]);
                                }
                                break;
                            }

                            case SingleCateg:
                            {
                                if (model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_cat_cat) {
                                    cond_clust["value_this"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                    cond_clust["value_comp"] = Rcpp::CharacterVector(1, cat_levels[model_outputs.all_trees[outl_col][parent_tree].col_num]
                                                                                                  [arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]]);
                                } else {
                                    cond_clust["value_this"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                    cond_clust["comparison"] = Rcpp::CharacterVector("=");
                                    cond_clust["value_comp"] = Rcpp::wrap((bool) arr_cat[row + model_outputs.all_trees[outl_col][parent_tree].col_num * nrows]);
                                }
                                break;
                            }

                            default:
                            {
                                assert(0);
                                break;
                            }
                        }

                        
                    }

                    /* https://github.com/RcppCore/Rcpp/issues/979 */
                    /* this comment below will fix Rcpp issue with having slashes in the comment above */
                    temp_list = lst_cond[row];
                    temp_list.push_back(Rcpp::clone(cond_clust));
                    lst_cond[row] = temp_list;
                    curr_tree = parent_tree;
                }
                
            }
            
        }
    }
    
    outp["suspicous_value"]  = outlier_val;
    outp["group_statistics"] = lst_stats;
    outp["conditions"]       = lst_cond;
    outp["tree_depth"]       = tree_depth;
    outp["uses_NA_branch"]   = has_na_col;
    outp["outlier_score"]    = outlier_score;
    return outp;
}

struct args_describe_outliers {
    ModelOutputs *model_outputs;
    double *arr_num;
    int    *arr_cat;
    int    *arr_ord;
    Rcpp::ListOf<Rcpp::StringVector> *cat_levels;
    Rcpp::ListOf<Rcpp::StringVector> *ord_levels;
    Rcpp::StringVector *colnames_num;
    Rcpp::StringVector *colnames_cat;
    Rcpp::StringVector *colnames_ord;
    Rcpp::NumericVector *min_date;
    Rcpp::NumericVector *min_ts;
};

SEXP describe_outliers_wrapper(void *args_)
{
    args_describe_outliers *args = (args_describe_outliers*)args_;
    return describe_outliers(*(args->model_outputs),
                             args->arr_num,
                             args->arr_cat,
                             args->arr_ord,
                             *(args->cat_levels),
                             *(args->ord_levels),
                             *(args->colnames_num),
                             *(args->colnames_cat),
                             *(args->colnames_ord),
                             *(args->min_date),
                             *(args->min_ts));
}


/* for extracting info about flaggable outliers */
Rcpp::List extract_outl_bounds(ModelOutputs &model_outputs,
                               Rcpp::ListOf<Rcpp::StringVector> cat_levels,
                               Rcpp::ListOf<Rcpp::StringVector> ord_levels,
                               Rcpp::NumericVector min_date,
                               Rcpp::NumericVector min_ts)
{
    size_t ncols_num     = model_outputs.ncols_numeric;
    size_t ncols_cat     = model_outputs.ncols_categ;
    size_t ncols_ord     = model_outputs.ncols_ord;
    size_t col_lim_num   = model_outputs.ncols_numeric - min_date.size() - min_ts.size();
    size_t col_lim_date  = model_outputs.ncols_numeric - min_ts.size();
    size_t ncols_cat_cat = cat_levels.size();
    size_t tot_cols = ncols_num + ncols_cat + ncols_ord;
    Rcpp::LogicalVector temp_bool;
    Rcpp::LogicalVector bool_choice(2, false); bool_choice[1] = true;
    Rcpp::List outp(tot_cols);
    
    for (size_t cl = 0; cl < tot_cols; cl++) {
        if (cl < col_lim_num) {
            /* numeric */
            outp[cl] = Rcpp::List::create(Rcpp::_["lb"] = Rcpp::wrap(model_outputs.min_outlier_any_cl[cl]),
                                          Rcpp::_["ub"] = Rcpp::wrap(model_outputs.max_outlier_any_cl[cl]));
        } else if (cl < col_lim_date) {
            /* date */
            outp[cl] = Rcpp::List::create(
                Rcpp::_["lb"] = Rcpp::Date(model_outputs.min_outlier_any_cl[cl] - 1 + min_date[cl - col_lim_num]),
                Rcpp::_["ub"] = Rcpp::Date(model_outputs.max_outlier_any_cl[cl] - 1 + min_date[cl - col_lim_num])
            );
        } else if (cl < ncols_num) {
            /* timestamp */
            outp[cl] = Rcpp::List::create(
                Rcpp::_["lb"] = Rcpp::Datetime(model_outputs.min_outlier_any_cl[cl] - 1 + min_ts[cl - col_lim_date]),
                Rcpp::_["ub"] = Rcpp::Datetime(model_outputs.max_outlier_any_cl[cl] - 1 + min_ts[cl - col_lim_date])
            );
        } else if (cl < (ncols_num + ncols_cat_cat)) {
            /* categorical */
            if (model_outputs.cat_outlier_any_cl[cl - ncols_num].size()) {
                temp_bool = Rcpp::wrap(model_outputs.cat_outlier_any_cl[cl - ncols_num]);
                outp[cl]  = cat_levels[cl - ncols_num][temp_bool];
             } else {
                outp[cl]  = Rcpp::StringVector();
             }
        } else if (cl < (ncols_num + ncols_cat)) {
            /* boolean */
            if (model_outputs.cat_outlier_any_cl[cl - ncols_num].size()) {
                temp_bool = Rcpp::wrap(model_outputs.cat_outlier_any_cl[cl - ncols_num]);
                outp[cl]  = bool_choice[temp_bool];
            } else {
                outp[cl]  = Rcpp::LogicalVector();
            }
        } else {
            /* ordinal */
            if (model_outputs.cat_outlier_any_cl[cl - ncols_num].size()) {
                temp_bool = Rcpp::wrap(model_outputs.cat_outlier_any_cl[cl - ncols_num]);
                outp[cl]  = ord_levels[cl - ncols_num - ncols_cat][temp_bool];
            } else {
                outp[cl]  = Rcpp::StringVector();
            }
        }
    }
    return outp;
}

struct args_extract_outl_bounds {
    ModelOutputs *model_outputs;
    Rcpp::ListOf<Rcpp::StringVector> *cat_levels;
    Rcpp::ListOf<Rcpp::StringVector> *ord_levels;
    Rcpp::NumericVector *min_date;
    Rcpp::NumericVector *min_ts;
};

SEXP extract_outl_bounds_wrapper(void *args_)
{
    args_extract_outl_bounds *args = (args_extract_outl_bounds*)args_;
    return extract_outl_bounds(*(args->model_outputs),
                               *(args->cat_levels),
                               *(args->ord_levels),
                               *(args->min_date),
                               *(args->min_ts));
}


/* external functions for fitting the model and predicting outliers */
// [[Rcpp::export(rng = false)]]
Rcpp::List fit_OutlierTree(Rcpp::NumericVector arr_num, size_t ncols_numeric,
                           Rcpp::IntegerVector arr_cat, size_t ncols_categ,   Rcpp::IntegerVector ncat,
                           Rcpp::IntegerVector arr_ord, size_t ncols_ord,     Rcpp::IntegerVector ncat_ord,
                           size_t nrows, Rcpp::LogicalVector cols_ignore_r, int nthreads,
                           bool categ_as_bin, bool ord_as_bin, bool cat_bruteforce_subset, bool categ_from_maj, bool take_mid,
                           size_t max_depth, double max_perc_outliers, size_t min_size_numeric, size_t min_size_categ,
                           double min_gain, bool follow_all, bool gain_as_pct, double z_norm, double z_outlier,
                           bool return_outliers,
                           Rcpp::ListOf<Rcpp::StringVector> cat_levels,
                           Rcpp::ListOf<Rcpp::StringVector> ord_levels,
                           Rcpp::StringVector colnames_num,
                           Rcpp::StringVector colnames_cat,
                           Rcpp::StringVector colnames_ord,
                           Rcpp::NumericVector min_date,
                           Rcpp::NumericVector min_ts)
{
    Rcpp::List outp = Rcpp::List::create(
        Rcpp::_["ptr_model"] = R_NilValue,
        Rcpp::_["serialized_obj"] = R_NilValue,
        Rcpp::_["bounds"] = R_NilValue,
        Rcpp::_["outliers_info"] = R_NilValue,
        Rcpp::_["ntrees"] = R_NilValue,
        Rcpp::_["nclust"] = R_NilValue,
        Rcpp::_["found_outliers"] = R_NilValue
    );

    bool found_outliers;
    size_t tot_cols = ncols_numeric + ncols_categ + ncols_ord;
    std::vector<char> cols_ignore;
    char *cols_ignore_ptr = NULL;
    if (cols_ignore_r.size() > 0) {
        cols_ignore.resize(tot_cols, false);
        for (size_t cl = 0; cl < tot_cols; cl++) cols_ignore[cl] = (bool) cols_ignore_r[cl];
        cols_ignore_ptr = &cols_ignore[0];
    }
    std::vector<double> Xcpp;
    double *arr_num_C = set_R_nan_as_C_nan(REAL(arr_num), Xcpp, arr_num.size(), nthreads);

    std::unique_ptr<ModelOutputs> model_outputs = std::unique_ptr<ModelOutputs>(new ModelOutputs());
    try {
    found_outliers = fit_outliers_models(*model_outputs,
                                         arr_num_C, ncols_numeric,
                                         INTEGER(arr_cat), ncols_categ, INTEGER(ncat),
                                         INTEGER(arr_ord), ncols_ord,   INTEGER(ncat_ord),
                                         nrows, cols_ignore_ptr, nthreads,
                                         categ_as_bin, ord_as_bin, cat_bruteforce_subset, categ_from_maj, take_mid,
                                         max_depth, max_perc_outliers, min_size_numeric, min_size_categ,
                                         min_gain, gain_as_pct, follow_all, z_norm, z_outlier);

    args_extract_outl_bounds temp = {
        model_outputs.get(),
        &cat_levels,
        &ord_levels,
        &min_date,
        &min_ts
    };
    outp["bounds"] = Rcpp::unwindProtect(extract_outl_bounds_wrapper, (void*)&temp);
    outp["serialized_obj"] = serialize_OutlierTree(model_outputs.get());
    } catch(std::bad_alloc &e) {
        Rcpp::stop("Insufficient memory.\n");
    }

    if (!Rf_xlength(outp["serialized_obj"]))
        return outp;
    if (return_outliers) {
        args_describe_outliers temp = {
            model_outputs.get(),
            arr_num_C,
            INTEGER(arr_cat),
            INTEGER(arr_ord),
            &cat_levels,
            &ord_levels,
            &colnames_num,
            &colnames_cat,
            &colnames_ord,
            &min_date,
            &min_ts
        };
        outp["outliers_info"] = Rcpp::unwindProtect(describe_outliers_wrapper, (void*)&temp);
    }
    forget_row_outputs(*model_outputs);

    /* add number of trees and clusters */
    size_t ntrees = 0, nclust = 0;
    for (size_t col = 0; col < model_outputs->all_trees.size(); col++) {
    	ntrees += model_outputs->all_trees[col].size();
    	nclust += model_outputs->all_clusters[col].size();
    }
    int ntrees_int = (int)ntrees;
    int nclust_int = (int)nclust;
    outp["ntrees"] = Rcpp::unwindProtect(safe_int, (void*)&ntrees_int);
    outp["nclust"] = Rcpp::unwindProtect(safe_int, (void*)&nclust_int);
    outp["found_outliers"] = Rcpp::unwindProtect(safe_bool, (void*)&found_outliers);
    
    outp["ptr_model"] = Rcpp::unwindProtect(safe_XPtr, model_outputs.get());
    model_outputs.release();
    return outp;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List predict_OutlierTree(SEXP ptr_model, size_t nrows, int nthreads,
                               Rcpp::NumericVector arr_num, Rcpp::IntegerVector arr_cat, Rcpp::IntegerVector arr_ord,
                               Rcpp::ListOf<Rcpp::StringVector> cat_levels,
                               Rcpp::ListOf<Rcpp::StringVector> ord_levels,
                               Rcpp::StringVector colnames_num,
                               Rcpp::StringVector colnames_cat,
                               Rcpp::StringVector colnames_ord,
                               Rcpp::NumericVector min_date,
                               Rcpp::NumericVector min_ts)
{
    Rcpp::NumericVector Xcpp;
    double *arr_num_C = set_R_nan_as_C_nan(REAL(arr_num), Xcpp, arr_num.size(), nthreads);

    ModelOutputs *model_outputs = static_cast<ModelOutputs*>(R_ExternalPtrAddr(ptr_model));
    bool found_outliers = find_new_outliers(arr_num_C, INTEGER(arr_cat), INTEGER(arr_ord),
                                            nrows, nthreads, *model_outputs);
    args_describe_outliers temp = {
        model_outputs,
        arr_num_C,
        INTEGER(arr_cat),
        INTEGER(arr_ord),
        &cat_levels,
        &ord_levels,
        &colnames_num,
        &colnames_cat,
        &colnames_ord,
        &min_date,
        &min_ts
    };

    Rcpp::List outp;
    try {
        outp = Rcpp::unwindProtect(describe_outliers_wrapper, (void*)&temp);
    } catch(...) {
        forget_row_outputs(*model_outputs);
        throw;
    }
    forget_row_outputs(*model_outputs);
    outp["found_outliers"] = Rcpp::LogicalVector(found_outliers);
    return outp;
}

// [[Rcpp::export(rng = false)]]
Rcpp::LogicalVector check_few_values(Rcpp::NumericVector arr_num, size_t nrows, size_t ncols, int nthreads)
{
    Rcpp::LogicalVector outp(ncols);
    std::vector<char> too_few_vals(ncols, 0);
    check_more_two_values(REAL(arr_num), nrows, ncols, nthreads, too_few_vals.data());
    for (size_t col = 0; col < ncols; col++) {
        outp[col] = (bool) too_few_vals[col];
    }
    return outp;
}


// [[Rcpp::export(rng = false)]]
bool R_has_openmp()
{
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

#endif /* _FOR_R */
