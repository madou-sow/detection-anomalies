//	MIT License
//
//  Copyright © 2017 Michael J Simms. All rights reserved.
//
//	Permission is hereby granted, free of charge, to any person obtaining a copy
//	of this software and associated documentation files (the "Software"), to deal
//	in the Software without restriction, including without limitation the rights
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//	copies of the Software, and to permit persons to whom the Software is
//	furnished to do so, subject to the following conditions:
//
//	The above copyright notice and this permission notice shall be included in all
//	copies or substantial portions of the Software.
//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//	SOFTWARE.

#pragma once

#include <map>
#include <random>
#include <set>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>
#include <math>
#include "Rcpp/Inst/include/Rcpp.h"
#include <cstdlib> //required for rand(), srand()
#include <iostream>
#include "ndarray.h"
#include<cstdlib>

int sample_size; 
float n_trees;
float random_state;
float height_limit = np.log2(sample_size);
float trees = [];

namespace Rcpp
{
	class Rcpp::List create_data_table() 
	{
    		Rcpp::List res;
  
    		// Populate the list
    		// ...
  
    		res.attr("class") = Rcpp::CharacterVector::create("data.table", "data.frame");
    		return res;
	};
};


using namespace Rcpp;

namespace DataFrame
{
	// [[Rcpp::export]]
	class DataFrame df(int n = 10, int nobs = 10000) {
    	// create a list with n slots
    	List res(n);
    	CharacterVector list_names = CharacterVector(n);

    	// populate the list elements
    	for (int i = 0; i < n; i++) {
        	list_names[i] = "col_" + std::to_string(i);
        	res[i] = rnorm(nobs);
    	}

    	// set the names for the list elements
    	res.names() = list_names;
    	return res;
	};
};


namespace IsolationTreeTreeEnsemble
{
	class fit
	{
		float (X);
	public:
		if (instanceof<X>(DataFrame df))
            		X = X.values
            		len_x = len(X);
            		col_x = X.shape[1];
            		trees = [];
	
	private:
        	if improved:
            		for (i:n_trees):

                                std::srand(std::time(0));
                		sample_idx = std::rand() % sample(list(range(len_x)), sample_size);
                		temp_tree = IsolationTree(height_limit, 0).fit_improved(X[sample_idx, :], improved=True);
                		trees.append(temp_tree);
        	else:
            		for (i:n_trees):

                		sample_idx = random(sample(list(j:len_x), sample_size));
                		temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :], improved=False);
                		trees.append(temp_tree);

        	return self;

	};


	class path_length
		float (X);
	{
	public:
		pl_vector = []
        	// if isinstance(X, pd.DataFrame):
        	if isinstance(X, DataFrame):
            		X = X.values;

        	for (x:X):
            		pl = std::vector<int>([path_length_tree(x, t, 0) for (t:trees)]);
            		pl = pl.mean();

            		pl_vector.append(pl);

        	pl_vector = std::vector<int>pl_vector = reshape(-1, 1);

        return pl_vector;

	private:
	};

	class anomaly_score
		float (X);
	{
	public:

		return 2.0 ** (-1.0 * path_length(X) / c(len(X)));
	};

	class predict_from_anomaly_scores
		float (scores, thresold);
	{
	public:
		predictions = [1 if p[0] >= threshold else 0 for (p:scores)]

        return predictions;

	};


	class predict
		float (X, threshold);
	{
	public:
		scores = 2.0 ** (-1.0 * path_length(X) / c(len(X)));
        	predictions = [1 if p[0] >= threshold else 0 for (p:cores)];

        return predictions;
	};

};

int height_limit;
int current_height;
char split_by;
char split_value;
char right;
char left;
int size;
int exnodes;
int n_nodes;

namespace IsolationTree
{
        class fit_improved
        {
                float (std:vector<int>X);
        public:
	        if len(X) <= 1 or current_height >= height_limit:
            		exnodes = 1;
            		size = len(X);

            	return current_height;

		//split_by = rand().choice(np.arange(X.shape[1]));
		split_by::rand().array[choice<int>X.shape];
		X.shape[1];
		//split_by = rand().choice(np.arange(X.cols));
		split_by::rand().array[choice(np.arange(X.cols))];
        	min_x = X[:, split_by].min();
        	max_x = X[:, split_by].max();

        	if (min_x == max_x)
            		exnodes = 1;
            		size = len(X);

            		return X;
        	condition = True;

        	while (condition)

            		//split_value = min_x + random.betavariate(0.5,0.5)*(max_x-min_x);
            		split_value = min_x + rand().betavariate[0.5,0.5]*(max_x-min_x);

            		a = X[X[:, split_by] < split_value];
            		b = X[X[:, split_by] >= split_value];
            		if (len(X) < 10 or a.shape[0] < 0.25 * b.shape[0] or b.shape[0] < 0.25 * a.shape[0] or (
                    			a.shape[0] > 0 and b.shape[0] > 0))
                		condition = False;

            		size = len(X);
            		split_by = split_by;
            		split_value = split_value;

            		left = IsolationTree(height_limit, current_height + 1).fit_improved(a, improved=False);
            		right = IsolationTree(height_limit, current_height + 1).fit_improved(b, improved=False);
            		n_nodes = left.n_nodes + right.n_nodes + 1;

        	return X;		

        };


        class fit
                float (X);
        {
        public:
		if (len(X) <= 1 or current_height >= height_limit)
            		exnodes = 1;
            		size = X.shape[0];

            		return X;

        	//split_by = random.choice(np.arange(X.shape[1]));
		split_by::rand().array[choice<int>X.shape];
        	X_col = X[:, split_by];
        	min_x = X_col.min();
        	max_x = X_col.max();

        	if (min_x == max_x)
            		exnodes = 1;
            	size = len(X);

            		return X;

        	else

            		//split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)
            		split_value = min_x + (rand().betavariate[0.5, 0.5]) * (max_x - min_x);

            		//w = np.where(X_col < split_value, True, False);
			w::array[where(X_col < split_value, True, False)];
            		del X_col;

            		size = X.shape[0];
            		split_by = split_by;
            		split_value = split_value;

            		left = IsolationTree(height_limit, current_height + 1).fit(X[w], improved=True);
            		right = IsolationTree(height_limit, current_height + 1).fit(X[~w], improved=True);
            		n_nodes = left.n_nodes + right.n_nodes + 1;

        	return X;


        };

        class find_TPR_threshold
                float (y, scores, desired_TPR);
        {
        public:
		threshold = 1;

    		while (threshold > 0)
        		y_pred = [1 if p[0] >= threshold else 0 for p in scores];
        		tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel();
        		TPR = tp / (tp + fn);
        		FPR = fp / (fp + tn);
        		if (TPR >= desired_TPR)
            			return threshold, FPR;

        		threshold = threshold - 0.001;

    		return threshold, FPR;

        };

        class c
                float (n);
        {
        public:
    		if (n > 2)
        		//return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0));
        		return 2.0*(array[log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))];
    		elif (n == 2)
        		return 1;
    		if (n == 1)
        		return 0;		
        };


        class path_length_tree
                float (x,t,e);
        {
        public:
		float e = e;
    		if (t.exnodes == 1)
        		e = e+ c(t.size);
        		return e;
    		else:
        		a = t.split_by;
        		if (x[a] < t.split_value)
            	return path_length_tree(x, t.left, e+1);

        	if (x[a] >= t.split_value)
            		return path_length_tree(x, t.right, e+1);

        };


};

