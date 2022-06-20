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

int sample_size; 
float n_trees;
float random_state;

namespace IsolationTreeTreeEnsemble
{
	class fit
	{
		float (X);
	public:

	protected:

	private:
	};


	class path_length
		float (X);
	{
	public:

	private:
	};

	class anomaly_score
		float (X);
	{
	public:

	private:
	};

	class predict_from_anomaly_scores
		float (scores, thresold);
	{
	public:

	private:
	};


	class predict
		float (X, threshold);
	{
	public:

	private:
	};

	class predict_proba
                float (X);
        {
        public:

        private:
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
                float (X);
        public:

        protected:

        private:
        };


        class fit
                float (X);
        {
        public:

        private:
        };

        class find_TPR_threshold
                float (X);
        {
        public:

        private:
        };

        class c
                float (scores, thresold);
        {
        public:

        private:
        };


        class path_length_tree
                float (X, threshold);
        {
        public:

        private:
        };


};

