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


	class _partial_fit
		float (X,y);
	{
	public:

	private:
	};

	class update_model
		float (window);
	{
	public:

	private:
	};

	class anomaly_scores_rate
		float (window);
	{
	public:

	private:
	};


	class predict
		float (X);
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
