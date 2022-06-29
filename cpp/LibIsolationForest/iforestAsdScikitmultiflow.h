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

int n_estimators;
int ensemble;
float random_state;
float window_size;
float samples_seen;
float anomaly_rate;
float anomaly_threshold;
float drift_threshold;
float window;
float prec_window;
int cpt;

namespace IsolationForestStream
{
	class partial_fit
	{
		float (X,y, classes, sample_weight);
	public:
		number_instances, X.shape

		if (samples_seen == 0)
			iforest = IsolationTreeEnsemble(window_size,n_estimators,random_state);
			ensemble = iforest;
			return ensembe;

		for (int i:number_instances)
			partial_fit(X[i], y[i]);
		return partial_fit;

	};


	class _partial_fit
		float (X,y);
	{
	public:

		std::vector<int> X.reshape(1,len(X));


	private:
		if (samples_seen % window_size == 0)
			prec_window = window;
			window =x;
		else
			window = window.append(X);

		if (samples_seen % windows_size == 0 and samples_seen !=0)
			if(cpt < n_estimators)
				ensemble.fit(prec_window);
				cpt += 1;
				anomaly_rate = anomaly_scores_rate(prec_window);
			
			if (anomaly_rate > drift_threshold)
				update_model(pred_window);

		samples_seen=1;
			

	};

	class update_model
		float (window);
	{
	public:
		is_learning_phase_on = True;
		iforest = IsolationTreeEnsemble(window_size,n_estimators,random_state);
    		ensemble = iforest;
    		ensemble.fit(window);
    		cout << ("Update");
	};

	class anomaly_scores_rate
		float (window);
	{
	public:
		score_tab = 2.0**(-1.0*ensemble.path_length(window)/c(len(window)));
		score = 0;
		for (auto x: score_tab)
			if (x > anomaly_thresold)
				score += 1;
		return score / len(score_tab);

	};

	class predict
		float (X);
	{
	public:
		if(samples_seen <= window_size)

      			return [-1]; 

    		X = np.reshape(X,(1,len(X[0])));
    		prec_window = prec_window.add(X);

    		prediction =  ensemble.predict_from_anomaly_scores(predict_proba(prec_window),anomaly_threshold) 

    		return [prediction];

	};

	class predict_proba
                float (X);
        {
        public:
		if(samples_seen <= window_size)
        		return [-1];
    		return ensemble.anomaly_score(prec_window)[-1];

        };

};
