import numpy as np
import optuna
from collections import namedtuple
import OptimizeHyperparameters as ohp

import numpy as np
import optuna
from collections import namedtuple

AnalysisResult = namedtuple("AnalysisResult",
	["param_names", "param_importance", "params_to_sample", "enqueued_params", "new_ranges"])

def analyze_study_refined_simple(study: optuna.study.Study, topk: int = 20, cv_thresh: float = 0.1, importance_thresh: float = 0.01) -> AnalysisResult:
	"""
	Refined analysis of Optuna study for next-stage search (simplified min-max ranges):
	- Computes parameter importance
	- Prepares enqueued trials (median of top-k)
	- Adjusts search ranges based on min/max of top-k trials

	Returns
	-------
	AnalysisResult namedtuple
	"""
	# 1. Get top-k trials
	best_trials = sorted(study.trials, key=lambda t: t.value)[:topk]
	param_names = list(best_trials[0].params.keys())
	param_types = best_trials[0].distributions
	n_params = len(param_names)
	values = np.array([[t.params[p] for t in best_trials] for p in param_names], dtype=float)
	
	means = values.mean(axis=1)
	stds = values.std(axis=1)
	cvs = np.divide(stds, np.abs(means), out=np.zeros_like(stds), where=np.abs(means) > 1e-12)
	param_importance = optuna.importance.get_param_importances(study)

	new_ranges = {}
	enqueued_params = {}
	for i, p in enumerate(param_names):
		dist = param_types[p]
		if isinstance(dist, (optuna.distributions.FloatDistribution, optuna.distributions.IntDistribution)):
			is_int = isinstance(dist, optuna.distributions.IntDistribution)
			islog = dist.log
			
			if islog:
				median_val = np.exp(np.log(values[i]))
			else:
				median_val = np.median(values[i])
			if is_int:
				median_val = int(median_val)
			
			enqueued_params[p] = median_val

			if cvs[i] > cv_thresh and param_importance.get(p, 0) > importance_thresh:
				new_ranges[p] = {
				"min": float(values[i].min()),
				"max": float(values[i].max()),
				"log": islog,
				"type": "int" if is_int else "float"
				}
		elif isinstance(dist, optuna.distributions.CategoricalDistribution):
			ptype = "categorical"
			values = [t.params[p] for t in best_trials]
			enqueued_params[p] = max(set(values), key=values.count)
			if cvs[i] > cv_thresh and param_importance.get(p, 0) > importance_thresh:
				new_ranges[p] = {"choices": param_types[p].choices, "type": ptype}

	return AnalysisResult(param_importance, enqueued_params, new_ranges)


