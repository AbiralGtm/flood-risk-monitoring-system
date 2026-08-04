import joblib
payload = joblib.load("model_random_forest.joblib")
rf = payload["model"]
print("n_estimators:", rf.n_estimators)
print("max_depth:", rf.max_depth)
print("max_leaf_nodes:", rf.max_leaf_nodes)
print("actual trees:", len(rf.estimators_))
print("avg tree depth:", sum(t.get_depth() for t in rf.estimators_) / len(rf.estimators_))
print("avg tree leaves:", sum(t.get_n_leaves() for t in rf.estimators_) / len(rf.estimators_))