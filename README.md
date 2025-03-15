# ML-arena

TODO:
- Winsorizer class
- play with boostings and random forests on synthetic data
    - check monotonic constraints and feature interaction constraints
    - try DART and RF parts of XGBoost
- check outliers=True case
- methods:
    - LGBM / Catboost / XGBoost / RandomForest
    - TabNet / [TabPFN](https://www.automl.org/tabpfn-a-transformer-that-solves-small-tabular-classification-problems-in-a-second/)
    - Lasso / LassoLars / LassoLarsIC / LassoLarsCV / ElasticNet (this and other linear/kernel/MLP models — with standard / robust / quantile scaling)
    - LinearRegression / Ridge / ARD / BayesianRidge
    - HuberRegressor / RANSACRegressor
    - SVR
    - PLS / PLS Canonical / CCA / PRS / other methods from the cross-decomposition page
    - KernelKNN (with CV scale?)
    - KernelRidge
    - KNN
    - GPRegressor
    - MLPRegressor / my variational MLP
    - my modifications: FeatureStacking and RobustRegressor
    - https://automl.github.io/auto-sklearn/master/api.html#regression
    - https://evalml.alteryx.com/en/stable/start.html
    - https://epistasislab.github.io/tpot/latest/Tutorial/2_Search_Spaces/