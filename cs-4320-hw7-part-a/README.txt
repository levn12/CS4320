CS 4320 - Assignment 7A Dataset (SMS Spam, Bag-of-Words)
===================================================

Source: SMS Spam Collection (UCI).

Files
-----
- messages.csv
    Columns: id (0..5573), label (0=ham, 1=spam), text (raw SMS message)

- X_counts.npz
    Sparse CSR matrix of token counts (n_samples x n_features)

- X_tfidf.npz
    Sparse CSR matrix of TF-IDF features (same vocab; fit on train split)

- vocab.json
    JSON mapping token -> column index in X_* matrices

- split.json
    Train/validation/test row indices (stratified), with random_state=4320

Vectorization (fixed)
---------------------
CountVectorizer(lowercase=True, token_pattern='(?u)\\b[a-zA-Z]{2,}\\b', min_df=2, max_df=0.95, max_features=5000)
Fit on TRAIN texts only; then applied to all splits to avoid leakage.

TF-IDF
------
TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
Fit on TRAIN count-matrix only; then applied to all splits.

Label meaning
-------------
0 = ham (not spam)
1 = spam
