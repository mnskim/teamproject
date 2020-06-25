
<h1>ATOMIC data generation</h1>

The original COMET code uses creates dynamic strings for all output filenames but we override this with simplified filenames. First, to create a dataset, set the output path like ```DATASET_PICKLE="data/atomic/processed/generation/pathcomet_1.pickle"```. We will pass this filename as arg to all the scripts, which will override the default filename handler.

<h3> Random paths without any restrictions </h3>

```
python scripts/data/make_atomic_data_loader.py --pickled_data "$DATASET_PICKLE" --n_train 100000 --n_dev 5000 --n_test 5000
```


