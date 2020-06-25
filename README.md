
<h1>ATOMIC data generation</h1>

The original COMET code creates dynamic strings based on configs, for all output filenames. We override this with simplified filenames. 

First, to create a dataset, set the output path like ```DATASET_PICKLE="data/atomic/processed/generation/pathcomet_1.pickle"```. 

We will pass this filename as arg to all the scripts, which will override the default filename handler.

<h3> Random paths without any restrictions </h3>
To create a dataset which consists of random walks sampled from a knowledge graph, run the following command. 
Here, the knowledge graph is generated by merging the triples provided in the ATOMIC dataset, with additional Inverse relations, into a single graph. An example path could have the form: Subject,Relation,Object,InverseRelation,Subject,Relation->Object.
You can set ```--n_train```,  ```--n_dev```,  ```--n_test```, to set the number of examples in each split, and set ```--n_per_node``` and ```--max_path_len``` to control the parameters of the path sampling.

```
python scripts/data/make_atomic_data_loader.py --pickled_data "$DATASET_PICKLE" --n_train 100000 --n_dev 5000 --n_test 5000 --n_per_node 3 --max_path_len 10
```

<h3> Original COMET format data </h3>
The original COMET data has the form, s,r->o. We provide the option to generate this format with the ```--comet``` flag. The other args are ignored here and the entire ATOMIC dataset will be generated.

<h3> Path COMET format data </h3>
We provide the option to sample a single path and prepend it to the s,r of the original COMET data, yielding the form, path,s,r->o. Use the ```--pathcomet``` flag. The other args are ignored here and the entire ATOMIC dataset will be generated.

