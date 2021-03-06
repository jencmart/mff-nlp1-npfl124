### Requirements
* Scripts expect folder `./dataset` in which two files `TEXTEN1.txt` and `TEXTCZ1.txt` are located. 
* You can change the folder location by param `--dataset_dir`
* But in this folder still must be the files `TEXTEN1.txt` and `TEXTCZ1.txt`


* beside that you only need `python3` 

### Part 1: Conditional Entropy of the text + Pen and Paper exercise

```
python3 ./text_entropy.py --dataset_dir path_to_dataset_folder
```
* script will create directory `results` with results in `res.csv`

* it also creates `lang_stats.csv` where basic properties of the both datasets are stored.


### Part 2: Cross Entropy
```
python3 ./cross_entropy.py --dataset_dir path_to_dataset_folder
```

* script will print all the results and also create directory `results` with results in `cross-ent.csv.params` (found $\lambda$ params) and `cross-ent.csv` with cross-entropy results
