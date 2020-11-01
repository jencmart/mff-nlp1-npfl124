### Requirements
* Scripts expect folder `./dataset` in which two files `TEXTEN1.txt` and `TEXTCZ1.txt` are located. (you can change the folder location by param `--dataset_dir`. But in this folder still must be the files `TEXTEN1.txt` and `TEXTCZ1.txt`


* then you also need `python3` 

### Part 1: Conditional Entropy of the text

#### How to run the script
```
python3 ./text_entropy.py --dataset_dir path_to_dataset_folder
```
* script will create directory `results` with results in `res.csv`

* it also creates `lang_stats.csv` where basic properties of the dataset are stored.

#### Analysis of the results
* There is Jupiter notebook `analysis-entropy-of-text.ipynb` with the analysis of the results.


* I have also exported it to pdf: `analysis-entropy-of-text.pdf` 


### Part 2: Cross Entropy

#### How to run the script

```
python3 ./cross_entropy.py --dataset_dir path_to_dataset_folder
```

* script will print all the results and also create directory `results` with results in `cross-ent.csv.params` (found $\lambda$ params) and `cross-ent.csv` with cross-entropy results

#### Analysis of the results
* There is Jupiter notebook `analysis-cross-entropy.ipynb` with the analysis of the results.


* I have also exported it to pdf: `analysis-cross-entropy.pdf`
