## 1. Best Friends
* find best word association pairs using the pointwise MI.
* data: TEXTEN1.txt and TEXTCZ1.txt

* POINTWISE MUTAL INFORMATION
   * poors man mutal information
   * Cheap_Mutal_Inf(a,b) = log2 ( P(a,b) / P(a) P(b)) = log2 ( p(a|b) / p(a)) ... if independent P(a,b) == P(a) P(b) thus log(1) == 0
   *   (.. we got rid of double sum over a and over b)
   * p(a,b) = #ab   [ not #ba]
   
   
* Compute pointwise mutual information for all word pairs appearing consecutively 
* (skip pairs in which one or both words < 10 times in the corpus)
* Sort the results from the best to the worst (did you get any negative values? Why?)
   * because joint probability p(a,b) is smaller than p(a)p(b), then lower than 1 ... two words are very rarely together !! (lower than randomly shuffling and been seen i.e. p(a) p(b))
* Tabulate the results, print best 20 pairs for both data sets.
* Do the same for distant words (at least 1 word apart, max 50 words (both directions)). ( a x1 b)  ... ( a x1 x2 ... x50 b)


## 2. Word Classes
* data: TEXTEN1.ptg and TEXTCZ1.ptg ( use strings before '/')


* Full class hierarchy of words
* using the first 8,000 words of those data, and only for words occurring 10 times or more. 
* Ignore the other words for building the classes, but keep them in the data for the bi-gram counts. 


* algorithm: Brown et al. paper distributed in the class (Class 12, formulas for Trick #4)
* Note the history of the merges, and attach it to your homework


* Now run the same algorithm again, but stop when reaching 15 classes
* Print out all the members of your 15 classes

## 3. Tag Classes
* this time, you will compute the classes for tags (the strings after slashes). 
* Compute tag classes for all tags appearing 5 times or more in the data. 
* Use as much data as time allows. 
* You will be graded relative to the other student's results. 
* Again, note the full history of merges, and attach it to your homework. 

* Pick three interesting classes as the algorithm goes (English data only; Czech optional), and comment on them (why you think you see those tags there together (or not), etc.).
