Determine the conditional entropy of the word distribution in a text given the previous word.
 
* First you need P(i,j),
    * the probability that at any position in the text you will find the word i followed immediately by the word j, 
* and P(j|i), which is 
    * the probability that if word i occurs in the text then word j will follow.
* Then the conditional entropy of the word distribution in a text given the previous word is
    * H(J|I) = - \sum_{i \in I,j \in J}P(i,j)\log_{2}P(j|i)H(J∣I)=− i∈I,j∈J ∑ ​	 P(i,j)log 2 ​	 P(j∣i)
The perplexity is then computed simply as
PX(P(J|I)) = 2^{H(J|I)}PX(P(J∣I))=2^{H(J|I)}
Compute this conditional entropy and perplexity for the file
TEXTEN1.txt

This file has every word on a separate line. 
(Punctuation is considered a word, as in many other cases.) 
The i,j above will also span sentence boundaries, where i is the last word of one sentence and j is the first word of the following sentence (but obviously, there will be a fullstop at the end of most sentences).

Next, you will mess up the text and measure how this alters the conditional entropy. For every character in the text, mess it up with a likelihood of 10%. If a character is chosen to be messed up, map it into a randomly chosen character from the set of characters that appear in the text. Since there is some randomness to the outcome of the experiment, run the experiment 10 times, each time measuring the conditional entropy of the resulting text, and give the min, max, and average entropy from these experiments. Be sure to use srand to reset the random number generator seed each time you run it. Also, be sure each time you are messing up the original text, and not a previously messed up text. Do the same experiment for mess up likelihoods of 5%, 1%, .1%, .01%, and .001%.
Next, for every word in the text, mess it up with a likelihood of 10%. If a word is chosen to be messed up, map it into a randomly chosen word from the set of words that appear in the text. Again run the experiment 10 times, each time measuring the conditional entropy of the resulting text, and give the min, max, and average entropy from these experiments. Do the same experiment for mess up likelihoods of 5%, 1%, .1%, .01%, and .001%.
Now do exactly the same for the file

TEXTCZ1.txt

which contains a similar amount of text in an unknown language (just FYI, that's Czech [*])

Tabulate, graph and explain your results. Also try to explain the differences between the two languages. To substantiate your explanations, you might want to tabulate also the basic characteristics of the two texts, such as the word count, number of characters (total, per word), the frequency of the most frequent words, the number of words with frequency 1, etc.

Attach your source code commented in such a way that it is sufficient to read the comments to understand what you have done and how you have done it.

Now assume two languages, L_1L 
1
​	  and L_2L 
2
​	  do not share any vocabulary items, and that the conditional entropy as described above of a text T_1T 
1
​	  in language L_1L 
1
​	  is EE and that the conditional entropy of a text T_2T 
2
​	  in language L_2L 
2
​	  is also EE. Now make a new text by appending T_2T 
2
​	  to the end of T_1T 
1
​	 . Will the conditional entropy of this new text be greater than, equal to, or less than EE? Explain (This is a paper-and-pencil exercise of course!)