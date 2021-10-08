
> ### Table of Contents
> - [**COURSE 3 - Natural Language Processing in TensorFlow**](#3)  
>    - [WEEK 1 Sentiment in text](#3-1)      
>    - [WEEK 2 Word Embeddings](#3-2)      
>    - [WEEK 3 Sequence models](#3-3)      
>    - [WEEK 4 Sequence models and literature](#3-4)      

</br>
</br>

<a name='3'></a>
## COURSE 3. Natural Language Processing in TensorFlow

</br>

<a name='3-1'></a>
### WEEK 1 Sentiment in text

- `Tokenizer` : the object used to tokenize sentences
- `fit_on_texts(sentences)` : the method used to tokenize a list of sentences
- `texts_to_sequences(sentences)` : Once you have the corpus tokenized, what’s the method used to encode a list of sentences to use those tokens?
- When initializing the tokenizer, how to you specify a token to use for unknown words?
- `oov_token=<Token>` : token for outer vocabulary to be used for words that aren't in the word index.
- If you don’t use a token for out of vocabulary words, the word isn’t encoded, and is skipped in the sequence
- If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
Process them on the input layer of the Neural Netword using the pad_sequences property
- If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?
They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones
- When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?
Pass padding=’post’ to pad_sequences when initializing it



</br>
</br>

<a name='3-2'></a>
### WEEK 2 Word Embeddings
-  the TensorFlow library containing common data that you can use to train and test neural networks?
TensorFlow Data Libraries
TensorFlow Data
There is no library of common data sets, you have to use your own
TensorFlow Datasets (TFDS)
-  [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/) : 50,000 records, 80/20 train/test split
- How are the labels for the IMDB dataset encoded? Reviews encoded as a number 0-1
- What is the purpose of the embedding dimension?
It is the number of dimensions for the vector representing the word encoding

- When tokenizing a corpus, what does the num_words=n parameter do?

It specifies the maximum number of words to be tokenized, and picks the most common ‘n’ words
- To use word embeddings in TensorFlow, in a sequential layer, what is the name of the class?

tf.keras.layers.Embedding

- IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?

Binary crossentropy
- `IMDB Sub Words dataset`, our results in classification were poor. Why?

Sequence becomes much more important when dealing with subwords, but we’re ignoring word positions



</br>
</br>

<a name='3-3'></a>
### WEEK 3 Sequence models

- Why does sequence make a large difference when determining semantics of language?
Because the order in which words appear dictate their meaning
Because the order of words doesn’t matter
It doesn’t
Because the order in which words appear dictate their impact on the meaning of the sentence
- How do Recurrent Neural Networks help you understand the impact of sequence on meaning?
They shuffle the words evenly
They look at the whole sentence at a time
They carry meaning from one cell to the next
They don’t

- How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
They don’t
They load all words into a cell state
They shuffle the words randomly
Values from earlier words can be carried to later ones via a cell state

- What keras layer type allows LSTMs to look forward and backward in a sentence?
Bothdirection
Bidirectional
Bilateral
Unilateral

- What’s the output shape of a bidirectional LSTM layer with 64 units?
(128,1)
(128,None)
(None, 64)
(None, 128)

- When stacking LSTMs, how do you instruct an LSTM to feed the next one in the sequence?
Do nothing, TensorFlow handles this automatically
Ensure that return_sequences is set to True only on units that feed to another LSTM
Ensure that they have the same number of units
Ensure that return_sequences is set to True on all units

- If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?
(None, 120, 124)
(None, 120, 128)
(None, 116, 124)
(None, 116, 128)
8.
Question 8
- What’s the best way to avoid overfitting in NLP datasets?
Use LSTMs
Use GRUs
Use Conv1D
None of the above


</br>
</br>

<a name='3-4'></a>
### WEEK 4 Sequence models and literature


</br>
</br>
