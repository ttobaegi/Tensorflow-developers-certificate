## COURSE 3 - Natural Language Processing in TensorFlow
> ### Table of Contents
>    - [WEEK 1 Sentiment in text](#3-1)      
>    - [WEEK 2 Word Embeddings](#3-2)      
>    - [WEEK 3 Sequence models](#3-3)      
>    - [WEEK 4 Sequence models and literature](#3-4)      

</br>
</br>

<a name='3-1'></a>
### WEEK 1. Sentiment in text

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
### WEEK 2. Word Embeddings
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
- `tf.keras.layers.Embedding` to use word embeddings in TensorFlow, in a sequential layer.
- The type of loss function for IMDB Reviews(either positive or negative) : `Binary crossentropy`
- The results of using IMDB Sub Words dataset in classification was poor. Because sequence becomes much more important when dealing with subwords, but `we’re ignoring word positions`.

</br>
</br>

<a name='3-3'></a>
### WEEK 3. Sequence models
- `Sequence` make a large difference when determining semantics of language. Because the `order` in which words appear dictate their impact on the meaning of the sentence.
- Recurrent Neural Networks help you understand the impact of sequence on meaning by `carrying meaning from one cell to the next`.
- LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
Values from earlier words can be carried to later ones via a cell state
- `Bidirectional` allows LSTMs to look forward and backward in a sentence?
- The output shape of a bidirectional LSTM layer with 64 units is (None, `128`)
- When stacking LSTMs, ensure that `return_sequences = True` only on units that feed to another LSTM to feed the next one in the sequence.
- The best way to avoid overfitting in NLP datasets : None of the above (Use LSTMs, Use GRUs, Use Conv1D)
Remember that with text, you'll probably get a bit more overfitting than you would have done with images. Not least because you'll almost always have out of vocabulary words in the validation data set. That is words in the validation dataset that weren't present in the training, naturally leading to overfitting. These words can't be classified and, of course, you're going to have these overfitting issues, but see what you can do to avoid them.

</br>
</br>

<a name='3-4'></a>
### WEEK 4. Sequence models and literature
- `fit_on_texts(sentences)` : the method used to tokenize a list of sentences.
- Sentence with 120 tokens & Conv1D with 128 filters with a Kernal size of 5 is passed over it, the output shape : (None, 116, 128)
- The purpose of the `embedding dimension` :  the number of dimensions for the vector representing the word encoding.
- `pad_sequences` object from the tensorflow.keras.preprocessing.sequence namespace : If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
- When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Because the probability that each word matches an existing phrase goes down the more words you create
- The major drawback of word-based training for text generation instead of character-based generation is that there are far more words in a typical corpus than characters, it is much more memory intensive.


</br>
</br>
