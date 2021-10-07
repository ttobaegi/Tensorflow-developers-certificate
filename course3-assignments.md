
>#### Table of Contents
> - [**COURSE 3 - Natural Language Processing in TensorFlow**](#3)  
>   
>      [WEEK 1 Sentiment in text](#3-1)
>      [WEEK 2 Word Embeddings](#3-2)
>      [WEEK 3 Sequence models](#3-3)
>      [WEEK 4 Sequence models and literature](#3-4)

</br>
</br>

<a name='3'></a>
## COURSE 3. Natural Language Processing in TensorFlow

</br>
</br>

<a name='3-1'></a>
### WEEK 1 Sentiment in text

- What is the name of the object used to tokenize sentences
CharacterTokenizer
WordTokenizer
Tokenizer
TextTokenizer


- What is the name of the method used to tokenize a list of sentences?
tokenize_on_text(sentences)
fit_to_text(sentences)
tokenize(sentences)
fit_on_texts(sentences)

- Once you have the corpus tokenized, what’s the method used to encode a list of sentences to use those tokens?
text_to_tokens(sentences)
text_to_sequences(sentences)
texts_to_sequences(sentences)
texts_to_tokens(sentences)

- When initializing the tokenizer, how to you specify a token to use for unknown words?
unknown_token=<Token>
unknown_word=<Token>
oov_token=<Token>
out_of_vocab=<Token>

- If you don’t use a token for out of vocabulary words, what happens at encoding?
The word is replaced by the most common token
The word isn’t encoded, and the sequencing ends
The word isn’t encoded, and is skipped in the sequence
The word isn’t encoded, and is replaced by a zero in the sequence

- If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
Use the pad_sequences object from the tensorflow.keras.preprocessing.sequence namespace
Process them on the input layer of the Neural Netword using the pad_sequences property
Make sure that they are all the same length using the pad_sequences method of the tokenizer
Specify the input layer of the Neural Network to expect different sizes with dynamic_length


- If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?
They’ll get cropped to the length of the shortest sequence
Nothing, they’ll remain unchanged
They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones
They’ll get padded to the length of the longest sequence by adding zeros to the end of shorter ones

  
- When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?
Pass padding=’after’ to pad_sequences when initializing it
Call the padding method of the pad_sequences object, passing it ‘post’
Pass padding=’post’ to pad_sequences when initializing it
Call the padding method of the pad_sequences object, passing it ‘after’



</br>
</br>

<a name='3-1'></a>
### WEEK 2 Word Embeddings

</br>
</br>

<a name='3-1'></a>
### WEEK 3 Sequence models

</br>
</br>

<a name='3-4'></a>
### WEEK 4 Sequence models and literature


</br>
</br>
