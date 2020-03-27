README
===========================
This project can help you simply search the documents in .../document with vector space model and Python3.

## Install python library

Before run there codes, you must install the following libraries:
* numpy
* nltk

If you don't have the following libraries, you can type following commands:
```sh
pip3 install --user -U numpy
pip3 install --user -U nltk
```
So now you have the tools to help you run these codes!

## How to use the codes

Please first get into the document folder, and run
```sh
unrar -e document.rar
```

And now you have the document to search!

For example, if you want to search "drill wood sharp".
You can just type the code below: 

```sh
python3 main.py --query "drill wood sharp"
```

And wait about 1 minute you will get your results!

## Code introduction

Here is the `Vector Space` model
```python
from VectorSpace import VectorSpace

vectorSpace = VectorSpace(documents)
```

`.split` can help you split your query into list.
```python
query = args.query.split(" ")
```

`.search` will return the Cosine Similarity and Euclidean Distance of all documents and your query.
```python
cosineResult, euclideanResult = vectorSpace.search(query)
```

Use the Nouns and the Verbs from the top document of method 3 for Pseudo Feedback.
`nltk` will help you exract the Nouns and Verbs
```python
f = open(realDir + '/' + method3_top, 'r')
tokens = nltk.word_tokenize(f.read())
result = nltk.pos_tag(tokens)
```

And the above Nouns and Verbs will add weight of 0.5 to the query vector to make your results more accurate!
```python
for word in feedbackList:
	if(word in self.vectorKeywordIndex):
		vector[self.vectorKeywordIndex[word]] += 0.5
```
