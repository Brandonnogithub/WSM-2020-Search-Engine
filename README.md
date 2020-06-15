# WSM-2020-Search-Engine

Final project of wsm. Build a search engine of wiki documents

## Requirement

* python 3.7

* nltk

## Index

We run this app in the server with 128G memory. This dataset is 73G. This index size is less than 128G. So it can be loaded into the memory.

We don't need to consider to put index file on disk.

`python -m index.indexer -i data/processed/wiki_00 -o data/index`
