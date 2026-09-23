# Sentence Embedding and Similarity

This project demonstrates how to generate sentence embeddings using Sentence Transformers and calculate the similarity between two sentences using cosine similarity.

## Technologies Used

- Python
- Sentence Transformers
- Scikit-learn
- all-MiniLM-L6-v2

## How It Works

The program:

1. Loads the `all-MiniLM-L6-v2` Sentence Transformer model.
2. Takes two sentences as input from the user.
3. Converts the sentences into numerical embeddings.
4. Displays the embeddings and their dimensions.
5. Calculates the cosine similarity between the two input sentences.
6. Displays the similarity score.

## Installation

Install the required libraries using:

```bash
python -m pip install sentence-transformers scikit-learn
