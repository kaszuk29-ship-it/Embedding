from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love learning music",
    "I like to do crafts",
    "The dress suits me perfectly",
    "I got success in my life",
    "I enjoy dancing"
]

sentence_embedding = []

sentence1 = input("Enter sentence 1: ")
sentence2 = input("Enter sentence 2: ")

sentences.append(sentence1)
sentences.append(sentence2)

for sentence in sentences:

    embedding = model.encode(sentence)

    sentence_embedding.append(embedding)

print("\n========== EMBEDDINGS ==========")

for i in range(len(sentences)):

    print("\nSentence:", sentences[i])
    print("Embedding:", sentence_embedding[i])
    print("Dimension:", len(sentence_embedding[i]))

embedding1 = sentence_embedding[-2]
embedding2 = sentence_embedding[-1]

similarity = cosine_similarity(
    [embedding1],
    [embedding2]
)

print("\n========== SIMILARITY ==========")

print("\nSentence 1:", sentence1)
print("Sentence 2:", sentence2)

print("\nSimilarity Score:")
print(similarity[0][0])