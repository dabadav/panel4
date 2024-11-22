# Return a sequence of content to visitors
from dataclasses import dataclass
from typing import List, Callable, Dict
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")


# ------------------ Content Representation ------------------
@dataclass
class Content:
    """
    A dataclass to store content information.
    """

    id: int
    text: str
    vector: list


class CorpusManager:
    """
    A class to handle the ingestion and embedding of a text corpus.
    """

    def __init__(self, embedding_function: Callable[[str], list]):
        """
        Initializes the CorpusManager with a specified embedding function.

        Args:
            embedding_function (Callable[[str], list]): A function that takes a string as input
                and returns an embedding vector as a list.
        """
        self.embedding_function = embedding_function

    def compute_text_embedding(self, text: str) -> list:
        """
        Computes the embedding vector for the given text using the specified embedding function.
        """
        return self.embedding_function(text)

    def ingest_corpus(self, df) -> Dict[int, Content]:
        """
        Reads the corpus from a CSV file and computes embeddings for each text entry.

        Args:
            file_path (str): Path to the CSV file with columns 'id' and 'text'.

        Returns:
            List[Content]: A list of Content instances with computed embeddings.
        """
        corpus = {
            row["id"]: Content(
                id=row["id"], text=row["text"], vector=self.compute_text_embedding(row["text"])
            )
            for _, row in df.iterrows()
        }
        return corpus


# spaCy embedding functions
def spacy_embedding(text: str) -> list:
    """
    Computes text embedding using SpaCy.
    """
    doc = nlp(text)
    return doc.vector


# BERT embedding functions
def huggingface_embedding(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()


# ------------------ Visitor Representation ------------------
class Visitor:
    """
    Tracks visitor interaction history and provides access to relevant vectors.
    """

    def __init__(self):
        self.history = {}  # Stores {content_id: {}}

    def update(self, content, event, timestamp, isrecommended):
        """
        Updates the visitor's history with new content interactions.
        """
        if content.id not in self.history:
            self.history[content.id] = []

        self.history[content.id].append(
            {
                "content": content,  # Store reference to the Content instance
                "event": event,
                "timestamp": timestamp,
                "isrecommended": isrecommended,
            }
        )

    def get_n_last_visited(self, n: int = 2):
        """
        Retrieves the Content instances of the n last visited content items based on the "close" event.

        Args:
            n (int): The number of last visited content items to retrieve.

        Returns:
            list: A list of Content instances for the last n visited content items based on the "close" event.

        Raises:
            ValueError: If n is greater than the number of visited unique contents or the history is empty.
        """
        if not self.history:
            raise ValueError("Visitor has no content history!")

        # Flatten all events into a list
        all_events = [
            {"content_id": content_id, **event}
            for content_id, events in self.history.items()
            for event in events
        ]
        # Filter for "close" events only
        close_events = [event for event in all_events if event["event"] == "close"]
        # Sort "close" events by timestamp in descending order
        close_events.sort(key=lambda x: x["timestamp"], reverse=True)
        # Check if there are enough unique "close" events
        unique_content_ids = {event["content_id"] for event in close_events}
        if n > len(unique_content_ids):
            n = len(unique_content_ids)
            # raise ValueError(f"Requested {n} items, but only {len(unique_content_ids)} unique 'close' events in history!")

        # Collect the last n unique Content instances
        visited_content = []
        seen_content_ids = set()
        for event in close_events:
            if event["content_id"] not in seen_content_ids:
                visited_content.append(event["content"])
                seen_content_ids.add(event["content_id"])
                if len(visited_content) == n:
                    break

        return visited_content


# ------------------ Similarity Engine Interface ------------------
class SimilarityEngine:
    """
    Abstract base class for similarity computation engines.
    """

    def compute_similarity(self, target_vector, corpus, top_k):
        """
        Computes similarity of the target vector against the corpus.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


# ------------------ Milvus Similarity Engine ------------------
class MilvusManager:
    """
    Handles interactions with the Milvus vector database.
    """

    def __init__(self, milvus_client):
        self.milvus_client = milvus_client

    def store_vectors(self, content_id, vector):
        """
        Stores a content vector in Milvus with an associated content ID.
        """
        # Placeholder for Milvus insertion logic
        pass

    def get_vector(self, content_id):
        """
        Retrieve the corresponding vector of a content ID
        """
        # Modify
        query_result = self.collection.query(
            expr=f"{self.user_id_field} == '{user_id}'", output_fields=[self.profile_vector_field]
        )

        # Check if true
        content_vector = query_result

        return content_vector

    def query_similar_vectors(self, content_id, top_k):
        """
        Queries Milvus for the top-k similar vectors to the target vector.
        """
        # Placeholder for Milvus query logic
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10},
        }

        # Get content vector
        emmbedding = self.get_vector(content_id)

        # Perform similarity search
        results = self.collection.search(
            data=[embedding],
            anns_field=self.profile_vector_field,
            param=search_params,
            limit=top_k,
            output_fields=[self.content_id_field],
        )

        # Return similar content
        similar_content = results  # list(list(str, float))
        return similar_content

    def delete_vector(self, content_id):
        """
        Deletes a content vector from Milvus using the content ID.
        """
        # Placeholder for Milvus deletion logic
        pass


class MilvusSimilarityEngine(SimilarityEngine):
    """
    Computes similarity using the Milvus vector database.
    """

    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager

    def compute_similarity(self, target_vector, corpus=None, top_k=100):
        """
        Queries Milvus for top-k similar vectors to the target vector.
        """
        return self.milvus_manager.query_similar_vectors(target_vector, top_k)


# ------------------ In-Memory Similarity Engine ------------------
class InMemorySimilarityEngine(SimilarityEngine):
    """
    Computes similarity in memory using vector-based operations.
    """

    def __init(self):
        pass

    def compute_similarity(self, content: Content, corpus: Dict[int, Content], top_k=100):
        """
        Computes similarity of the target vector against an in-memory corpus of vectors.
        """
        similarities = []
        for corpus_content in corpus.values():
            similarity_score = self.cosine_similarity(content.vector, corpus_content.vector)
            similarities.append((corpus_content.id, similarity_score))
        # Sort by similarity score and return top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def cosine_similarity(vector1, vector2):
        """
        Computes cosine similarity between two vectors.
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm1 * norm2 + 1e-10)  # To avoid division by zero
        return np.array(similarity)


# ------------------ Recommendation System ------------------
class RecSys:
    """
    Core recommendation system logic that integrates with a similarity engine.
    """

    def __init__(self, similarity_engine: SimilarityEngine, contents: Dict[int, Content]):
        self.similarity_engine = similarity_engine  # Either Milvus or In-Memory engine
        self.contents = contents  # Corpus of content items

    def recommend(self, visitor: Visitor, num_recommendations: int = 12):
        """Returns sequence of 12 content pieces

        Look at the last content history of visitor

        Compute similarity for visited content
        Turn similarity into probability distribution

        Merge probability distributions (could use normalized time spent as a weight)

        Randomly sample 12 items
        """

        # Get last n visited content
        visited_content = visitor.get_n_last_visited()

        if not visited_content:
            raise ValueError("Visitor has no content history.")

        # Initialize probability distribution list
        prob_dist_list = []

        # TODO: Visited content to ~0 prob

        for content in visited_content:
            # Corpus similarity of content visited
            similarities = self.similarity_engine.compute_similarity(
                content=content,
                corpus=(
                    self.contents
                    if isinstance(self.similarity_engine, InMemorySimilarityEngine)
                    else None
                ),
                top_k=100,
            )
            # Extract labels and similarity scores
            labels = [sim[0] for sim in similarities]
            similarity_scores = np.array([sim[1] for sim in similarities])

            # Convert to probability distribution
            total_similarity = similarity_scores.sum()
            prob_dist = similarity_scores / total_similarity

            # Ensure probabilities sum to 1 using numpy
            prob_dist = prob_dist / prob_dist.sum()

            # Store probability distribution
            prob_dist_list.append((labels, prob_dist))

        # Merge the PMFs using multiplication method
        unified_labels, merged_probabilities = self.merge_pmf(prob_dist_list)

        # Randomly sample items based on the probability distribution
        recommended = np.random.choice(
            unified_labels, size=num_recommendations, replace=False, p=merged_probabilities
        )

        return recommended

    @staticmethod
    def merge_pmf(prob_dist_list):
        """
        Merges multiple probability mass functions (PMFs) by element-wise multiplication
        and then normalizes the resulting distribution.

        :param prob_dist_list: A list of tuples where each tuple contains:
                                - labels (list): The list of labels for the PMF.
                                - prob_dist (numpy array): The probability distribution corresponding to the labels.
        :return: A tuple with the merged labels and their corresponding normalized probabilities.
        """
        # Initialize the unified label set and merged probabilities array
        unified_labels = set()
        for labels, _ in prob_dist_list:
            unified_labels.update(labels)

        unified_labels = list(unified_labels)  # Convert to list to maintain consistent order
        merged_probabilities = np.ones(len(unified_labels))

        # Multiply the probability distributions
        for labels, prob_dist in prob_dist_list:
            aligned_probs = np.zeros(len(unified_labels))
            for i, label in enumerate(unified_labels):
                if label in labels:
                    aligned_probs[i] = prob_dist[labels.index(label)]
            # Multiply aligned probabilities into the merged probabilities
            merged_probabilities *= aligned_probs

        # Normalize the merged probabilities so that they sum to 1
        merged_probabilities /= merged_probabilities.sum()

        return unified_labels, merged_probabilities
