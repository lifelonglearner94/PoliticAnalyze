# TODO s

import json
import os
import spacy
from spacy.language import Language
import re
import PyPDF2
from collections import defaultdict, Counter

def get_all_filenames_from_a_given_path(path):
    # get all filenames from a given path
    import os
    return os.listdir(path)


def read_pdf(file_path):
    # read pdf file
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        num_pages = len(pdf.pages)
        text = ''
        for page_num in range(num_pages):
            page = pdf.pages[page_num]
            text += page.extract_text()
    return text






def create_german_nlp_pipeline():
    # Deutsches Sprachmodell laden
    nlp = spacy.load('de_core_news_sm')

    # Benutzerdefinierte Komponente für Textbereinigung
    @Language.component("text_cleanup")
    def text_cleanup(doc):
        text = doc.text

        # URLs entfernen
        text = re.sub(r'http\S+|www\S+', '', text)

        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text)

        # Unnötige Zahlen und Nummerierungen entfernen
        text = re.sub(r'\b\d+\b', '', text)

        # Unnötige Punkte entfernen
        text = re.sub(r'\.{2,}', '.', text)

        # Bindestriche entfernen und Wörter zusammenfügen
        text = re.sub(r'-\s+', '', text)

        # Getrennte Buchstaben wieder zu Wörtern zusammensetzen (z. B. "Inhal t" -> "Inhalt")
        text = re.sub(r'(\b\w)\s+(\w\b)', r'\1\2', text)

        return nlp.make_doc(text.strip())

    # Benutzerdefinierte Komponente zur Pipeline hinzufügen
    nlp.add_pipe("text_cleanup", first=True)

    return nlp

def preprocess_text(text: str, nlp) -> dict:
    doc = nlp(text)

    # Filter tokens based on conditions
    tokens = [
        token for token in doc
        if not token.is_space
        and not token.is_stop
        and not token.is_digit
        and not token.is_punct
        and not token.is_quote
        and not token.is_bracket
        and token.text != "--"
        and token.text != "\uf0b7"
        and token.text != "|"
    ]

    # Lemmas lowercased
    lemmas_lowercased = [token.lemma_.lower() for token in tokens]

    # Lemmas with POS tags
    lemmas_with_pos = [(token.lemma_.lower(), token.pos_) for token in tokens]

    # Sentences
    sentences = [sent.text.strip() for sent in doc.sents]

    processed_sentences = []

    for sentence in sentences:
        # Process the sentence using spaCy
        doc = nlp(sentence)

        # Check if the sentence has 3 or more words (tokens)
        if len([token.text for token in doc if not token.is_stop]) >= 3:
            processed_sentences.append(sentence)

    return {
        'lemmas_lowercased': lemmas_lowercased,
        'lemmas_with_pos': lemmas_with_pos,
        'sentences': processed_sentences,
    }






from collections import Counter, defaultdict

def calculate_lemma_frequencies(data, top_n_overall=5, top_n_per_category=10):
    """
    Calculate lemma frequencies overall and per POS category.

    Parameters:
        data (dict): A dictionary containing 'lemmas_lowercased' and 'lemmas_with_pos'.
        top_n_overall (int, optional): Number of top overall lemmas to return.
        top_n_per_category (int, optional): Number of top lemmas per POS category to return.

    Returns:
        dict: A dictionary containing:
            - 'top_overall': Top N overall lemmas and their frequencies.
            - 'top_per_category': Top N lemmas per POS category and their frequencies.
    """
    # Extract the necessary data from the input dictionary
    lemmas_lowercased = data['lemmas_lowercased']
    lemmas_with_pos = data['lemmas_with_pos']

    # Calculate overall lemma frequencies
    lemma_frequencies = Counter(lemmas_lowercased)

    # Get top N overall lemmas if specified
    top_overall = (lemma_frequencies.most_common(top_n_overall)
                   if top_n_overall else lemma_frequencies.items())

    # Initialize a dictionary to store lemma frequencies per POS category
    pos_frequencies = {
        'NOUN': defaultdict(int),
        'VERB': defaultdict(int),
        'ADJ': defaultdict(int),
    }

    # Iterate through lemmas_with_pos to count frequencies per POS category
    for lemma, pos in lemmas_with_pos:
        if pos in pos_frequencies:
            pos_frequencies[pos][lemma] += 1

    # Get top N per category if specified
    top_per_category = {}
    for pos, frequencies in pos_frequencies.items():
        sorted_frequencies = Counter(frequencies)
        top_per_category[pos] = (sorted_frequencies.most_common(top_n_per_category)
                                  if top_n_per_category else sorted_frequencies.items())

    return {
        'lemmas_top_overall': dict(top_overall),
        'lemmas_top_per_category': {pos: dict(top) for pos, top in top_per_category.items()},
    }


from pprint import pprint

def pretty_print_frequencies(result, top_n_overall=20, top_n_per_category=15):
    # Print the top N most frequent lemmas overall
    print("Top {} most frequent lemmas overall:".format(top_n_overall))
    overall_freq = result['lemma_frequencies']
    sorted_overall = sorted(overall_freq.items(), key=lambda x: x[1], reverse=True)[:top_n_overall]
    pprint(sorted_overall)
    print("\n")

    # Print the top N most frequent lemmas for each POS category
    pos_freq = result['pos_frequencies']
    for pos, lemma_counts in pos_freq.items():
        print("Top {} most frequent {} lemmas:".format(top_n_per_category, pos))
        sorted_pos = sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_per_category]
        pprint(sorted_pos)
        print("\n")




def analyze_german_sentiments(sentences_list):
    """
    Perform sentiment analysis on German sentences and calculate sentiment distribution.

    Args:
        sentences_dict (dict): Dictionary containing preprocessed German sentences

    Returns:
        tuple: (sentiment_results, sentiment_percentages)
    """
    from transformers import pipeline
    # Initialize the sentiment analysis pipeline for German text
    classifier = pipeline('sentiment-analysis',
                        model='oliverguhr/german-sentiment-bert',
                        tokenizer='oliverguhr/german-sentiment-bert')

    # Store results
    results = []

    # Process each sentence
    for sentence in sentences_list:
        try:
            # Perform sentiment analysis
            result = classifier(sentence)[0]
            sentiment = result['label']
            confidence = result['score']
            # print()
            # print(f"Sentence: {sentence} | Sentiment: {sentiment} | Confidence: {confidence}")
            # Store results with additional information
            results.append({
                'sentence': sentence,
                'sentiment': sentiment,
                'confidence': confidence
            })

        except Exception as e:
            print(f"Error processing sentence: {str(e)}")

    # Calculate sentiment distribution
    sentiment_counts = Counter([r['sentiment'] for r in results])
    total_sentences = len(results)

    # Calculate percentages
    sentiment_percentages = {
        sentiment: (count / total_sentences) * 100
        for sentiment, count in sentiment_counts.items()
    }

    # Print results
    # print("\nSentiment Analysis Results:")
    # print("-" * 50)
    # print(f"Total sentences analyzed: {total_sentences}\n")

    # for sentiment, percentage in sentiment_percentages.items():
    #     print(f"{sentiment}: {percentage:.2f}%")

    return results, sentiment_percentages




def most_common_topics(sentences_list, german_stopwords, num_clusters=None, min_sentences_per_cluster=3):
    """
    Extrahiert häufige Themen aus einer Liste deutscher Sätze mittels Clustering.

    Args:
        sentences_list (list): Liste von deutschen Sätzen
        num_clusters (int, optional): Anzahl der Cluster. Falls None, wird automatisch bestimmt
        min_sentences_per_cluster (int): Minimale Anzahl Sätze pro Cluster

    Returns:
        dict: Cluster-IDs mit zugehörigen Keywords
        list: Häufigste Themen über alle Cluster
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from collections import Counter
    import numpy as np
    # Input-Validierung
    if not sentences_list or not isinstance(sentences_list, list):
        raise ValueError("Eingabe muss eine nicht-leere Liste von Sätzen sein")

    # Entferne leere Sätze und doppelte Einträge
    sentences_list = list(filter(None, set(sentences_list)))

    # Automatische Cluster-Anzahl bestimmen
    if num_clusters is None:
        num_clusters = max(2, min(len(sentences_list) // min_sentences_per_cluster, 15))

    try:
        # Vektorisierung
        model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

        # Batch-Processing für große Listen
        batch_size = 32
        sentence_embeddings = []
        for i in range(0, len(sentences_list), batch_size):
            batch = sentences_list[i:i + batch_size]
            embeddings = model.encode(batch)
            sentence_embeddings.append(embeddings)

        sentence_embeddings = np.vstack(sentence_embeddings)

        # Clustering
        clustering_model = KMeans(n_clusters=num_clusters,
                                random_state=42,
                                n_init=10)
        cluster_assignments = clustering_model.fit_predict(sentence_embeddings)

        # Gruppieren nach Clustern
        clustered_sentences = {i: [] for i in range(num_clusters)}
        for sentence, cluster in zip(sentences_list, cluster_assignments):
            clustered_sentences[cluster].append(sentence)

        # Erweiterte Stop-Wörter
        custom_german_stop_words = german_stopwords

        # Keyword Extraction mit Fehlerbehandlung
        def extract_keywords(sentences):
            if not sentences:
                return []
            try:
                vectorizer = CountVectorizer(max_features=5,
                                          stop_words=custom_german_stop_words,
                                          min_df=2)  # Mindestens 2 Vorkommen
                X = vectorizer.fit_transform(sentences)
                return vectorizer.get_feature_names_out()
            except Exception as e:
                print(f"Fehler bei Keyword-Extraktion: {e}")
                return []

        # Themen extrahieren
        topics = {}
        for cluster_id, cluster_sentences in clustered_sentences.items():
            if len(cluster_sentences) >= min_sentences_per_cluster:
                topics[cluster_id] = extract_keywords(cluster_sentences)

        # Häufigste Themen analysieren
        all_keywords = [keyword for topic in topics.values() for keyword in topic]
        top_themes = Counter(all_keywords).most_common()

        return topics, top_themes

    except Exception as e:
        raise Exception(f"Fehler bei der Themenextraktion: {e}")


def most_common_topics_with_summ(sentences_list, german_stopwords, num_clusters=None, min_sentences_per_cluster=3):
    """
    Extracts common topics from a list of German sentences using clustering.

    Args:
        sentences_list (list): List of German sentences
        german_stopwords (list): List of stop words
        num_clusters (int, optional): Number of clusters. If None, it is determined automatically
        min_sentences_per_cluster (int): Minimum number of sentences per cluster

    Returns:
        dict: Cluster IDs with their keywords
        list: List of summary headings (top_themes) for each cluster
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from transformers import pipeline
    import numpy as np
    # Input Validation
    if not sentences_list or not isinstance(sentences_list, list):
        raise ValueError("Input must be a non-empty list of sentences")

    # Remove empty sentences and duplicate entries
    sentences_list = list(filter(None, set(sentences_list)))

    # Automatically determine the number of clusters
    if num_clusters is None:
        num_clusters = max(2, min(len(sentences_list) // min_sentences_per_cluster, 15))

    try:
        # Vectorization
        model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')

        # Batch processing for large lists
        batch_size = 32
        sentence_embeddings = []
        for i in range(0, len(sentences_list), batch_size):
            batch = sentences_list[i:i + batch_size]
            embeddings = model.encode(batch)
            sentence_embeddings.append(embeddings)

        sentence_embeddings = np.vstack(sentence_embeddings)

        # Clustering
        clustering_model = KMeans(n_clusters=num_clusters,
                                  random_state=42,
                                  n_init=10)
        cluster_assignments = clustering_model.fit_predict(sentence_embeddings)

        # Group by clusters
        clustered_sentences = {i: [] for i in range(num_clusters)}
        for sentence, cluster in zip(sentences_list, cluster_assignments):
            clustered_sentences[cluster].append(sentence)

        # Extended Stop Words
        custom_german_stop_words = german_stopwords

        # Keyword Extraction with Error Handling
        def extract_keywords(sentences):
            if not sentences:
                return []
            try:
                vectorizer = CountVectorizer(max_features=5,
                                             stop_words=custom_german_stop_words,
                                             min_df=2)  # At least 2 occurrences
                X = vectorizer.fit_transform(sentences)
                return vectorizer.get_feature_names_out()
            except Exception as e:
                print(f"Error during keyword extraction: {e}")
                return []

        # Extract topics
        topics = {}
        for cluster_id, cluster_sentences in clustered_sentences.items():
            if len(cluster_sentences) >= min_sentences_per_cluster:
                topics[cluster_id] = extract_keywords(cluster_sentences)

        # Load summarization pipeline
        summarizer = pipeline('summarization', model='google/pegasus-multi-xsum')

        # Generate summary headings for each cluster
        headings = {}
        for cluster_id, cluster_sentences in clustered_sentences.items():
            if len(cluster_sentences) >= min_sentences_per_cluster:
                text = ' '.join(cluster_sentences)
                # Adjust max_length and min_length as needed
                summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
                heading = summary[0]['summary_text']
                headings[cluster_id] = heading

        # Sort clusters by number of sentences descending
        sorted_clusters = sorted(clustered_sentences.keys(), key=lambda x: len(clustered_sentences[x]), reverse=True)

        # Collect top_themes in order of sorted clusters
        top_themes = [headings[cluster_id] for cluster_id in sorted_clusters if cluster_id in headings]

        return topics, top_themes

    except Exception as e:
        raise Exception(f"Error during topic extraction: {e}")




def translate_sentences(sentences_list, target_language="en"):
    from easynmt import EasyNMT
    model = EasyNMT('opus-mt')

    translated_sentences = []
    for sentence in sentences_list:
        translation = model.translate(sentence, target_lang=target_language)
        translated_sentences.append(translation)

    return translated_sentences

from openai import OpenAI
import time

def classify_statements_as_claim(sentences, openrouter_api_key, batch_size=10):
    """
    Klassifiziert eine Liste von Sätzen als Behauptung oder keine Behauptung mithilfe der LLM API.

    Args:
        sentences (list of str): Die Liste der zu klassifizierenden Sätze.
        openrouter_api_key (str): Der API-Schlüssel für OpenRouter.
        batch_size (int): Die Anzahl der Sätze, die pro Anfrage gesendet werden sollen.

    Returns:
        dict: Ein Dictionary mit den Sätzen als Schlüssel und ihren Klassifikationen als Werte.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    results = {}
    for i in range(0, len(sentences), batch_size):
        print(f"Processing batch {i // batch_size + 1}...")
        batch = sentences[i:i + batch_size]
        # Formulieren Sie den Prompt für die aktuelle Batch.
        prompt_2 = (
            "Classify the following statements. Return each sentence followed by a \"#\" and the classification \"Claim\" or \"No Claim\". " +
            "Definition of \"Claim\": " +
            "An claim is a statement in sentence form that presents a fact or situation as given or true. " +
            "At the same time, it implies that the speaker is convinced of the truth of this fact. " +
            "Definition of \"Not a Claim\": " +
            "A statement that is not an claim includes, for example: " +
            "- A question " +
            "- An expression of opinion without a clear statement of fact " +
            "- A condition (e.g., if-then sentences) " +
            "- An exclamation or a command " +
            "Examples to distinguish: " +
            "1. Claim: " +
            "- \"The Earth is a sphere.\" " +
            "- \"Climate change is caused by human activity.\" " +
            "2. Not a Claim: " +
            "- \"Is the Earth a sphere?\" " +
            "- \"I think it’s cold today.\" " +
            "- \"If it rains tomorrow, we’ll stay home.\" " +
            "- \"Look out the window!\" " +
            "Statements to classify: " +
            '\n'.join([f'{i + idx + 1}. {sentence}' for idx, sentence in enumerate(batch)]) + '\n\n' +
            "Output format: " +
            "Return the output in the following format: \"Sentence # Classification\". No other text or whitespaces!"
        )


        try:
            # Anfrage an die API
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "<YOUR_SITE_URL>",
                    "X-Title": "<YOUR_SITE_NAME>",
                },
                model="google/gemma-2-9b-it:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_2,
                    }
                ]
            )
            # API-Antwort verarbeiten
            response = completion.choices[0].message.content.strip().split("\n")

            # Antwort korrekt zuordnen
            current_classifications = []
            for line in response:
                if "#" in line:
                    sentence, classification = map(str.strip, line.split("#", 1))
                    # cleaned_sentence = sentence.lstrip("1234567890. ").strip()
                    current_classifications.append(classification)
                else:
                    print(f"Fehlerhafte Zeile: {line}")  # Debugging
            for idx, sentence in enumerate(batch):
                results[sentence] = current_classifications[idx]

        except Exception as e:
            print(f"Fehler bei der Verarbeitung der Batch {i // batch_size + 1}: {e}")
            for sentence in batch:
                results[sentence] = "Fehler"

        # Optional: Zeitverzögerung, um API-Rate-Limits zu vermeiden
        time.sleep(1)

    return results


def write_to_json(file_path, data):
    """
    Writes a list of strings to a JSON file.

    :param file_path: Path to the JSON file.
    :param data: List of strings to write.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(data, file)
        print(f"List successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to {file_path}: {e}")


def read_from_json(file_path):
    """
    Reads a list of strings from a JSON file.

    :param file_path: Path to the JSON file.
    :return: List of strings read from the file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        print(f"List successfully read from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while reading from {file_path}: {e}")
        return None


def calculate_classification_percentages(data):
    """
    Calculate the percentage of each classification in the given data.

    :param data: List of lists where each sublist contains a statement and a classification.
    :return: Dictionary with classifications as keys and their percentages as values.
    """
    from collections import Counter

    # Extract the classifications from the data
    classifications = [entry[1] for entry in data]

    # Count the occurrences of each classification
    classification_counts = Counter(classifications)

    # Calculate the total number of classifications
    total_count = sum(classification_counts.values())

    # Calculate the percentage of each classification
    percentages = {key: (value / total_count) * 100 for key, value in classification_counts.items()}

    return percentages


if __name__ == "__main__":
    print(calculate_classification_percentages([('Klimawandel verursacht durch menschliche Aktivitäten', True), ('Situation in Deutschland', False)]))
