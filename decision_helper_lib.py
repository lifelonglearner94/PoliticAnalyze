import json
import os
import spacy
from spacy.language import Language
import re
import PyPDF2
from collections import defaultdict, Counter

def generate_test_pdf():
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from lorem_text import lorem

    # Generate random German text with sentences ending in dots
    sentence_count = 15  # Determines how many sentences are generated
    german_text = " Die Sonne ist genau 15 Millionen Kilometer von der Erde entfernt.".join([lorem.sentence() for i in range(sentence_count)])

    # Create a PDF
    pdf_filename = os.path.join("data_test", "random_german_text.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    # Configure font and text
    c.setFont("Helvetica", 12)
    text_object = c.beginText(40, 750)  # Position (x, y) in points

    # Split text into lines to fit the page
    lines = []
    words = german_text.split()
    current_line = []
    max_width = 500  # Adjust based on page width

    for word in words:
        current_line.append(word)
        line_width = c.stringWidth(" ".join(current_line), "Helvetica", 12)
        if line_width > max_width:
            lines.append(" ".join(current_line[:-1]))
            current_line = [word]
    lines.append(" ".join(current_line))

    # Add lines to the PDF
    for line in lines:
        text_object.textLine(line)

    c.drawText(text_object)
    c.save()

    print(f"PDF created: {pdf_filename}")


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
        and token.text != "eichhorster"
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
    # classifier = pipeline('sentiment-analysis',
    #                     model='oliverguhr/german-sentiment-bert',
    #                     tokenizer='oliverguhr/german-sentiment-bert')

    classifier = pipeline('sentiment-analysis',
                        model='Commandante/german-party-sentiment-bert',
                        tokenizer='Commandante/german-party-sentiment-bert')

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


def calculate_classification_percentages_facticity_output(data):
    """
    Calculate the percentage of each classification in the given data.

    :param data: List of lists where each sublist contains a statement and a classification.
    :return: Dictionary with classifications as keys and their percentages as values.
    """
    from collections import Counter

    # Extract the classifications from the data
    # classifications = [entry[1] for entry in data]
    classifications = []

    for dict_ in data:
        classifications.append(dict_["Classification"])

    # Count the occurrences of each classification
    classification_counts = Counter(classifications)

    # Calculate the total number of classifications
    total_count = sum(classification_counts.values())

    # Calculate the percentage of each classification
    percentages = {key: (value / total_count) * 100 for key, value in classification_counts.items()}

    return percentages


def calculate_classification_percentages_zyla_output(fact_checks_data):
    def count_occurrences_in_structure(data, search_string):
        """
        Recursively traverses a nested structure of lists, dictionaries, and other values,
        counting the occurrences of the specified search string.

        :param data: The input structure (list, dict, str, etc.)
        :param search_string: The string to search for and count occurrences of.
        :return: The count of occurrences of the search string.
        """
        if data == search_string:
            return 1

        if isinstance(data, dict):
            # Recursively count in each value of the dictionary
            return sum(count_occurrences_in_structure(value, search_string) for value in data.values())

        if isinstance(data, list):
            # Recursively count in each element of the list
            return sum(count_occurrences_in_structure(item, search_string) for item in data)

        return 0

    true_count = count_occurrences_in_structure(fact_checks_data, "True")
    false_count = count_occurrences_in_structure(fact_checks_data, "False")
    partially_true_count = count_occurrences_in_structure(fact_checks_data, "Partially true")

    all_verifiable_facts = true_count + false_count + partially_true_count

    percentages_dict = {"True": true_count / all_verifiable_facts * 100,
                        "Partially true": partially_true_count / all_verifiable_facts * 100,
                        "False": false_count / all_verifiable_facts * 100}
    return percentages_dict

if __name__ == "__main__":
    generate_test_pdf()
