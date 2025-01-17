# PoliticAnalyze

**PoliticAnalyze** is a Python project designed to analyze political documents in PDF format. It preprocesses the text data, performs various analyses including lemma frequency calculation, sentiment analysis, claim extraction, and fact-checking, and saves the results in JSON format for further use.

## Project Workflow

1. **Reading and Preprocessing PDFs:**
   - Reads all PDF files from the `data` directory.
   - Extracts text from each PDF and preprocesses it using a German NLP pipeline.
   - Saves the preprocessed data in a dictionary format to `full_data_dict.json`.

2. **Analyzing Lemma Frequencies:**
   - Calculates the frequency of lemmas in the preprocessed text.
   - Stores the lemma frequency data in `full_results_dict.json`.

3. **Analyzing German Sentiments:**
   - Analyzes the sentiment of the text data.
   - Adds sentiment percentages to `full_results_dict.json`.

4. **Extracting Claims:**
   - Extracts claims from the text sentences.
   - Stores the extracted claims in `full_results_dict.json`.

5. **Fact-Checking:**
   - Performs fact-checking on the extracted claims.
   - Calculates the percentage of claim classifications (e.g., true, false, neutral).
   - Saves the final results in `final_results_dict.json`.

## Directory Structure

- `data/`: Contains the input PDF files.
- `saved_on_harddrive/`: Stores the JSON output files:
  - `full_data_dict.json`: Preprocessed data dictionary.
  - `full_results_dict.json`: Intermediate analysis results.
  - `final_results_dict.json`: Final analysis results including fact-checking.

## Dependencies

- `spacy` for NLP processing.
- `pdfminer.six` for PDF text extraction.
- `sentiment_analysis_lib` for sentiment analysis.
- `claim_extraction_lib` for claim extraction.
- `fact_checking_lib` for fact-checking.
