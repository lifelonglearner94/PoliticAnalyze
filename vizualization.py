import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_political_data(data):
    """
    Visualize fact-check percentages, lemma frequencies, and sentiments for political parties.

    Parameters:
        data (dict): Nested dictionary with political party data.
    """
    # Set up for visualization
    sns.set_theme(style="whitegrid")

    # FACT CHECK PERCENTAGES
    fact_check_df = pd.DataFrame({party: stats['fact_checks_percentage'] for party, stats in data.items()})
    fact_check_df = fact_check_df.reset_index().rename(columns={'index': 'Fact Check'}).melt(
        id_vars=['Fact Check'], var_name='Party', value_name='Percentage'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Party', y='Percentage', hue='Fact Check', data=fact_check_df, palette='pastel')
    plt.title('Fact Check Percentages by Party', fontsize=16)
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.legend(title='Fact Check')
    plt.tight_layout()
    plt.show()

    # LEMMAS - TOP OVERALL
    lemma_overall_df = pd.DataFrame({party: stats['lemmas_top_overall'] for party, stats in data.items()}).fillna(0)
    lemma_overall_df = lemma_overall_df.stack().reset_index()
    lemma_overall_df.columns = ['Lemma', 'Party', 'Frequency']

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Lemma', hue='Party', data=lemma_overall_df, palette='tab10')
    plt.title('Top Lemmas Overall by Party', fontsize=16)
    plt.xlabel('Frequency')
    plt.ylabel('Lemma')
    plt.tight_layout()
    plt.show()

    # SENTIMENTS
    sentiment_df = pd.DataFrame({party: stats['sentiments'] for party, stats in data.items()}).reset_index()
    sentiment_df = sentiment_df.rename(columns={'index': 'Sentiment'}).melt(
        id_vars=['Sentiment'], var_name='Party', value_name='Percentage'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Party', y='Percentage', hue='Sentiment', data=sentiment_df, palette='coolwarm')
    plt.title('Sentiment Distribution by Party', fontsize=16)
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

    # LEMMAS - TOP PER CATEGORY
    for category in ['ADJ', 'NOUN', 'VERB']:
        lemma_category_df = pd.DataFrame({party: stats['lemmas_top_per_category'][category] for party, stats in data.items()}).fillna(0)
        lemma_category_df = lemma_category_df.stack().reset_index()
        lemma_category_df.columns = ['Lemma', 'Party', 'Frequency']

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Frequency', y='Lemma', hue='Party', data=lemma_category_df, palette='viridis')
        plt.title(f'Top {category} Lemmas by Party', fontsize=16)
        plt.xlabel('Frequency')
        plt.ylabel('Lemma')
        plt.tight_layout()
        plt.show()
