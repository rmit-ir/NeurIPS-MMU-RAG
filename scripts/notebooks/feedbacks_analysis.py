# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "plotly",
#   "wordcloud",
#   "scikit-learn",
#   "nltk",
#   "textstat",
#   "openai",
#   "aiohttp",
#   "structlog",
#   "tqdm",
# ]
# ///

"""
Manual Feedbacks Analysis

Code written by LLM and not enough reviewed by humans. Use with caution.

Comprehensive analysis of manual feedbacks on different RAG systems including:
- Basic statistics (users, feedbacks, text coverage)
- Sentiment analysis
- System comparisons
- Query preference analysis
- Text analysis (word cloud, clustering)
- Actionable insights clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from openai.types.chat import ChatCompletionMessageParam
import re
import os
import sys
import asyncio

# Add the src directory to the path for imports
sys.path.append('./src')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def load_and_prepare_data():
    """Load and prepare the feedback data"""
    print("Loading feedback data...")

    # Load the data
    feedbacks_file = "./data/evaluation_results/manual_feedbacks_clean.csv"
    df = pd.read_csv(feedbacks_file)

    # Filter to only include specified search engines
    target_engines = ['mmu_rag_vanilla', 'decomposition_rag', 'mmu_rag_router_llm']
    df = df[df['search_engine'].isin(target_engines)]
    
    print(f"Dataset shape after filtering: {df.shape}")
    print(f"Filtered to search engines: {target_engines}")
    print(f"Column names: {list(df.columns)}")

    # Data overview and basic cleaning
    print("\nData Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nUnique values per column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")

    # Clean feedback_text - handle empty strings
    df['feedback_text'] = df['feedback_text'].fillna('')
    df['has_feedback_text'] = df['feedback_text'].str.strip() != ''

    print(
        f"\nFeedbacks with text: {df['has_feedback_text'].sum()} out of {len(df)} ({df['has_feedback_text'].mean():.1%})")

    return df


def analyze_basic_statistics(df):
    """Analyze basic statistics about users and feedbacks"""
    print("\n" + "="*60)
    print("BASIC STATISTICS ANALYSIS")
    print("="*60)

    # 1. How many users participated
    unique_users = df['user_id'].nunique()
    print(f"Number of unique users: {unique_users}")

    # 2. How many feedbacks each user left
    user_feedback_counts = df.groupby(
        'user_id').size().sort_values(ascending=False)
    print(f"\nFeedbacks per user:")
    print(f"Average: {user_feedback_counts.mean():.1f}")
    print(f"Median: {user_feedback_counts.median():.1f}")
    print(f"Min: {user_feedback_counts.min()}")
    print(f"Max: {user_feedback_counts.max()}")

    print(f"\nTop 10 most active users:")
    print(user_feedback_counts.head(10))

    # 3. Percentage of feedbacks with text
    text_percentage = df['has_feedback_text'].mean() * 100
    print(f"\nPercentage of feedbacks with text: {text_percentage:.1f}%")

    # 4. Overall sentiment distribution
    print(f"\nOverall feedback sentiment:")
    sentiment_counts = df['feedback_type'].value_counts()
    print(sentiment_counts)
    print(
        f"Positive (helpful): {sentiment_counts.get('helpful', 0)} ({sentiment_counts.get('helpful', 0)/len(df)*100:.1f}%)")
    print(
        f"Negative (not-helpful): {sentiment_counts.get('not-helpful', 0)} ({sentiment_counts.get('not-helpful', 0)/len(df)*100:.1f}%)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # User participation distribution
    axes[0, 0].hist(user_feedback_counts, bins=20,
                    edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Feedbacks per User')
    axes[0, 0].set_xlabel('Number of Feedbacks')
    axes[0, 0].set_ylabel('Number of Users')

    # Feedback type distribution
    sentiment_counts.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
    axes[0, 1].set_title('Overall Feedback Sentiment Distribution')
    axes[0, 1].set_ylabel('')

    # Search engine usage
    engine_counts = df['search_engine'].value_counts()
    engine_counts.plot(kind='barh', ax=axes[1, 0])
    axes[1, 0].set_title('Feedback Count by Search Engine')
    axes[1, 0].set_xlabel('Number of Feedbacks')

    # Text vs no-text feedbacks
    text_dist = df['has_feedback_text'].value_counts()
    text_dist.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', labels=[
                   'Has Text', 'No Text'])
    axes[1, 1].set_title('Feedbacks with/without Text')
    axes[1, 1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Downloads/feedback_basic_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return user_feedback_counts, sentiment_counts


def analyze_system_performance(df):
    """Analyze performance per system (search engine)"""
    print("\n" + "="*60)
    print("SYSTEM COMPARISON ANALYSIS")
    print("="*60)

    # Per system analysis
    system_analysis = df.groupby('search_engine').agg({
        'feedback_type': ['count', lambda x: (x == 'helpful').sum(), lambda x: (x == 'helpful').mean()],
        'has_feedback_text': 'mean',
        'user_id': 'nunique'
    })

    system_analysis.columns = [
        'Total_Feedbacks', 'Helpful_Count', 'Helpful_Rate', 'Text_Rate', 'Unique_Users']
    system_analysis = system_analysis.sort_values(
        'Helpful_Rate', ascending=False)

    print("System performance comparison:")
    print(system_analysis)

    # Create detailed system comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Helpful rate by system
    system_analysis['Helpful_Rate'].plot(
        kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Helpful Rate by Search Engine')
    axes[0, 0].set_ylabel('Helpful Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Total feedbacks by system
    system_analysis['Total_Feedbacks'].plot(
        kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Total Feedbacks by Search Engine')
    axes[0, 1].set_ylabel('Number of Feedbacks')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Text rate by system
    system_analysis['Text_Rate'].plot(
        kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Text Feedback Rate by Search Engine')
    axes[1, 0].set_ylabel('Rate of Feedbacks with Text')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Unique users by system
    system_analysis['Unique_Users'].plot(
        kind='bar', ax=axes[1, 1], color='gold')
    axes[1, 1].set_title('Unique Users by Search Engine')
    axes[1, 1].set_ylabel('Number of Unique Users')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Downloads/feedback_system_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return system_analysis


def analyze_query_preferences(df):
    """Analyze query preferences across different systems"""
    print("\n" + "="*60)
    print("QUERY PREFERENCE ANALYSIS")
    print("="*60)

    # Find queries that were tested on multiple systems
    query_systems = df.groupby(
        'query')['search_engine'].nunique().sort_values(ascending=False)
    multi_system_queries = query_systems[query_systems > 1]

    print(f"Queries tested on multiple systems: {len(multi_system_queries)}")
    print(f"Most tested queries:")
    print(multi_system_queries.head(10))

    # Analyze preference for queries with multiple systems
    preference_analysis = []
    for query in multi_system_queries.index:
        query_data = df[df['query'] == query]

        # Count helpful vs not-helpful for each system on this query
        query_summary = query_data.groupby(
            ['search_engine', 'feedback_type']).size().unstack(fill_value=0)
        if 'helpful' in query_summary.columns:
            query_summary['helpful_rate'] = query_summary['helpful'] / \
                (query_summary['helpful'] +
                 query_summary.get('not-helpful', 0))
        else:
            query_summary['helpful_rate'] = 0

        # Find the preferred system (highest helpful rate)
        if len(query_summary) > 0:
            preferred_system = query_summary['helpful_rate'].idxmax()
            max_rate = query_summary['helpful_rate'].max()

            preference_analysis.append({
                'query': query,
                'systems_tested': len(query_summary),
                'preferred_system': preferred_system,
                'preferred_rate': max_rate,
                'total_feedbacks': len(query_data)
            })

    preference_df = pd.DataFrame(preference_analysis)
    print(f"\nSystem preference summary:")
    if not preference_df.empty:
        system_preference_counts = preference_df['preferred_system'].value_counts(
        )
        print(system_preference_counts)

        # Show some examples
        print(f"\nExample queries with clear preferences:")
        high_preference = preference_df[preference_df['preferred_rate'] >= 0.8].head(
            5)
        for _, row in high_preference.iterrows():
            print(
                f"'{row['query'][:60]}...' -> {row['preferred_system']} ({row['preferred_rate']:.1%} helpful)")

    # Cross-user query analysis
    print("\n" + "="*60)
    print("CROSS-USER QUERY ANALYSIS")
    print("="*60)

    query_user_analysis = []
    for query in multi_system_queries.index:
        query_data = df[df['query'] == query]
        unique_users = query_data['user_id'].nunique()
        unique_systems = query_data['search_engine'].nunique()

        if unique_users > 1:  # Multiple users tested this query
            query_user_analysis.append({
                'query': query,
                'unique_users': unique_users,
                'unique_systems': unique_systems,
                'total_feedbacks': len(query_data),
                'avg_helpful_rate': (query_data['feedback_type'] == 'helpful').mean()
            })

    cross_user_df = pd.DataFrame(query_user_analysis)
    if not cross_user_df.empty:
        cross_user_df = cross_user_df.sort_values(
            'unique_users', ascending=False)
        print(f"Queries tested by multiple users: {len(cross_user_df)}")
        print(cross_user_df.head(10))
    else:
        print("No queries were tested by multiple users on different systems.")

    return preference_df, cross_user_df


def create_word_cloud(df):
    """Create word cloud from feedback text"""
    print("\n" + "="*60)
    print("TEXT ANALYSIS - WORD CLOUD")
    print("="*60)

    # Prepare text for analysis
    text_feedbacks = df[df['has_feedback_text']].copy()
    print(f"Analyzing {len(text_feedbacks)} feedbacks with text")

    # Combine all feedback text
    all_text = ' '.join(text_feedbacks['feedback_text'].astype(str))

    # Clean text for word cloud
    def clean_text_for_wordcloud(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    clean_text = clean_text_for_wordcloud(all_text)

    # Create word cloud
    stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'a', 'an', 'as', 'if', 'then', 'than', 'so', 'very', 'can', 'just', 'not',
        'answer', 'question', 'system', 'response', 'good', 'better', 'best'
    }

    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=stopwords,
                          max_words=100,
                          colormap='viridis').generate(clean_text)

    # Display word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Feedback Text', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/Downloads/feedback_wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Most common words
    words = clean_text.split()
    filtered_words = [word for word in words if len(
        word) > 3 and word not in stopwords]
    word_freq = Counter(filtered_words)
    print(f"\nMost common words in feedback:")
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")

    return word_freq


async def cluster_feedback_llm(df):
    """Use LLM to help identify common feedback themes"""
    print("\n" + "="*60)
    print("FEEDBACK CLUSTERING FOR ACTIONABLE INSIGHTS (LLM-BASED)")
    print("="*60)

    try:
        from tools.llm_servers.general_openai_client import GeneralOpenAIClient

        client = GeneralOpenAIClient(
            api_key=os.environ.get("MMU_OPENAI_API_KEY", ""),
            api_base="https://mmu-proxy-server-llm-proxy.rankun.org/v1",
            model_id="qwen.qwen3-235b-a22b-2507-v1:0"
        )

        # Prepare sample feedback text for analysis
        text_feedbacks = df[df['has_feedback_text']].copy()
        sample_feedbacks = text_feedbacks['feedback_text'].tolist()[
            :100]  # Use first 100 for analysis
        feedback_sample = '\n\n---\n\n'.join(
            [f"Feedback {i+1}: {fb}" for i, fb in enumerate(sample_feedbacks)])

        clustering_prompt = f"""
Analyze the following user feedback on RAG (Retrieval-Augmented Generation) systems and identify distinct categories of issues that users commonly mention. Each category should represent an actionable improvement area.

Based on manual inspection, some known categories include:
- Imbalanced viewpoint / bias issues
- Citation and referencing problems
- Response length issues (too long/verbose)
- Relevance and accuracy problems
- Speed/performance issues
- Source quality concerns

Please analyze all the feedback below and:
1. Identify ALL distinct categories of issues (including but not limited to the ones mentioned above)
2. For each category, provide:
   - Category name
   - Brief description
   - 2-3 example quotes from the feedback
   - Suggested actionable improvement
3. Count how many feedbacks mention each type of issue

Feedback data:
{feedback_sample}

Please format your response as a structured analysis.
"""

        print("Analyzing feedback themes using LLM...")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are an expert data analyst specializing in user feedback analysis and categorization."},
            {"role": "user", "content": clustering_prompt}
        ]
        
        # Add progress bar for LLM request
        response, _ = await client.complete_chat(messages)
        
        print("\n=== LLM FEEDBACK CATEGORIES ANALYSIS ===")
        print(response)

    except Exception as e:
        print(f"Error analyzing with LLM: {e}")
        print("Continuing with rule-based analysis...")


def cluster_feedback_rules(df):
    """Rule-based feedback categorization as backup"""
    print("\n" + "="*60)
    print("RULE-BASED FEEDBACK CATEGORIZATION")
    print("="*60)

    # Define categories and keywords
    categories = {
        'Citation/References': ['citation', 'reference', 'cited', 'citing', 'source', 'sources', 'link', 'links'],
        'Length/Verbosity': ['long', 'verbose', 'short', 'lengthy', 'brief', 'concise', 'too much'],
        'Bias/Balance': ['bias', 'balanced', 'imbalanced', 'one-sided', 'perspective', 'viewpoint'],
        'Accuracy/Relevance': ['wrong', 'incorrect', 'accurate', 'relevant', 'irrelevant', 'hallucination'],
        'Speed/Performance': ['slow', 'fast', 'delay', 'speed', 'time', 'took ages', 'quick'],
        'Quality/Content': ['quality', 'good', 'better', 'comprehensive', 'detailed', 'thorough']
    }

    # Categorize feedbacks
    text_feedbacks = df[df['has_feedback_text']].copy()
    feedback_categories = {cat: [] for cat in categories.keys()}
    feedback_categories['Other'] = []

    for idx, feedback in text_feedbacks.iterrows():
        text = feedback['feedback_text'].lower()
        found_category = False

        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                feedback_categories[category].append({
                    'text': feedback['feedback_text'],
                    'system': feedback['search_engine'],
                    'sentiment': feedback['feedback_type']
                })
                found_category = True
                break

        if not found_category:
            feedback_categories['Other'].append({
                'text': feedback['feedback_text'],
                'system': feedback['search_engine'],
                'sentiment': feedback['feedback_type']
            })

    # Print category analysis
    print("Feedback categorization results:")
    for category, feedbacks in feedback_categories.items():
        count = len(feedbacks)
        if count > 0:
            helpful_count = sum(
                1 for fb in feedbacks if fb['sentiment'] == 'helpful')
            helpful_rate = helpful_count / count if count > 0 else 0
            print(
                f"\n{category}: {count} feedbacks (helpful rate: {helpful_rate:.1%})")

            # Show ALL examples
            examples = feedbacks  # Show all examples
            for i, example in enumerate(examples, 1):
                print(f"  Example {i}: '{example['text'][:100]}...'")

    return feedback_categories


def advanced_clustering(df):
    """Advanced clustering using TF-IDF and machine learning"""
    print("\n" + "="*60)
    print("ADVANCED FEEDBACK CLUSTERING")
    print("="*60)

    text_feedbacks = df[df['has_feedback_text']].copy()

    if len(text_feedbacks) < 10:
        print("Not enough text feedbacks for clustering analysis")
        return None

    # Prepare text data
    texts = text_feedbacks['feedback_text'].tolist()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Determine optimal number of clusters (3-8 range)
        n_clusters = min(8, max(3, len(texts) // 20))

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Add cluster labels to dataframe
        text_feedbacks_copy = text_feedbacks.copy()
        text_feedbacks_copy['cluster'] = cluster_labels

        # Analyze clusters
        print(f"Created {n_clusters} clusters:")
        for i in range(n_clusters):
            cluster_feedbacks = text_feedbacks_copy[text_feedbacks_copy['cluster'] == i]
            helpful_rate = (
                cluster_feedbacks['feedback_type'] == 'helpful').mean()

            print(
                f"\nCluster {i}: {len(cluster_feedbacks)} feedbacks (helpful rate: {helpful_rate:.1%})")

            # Show example feedbacks from this cluster
            examples = cluster_feedbacks['feedback_text'].head(3).tolist()
            for j, example in enumerate(examples, 1):
                print(f"  Example {j}: '{example[:80]}...'")

        # Feature words for each cluster
        feature_names = vectorizer.get_feature_names_out()
        print(f"\nTop terms per cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        for i in range(n_clusters):
            print(f"Cluster {i}: ", end="")
            top_terms = [str(feature_names[ind])
                         for ind in order_centroids[i, :5]]
            print(", ".join(top_terms))

        return text_feedbacks_copy, kmeans, vectorizer

    except Exception as e:
        print(f"Error in clustering: {e}")
        return None


def generate_summary_report(df, user_feedback_counts, sentiment_counts, system_analysis,
                            preference_df, cross_user_df, word_freq, feedback_categories):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEEDBACK ANALYSIS SUMMARY")
    print("="*80)

    # Key metrics summary
    print(f"\n📊 KEY METRICS:")
    print(f"- Total feedbacks: {len(df)}")
    print(f"- Unique users: {df['user_id'].nunique()}")
    print(
        f"- Feedbacks with text: {df['has_feedback_text'].sum()} ({df['has_feedback_text'].mean():.1%})")
    print(
        f"- Overall helpful rate: {(df['feedback_type'] == 'helpful').mean():.1%}")
    print(f"- Most active user gave {user_feedback_counts.max()} feedbacks")
    print(f"- Average feedbacks per user: {user_feedback_counts.mean():.1f}")

    # System performance summary
    print(f"\n🔍 SYSTEM PERFORMANCE RANKING:")
    for i, (system, data) in enumerate(system_analysis.iterrows(), 1):
        print(
            f"{i}. {system}: {data['Helpful_Rate']:.1%} helpful rate ({data['Total_Feedbacks']} feedbacks)")

    # Most tested queries
    if not preference_df.empty:
        print(f"\n❓ MOST TESTED QUERIES:")
        for _, row in preference_df.head(5).iterrows():
            print(
                f"- '{row['query'][:50]}...' (tested on {row['systems_tested']} systems)")

    # Category insights
    print(f"\n📋 TOP FEEDBACK CATEGORIES:")
    sorted_categories = sorted([(cat, len(feedbacks)) for cat, feedbacks in feedback_categories.items()],
                               key=lambda x: x[1], reverse=True)
    for cat, count in sorted_categories[:5]:
        if count > 0:
            helpful_rate = sum(
                1 for fb in feedback_categories[cat] if fb['sentiment'] == 'helpful') / count
            print(f"- {cat}: {count} mentions (helpful rate: {helpful_rate:.1%})")

    # Most common words
    print(f"\n🔤 TOP FEEDBACK TERMS:")
    for word, count in word_freq.most_common(10):
        print(f"- {word}: {count} occurrences")

    print(f"\n🎯 ACTIONABLE RECOMMENDATIONS:")
    print("1. **Citation Issues**: Improve in-text citation formatting and source attribution")
    print("2. **Response Length**: Implement adaptive response length based on query complexity")
    print("3. **Bias Detection**: Add bias detection and balanced perspective mechanisms")
    print("4. **Speed Optimization**: Focus on systems with slow performance complaints")
    print("5. **Source Quality**: Improve source filtering and quality assessment")
    print("6. **User Interface**: Better presentation of multi-dimensional answers")

    print(f"\n📈 NEXT STEPS:")
    print("- Implement fixes for top 3 categories with highest complaint rates")
    print("- A/B test improved citation formatting")
    print("- Develop adaptive response length algorithms")
    print("- Create bias detection metrics and alerts")
    print("- Regular monitoring of feedback categories for trends")


async def main():
    """Main function to run all analyses"""
    print("Starting comprehensive feedback analysis...")

    # Load and prepare data
    df = load_and_prepare_data()

    # Run analyses
    user_feedback_counts, sentiment_counts = analyze_basic_statistics(df)
    system_analysis = analyze_system_performance(df)
    preference_df, cross_user_df = analyze_query_preferences(df)
    word_freq = create_word_cloud(df)

    # Clustering analyses
    await cluster_feedback_llm(df)
    feedback_categories = cluster_feedback_rules(df)
    advanced_clustering(df)

    # Generate summary report
    generate_summary_report(df, user_feedback_counts, sentiment_counts, system_analysis,
                            preference_df, cross_user_df, word_freq, feedback_categories)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("Generated files saved to ~/Downloads/:")
    print("- feedback_basic_statistics.png")
    print("- feedback_system_performance.png") 
    print("- feedback_wordcloud.png")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
