#!/usr/bin/env python3
"""
Usage example for the saved qdecompose_model.joblib

This script demonstrates how to load and use the trained query complexity model
for making predictions on new queries to determine if they are simple or complex.
"""

import time
import joblib
import spacy
from spacy.language import Language
from spacy.cli.download import download as spacy_download
import numpy as np
import torch
from typing import NamedTuple, List, TypedDict
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

from tools.logging_utils import get_logger


class PredictionResult(NamedTuple):
    """Query complexity prediction result."""
    query: str
    is_simple_prob: float
    is_simple: bool
    confidence: float
    infer_time: float


class QueryComplexityModel(TypedDict):
    """Loaded model components."""
    log_reg_model: LogisticRegression
    model_key: str
    tokenizer_key: str
    embedding_dim: int
    POS_LIST: List[str]
    coefficients: np.ndarray
    """For binary classification, it's typically a 1D array"""
    intercept: float


class QueryComplexity:
    """Class to load and use the saved query complexity model."""

    def __init__(self, model_path: str = 'models/qdecompose_model.joblib'):
        """
        Args:
            model_path (str): Path to the saved joblib model file
        """
        self.model_path = model_path
        self.model_data: QueryComplexityModel | None = None
        self.log_reg_model: LogisticRegression
        self.tokenizer: AutoTokenizer
        self.model: AutoModel
        self.nlp: Language
        self.embedding_dim: int = 0
        self.POS_LIST: List[str] = []
        self.logger = get_logger('QueryComplexity')

        self._load_model()
        self._setup_spacy()

    def _load_model(self) -> None:
        """Load the saved model and components."""
        print(f"Loading model from: {self.model_path}")

        self.model_data = joblib.load(self.model_path)
        if not self.model_data:
            raise ValueError(
                f"Failed to load model data from {self.model_path}")

        # Extract components
        self.log_reg_model = self.model_data['log_reg_model']
        self.embedding_dim = self.model_data['embedding_dim']
        self.POS_LIST = self.model_data['POS_LIST']
        model_key = self.model_data['model_key']
        tokenizer_key = self.model_data['tokenizer_key']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_key)
        self.model = AutoModel.from_pretrained(model_key)

        self.logger.info(f"Query complexity model loaded",
                         model_path=self.model_path, model_data=self.model_data)

    def _setup_spacy(self, is_2nd_try=False) -> None:
        """Setup spaCy NLP pipeline."""
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            print("spaCy model loaded successfully!")
        except OSError:
            if is_2nd_try:
                raise OSError(
                    "spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            spacy_download("en_core_web_sm")
            self._setup_spacy(is_2nd_try=True)

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Generate BERT embedding for the given text.

        Args:
            text (str): Input text to embed

        Returns:
            numpy.ndarray: BERT embedding vector
        """
        inputs = self.tokenizer(text,  # type: ignore
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)  # type: ignore
        # Extract the [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return cls_embedding.numpy()

    def _extract_pos_features(self, text: str) -> np.ndarray:
        """
        Extract POS tag features from text.

        Args:
            text (str): Input text

        Returns:
            numpy.ndarray: Combined BERT + POS features
        """
        # Get BERT embedding
        bert_embedding = self._get_bert_embedding(text)

        # Initialize feature vector with POS tag counts
        feature_vector = np.concatenate(
            (bert_embedding, np.zeros(len(self.POS_LIST))))

        # Count POS tags
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ in self.POS_LIST:
                pos_idx = self.POS_LIST.index(token.pos_)
                feature_vector[self.embedding_dim + pos_idx] += 1

        return feature_vector

    def predict(self, query: str) -> PredictionResult:
        """
        Predict if a single query is simple or complex.
        
        Returns True for simple queries (no decomposition needed),
        Returns False for complex queries (decomposition beneficial).

        Args:
            query (str): The query to analyze

        Returns:
            PredictionResult: Prediction results including probability and binary prediction
        """
        start_time = time.perf_counter()
        # Extract features
        features = self._extract_pos_features(query)
        # features is 1D, like (800,), for a single query,
        #   however sklearn expects 2D input, so we reshape it to (1, 800), ONE row, 800 columns
        features_2d = features.reshape(1, -1)

        # Make prediction
        probability = self.log_reg_model.predict_proba(features_2d)[0]
        binary_prediction = self.log_reg_model.predict(features_2d)[0]

        return PredictionResult(
            query=query,
            is_simple_prob=round(probability[1], 4),
            is_simple=bool(binary_prediction),
            confidence=float(max(probability)),
            infer_time=time.perf_counter() - start_time,
        )


def main():
    """Example usage of the QueryComplexity classifier."""

    # Initialize the model
    try:
        complexity_classifier = QueryComplexity()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print("\n1. Single Query Complexity Prediction:")
    sample_queries = [
        "how the american judicial system works", # from Lida's output, Yes
        "which german state benefited most from the territorial changes made by the congress of vienna", # No
        "the first one to see the ghost of king hamlet is", # No
        "how does watching the news affect your health", # Yes
        "how does population density affect life", # Yes
        "who dies in season 6 of sons of anarchy", # No
        "I'm hoping to understand the arguments for and against legalizing euthanasia, including how it differs from physician-assisted suicide. I'm also interested in how cultural values and Western countries' perspectives influence these debates, and the moral justifications from both sides.", # TREC RAG 2025, complex query
        "I want to understand why deforestation is such a major problem. Specifically, how does it impact the environment, climate, animals, and humans? Could you also explain its main causes, effects on rainforests like the Amazon, and what actions can prevent it?", # another complex query
        "What are the main faktors that contribute to the US dollar's role as the dominant reserve currancy in international trade?", # LiveRAG, another long but not certain complex query, ambiguous
        "where did choan seng song get phd", # LiveRAG, short and simple
    ]

    for query in sample_queries:
        result = complexity_classifier.predict(query)
        print(f"Query: '{result.query}'")
        print(f"  Is simple: {result.is_simple}")
        print(f"  Simple probability: {result.is_simple_prob}")
        print(f"  Confidence: {result.confidence:.4f}")
        # print(f"  Inference Time: {result.infer_time:.4f} seconds")
        print()


if __name__ == "__main__":
    main()
