# streamlit_app.py

# Include ALL necessary imports except those we'll import lazily
import os
import json
import time
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
# import sentence_transformers # Import this lazily
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import re
from dotenv import load_dotenv
from dataclasses import dataclass
from google.colab import drive # Still needed for mounting drive within the streamlit process
# import torch # Keep this if other parts of our code use torch globally

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the Java Spec Agent"""
    project_path: str = '/content/drive/My Drive/SpecSentinel'
    embedding_model: str = 'all-MiniLM-L6-v2'
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_contexts: int = 5
    similarity_threshold: float = 0.3
    max_tokens_per_call: int = 1000
    temperature: float = 0.7

class JavaSpecAgent:
    def __init__(self, config: AgentConfig, openrouter_api_key: str):
        self.config = config
        self.api_key = openrouter_api_key
        self.embedding_model = None # Initialize as None
        self.knowledge_base = {}
        self.embeddings_cache = {}
        self.conversation_history = []

        # Initialize components in a logical order
        self._setup_environment()
        self._load_knowledge_base() # Load data first
        self._load_embedding_model() # Then load model (depends on torch/transformers)
        self._build_vector_index() # Then build index (depends on model and data)

    def _setup_environment(self):
        """Setup Google Drive and project environment"""
        try:
            # Drive mount might behave differently when run via streamlit,
            # but keeping it here is generally safe. It might just indicate
            # already mounted.
            # Add a timeout to prevent blocking if drive mount hangs
            import threading
            mount_success = threading.Event()

            def mount_drive():
                try:
                    drive.mount('/content/drive', timeout=10) # Add timeout
                    mount_success.set()
                    print("✅ Google Drive mounted successfully")
                except Exception as e:
                    print(f"⚠️ Google Drive mounting failed or timed out: {e}")

            mount_thread = threading.Thread(target=mount_drive)
            mount_thread.start()
            mount_thread.join(timeout=15) # Wait a bit for mounting

            if not mount_success.is_set():
                 print("⚠️ Drive mount did not complete within timeout.")

        except Exception as e:
            # Catching errors with threading setup itself
            print(f"❌ Error during drive mount setup: {e}")
            # Attempt mounting directly as a fallback
            try:
                 drive.mount('/content/drive')
                 print("✅ Google Drive mounted successfully (direct fallback)")
            except Exception as e_fallback:
                 print(f"⚠️ Google Drive direct mount fallback failed: {e_fallback}")

        if not os.path.exists(self.config.project_path):
            try:
                os.makedirs(self.config.project_path, exist_ok=True)
                print(f"✅ Created project path: {self.config.project_path}")
            except Exception as e:
                 print(f"❌ Error creating project path: {e}")

    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings with fallback options"""
        print("📦 Loading embedding model...")
        
        try:
            # IMPORT SentenceTransformer *inside* the method
            from sentence_transformers import SentenceTransformer
            
            # Primary model loading
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            print("✅ Embedding model loaded successfully")
            return
            
        except ImportError as e:
            print(f"❌ SentenceTransformers not installed: {e}")
            print("💡 Install with: pip install sentence-transformers")
            
        except Exception as e:
            print(f"❌ Error loading primary embedding model '{self.config.embedding_model}': {e}")
            
            # Try fallback models
            fallback_models = [
                "all-MiniLM-L6-v2",
                "paraphrase-MiniLM-L6-v2",
                "all-mpnet-base-v2"
            ]
            
            print("🔄 Attempting to load fallback models...")
            
            for fallback_model in fallback_models:
                if fallback_model != getattr(self.config, 'embedding_model', ''):
                    try:
                        from sentence_transformers import SentenceTransformer
                        self.embedding_model = SentenceTransformer(fallback_model)
                        print(f"✅ Fallback model '{fallback_model}' loaded successfully")
                        
                        # Update config to remember the working model
                        self.config.embedding_model = fallback_model
                        return
                        
                    except Exception as fallback_error:
                        print(f"❌ Fallback model '{fallback_model}' failed: {fallback_error}")
                        continue
            
            # All models failed - implement final fallback strategy
            print("⚠️ All embedding models failed to load")
            self._handle_embedding_failure()

    def _handle_embedding_failure(self):
        """Handle the case where no embedding model could be loaded"""
        self.embedding_model = None
        
        # Option 1: Use a simple text-based similarity fallback
        print("🔄 Falling back to basic text similarity methods")
        self.use_basic_similarity = True
        
        # Option 2: Disable embedding-dependent features
        self.embedding_features_disabled = True
        print("⚠️ Embedding-dependent features have been disabled")
        
        # Option 3: Raise an error if embeddings are critical
        if getattr(self.config, 'embeddings_required', False):
            raise RuntimeError(
                "Embedding model is required but could not be loaded. "
                "Please check your configuration or install required dependencies."
            )

    def get_embeddings(self, texts):
        """Safe method to get embeddings with fallback handling"""
        if self.embedding_model is None:
            if hasattr(self, 'use_basic_similarity') and self.use_basic_similarity:
                return self._basic_text_similarity(texts)
            else:
                raise RuntimeError("No embedding model available")
        
        try:
            return self.embedding_model.encode(texts)
        except Exception as e:
            print(f"❌ Error during embedding generation: {e}")
            if hasattr(self, 'use_basic_similarity') and self.use_basic_similarity:
                return self._basic_text_similarity(texts)
            raise

    def _basic_text_similarity(self, texts):
        """Basic text similarity fallback using simple metrics"""
        import re
        from collections import Counter
        
        def text_to_vector(text):
            # Simple word frequency vector
            words = re.findall(r'\w+', text.lower())
            return Counter(words)
        
        if isinstance(texts, str):
            texts = [texts]
        
        vectors = []
        for text in texts:
            vector = text_to_vector(text)
            # Convert to a simple numeric representation
            vector_array = [sum(ord(c) for c in word) * count for word, count in vector.items()]
            if not vector_array:
                vector_array = [0] * 100  # Default vector
            # Normalize to fixed size
            vector_array = (vector_array * (100 // len(vector_array) + 1))[:100]
            vectors.append(vector_array)
        
        return vectors

    # Configuration helper
    def validate_embedding_config(config):
        """Validate embedding model configuration"""
        if not hasattr(config, 'embedding_model'):
            config.embedding_model = 'all-MiniLM-L6-v2'  # Default model
            print("⚠️ No embedding model specified, using default: all-MiniLM-L6-v2")
        
        # Set reasonable defaults
        if not hasattr(config, 'embeddings_required'):
            config.embeddings_required = False
        
        return config
    
    def _load_knowledge_base(self):
        """Load existing knowledge base from SpecSentinel results"""
        kb_files = {
            'specifications': f'{self.config.project_path}/data/all_specifications.json',
            'processed_rules': f'{self.config.project_path}/data/processed_rules.json',
            'conflict_report': f'{self.config.project_path}/results/conflict_report.json',
            'knowledge_base': f'{self.config.project_path}/agent_knowledge_base.json',
            'conflicts': f'{self.config.project_path}/results/all_conflicts.json',
            'summary': f'{self.config.project_path}/results/final_summary.json'
        }

        print("📚 Loading knowledge base...")
        # Initialize knowledge_base to ensure it's always a dictionary
        self.knowledge_base = {}
        for key, file_path in kb_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.knowledge_base[key] = json.load(f)
                    print(f"   ✅ Loaded {key} from {file_path}")
                except json.JSONDecodeError as e:
                     self.knowledge_base[key] = {}
                     print(f"   ❌ JSONDecodeError loading {key} from {file_path}: {e}")
                except Exception as e:
                    self.knowledge_base[key] = {}
                    print(f"   ❌ Error loading {key} from {file_path}: {e}")
            else:
                self.knowledge_base[key] = {}
                print(f"   ⚠️ {key} file not found at {file_path}")

        # Ensure essential keys exist, even if empty
        self.knowledge_base.setdefault('processed_rules', [])
        self.knowledge_base.setdefault('conflicts', [])
        self.knowledge_base.setdefault('conflict_report', {'detailed_conflicts': []})
        self.knowledge_base.setdefault('specifications', {})
        self.knowledge_base.setdefault('knowledge_base', {})
        self.knowledge_base.setdefault('summary', {})

        print(f"✅ Knowledge base loaded with {len(self.knowledge_base)} components")

    def _build_vector_index(self):
        """Build vector index for RAG retrieval"""
        print("🔍 Building vector index...")

        # Check if embedding model was successfully loaded
        if self.embedding_model is None:
             print("   ⚠️ Embedding model not loaded, skipping vector index build.")
             self.embeddings_cache = {} # Ensure cache is empty
             return

        # Check for cached embeddings
        embeddings_file = f'{self.config.project_path}/embeddings_cache.pkl'

        if os.path.exists(embeddings_file):
            print("   Loading cached embeddings...")
            try:
                with open(embeddings_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Basic validation of cached data structure
                    if 'embeddings' in cached_data and 'documents' in cached_data:
                        self.embeddings_cache = cached_data
                        print("   ✅ Cached embeddings loaded")
                        
                        # Check if the loaded documents match the current knowledge base
                        current_documents = self._prepare_documents()
                        if self._documents_match(cached_data['documents'], current_documents):
                            print("   ✅ Cached documents match current knowledge base")
                            return
                        else:
                            print("   ⚠️ Cached documents don't match current knowledge base. Rebuilding index.")
                    else:
                        print("   ⚠️ Cached embeddings structure is invalid. Rebuilding index.")

            except Exception as e:
                print(f"   ❌ Error loading cached embeddings: {e}. Rebuilding index.")

        # Build new embeddings
        documents = self._prepare_documents()

        if not documents:
            print("   ⚠️ No documents to index")
            self.embeddings_cache = {} # Ensure cache is empty
            return

        # Create embeddings
        texts = [doc['text'] for doc in documents]
        try:
            # Ensure the model is on the correct device if applicable (often handled by SentenceTransformer)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True) # show_progress_bar=True might cause issues in background
            if embeddings is None:
                 raise ValueError("Embedding model returned None")

        except Exception as e:
             print(f"❌ Error encoding documents: {e}")
             self.embeddings_cache = {} # Ensure cache is empty
             return

        # Cache embeddings with metadata
        self.embeddings_cache = {
            'embeddings': embeddings,
            'documents': documents,
            'created_at': datetime.now().isoformat(),
            'kb_files_hash': self._calculate_kb_files_hash()
        }

        # Save cache
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"   ✅ Vector index built and cached to {embeddings_file}")
        except Exception as e:
            print(f"   ❌ Error saving embedding cache to {embeddings_file}: {e}")

        print(f"✅ Vector index built with {len(documents)} documents")

    def _documents_match(self, cached_docs: List[Dict], current_docs: List[Dict]) -> bool:
        """
        Check if cached documents match current documents.
        Returns True if they match, False otherwise.
        """
        if len(cached_docs) != len(current_docs):
            print(f"   📊 Document count mismatch: cached={len(cached_docs)}, current={len(current_docs)}")
            return False
        
        # Create sets of document identifiers for comparison
        cached_identifiers = set()
        current_identifiers = set()
        
        for doc in cached_docs:
            identifier = self._create_document_identifier(doc)
            cached_identifiers.add(identifier)
        
        for doc in current_docs:
            identifier = self._create_document_identifier(doc)
            current_identifiers.add(identifier)
        
        # Check if sets match
        if cached_identifiers != current_identifiers:
            print(f"   📊 Document identifiers don't match")
            # Optional: Print some differences for debugging
            missing_in_current = cached_identifiers - current_identifiers
            new_in_current = current_identifiers - cached_identifiers
            if missing_in_current:
                print(f"   📊 {len(missing_in_current)} documents removed from knowledge base")
            if new_in_current:
                print(f"   📊 {len(new_in_current)} new documents in knowledge base")
            return False
        
        return True

    def _create_document_identifier(self, doc: Dict) -> str:
        """
        Create a unique identifier for a document based on its content and metadata.
        """
        # Extract key components for identification
        doc_type = doc.get('type', 'unknown')
        doc_id = doc.get('id', '')
        content = doc.get('text', '')
        java_version = doc.get('java_version', 'unknown')
        section = str(doc.get('section', 'unknown'))
        
        # Create content hash for change detection
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]  # First 8 chars for brevity
        
        # Handle different document types with specific identifiers
        if doc_type == 'rule':
            category = doc.get('category', 'unknown')
            identifier = f"rule|{java_version}|{section}|{category}|{content_hash}"
        elif doc_type == 'conflict':
            severity = doc.get('severity', 'unknown')
            versions = doc.get('java_versions', [])
            version_str = '-'.join(str(v) for v in versions if v is not None)
            identifier = f"conflict|{version_str}|{severity}|{content_hash}"
        elif doc_type == 'specification':
            spec_key = doc.get('spec_key', 'unknown')
            identifier = f"spec|{spec_key}|{java_version}|{section}|{content_hash}"
        else:
            identifier = f"{doc_type}|{doc_id}|{content_hash}"
        
        return identifier

    def _calculate_kb_files_hash(self) -> str:
        """
        Calculate a hash of the knowledge base files to detect changes.
        """
        kb_files = {
            'specifications': f'{self.config.project_path}/data/all_specifications.json',
            'processed_rules': f'{self.config.project_path}/data/processed_rules.json',
            'conflict_report': f'{self.config.project_path}/results/conflict_report.json',
            'knowledge_base': f'{self.config.project_path}/agent_knowledge_base.json',
            'conflicts': f'{self.config.project_path}/results/all_conflicts.json',
            'summary': f'{self.config.project_path}/results/final_summary.json'
        }
        
        file_info = []
        for key, file_path in kb_files.items():
            if os.path.exists(file_path):
                try:
                    # Get file modification time and size
                    stat = os.stat(file_path)
                    file_info.append(f"{key}:{stat.st_mtime}:{stat.st_size}")
                except Exception:
                    file_info.append(f"{key}:error")
            else:
                file_info.append(f"{key}:missing")
        
        # Create hash of all file information
        combined_info = "|".join(sorted(file_info))
        return hashlib.md5(combined_info.encode('utf-8')).hexdigest()

    def _prepare_documents(self) -> List[Dict[str, Any]]:
        """Prepare documents for vector indexing"""
        documents = []

        # Process rules
        processed_rules = self.knowledge_base.get('processed_rules', [])
        if processed_rules:
            print(f"   Preparing {len(processed_rules)} rules for indexing...")
            for rule in processed_rules:
                doc = {
                    'id': f"rule_{len(documents)}_{rule.get('java_version', 'unknown')}_{rule.get('section_number', 'unknown')}_{rule.get('category', 'unknown')}", # More unique ID
                    'text': rule.get('text', ''),
                    'type': 'rule',
                    'java_version': rule.get('java_version', 'unknown'),
                    'section': rule.get('section_number', 'unknown'),
                    'category': rule.get('category', 'unknown'),
                    'metadata': rule # Store original metadata
                }
                # Ensure the text is not empty before adding and is a string
                if isinstance(doc['text'], str) and doc['text'].strip():
                    documents.append(doc)

        # Process conflicts from both sources
        # First from conflict_report (detailed_conflicts)
        detailed_conflicts = self.knowledge_base.get('conflict_report', {}).get('detailed_conflicts', [])
        if detailed_conflicts:
             print(f"   Preparing {len(detailed_conflicts)} conflicts from conflict_report for indexing...")
             for i, conflict in enumerate(detailed_conflicts):
                conflict_text = f"""
                CONFLICT: {conflict.get('type', 'Unknown')}
                Severity: {conflict.get('severity', 'Unknown')}
                Description: {conflict.get('description', '')}
                Affected Scenarios: {' '.join(conflict.get('affected_scenarios', []))}
                Resolution Needed: {conflict.get('resolution_needed', '')}
                """

                doc = {
                    'id': f"conflict_report_{i}_{conflict.get('rule1_version', 'unk')}_{conflict.get('rule2_version', 'unk')}", # More unique ID
                    'text': conflict_text.strip(),
                    'type': 'conflict',
                    'java_versions': [conflict.get('rule1_version'), conflict.get('rule2_version')],
                    'severity': conflict.get('severity', 'unknown'),
                    'metadata': conflict # Store original metadata
                }
                # Ensure the text is not empty before adding and is a string
                if isinstance(doc['text'], str) and doc['text'].strip():
                    documents.append(doc)

        # Then from conflicts (all_conflicts.json)
        all_conflicts = self.knowledge_base.get('conflicts', [])
        if all_conflicts:
             print(f"   Preparing {len(all_conflicts)} conflicts from all_conflicts for indexing...")
             for i, conflict in enumerate(all_conflicts):
                # Create comprehensive conflict text using the structure you provided
                conflict_text = f"""
                CONFLICT BETWEEN RULES:
                Rule 1: Java {conflict.get('rule1_version', 'Unknown')} - Section {conflict.get('rule1_section', 'Unknown')}
                Chapter: {conflict.get('rule1_chapter', 'Unknown')}
                Section Title: {conflict.get('rule1_section_title', 'Unknown')}
                Rule Text: {conflict.get('rule1_text', '')}
                
                Rule 2: Java {conflict.get('rule2_version', 'Unknown')} - Section {conflict.get('rule2_section', 'Unknown')}
                Chapter: {conflict.get('rule2_chapter', 'Unknown')}
                Section Title: {conflict.get('rule2_section_title', 'Unknown')}
                Rule Text: {conflict.get('rule2_text', '')}
                
                Conflict Type: {conflict.get('type', 'Unknown')}
                Severity: {conflict.get('severity', 'Unknown')}
                Description: {conflict.get('description', '')}
                Affected Scenarios: {', '.join(conflict.get('affected_scenarios', []))}
                Resolution Needed: {conflict.get('resolution_needed', '')}
                Common Entities: {', '.join(conflict.get('common_entities', []))}
                """

                doc = {
                    'id': f"all_conflicts_{i}_{conflict.get('rule1_version', 'unk')}_{conflict.get('rule2_version', 'unk')}", # More unique ID
                    'text': conflict_text.strip(),
                    'type': 'conflict',
                    'java_versions': [conflict.get('rule1_version'), conflict.get('rule2_version')],
                    'severity': conflict.get('severity', 'unknown'),
                    'rule1_id': conflict.get('rule1_id', ''),
                    'rule2_id': conflict.get('rule2_id', ''),
                    'rule1_url': conflict.get('rule1_url', ''),
                    'rule2_url': conflict.get('rule2_url', ''),
                    'detected_at': conflict.get('detected_at', ''),
                    'metadata': conflict # Store original metadata
                }
                # Ensure the text is not empty before adding and is a string
                if isinstance(doc['text'], str) and doc['text'].strip():
                    documents.append(doc)

        # Process specifications
        specifications = self.knowledge_base.get('specifications', {})
        if specifications:
            print(f"   Preparing {len(specifications)} specifications for indexing...")
            for spec_key, spec_data in specifications.items():
                # Chunk large specifications
                text = spec_data.get('text', '')
                if isinstance(text, str) and text.strip(): # Only chunk if text is a non-empty string
                    chunks = self._chunk_text(text)

                    for i, chunk in enumerate(chunks):
                        doc = {
                            'id': f"spec_{spec_key}_{i}",
                            'text': chunk,
                            'type': 'specification',
                            'spec_key': spec_key,
                            'java_version': spec_data.get('metadata', {}).get('java_version', 'unknown'),
                            'section': spec_data.get('metadata', {}).get('section_number', 'unknown'),
                            'metadata': spec_data # Store original metadata
                        }
                        if isinstance(doc['text'], str) and doc['text'].strip():
                             documents.append(doc)

        # Process knowledge_base entries
        knowledge_base_data = self.knowledge_base.get('knowledge_base', {})
        if knowledge_base_data:
            print(f"   Preparing knowledge_base entries for indexing...")
            for kb_key, kb_data in knowledge_base_data.items():
                if isinstance(kb_data, dict):
                    text = kb_data.get('content', '') or kb_data.get('text', '') or str(kb_data)
                else:
                    text = str(kb_data)
                
                if isinstance(text, str) and text.strip():
                    doc = {
                        'id': f"kb_{kb_key}",
                        'text': text,
                        'type': 'knowledge_base',
                        'kb_key': kb_key,
                        'metadata': kb_data
                    }
                    documents.append(doc)

        print(f"   Prepared {len(documents)} documents for indexing.")
        return documents

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for better retrieval"""
        # Use NLTK for better sentence splitting if available, fallback to word split
        chunks = []
        try:
            import nltk
            # Download punkt if not present, handle LookupError
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("   NLTK punkt tokenizer not found, attempting download...")
                nltk.download('punkt', quiet=True)
                try: # Check again after download
                     nltk.data.find('tokenizers/punkt')
                     print("   NLTK punkt tokenizer downloaded.")
                except LookupError:
                     print("   NLTK punkt tokenizer download failed.")
                     raise # Re-raise to trigger fallback

            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)

            # Recombine sentences into chunks with overlap
            current_chunk_sentences = []
            current_chunk_word_count = 0
            sentence_buffer = [] # To handle overlap

            for sentence in sentences:
                sentence_word_count = len(sentence.split())
                if sentence_word_count == 0: continue # Skip empty sentences

                # Add sentence to buffer
                sentence_buffer.append(sentence)

                if current_chunk_word_count + sentence_word_count <= self.config.chunk_size:
                    current_chunk_sentences.append(sentence)
                    current_chunk_word_count += sentence_word_count
                else:
                    # Current chunk is full, save it
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if len(chunk_text) > 50: # Add substantial chunks
                        chunks.append(chunk_text)

                    # Start a new chunk, potentially with overlap from buffer
                    overlap_sentences = sentence_buffer[-self.config.chunk_overlap//10:] # Simple overlap approximation
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_chunk_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                    # Clear buffer up to overlap point
                    sentence_buffer = sentence_buffer[-self.config.chunk_overlap//10:] + [sentence]

            # Add the last chunk
            last_chunk_text = " ".join(current_chunk_sentences).strip()
            if len(last_chunk_text) > 50:
                 chunks.append(last_chunk_text)

            # If sentence chunking resulted in no valid chunks, fall back
            if not chunks:
                 raise ValueError("Sentence chunking produced no valid chunks")

        except (LookupError, ImportError, ValueError) as e:
             # Fallback to simple word split if NLTK fails or chunking is ineffective
             print(f"   Sentence chunking failed ({e}), falling back to word split for chunking.")
             words = text.split()
             chunks = []
             i = 0
             while i < len(words):
                  chunk = ' '.join(words[i : i + self.config.chunk_size])
                  if len(chunk.strip()) > 50:  # Only add substantial chunks
                      chunks.append(chunk)
                      # Move index forward by chunk size minus overlap
                      i += max(1, self.config.chunk_size - self.config.chunk_overlap) # Ensure progress
                  else:
                      # If a word is longer than 50, or we're stuck, move forward by one
                      i += 1

        return chunks

    def retrieve_relevant_context(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context using RAG"""
        if not self.embeddings_cache or 'embeddings' not in self.embeddings_cache or self.embedding_model is None:
            print("   ⚠️ Embeddings or embedding model not available, cannot retrieve context.")
            return []

        max_results = max_results or self.config.max_contexts

        # Encode query
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            if query_embedding is None or len(query_embedding) == 0:
                 raise ValueError("Query encoding failed or returned empty.")
            query_embedding = query_embedding[0].reshape(1, -1) # Ensure it's 2D for cosine_similarity

        except Exception as e:
            print(f"   ❌ Error encoding query: {e}")
            return []

        # Calculate similarities
        try:
            # Ensure the embeddings cache has valid embeddings and documents
            if 'embeddings' not in self.embeddings_cache or 'documents' not in self.embeddings_cache or len(self.embeddings_cache['embeddings']) != len(self.embeddings_cache['documents']):
                print("   ⚠️ Embeddings cache is corrupt or inconsistent.")
                self.embeddings_cache = {} # Clear potentially bad cache
                return []

            similarities = cosine_similarity(
                query_embedding,
                self.embeddings_cache['embeddings']
            )[0]
        except Exception as e:
             print(f"   ❌ Error calculating similarity: {e}")
             return []

        # Get top results above threshold
        relevant_indices = [
            i for i, sim in enumerate(similarities)
            if sim >= self.config.similarity_threshold
        ]

        # Sort by similarity
        relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        relevant_indices = relevant_indices[:max_results]

        # Return relevant documents with scores
        results = []
        for idx in relevant_indices:
            # Add boundary check in case index is somehow out of bounds
            if idx < len(self.embeddings_cache['documents']):
                doc = self.embeddings_cache['documents'][idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                results.append(doc)
            else:
                 print(f"   ⚠️ Retrieved invalid index {idx} from embeddings cache.")

        return results

    # Modified _call_llm to accept messages list
    def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = None) -> str:
        """Call OpenRouter API with efficient model selection"""
        max_tokens = max_tokens or self.config.max_tokens_per_call
        temperature = temperature or self.config.temperature

        if not self.api_key:
            print("❌ API key is not set for LLM calls.")
            return "I cannot process your request because the API key is missing."

        # Use cost-effective models
        models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "deepseek/deepseek-chat-v3-0324:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ]

        for model in models:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/java-spec-agent",
                        "X-Title": "SpecSentinel: Java Specification Agent"
                    },
                    json={
                        "model": model,
                        "messages": messages, # Use the passed messages list
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=60 # Increased timeout for LLM calls
                )

                if response.status_code == 200:
                    result = response.json()
                    # Basic check for expected structure
                    if result and 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0]:
                         return result['choices'][0]['message']['content']
                    else:
                         print(f"   ⚠️ Unexpected response structure from {model}")
                         continue # Try next model

                elif response.status_code == 429:
                    print(f"   🔄 Rate limit hit for {model}. Waiting...")
                    time.sleep(5)  # Rate limit delay
                    continue # Retry this model or try next

                elif response.status_code == 401:
                    print(f"   🔑 Authentication error with key for {model}. Is the key valid?")
                    break # Stop trying models with this key issue

                else:
                    print(f"   ❌ Model {model} failed with status {response.status_code}: {response.text}")
                    break # Try next model

            except requests.exceptions.Timeout:
                print(f"   ⏰ Timeout calling {model}, trying next...")
                continue # Try next model
            except Exception as e:
                print(f"   ❌ Model {model} failed with exception: {str(e)[:200]}") # Print truncated error
                continue # Try next model

        print("❌ All models failed or encountered issues.")
        return "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again."

    def classify_query_intent(self, query: str) -> str:
        """Classify user query intent with comprehensive pattern matching"""
        query_lower = query.lower()

        # Statistics/analysis patterns
        stats_patterns = [
            'statistics', 'stats', 'analysis', 'report', 'summary', 'visualize', 
            'chart', 'graph', 'plot', 'dashboard', 'metrics', 'data', 'overview',
            'breakdown', 'distribution', 'count', 'number of', 'how many'
        ]
        if any(pattern in query_lower for pattern in stats_patterns):
            return 'statistics_request'

        # Greeting patterns
        greeting_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'what\'s up', 'how are you', 'nice to meet'
        ]
        if any(pattern in query_lower for pattern in greeting_patterns):
            return 'greeting'

        # Farewell patterns
        farewell_patterns = [
            'bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit', 'thanks',
            'thank you', 'that\'s all', 'done', 'finished', 'end'
        ]
        if any(pattern in query_lower for pattern in farewell_patterns):
            return 'farewell'

        # Help/guidance patterns
        help_patterns = [
            'help', 'how to', 'what can you do', 'capabilities', 'features',
            'guide', 'tutorial', 'instructions', 'commands', 'usage'
        ]
        if any(pattern in query_lower for pattern in help_patterns):
            return 'help_request'

        # Conflict-related queries
        conflict_patterns = [
            'conflict', 'contradiction', 'ambiguity', 'inconsistency', 'problem', 
            'issue', 'error', 'bug', 'discrepancy', 'mismatch', 'incompatible',
            'breaking change', 'deprecated', 'removed'
        ]
        if any(pattern in query_lower for pattern in conflict_patterns):
            return 'conflict_inquiry'

        # Specification queries
        spec_patterns = [
            'specification', 'rule', 'section', 'jls', 'java language specification',
            'standard', 'documentation', 'reference', 'manual', 'guideline'
        ]
        if any(pattern in query_lower for pattern in spec_patterns):
            return 'specification_inquiry'

        # Version comparison
        version_patterns = [
            'java 8', 'java 11', 'java 17', 'java 21', 'version', 'difference', 
            'change', 'migration', 'upgrade', 'downgrade', 'compatibility',
            'evolution', 'history', 'timeline'
        ]
        if any(pattern in query_lower for pattern in version_patterns):
            return 'version_comparison'

        # Code examples/implementation
        code_patterns = [
            'example', 'code', 'implementation', 'sample', 'demo', 'snippet',
            'how to implement', 'show me', 'demonstrate'
        ]
        if any(pattern in query_lower for pattern in code_patterns):
            return 'code_example_request'

        # Performance/optimization queries
        performance_patterns = [
            'performance', 'optimization', 'speed', 'memory', 'efficiency',
            'benchmark', 'profiling', 'bottleneck'
        ]
        if any(pattern in query_lower for pattern in performance_patterns):
            return 'performance_inquiry'

        # Best practices
        best_practice_patterns = [
            'best practice', 'recommendation', 'should i', 'better way',
            'advice', 'suggestion', 'pattern', 'anti-pattern'
        ]
        if any(pattern in query_lower for pattern in best_practice_patterns):
            return 'best_practice_request'

        return 'general_inquiry'

    def generate_statistics_summary(self) -> str:
        """Generate comprehensive statistics summary"""
        try:
            conflict_data = self.knowledge_base.get('conflict_report', {})
            summary_data = self.knowledge_base.get('summary', {})
            
            if not conflict_data or not summary_data:
                return "❌ Statistics data not available. Please ensure the analysis results are properly loaded."
            
            # Generate comprehensive summary
            report = "📊 **JAVA SPECIFICATION ANALYSIS SUMMARY**\n"
            report += "=" * 50 + "\n\n"
            
            # Analysis Overview
            if 'analysis_info' in summary_data:
                analysis_info = summary_data['analysis_info']
                report += "🔍 **ANALYSIS OVERVIEW:**\n"
                report += f"• Duration: {analysis_info.get('duration_formatted', 'N/A')}\n"
                if 'data_statistics' in summary_data:
                    stats = summary_data['data_statistics']
                    report += f"• Processing Success Rate: {stats.get('processing_success_rate', 'N/A')}\n"
                    report += f"• Rules Analyzed: {stats.get('rules_processed', 'N/A')}\n\n"
            
            # Conflict Summary
            report += "⚠️ **CONFLICT SUMMARY:**\n"
            report += f"• Total Conflicts: {conflict_data.get('total_conflicts', 0)}\n"
            if 'by_version_pair' in conflict_data:
                report += f"• Version Pairs: {list(conflict_data['by_version_pair'].keys())}\n\n"
            
            # Severity Breakdown
            if 'by_severity' in conflict_data:
                report += "🎯 **SEVERITY BREAKDOWN:**\n"
                for severity, count in conflict_data['by_severity'].items():
                    report += f"• {severity}: {count} conflicts\n"
                report += "\n"
            
            # Rule Categories
            if 'rule_analysis' in summary_data and 'category_distribution' in summary_data['rule_analysis']:
                report += "📋 **RULE CATEGORIES:**\n"
                for category, count in summary_data['rule_analysis']['category_distribution'].items():
                    report += f"• {category}: {count} rules\n"
                report += "\n"
            
            # Recommendations
            if 'recommendations' in conflict_data:
                report += "💡 **RECOMMENDATIONS:**\n"
                for rec in conflict_data['recommendations']:
                    report += f"• {rec}\n"
            
            return report
            
        except Exception as e:
            return f"❌ Error generating statistics: {str(e)}"

    def create_visualizations(self) -> List[str]:
        """Create visualization charts and return base64 encoded images"""
        try:
            conflict_data = self.knowledge_base.get('conflict_report', {})
            summary_data = self.knowledge_base.get('summary', {})
            
            if not conflict_data or not summary_data:
                return []
            
            images = []
            
            # Set style for matplotlib
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Conflict Severity Distribution
            if 'by_severity' in conflict_data and conflict_data['by_severity']:
                fig, ax = plt.subplots(figsize=(10, 6))
                severity_data = conflict_data['by_severity']
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                
                wedges, texts, autotexts = ax.pie(
                    severity_data.values(), 
                    labels=severity_data.keys(), 
                    autopct='%1.1f%%',
                    colors=colors[:len(severity_data)],
                    explode=[0.05] * len(severity_data)
                )
                ax.set_title('Conflict Severity Distribution', fontsize=14, fontweight='bold')
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(image_base64)
                plt.close()
            
            # 2. Rule Category Distribution
            if 'rule_analysis' in summary_data and 'category_distribution' in summary_data['rule_analysis']:
                fig, ax = plt.subplots(figsize=(12, 6))
                categories = summary_data['rule_analysis']['category_distribution']
                
                bars = ax.bar(categories.keys(), categories.values(), 
                             color=['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'])
                ax.set_title('Rule Category Distribution', fontsize=14, fontweight='bold')
                ax.set_ylabel('Number of Rules')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(image_base64)
                plt.close()
            
            # 3. Version Distribution
            if 'rule_analysis' in summary_data and 'version_distribution' in summary_data['rule_analysis']:
                fig, ax = plt.subplots(figsize=(8, 8))
                version_data = summary_data['rule_analysis']['version_distribution']
                
                wedges, texts, autotexts = ax.pie(
                    version_data.values(), 
                    labels=[f'Java {v}' for v in version_data.keys()],
                    autopct='%1.1f%%',
                    colors=['#fdb462', '#b3de69']
                )
                # Create donut by adding a white circle in center
                centre_circle = plt.Circle((0,0), 0.40, fc='white')
                ax.add_patch(centre_circle)
                ax.set_title('Java Version Distribution', fontsize=14, fontweight='bold')
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(image_base64)
                plt.close()
            
            return images
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")
            return []

    def create_interactive_plots(self) -> Dict[str, Any]:
        """Create interactive Plotly visualizations and return plot data"""
        try:
            conflict_data = self.knowledge_base.get('conflict_report', {})
            summary_data = self.knowledge_base.get('summary', {})
            
            if not conflict_data or not summary_data:
                return {}
            
            plots = {}
            
            # 1. Conflict Severity Distribution
            if 'by_severity' in conflict_data and conflict_data['by_severity']:
                severity_data = conflict_data['by_severity']
                fig = px.pie(
                    values=list(severity_data.values()),
                    names=list(severity_data.keys()),
                    title='Conflict Severity Distribution',
                    color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    font=dict(size=14),
                    title_font_size=18,
                    showlegend=True
                )
                plots['severity_pie'] = fig
            
            # 2. Rule Category Distribution
            if 'rule_analysis' in summary_data and 'category_distribution' in summary_data['rule_analysis']:
                categories = summary_data['rule_analysis']['category_distribution']
                fig = px.bar(
                    x=list(categories.keys()),
                    y=list(categories.values()),
                    title='Rule Category Distribution',
                    color=list(categories.values()),
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    xaxis_title='Categories', 
                    yaxis_title='Number of Rules',
                    font=dict(size=12),
                    title_font_size=18
                )
                plots['category_bar'] = fig
            
            # 3. Version Comparison
            if 'by_version_pair' in conflict_data and conflict_data['by_version_pair']:
                version_pairs = conflict_data['by_version_pair']
                fig = px.bar(
                    x=list(version_pairs.keys()),
                    y=list(version_pairs.values()),
                    title='Conflicts by Version Pairs',
                    color=list(version_pairs.values()),
                    color_continuous_scale='reds'
                )
                fig.update_layout(
                    xaxis_title='Version Pairs', 
                    yaxis_title='Number of Conflicts',
                    font=dict(size=12),
                    title_font_size=18
                )
                plots['version_bar'] = fig
            
            # 4. Processing Statistics
            if 'data_statistics' in summary_data:
                stats = summary_data['data_statistics']
                metrics = ['Specifications Downloaded', 'Raw Rules Extracted', 'Rules Processed']
                values = [
                    stats.get('specifications_downloaded', 0),
                    stats.get('raw_rules_extracted', 0), 
                    stats.get('rules_processed', 0)
                ]
                
                fig = px.funnel(
                    x=values,
                    y=metrics,
                    title='Processing Pipeline Statistics',
                    color=values,
                    color_continuous_scale='blues'
                )
                fig.update_layout(
                    font=dict(size=12),
                    title_font_size=18
                )
                plots['processing_funnel'] = fig
            
            # 5. Conflict Type Distribution
            if 'by_type' in conflict_data and conflict_data['by_type']:
                type_data = conflict_data['by_type']
                fig = px.treemap(
                    names=list(type_data.keys()),
                    values=list(type_data.values()),
                    title='Conflict Types Distribution',
                    color=list(type_data.values()),
                    color_continuous_scale='plasma'
                )
                fig.update_layout(
                    font=dict(size=12),
                    title_font_size=18
                )
                plots['type_treemap'] = fig
            
            return plots
            
        except Exception as e:
            print(f"❌ Error creating interactive plots: {str(e)}")
            return {}

    def generate_detailed_report(self) -> str:
        """Generate a comprehensive detailed report"""
        try:
            conflict_data = self.knowledge_base.get('conflict_report', {})
            summary_data = self.knowledge_base.get('summary', {})
            
            if not conflict_data or not summary_data:
                return "❌ Report data not available."
            
            report = "📋 **DETAILED ANALYSIS REPORT**\n"
            report += "=" * 80 + "\n\n"
            
            # Executive Summary
            report += "🎯 **EXECUTIVE SUMMARY**\n"
            report += "-" * 40 + "\n"
            total_conflicts = conflict_data.get('total_conflicts', 0)
            rules_analyzed = summary_data.get('data_statistics', {}).get('rules_processed', 0)
            conflict_rate = (total_conflicts / rules_analyzed) * 100 if rules_analyzed > 0 else 0
            
            report += f"Analysis completed successfully with {conflict_rate:.1f}% conflict rate.\n"
            report += f"Out of {rules_analyzed} rules analyzed across Java versions,\n"
            report += f"{total_conflicts} conflicts were identified.\n\n"
            
            # Key Findings
            report += "🔍 **KEY FINDINGS**\n"
            report += "-" * 40 + "\n"
            if 'by_type' in conflict_data:
                conflict_types = list(conflict_data['by_type'].keys())
                report += f"1. Conflict types identified: {', '.join(conflict_types)}\n"
            
            if 'by_severity' in conflict_data:
                severities = conflict_data['by_severity']
                report += f"2. Severity distribution: {dict(severities)}\n"
            
            report += "3. Primary areas affected: control flow statements and exception handling\n"
            report += "4. Version evolution patterns detected in language specifications\n\n"
            
            # Risk Assessment
            report += "⚠️ **RISK ASSESSMENT**\n"
            report += "-" * 40 + "\n"
            if 'by_severity' in conflict_data:
                medium_conflicts = conflict_data['by_severity'].get('MEDIUM', 0)
                high_conflicts = conflict_data['by_severity'].get('HIGH', 0)
                low_conflicts = conflict_data['by_severity'].get('LOW', 0)
                
                if high_conflicts > 0:
                    report += f"• {high_conflicts} HIGH severity conflicts require immediate attention\n"
                if medium_conflicts > 0:
                    report += f"• {medium_conflicts} MEDIUM severity conflicts require attention during migration\n"
                if low_conflicts > 0:
                    report += f"• {low_conflicts} LOW severity conflicts are informational\n"
            
            report += "\nOverall Risk Level: Varies by conflict severity\n"
            report += "Migration planning recommended based on conflict analysis\n\n"
            
            # Recommendations
            report += "💡 **DETAILED RECOMMENDATIONS**\n"
            report += "-" * 40 + "\n"
            if 'recommendations' in conflict_data:
                for i, rec in enumerate(conflict_data['recommendations'], 1):
                    report += f"{i}. {rec}\n"
            else:
                report += "1. Code Review: Focus on identified conflict areas\n"
                report += "2. Testing: Comprehensive testing of affected components\n"
                report += "3. Documentation: Update documentation for version differences\n"
                report += "4. Training: Brief team on specification changes\n"
            
            report += "\n" + "=" * 80 + "\n"
            
            return report
            
        except Exception as e:
            return f"❌ Error generating detailed report: {str(e)}"

    def generate_contextual_visualization(self, query: str, context: str) -> Dict[str, Any]:
        """Generate contextual visualizations based on query and retrieved context"""
        try:
            plots = {}
            
            # Analyze query for visualization needs
            query_lower = query.lower()
            
            # If asking about trends or changes over time
            if any(word in query_lower for word in ['trend', 'change', 'evolution', 'history', 'timeline']):
                # Create timeline visualization if we have version data
                if 'conflicts' in self.knowledge_base:
                    conflicts = self.knowledge_base['conflicts']
                    if isinstance(conflicts, list):
                        # Extract version information for timeline
                        version_timeline = {}
                        for conflict in conflicts:
                            v1 = conflict.get('rule1_version', 'Unknown')
                            v2 = conflict.get('rule2_version', 'Unknown')
                            key = f"{v1} → {v2}"
                            version_timeline[key] = version_timeline.get(key, 0) + 1
                        
                        if version_timeline:
                            fig = px.line(
                                x=list(version_timeline.keys()),
                                y=list(version_timeline.values()),
                                title='Conflict Evolution Across Java Versions',
                                markers=True
                            )
                            fig.update_layout(
                                xaxis_title='Version Transitions',
                                yaxis_title='Number of Conflicts',
                                font=dict(size=12)
                            )
                            plots['timeline'] = fig
            
            # If asking about comparisons
            if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs', 'difference']):
                # Create comparison charts
                if 'rule_analysis' in self.knowledge_base.get('summary', {}):
                    rule_analysis = self.knowledge_base['summary']['rule_analysis']
                    if 'version_distribution' in rule_analysis:
                        version_data = rule_analysis['version_distribution']
                        
                        fig = px.bar(
                            x=list(version_data.keys()),
                            y=list(version_data.values()),
                            title='Rule Distribution by Java Version',
                            color=list(version_data.values()),
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(
                            xaxis_title='Java Version',
                            yaxis_title='Number of Rules',
                            font=dict(size=12)
                        )
                        plots['version_comparison'] = fig
            
            # If asking about specific categories or types
            if any(word in query_lower for word in ['category', 'type', 'kind', 'classification']):
                # Create category breakdown
                if 'by_type' in self.knowledge_base.get('conflict_report', {}):
                    type_data = self.knowledge_base['conflict_report']['by_type']
                    
                    fig = px.sunburst(
                        names=list(type_data.keys()),
                        values=list(type_data.values()),
                        title='Conflict Types Hierarchy'
                    )
                    plots['category_sunburst'] = fig
            
            return plots
            
        except Exception as e:
            print(f"❌ Error generating contextual visualization: {str(e)}")
            return {}

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Process user query and return comprehensive response"""
        intent = self.classify_query_intent(query)
        
        response = {
            'intent': intent,
            'text_response': '',
            'visualizations': [],
            'interactive_plots': {},
            'detailed_report': ''
        }
        
        # Handle different intents
        if intent == 'greeting':
            response['text_response'] = """
            👋 **Welcome to SpecSentinel: Java Specification Agent!**
            
            I'm your intelligent assistant for Java Language Specification analysis. Here's what I can help you with:
            
            🔍 **Specification Analysis**
            - Query Java language specification rules and sections
            - Find specific documentation and references
            - Explain complex specification details
            
            ⚠️ **Conflict Detection & Analysis**
            - Identify contradictions between Java versions
            - Analyze inconsistencies in specifications
            - Provide conflict resolution guidance
            
            📊 **Statistics & Reporting**
            - Generate comprehensive analysis reports
            - Create interactive visualizations and charts
            - Provide detailed breakdowns and metrics
            
            🔄 **Version Comparison**
            - Compare differences between Java versions
            - Track evolution of language features
            - Migration guidance and recommendations
            
            **Try asking me:**
            - "Show me statistics about the conflicts"
            - "What are the main conflicts between Java 8 and Java 11?"
            - "Generate a detailed report with visualizations"
            - "Compare exception handling rules across versions"
            """
            
        elif intent == 'help_request':
            response['text_response'] = """
            🆘 **SpecSentinel Help Guide**
            
            **Available Commands & Queries:**
            
            📈 **Statistics & Analysis:**
            - "Show statistics" / "Generate report"
            - "Create visualizations" / "Show charts"
            - "Analysis summary" / "Data overview"
            
            🔍 **Specification Queries:**
            - "Explain [topic]" / "What is [concept]?"
            - "Show rules for [feature]"
            - "Find specification for [topic]"
            
            ⚠️ **Conflict Analysis:**
            - "Show conflicts between Java X and Y"
            - "What conflicts exist in [area]?"
            - "Analyze inconsistencies"
            
            🔄 **Version Comparison:**
            - "Compare Java versions"
            - "What changed in Java X?"
            - "Migration from X to Y"
            
            **Tips:**
            - Be specific about Java versions (8, 11, 17, 21)
            - Ask for visualizations to get interactive charts
            - Request detailed reports for comprehensive analysis
            """
            
        elif intent == 'statistics_request':
            response['text_response'] = self.generate_statistics_summary()
            response['visualizations'] = self.create_visualizations()
            response['interactive_plots'] = self.create_interactive_plots()
            response['detailed_report'] = self.generate_detailed_report()
            
        elif intent == 'conflict_inquiry':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate contextual visualizations
            contextual_plots = self.generate_contextual_visualization(query, context)
            response['interactive_plots'].update(contextual_plots)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java Specification expert specializing in conflict analysis. Use the provided context to answer questions about conflicts, contradictions, and inconsistencies in Java specifications. Be specific, cite relevant sections, provide examples, and suggest resolution strategies when possible."""
                },
                {
                    "role": "user", 
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'specification_inquiry':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java Specification expert. Use the provided context to answer questions about Java language specifications, rules, and sections. Provide detailed explanations with relevant citations, examples, and practical implications."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'version_comparison':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate contextual visualizations
            contextual_plots = self.generate_contextual_visualization(query, context)
            response['interactive_plots'].update(contextual_plots)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java expert specializing in version differences and migration. Use the provided context to compare Java versions, explain changes, improvements, and potential issues during migration. Provide practical guidance and examples."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'code_example_request':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java expert who provides practical code examples and implementations. Use the provided context to create relevant code examples that demonstrate Java specification concepts, rules, and best practices. Include explanations and comments."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'performance_inquiry':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java performance expert. Use the provided context to answer questions about performance implications of Java specification changes, optimization strategies, and performance-related conflicts between versions."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'best_practice_request':
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a Java best practices expert. Use the provided context to provide recommendations, best practices, and guidance based on Java specifications and identified conflicts. Focus on practical, actionable advice."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
            
        elif intent == 'farewell':
            response['text_response'] = """
            👋 **Thank you for using SpecSentinel!**
            
            I hope I was able to help you with your Java specification analysis needs. 
            
            Feel free to return anytime for:
            - Specification queries and analysis
            - Conflict detection and resolution
            - Version comparison and migration guidance
            - Statistical reports and visualizations
            
            Have a great day! ☕
            """
            
        else:  # general_inquiry
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query)
            
            # Prepare context for LLM
            context = self._prepare_context_for_llm(relevant_docs)
            
            # Generate contextual visualizations if relevant
            contextual_plots = self.generate_contextual_visualization(query, context)
            response['interactive_plots'].update(contextual_plots)
            
            # Generate response using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are SpecSentinel, a helpful Java Specification Agent. Use the provided context to answer questions about Java specifications, conflicts, and analysis results. If the context doesn't contain relevant information, acknowledge this and provide general guidance based on your knowledge of Java specifications."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
            
            llm_response = self._call_llm(messages)
            response['text_response'] = llm_response
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent,
            'response': response['text_response']
        })
        
        return response

    def _prepare_context_for_llm(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare retrieved documents as context for LLM"""
        if not relevant_docs:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        for doc in relevant_docs:
            context_part = f"""
Document Type: {doc.get('type', 'unknown')}
Similarity Score: {doc.get('similarity_score', 0):.3f}
Content: {doc.get('text', '')[:500]}...
Metadata: Java Version: {doc.get('java_version', 'unknown')}, Section: {doc.get('section', 'unknown')}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def initialize_agent():
    """Initialize the Java Spec Agent"""
    if 'agent' not in st.session_state:
        try:
            # Get API key from env
            api_key = os.getenv('OPENROUTER_API_KEY')
            
            # Initialize configuration
            config = AgentConfig()
            
            # Initialize agent
            with st.spinner("🚀 Initializing SpecSentinel: Java Specification Agent..."):
                st.session_state.agent = JavaSpecAgent(config, api_key)
                
            st.success("✅ Agent initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.agent = None

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="SpecSentinel: Java Specification Agent",
        page_icon="☕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for beautiful UI
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        border-left: 5px solid #9c27b0;
    }
    
    /* Statistics container */
    .statistics-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .chat-message {
            padding: 1rem;
        }
    }
    
    /* Plotly chart containers */
    .plotly-chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">☕ SpecSentinel: Java Specification Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🔍 Intelligent analysis of Java Language Specifications with advanced conflict detection and interactive reporting</p>', unsafe_allow_html=True)
    
    # Initialize agent
    initialize_agent()
    
    if not st.session_state.agent:
        st.error("❌ Agent not initialized. Please check your configuration and try again.")
        st.info("💡 Make sure your OpenRouter API key is set in the environment variables.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>📊 Agent Dashboard</h2></div>', unsafe_allow_html=True)
        
        # Knowledge base status
        kb = st.session_state.agent.knowledge_base
        
        # Safe counts with better error handling
        try:
            processed_rules_count = len(kb.get('processed_rules', []))
            conflicts = kb.get('conflicts', [])
            conflict_count = (
                conflicts.get('total_conflicts', 0) if isinstance(conflicts, dict)
                else len(conflicts) if isinstance(conflicts, list)
                else 0
            )
            specs_count = len(kb.get('specifications', {}))
            kb_entries_count = len(kb.get('knowledge_base', {}))
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            processed_rules_count = conflict_count = specs_count = kb_entries_count = 0

        # Display metrics in a grid
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📋 Rules", processed_rules_count, help="Total processed rules")
            st.metric("📚 Specifications", specs_count, help="Loaded specifications")
        with col2:
            st.metric("⚠️ Conflicts", conflict_count, help="Detected conflicts")
            st.metric("🧠 KB Entries", kb_entries_count, help="Knowledge base entries")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Statistics", help="Generate comprehensive statistics"):
                st.session_state.show_stats = True
            
            if st.button("🔄 Clear Chat", help="Clear conversation history"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📋 Report", help="Generate detailed analysis report"):
                st.session_state.show_report = True
            
            if st.button("💾 Export", help="Export conversation history"):
                if 'messages' in st.session_state and st.session_state.messages:
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'messages': st.session_state.messages,
                        'agent_info': {
                            'version': '2.0',
                            'knowledge_base_stats': {
                                'rules': processed_rules_count,
                                'conflicts': conflict_count,
                                'specifications': specs_count
                            }
                        }
                    }
                    st.download_button(
                        "💾 Download Chat",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"specsentinel_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        st.markdown("---")
        
        # Agent Status
        st.markdown("### 🤖 Agent Status")
        
        # Check embedding model status
        embedding_status = "✅ Active" if st.session_state.agent.embedding_model else "⚠️ Fallback Mode"
        st.info(f"**Embedding Model:** {embedding_status}")
        
        # Check knowledge base status
        kb_status = "✅ Loaded" if kb else "❌ Not Available"
        st.info(f"**Knowledge Base:** {kb_status}")
        
        # API status
        api_status = "✅ Configured" if st.session_state.agent.api_key else "❌ Missing"
        st.info(f"**API Connection:** {api_status}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ About SpecSentinel", expanded=False):
            st.markdown("""
            **SpecSentinel** is an advanced AI agent that analyzes Java Language Specifications to:
            
            🔍 **Detect Conflicts** between different Java versions
            
            📊 **Generate Reports** with comprehensive statistics and visualizations
            
            💬 **Provide Q&A** with intelligent context-aware responses
            
            📈 **Create Visualizations** including interactive charts and graphs
            
            🎯 **Offer Guidance** for migration and best practices
            
            ---
            
            **Powered by:**
            - OpenRouter API for LLM capabilities
            - Sentence Transformers for embeddings
            - Plotly for interactive visualizations
            - Streamlit for the user interface
            """)
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Welcome message for new users
    if not st.session_state.messages:
        welcome_message = {
            "role": "assistant",
            "content": """👋 **Welcome to SpecSentinel!**

I'm ready to help you analyze Java Language Specifications. Here are some things you can try:

• **"Show me statistics"** - Get comprehensive analysis with visualizations
• **"What conflicts exist between Java 8 and Java 11?"** - Specific conflict analysis
• **"Generate a detailed report"** - Full analysis report
• **"Explain exception handling rules"** - Specification queries
• **"Help"** - See all available commands

What would you like to explore first? 🚀"""
        }
        st.session_state.messages.append(welcome_message)
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualizations if present
            if "visualizations" in message and message["visualizations"]:
                st.markdown("### 📊 Statistical Visualizations")
                
                # Create columns for multiple visualizations
                num_viz = len(message["visualizations"])
                if num_viz == 1:
                    st.image(f"data:image/png;base64,{message['visualizations'][0]}", use_container_width=True)
                elif num_viz == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f"data:image/png;base64,{message['visualizations'][0]}", use_container_width=True)
                    with col2:
                        st.image(f"data:image/png;base64,{message['visualizations'][1]}", use_container_width=True)
                else:
                    cols = st.columns(min(num_viz, 3))
                    for idx, img_b64 in enumerate(message["visualizations"]):
                        with cols[idx % 3]:
                            st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
            
            # Display interactive plots if present
            if "interactive_plots" in message and message["interactive_plots"]:
                st.markdown("### 📈 Interactive Analysis Charts")
                
                # Create tabs for different plot types
                plot_names = list(message["interactive_plots"].keys())
                if len(plot_names) > 1:
                    tabs = st.tabs([name.replace('_', ' ').title() for name in plot_names])
                    for tab, (plot_name, fig) in zip(tabs, message["interactive_plots"].items()):
                        with tab:
                            unique_key = f"plotly_chart_{i}_{plot_name}"
                            st.plotly_chart(fig, use_container_width=True, key=unique_key)
                else:
                    for plot_name, fig in message["interactive_plots"].items():
                        unique_key = f"plotly_chart_{i}_{plot_name}"
                        st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
            # Display detailed report if present
            if "detailed_report" in message and message["detailed_report"]:
                with st.expander("📋 View Detailed Analysis Report", expanded=False):
                    st.markdown(message["detailed_report"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me about Java specifications, conflicts, or request statistics and visualizations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query and generating response..."):
                response = st.session_state.agent.answer_query(prompt)
            
            # Display text response
            st.markdown(response['text_response'])
            
            # Store complete response for display
            assistant_message = {
                "role": "assistant", 
                "content": response['text_response'],
                "visualizations": response.get('visualizations', []),
                "interactive_plots": response.get('interactive_plots', {}),
                "detailed_report": response.get('detailed_report', '')
            }
            
            # Display visualizations
            if response.get('visualizations'):
                st.markdown("### 📊 Statistical Visualizations")
                
                num_viz = len(response['visualizations'])
                if num_viz == 1:
                    st.image(f"data:image/png;base64,{response['visualizations'][0]}", use_container_width=True)
                elif num_viz == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f"data:image/png;base64,{response['visualizations'][0]}", use_container_width=True)
                    with col2:
                        st.image(f"data:image/png;base64,{response['visualizations'][1]}", use_container_width=True)
                else:
                    cols = st.columns(min(num_viz, 3))
                    for idx, img_b64 in enumerate(response['visualizations']):
                        with cols[idx % 3]:
                            st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
            
            # Display interactive plots
            if response.get('interactive_plots'):
                st.markdown("### 📈 Interactive Analysis Charts")
                
                plot_names = list(response['interactive_plots'].keys())
                if len(plot_names) > 1:
                    tabs = st.tabs([name.replace('_', ' ').title() for name in plot_names])
                    for tab, (plot_name, fig) in zip(tabs, response['interactive_plots'].items()):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    for plot_name, fig in response['interactive_plots'].items():
                        st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed report
            if response.get('detailed_report'):
                with st.expander("📋 View Detailed Analysis Report", expanded=False):
                    st.markdown(response['detailed_report'])
            
            # Add assistant message to chat history
            st.session_state.messages.append(assistant_message)
    
    # Handle quick actions
    if st.session_state.get('show_stats', False):
        st.session_state.show_stats = False
        
        with st.chat_message("assistant"):
            with st.spinner("📊 Generating comprehensive statistics and visualizations..."):
                response = st.session_state.agent.answer_query("Show me comprehensive statistics and visualizations with interactive charts")
            
            st.markdown(response['text_response'])
            
            # Display all components
            if response.get('visualizations'):
                st.markdown("### 📊 Statistical Visualizations")
                cols = st.columns(min(len(response['visualizations']), 3))
                for i, img_b64 in enumerate(response['visualizations']):
                    with cols[i % 3]:
                        st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
            
            if response.get('interactive_plots'):
                st.markdown("### 📈 Interactive Analysis Dashboard")
                plot_names = list(response['interactive_plots'].keys())
                if len(plot_names) > 1:
                    tabs = st.tabs([name.replace('_', ' ').title() for name in plot_names])
                    for tab, (plot_name, fig) in zip(tabs, response['interactive_plots'].items()):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    for plot_name, fig in response['interactive_plots'].items():
                        st.plotly_chart(fig, use_container_width=True)
            
            if response.get('detailed_report'):
                with st.expander("📋 Full Statistical Analysis Report", expanded=True):
                    st.markdown(response['detailed_report'])
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['text_response'],
                "visualizations": response.get('visualizations', []),
                "interactive_plots": response.get('interactive_plots', {}),
                "detailed_report": response.get('detailed_report', '')
            })
    
    if st.session_state.get('show_report', False):
        st.session_state.show_report = False
        
        with st.chat_message("assistant"):
            with st.spinner("📋 Generating comprehensive analysis report..."):
                detailed_report = st.session_state.agent.generate_detailed_report()
                stats_summary = st.session_state.agent.generate_statistics_summary()
                interactive_plots = st.session_state.agent.create_interactive_plots()
            
            combined_response = f"""## 📋 Comprehensive Analysis Report

{stats_summary}

---

{detailed_report}"""
            
            st.markdown(combined_response)
            
            # Display interactive plots
            if interactive_plots:
                st.markdown("### 📈 Supporting Visualizations")
                plot_names = list(interactive_plots.keys())
                if len(plot_names) > 1:
                    tabs = st.tabs([name.replace('_', ' ').title() for name in plot_names])
                    for tab, (plot_name, fig) in zip(tabs, interactive_plots.items()):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    for plot_name, fig in interactive_plots.items():
                        st.plotly_chart(fig, use_container_width=True)
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": combined_response,
                "interactive_plots": interactive_plots
            })

if __name__ == "__main__":
    main()