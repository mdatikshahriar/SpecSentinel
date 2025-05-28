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
                "deepseek/deepseek-chat-v3-0324:free",
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemma-2-9b-it:free",
                "microsoft/phi-3-mini-128k-instruct:free"
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
        elif doc_type == 'knowledge_base':
            category = doc.get('category', 'unknown')
            identifier = f"kb|{category}|{java_version}|{content_hash}"
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

        # Process knowledge base (new file)
        knowledge_base_data = self.knowledge_base.get('knowledge_base', {})
        if knowledge_base_data:
            print(f"   Preparing knowledge base entries for indexing...")
            # Handle different structures in the knowledge base file
            if isinstance(knowledge_base_data, dict):
                for key, value in knowledge_base_data.items():
                    if isinstance(value, dict):
                        text_content = value.get('content', '') or value.get('text', '') or str(value)
                    else:
                        text_content = str(value)
                    
                    if text_content and text_content.strip():
                        doc = {
                            'id': f"knowledge_base_{len(documents)}_{key}",
                            'text': text_content,
                            'type': 'knowledge_base',
                            'category': key,
                            'java_version': value.get('java_version', 'unknown') if isinstance(value, dict) else 'unknown',
                            'metadata': value
                        }
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
            doc = self.embeddings_cache['documents'][idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            results.append(doc)

        return results

    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Enhanced query intent classification with comprehensive scenarios"""
        query_lower = query.lower().strip()
        
        # Initialize classification result
        classification = {
            'primary_intent': 'general',
            'secondary_intents': [],
            'confidence': 0.5,
            'java_versions': [],
            'entities': [],
            'query_type': 'informational',
            'urgency': 'normal',
            'complexity': 'simple'
        }
        
        # Define patterns for different intents
        intent_patterns = {
            'conflict_analysis': [
                r'conflict', r'contradiction', r'inconsistent', r'disagree', r'differ',
                r'incompatible', r'clash', r'oppose', r'violation', r'mismatch'
            ],
            'version_comparison': [
                r'java \d+', r'jdk \d+', r'compare.*version', r'difference.*between',
                r'changed.*from', r'evolution', r'migration', r'upgrade', r'downgrade'
            ],
            'rule_lookup': [
                r'rule', r'regulation', r'specification', r'requirement', r'standard',
                r'guideline', r'constraint', r'restriction', r'policy'
            ],
            'explanation': [
                r'what.*is', r'how.*does', r'why.*is', r'explain', r'describe',
                r'define', r'meaning', r'purpose', r'rationale'
            ],
            'example_request': [
                r'example', r'sample', r'demonstration', r'show.*me', r'illustrate',
                r'instance', r'case.*study', r'scenario'
            ],
            'best_practices': [
                r'best.*practice', r'recommendation', r'should.*i', r'how.*to',
                r'proper.*way', r'correct.*approach', r'optimal', r'advice'
            ],
            'troubleshooting': [
                r'error', r'problem', r'issue', r'bug', r'fail', r'wrong',
                r'not.*work', r'troubleshoot', r'debug', r'fix'
            ],
            'compatibility': [
                r'compatible', r'support', r'work.*with', r'integrate',
                r'interoperable', r'backward.*compatible', r'forward.*compatible'
            ],
            'performance': [
                r'performance', r'speed', r'optimization', r'efficiency',
                r'memory', r'cpu', r'benchmark', r'profiling'
            ],
            'security': [
                r'security', r'vulnerable', r'exploit', r'attack', r'safe',
                r'encryption', r'authentication', r'authorization'
            ]
        }
        
        # Calculate intent scores
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = 0
            matches = []
            for pattern in patterns:
                pattern_matches = re.findall(pattern, query_lower)
                if pattern_matches:
                    score += len(pattern_matches) * (1.0 / len(patterns))
                    matches.extend(pattern_matches)
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'matches': matches
                }
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
            classification['primary_intent'] = primary_intent
            classification['confidence'] = min(intent_scores[primary_intent]['score'], 1.0)
            
            # Add secondary intents
            for intent, data in intent_scores.items():
                if intent != primary_intent and data['score'] > 0.3:
                    classification['secondary_intents'].append(intent)
        
        # Extract Java versions
        java_versions = re.findall(r'java\s*(\d+(?:\.\d+)*)', query_lower)
        jdk_versions = re.findall(r'jdk\s*(\d+(?:\.\d+)*)', query_lower)
        classification['java_versions'] = list(set(java_versions + jdk_versions))
        
        # Extract entities (keywords)
        entity_patterns = [
            r'class(?:es)?', r'method(?:s)?', r'interface(?:s)?', r'package(?:s)?',
            r'annotation(?:s)?', r'generic(?:s)?', r'lambda(?:s)?', r'stream(?:s)?',
            r'module(?:s)?', r'record(?:s)?', r'switch', r'pattern', r'sealed'
        ]
        
        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, query_lower)
            entities.extend(matches)
        classification['entities'] = list(set(entities))
        
        # Determine query type
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
            classification['query_type'] = 'informational'
        elif any(word in query_lower for word in ['should', 'can', 'may', 'could']):
            classification['query_type'] = 'advisory'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            classification['query_type'] = 'comparative'
        elif any(word in query_lower for word in ['fix', 'solve', 'resolve', 'debug']):
            classification['query_type'] = 'procedural'
        
        # Determine urgency
        urgency_indicators = {
            'critical': ['urgent', 'critical', 'emergency', 'asap', 'immediately'],
            'high': ['important', 'priority', 'soon', 'quickly'],
            'low': ['later', 'whenever', 'eventually', 'sometime']
        }
        
        for level, indicators in urgency_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                classification['urgency'] = level
                break
        
        # Determine complexity
        complexity_indicators = {
            'complex': ['multiple', 'various', 'several', 'complex', 'advanced', 'detailed'],
            'simple': ['simple', 'basic', 'quick', 'brief', 'short']
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                classification['complexity'] = level
                break
        
        return classification

    def generate_response(self, query: str, context: List[Dict], intent: Dict) -> str:
        """Generate enhanced response using OpenRouter with improved context awareness"""
        
        # Build context string with better formatting
        context_parts = []
        for i, doc in enumerate(context):
            doc_type = doc.get('type', 'unknown')
            similarity = doc.get('similarity_score', 0)
            
            context_header = f"[Context {i+1} - {doc_type.title()} (Relevance: {similarity:.2f})]"
            context_parts.append(f"{context_header}\n{doc['text']}\n")
        
        context_str = "\n".join(context_parts)
        
        # Enhanced system prompt based on intent
        system_prompts = {
            'conflict_analysis': """You are a Java specification expert specializing in conflict analysis. 
            Analyze conflicts between different Java versions, identify incompatibilities, and provide resolution strategies.
            Focus on practical implications and migration paths.""",
            
            'version_comparison': """You are a Java version comparison specialist. 
            Compare features, syntax, and behaviors across Java versions. Highlight key differences and evolution patterns.
            Provide migration guidance and compatibility considerations.""",
            
            'rule_lookup': """You are a Java specification rule expert. 
            Provide precise rule interpretations, cite specific sections, and explain compliance requirements.
            Include practical examples and edge cases.""",
            
            'explanation': """You are a Java concept explainer. 
            Provide clear, comprehensive explanations with examples. Break down complex topics into understandable parts.
            Use analogies and practical scenarios to illustrate concepts.""",
            
            'troubleshooting': """You are a Java troubleshooting expert. 
            Diagnose problems, provide step-by-step solutions, and suggest preventive measures.
            Include debugging techniques and common pitfalls.""",
            
            'best_practices': """You are a Java best practices advisor. 
            Provide actionable recommendations, explain trade-offs, and suggest optimal approaches.
            Include industry standards and proven patterns."""
        }
        
        system_prompt = system_prompts.get(
            intent['primary_intent'], 
            "You are a comprehensive Java specification expert assistant."
        )
        
        # Build user prompt with intent-aware formatting
        user_prompt = f"""
Query Intent Analysis:
- Primary Intent: {intent['primary_intent']}
- Query Type: {intent['query_type']}
- Complexity: {intent['complexity']}
- Java Versions Mentioned: {', '.join(intent['java_versions']) if intent['java_versions'] else 'None'}
- Entities: {', '.join(intent['entities']) if intent['entities'] else 'None'}

User Query: {query}

Relevant Context:
{context_str}

Please provide a comprehensive response based on the context and intent analysis. 
Structure your response appropriately for the identified intent and complexity level.
If conflicts are mentioned in the context, highlight them prominently.
Include practical examples where relevant.
"""
        return self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    
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
            # "openai/gpt-4.1-mini",
            # "anthropic/claude-3.5-haiku",
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
                        "max_tokens": self.config.max_tokens_per_call,
                        "temperature": self.config.temperature
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

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Main query answering method with comprehensive response"""
        
        # Classify query intent
        intent = self.classify_query_intent(query)
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(query)
        
        # Generate response
        response = self.generate_response(query, context, intent)
        
        # Add to conversation history
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent,
            'context_count': len(context),
            'response': response
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            'response': response,
            'intent': intent,
            'context': context,
            'metadata': {
                'context_sources': [doc['type'] for doc in context],
                'java_versions_involved': intent['java_versions'],
                'confidence': intent['confidence']
            }
        }

    def generate_conflict_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive conflict analysis report with interactive data"""
        
        conflicts = self.knowledge_base.get('conflicts', [])
        detailed_conflicts = self.knowledge_base.get('conflict_report', {}).get('detailed_conflicts', [])
        
        if not conflicts and not detailed_conflicts:
            return {'error': 'No conflicts found in knowledge base'}
        
        # Combine all conflicts
        all_conflicts = conflicts + detailed_conflicts
        
        # Analyze conflicts by various dimensions
        analysis = {
            'total_conflicts': len(all_conflicts),
            'by_severity': {},
            'by_java_version': {},
            'by_type': {},
            'by_affected_scenarios': {},
            'resolution_status': {},
            'timeline_data': [],
            'network_data': {'nodes': [], 'edges': []},
            'recommendations': []
        }
        
        # Process each conflict
        for i, conflict in enumerate(all_conflicts):
            severity = conflict.get('severity', 'unknown')
            conflict_type = conflict.get('type', 'unknown')
            
            # Count by severity
            analysis['by_severity'][severity] = analysis['by_severity'].get(severity, 0) + 1
            
            # Count by type
            analysis['by_type'][conflict_type] = analysis['by_type'].get(conflict_type, 0) + 1
            
            # Count by Java versions
            versions = []
            if 'rule1_version' in conflict:
                versions.append(str(conflict['rule1_version']))
            if 'rule2_version' in conflict:
                versions.append(str(conflict['rule2_version']))
            if 'java_versions' in conflict:
                versions.extend([str(v) for v in conflict['java_versions']])
            
            for version in set(versions):
                analysis['by_java_version'][version] = analysis['by_java_version'].get(version, 0) + 1
            
            # Count affected scenarios
            scenarios = conflict.get('affected_scenarios', [])
            for scenario in scenarios:
                analysis['by_affected_scenarios'][scenario] = analysis['by_affected_scenarios'].get(scenario, 0) + 1
            
            # Timeline data
            detected_at = conflict.get('detected_at', datetime.now().isoformat())
            analysis['timeline_data'].append({
                'date': detected_at,
                'conflict_id': i,
                'severity': severity,
                'type': conflict_type,
                'versions': versions
            })
            
            # Network data for visualization
            if len(versions) >= 2:
                for j, v1 in enumerate(versions):
                    # Add nodes
                    if not any(node['id'] == v1 for node in analysis['network_data']['nodes']):
                        analysis['network_data']['nodes'].append({
                            'id': v1,
                            'label': f'Java {v1}',
                            'group': 'version'
                        })
                    
                    for k, v2 in enumerate(versions[j+1:], j+1):
                        if not any(node['id'] == v2 for node in analysis['network_data']['nodes']):
                            analysis['network_data']['nodes'].append({
                                'id': v2,
                                'label': f'Java {v2}',
                                'group': 'version'
                            })
                        
                        # Add edges
                        analysis['network_data']['edges'].append({
                            'from': v1,
                            'to': v2,
                            'label': conflict_type,
                            'color': {'color': '#ff0000' if severity == 'high' else '#ffaa00' if severity == 'medium' else '#ffff00'},
                            'width': 3 if severity == 'high' else 2 if severity == 'medium' else 1
                        })
        
        # Generate recommendations
        if analysis['by_severity'].get('high', 0) > 0:
            analysis['recommendations'].append({
                'priority': 'high',
                'title': 'Critical Conflicts Require Immediate Attention',
                'description': f"Found {analysis['by_severity']['high']} high-severity conflicts that need immediate resolution.",
                'action': 'Review and resolve high-severity conflicts first'
            })
        
        if len(analysis['by_java_version']) > 2:
            analysis['recommendations'].append({
                'priority': 'medium',
                'title': 'Multiple Java Version Conflicts',
                'description': f"Conflicts span across {len(analysis['by_java_version'])} Java versions.",
                'action': 'Consider creating version-specific migration guides'
            })
        
        most_affected_scenario = max(analysis['by_affected_scenarios'].items(), key=lambda x: x[1])[0] if analysis['by_affected_scenarios'] else None
        if most_affected_scenario:
            analysis['recommendations'].append({
                'priority': 'medium',
                'title': f'Focus on {most_affected_scenario} Scenarios',
                'description': f"The '{most_affected_scenario}' scenario is most frequently affected by conflicts.",
                'action': f'Prioritize testing and documentation for {most_affected_scenario} scenarios'
            })
        
        return analysis

    def create_interactive_visualizations(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive visualizations for the conflict analysis"""
        
        visualizations = {}
        
        # 1. Severity Distribution Pie Chart
        if analysis_data.get('by_severity'):
            severity_fig = px.pie(
                values=list(analysis_data['by_severity'].values()),
                names=list(analysis_data['by_severity'].keys()),
                title="Conflicts by Severity Level",
                color_discrete_map={
                    'high': '#ff4444',
                    'medium': '#ffaa44',
                    'low': '#44ff44',
                    'unknown': '#888888'
                }
            )
            severity_fig.update_traces(textposition='inside', textinfo='percent+label')
            visualizations['severity_distribution'] = severity_fig
        
        # 2. Java Version Conflicts Bar Chart
        if analysis_data.get('by_java_version'):
            version_fig = px.bar(
                x=list(analysis_data['by_java_version'].keys()),
                y=list(analysis_data['by_java_version'].values()),
                title="Conflicts by Java Version",
                labels={'x': 'Java Version', 'y': 'Number of Conflicts'},
                color=list(analysis_data['by_java_version'].values()),
                color_continuous_scale='Reds'
            )
            version_fig.update_layout(showlegend=False)
            visualizations['version_conflicts'] = version_fig
        
        # 3. Conflict Types Horizontal Bar Chart
        if analysis_data.get('by_type'):
            type_fig = px.bar(
                x=list(analysis_data['by_type'].values()),
                y=list(analysis_data['by_type'].keys()),
                orientation='h',
                title="Conflicts by Type",
                labels={'x': 'Number of Conflicts', 'y': 'Conflict Type'},
                color=list(analysis_data['by_type'].values()),
                color_continuous_scale='Blues'
            )
            visualizations['conflict_types'] = type_fig
        
        # 4. Timeline Analysis
        if analysis_data.get('timeline_data'):
            timeline_df = pd.DataFrame(analysis_data['timeline_data'])
            timeline_df['date'] = pd.to_datetime(timeline_df['date'])
            timeline_df['month'] = timeline_df['date'].dt.to_period('M')
            
            monthly_counts = timeline_df.groupby(['month', 'severity']).size().reset_index(name='count')
            monthly_counts['month'] = monthly_counts['month'].astype(str)
            
            timeline_fig = px.bar(
                monthly_counts,
                x='month',
                y='count',
                color='severity',
                title="Conflict Detection Timeline",
                labels={'month': 'Month', 'count': 'Number of Conflicts'},
                color_discrete_map={
                    'high': '#ff4444',
                    'medium': '#ffaa44',
                    'low': '#44ff44',
                    'unknown': '#888888'
                }
            )
            visualizations['timeline'] = timeline_fig
        
        # 5. Affected Scenarios Treemap
        if analysis_data.get('by_affected_scenarios'):
            scenarios_fig = px.treemap(
                names=list(analysis_data['by_affected_scenarios'].keys()),
                values=list(analysis_data['by_affected_scenarios'].values()),
                title="Affected Scenarios Distribution"
            )
            visualizations['affected_scenarios'] = scenarios_fig
        
        # 6. Network Graph (using networkx and plotly)
        if analysis_data.get('network_data', {}).get('nodes'):
            G = nx.Graph()
            
            # Add nodes
            for node in analysis_data['network_data']['nodes']:
                G.add_node(node['id'], label=node['label'])
            
            # Add edges
            for edge in analysis_data['network_data']['edges']:
                G.add_edge(edge['from'], edge['to'], label=edge['label'])
            
            # Create layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Extract node and edge traces
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=[G.nodes[node]['label'] for node in G.nodes()],
                textposition="middle center",
                marker=dict(size=30, color='lightblue', line=dict(width=2, color='black')),
                name='Java Versions'
            )
            
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='red'),
                        showlegend=False
                    )
                )
            
            network_fig = go.Figure(data=[node_trace] + edge_traces)
            network_fig.update_layout(
                title="Java Version Conflict Network",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Network showing conflicts between Java versions",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            visualizations['network_graph'] = network_fig
        
        return visualizations

    def export_analysis_report(self, format_type: str = 'json') -> str:
        """Export analysis report in various formats"""
        
        analysis = self.generate_conflict_analysis_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == 'json':
            filename = f"conflict_analysis_{timestamp}.json"
            filepath = os.path.join(self.config.project_path, 'reports', filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            return filepath
        
        elif format_type.lower() == 'csv':
            # Convert conflict data to CSV format
            filename = f"conflict_analysis_{timestamp}.csv"
            filepath = os.path.join(self.config.project_path, 'reports', filename)
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare data for CSV
            csv_data = []
            conflicts = self.knowledge_base.get('conflicts', [])
            
            for i, conflict in enumerate(conflicts):
                csv_data.append({
                    'conflict_id': i,
                    'type': conflict.get('type', ''),
                    'severity': conflict.get('severity', ''),
                    'rule1_version': conflict.get('rule1_version', ''),
                    'rule2_version': conflict.get('rule2_version', ''),
                    'description': conflict.get('description', ''),
                    'affected_scenarios': ', '.join(conflict.get('affected_scenarios', [])),
                    'resolution_needed': conflict.get('resolution_needed', ''),
                    'detected_at': conflict.get('detected_at', '')
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False)
            
            return filepath
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_documents': 0,
            'by_type': {},
            'java_versions': set(),
            'total_conflicts': 0,
            'high_severity_conflicts': 0,
            'last_updated': None
        }
        
        # Count documents in embeddings cache
        if self.embeddings_cache and 'documents' in self.embeddings_cache:
            documents = self.embeddings_cache['documents']
            stats['total_documents'] = len(documents)
            
            for doc in documents:
                doc_type = doc.get('type', 'unknown')
                stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
                
                if doc.get('java_version'):
                    stats['java_versions'].add(str(doc['java_version']))
        
        # Count conflicts
        conflicts = self.knowledge_base.get('conflicts', [])
        detailed_conflicts = self.knowledge_base.get('conflict_report', {}).get('detailed_conflicts', [])
        
        all_conflicts = conflicts + detailed_conflicts
        stats['total_conflicts'] = len(all_conflicts)
        stats['high_severity_conflicts'] = sum(1 for c in all_conflicts if c.get('severity') == 'high')
        
        # Get last updated time
        if self.embeddings_cache and 'created_at' in self.embeddings_cache:
            stats['last_updated'] = self.embeddings_cache['created_at']
        
        stats['java_versions'] = list(stats['java_versions'])
        
        return stats

    def search_knowledge_base(self, search_term: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search knowledge base with optional filters"""
        if not self.embeddings_cache or 'documents' not in self.embeddings_cache:
            return []
        
        results = []
        documents = self.embeddings_cache['documents']
        search_term_lower = search_term.lower()
        
        for doc in documents:
            # Text search
            if search_term_lower in doc.get('text', '').lower():
                match_score = doc.get('text', '').lower().count(search_term_lower)
                
                # Apply filters if provided
                if filters:
                    if 'doc_type' in filters and doc.get('type') != filters['doc_type']:
                        continue
                    if 'java_version' in filters and str(doc.get('java_version')) != str(filters['java_version']):
                        continue
                    if 'severity' in filters and doc.get('severity') != filters['severity']:
                        continue
                
                doc_result = doc.copy()
                doc_result['match_score'] = match_score
                results.append(doc_result)
        
        # Sort by match score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def initialize_agent():
    """Initialize the Java Spec Agent"""
    if 'agent' not in st.session_state:
        try:
            # Get API key from env
            api_key = os.getenv('OPENROUTER_API_KEY')
            
            if not api_key:
                st.error("❌ OPENROUTER_API_KEY not found in environment variables")
                st.session_state.agent = None
                return
            
            # Initialize configuration
            config = AgentConfig()
            
            # Initialize agent
            with st.spinner("🚀 Initializing SpecSentinel: Java Specification Agent..."):
                st.session_state.agent = JavaSpecAgent(config, api_key)
                
            st.success("✅ Agent initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.agent = None

def display_knowledge_base_stats():
    """Display knowledge base statistics in sidebar"""
    if not st.session_state.agent:
        return
    
    try:
        # Try multiple approaches to get stats
        stats = st.session_state.agent.get_knowledge_base_stats()
        
        # Also try to access knowledge base directly
        kb = getattr(st.session_state.agent, 'knowledge_base', {})
        
        # Helper function to safely extract metric values
        def safe_metric_value(value, default=0):
            if isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                try:
                    return int(value)
                except:
                    return value
            elif isinstance(value, list):
                return len(value)
            elif isinstance(value, dict):
                return len(value)
            else:
                return default
        
        # Try to get more accurate counts
        total_docs = safe_metric_value(stats.get('total_documents', len(kb.get('documents', []))))
        st.metric("Total Documents", total_docs)
        
        # Check for chunks in different possible locations
        total_chunks = 0
        chunk_sources = [
            stats.get('total_chunks', 0),
            len(kb.get('chunks', [])),
            len(kb.get('processed_rules', [])),
            sum(len(doc.get('chunks', [])) for doc in kb.get('documents', []) if isinstance(doc, dict))
        ]
        total_chunks = max(chunk_sources)
        st.metric("Total Chunks", total_chunks)
        
        # Check for conflicts in different possible locations
        conflicts = 0
        conflict_sources = [
            stats.get('conflict_count', 0),
            stats.get('conflicts', 0),
            len(kb.get('conflicts', [])),
            kb.get('conflicts', {}).get('total_conflicts', 0) if isinstance(kb.get('conflicts'), dict) else 0
        ]
        conflicts = max(conflict_sources)
        
        # If still 0, try to count conflicts from other possible structures
        if conflicts == 0:
            # Check if conflicts are stored differently
            if 'conflict_analysis' in kb:
                conflicts = len(kb['conflict_analysis'])
            elif 'detected_conflicts' in kb:
                conflicts = len(kb['detected_conflicts'])
            elif hasattr(st.session_state.agent, 'conflicts'):
                conflicts = len(getattr(st.session_state.agent, 'conflicts', []))
        
        st.metric("Detected Conflicts", conflicts)
        
        # Handle Java versions specially
        java_versions = stats.get('java_versions', kb.get('java_versions', []))
        
        # Filter and clean Java versions
        valid_versions = []
        if isinstance(java_versions, list):
            for version in java_versions:
                # Try to extract valid Java version numbers
                if isinstance(version, (int, float)):
                    if 1 <= version <= 50:  # Reasonable Java version range
                        valid_versions.append(int(version))
                elif isinstance(version, str):
                    # Try to extract numeric version from string
                    # Handle formats like "Java 17", "JDK-11", "1.8", etc.
                    import re
                    # Look for numeric patterns
                    numeric_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', version)
                    for match in numeric_matches:
                        try:
                            num_version = float(match)
                            # Convert 1.x format to x (e.g., 1.8 -> 8)
                            if 1.0 <= num_version < 2.0:
                                num_version = int((num_version - 1) * 10 + 1)
                            elif num_version >= 8:  # Modern Java versions
                                num_version = int(num_version)
                            else:
                                continue
                            
                            if 1 <= num_version <= 50:  # Reasonable range
                                valid_versions.append(num_version)
                                break  # Take first valid match
                        except (ValueError, TypeError):
                            continue
        
        # Remove duplicates and sort
        valid_versions = sorted(list(set(valid_versions)))
        
        if valid_versions:
            st.metric("Java Versions", len(valid_versions))
            # Show versions in a compact format
            versions_str = ', '.join(str(v) for v in valid_versions[:8])
            if len(valid_versions) > 8:
                versions_str += f" +{len(valid_versions) - 8} more"
            st.caption(f"Versions: {versions_str}")
        else:
            # Fallback if no valid versions found
            original_count = len(java_versions) if isinstance(java_versions, list) else safe_metric_value(java_versions)
            st.metric("Java Versions", original_count)
            if isinstance(java_versions, list) and java_versions:
                st.caption(f"Raw: {', '.join(str(v) for v in java_versions[:3])}{'...' if len(java_versions) > 3 else ''}")
        
        # Additional stats if available
        specs_count = 0
        if 'specifications' in stats:
            specs_count = safe_metric_value(stats['specifications'])
        elif 'specifications' in kb:
            specs_count = safe_metric_value(kb['specifications'])
        
        if specs_count > 0:
            st.metric("Specifications Loaded", specs_count)
            
    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")
        
        # Enhanced fallback - try to access agent attributes directly
        try:
            kb = getattr(st.session_state.agent, 'knowledge_base', {})
            if kb:
                st.metric("Status", "Loaded")
                st.metric("KB Keys", len(kb.keys()))
            else:
                st.metric("Status", "Empty")
        except:
            st.metric("Status", "Error")

def handle_query_response(prompt: str):
    """Handle user query and generate response"""
    try:
        # Get agent response
        response = st.session_state.agent.answer_query(prompt)
        
        # Create assistant message structure
        assistant_message = {
            "role": "assistant", 
            "content": response.get('response', response.get('text_response', 'No response generated')),
            "context_used": response.get('context_used', []),
            "intent": response.get('intent', {}),
            "visualizations": response.get('visualizations', []),
            "interactive_plots": response.get('interactive_plots', {}),
            "detailed_report": response.get('detailed_report', ''),
            "metadata": response.get('metadata', {})
        }
        
        return assistant_message
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return {
            "role": "assistant",
            "content": f"I encountered an error while processing your query: {str(e)}",
            "error": True
        }

def display_response_components(response: Dict[str, Any]):
    """Display various components of the agent response"""
    
    # Display main text response
    st.markdown(response['content'])
    
    # Display context information if available
    if response.get('context_used') and st.checkbox("Show Context Used", key=f"context_{id(response)}"):
        with st.expander("📚 Retrieved Context"):
            for i, context in enumerate(response['context_used']):
                st.write(f"**Source {i+1}:** {context.get('source', 'Unknown')}")
                st.write(f"**Relevance:** {context.get('similarity', 'N/A')}")
                st.write(f"**Content:** {context.get('content', '')[:200]}...")
                st.divider()
    
    # Display visualizations
    if response.get('visualizations'):
        st.subheader("📊 Visualizations")
        cols = st.columns(min(len(response['visualizations']), 3))
        for i, img_b64 in enumerate(response['visualizations']):
            with cols[i % 3]:
                st.image(f"data:image/png;base64,{img_b64}")
    
    # Display interactive plots
    if response.get('interactive_plots'):
        st.subheader("📈 Interactive Charts")
        for plot_name, fig in response['interactive_plots'].items():
            # Create unique key for each plot
            unique_key = f"plotly_chart_{id(response)}_{plot_name}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Display detailed report
    if response.get('detailed_report'):
        with st.expander("📋 View Detailed Report"):
            st.markdown(response['detailed_report'])

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="SpecSentinel: Java Specification Agent",
        page_icon="☕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .statistics-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">☕ SpecSentinel: Java Specification Agent</h1>', unsafe_allow_html=True)
    st.markdown("*Intelligent analysis of Java Language Specifications with conflict detection and reporting*")
    
    # Initialize agent
    initialize_agent()
    
    if not st.session_state.agent:
        st.error("Agent not initialized. Please check your configuration and API key.")
        st.info("Make sure OPENROUTER_API_KEY is set in your environment variables.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Agent Status")
        
        # Knowledge base status
        display_knowledge_base_stats()
        
        st.header("🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Statistics", use_container_width=True):
                st.session_state.show_stats = True
                st.rerun()
        
        with col2:
            if st.button("📋 Report", use_container_width=True):
                st.session_state.show_report = True
                st.rerun()
        
        if st.button("🔍 Search KB", use_container_width=True):
            st.session_state.show_search = True
            st.rerun()
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Export options
        st.header("📤 Export")
        export_format = st.selectbox("Format", ["json", "markdown", "html"])
        if st.button("Export Report", use_container_width=True):
            try:
                exported_data = st.session_state.agent.export_analysis_report(export_format)
                st.download_button(
                    label=f"Download {export_format.upper()}",
                    data=exported_data,
                    file_name=f"java_spec_analysis.{export_format}",
                    mime=f"application/{export_format}"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
        
        st.header("ℹ️ About")
        st.info("""
        This agent analyzes Java Language Specifications to:
        - Detect conflicts between versions
        - Generate comprehensive reports
        - Provide intelligent Q&A
        - Create visualizations
        - Search knowledge base
        """)
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_response_components(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me about Java specifications, conflicts, or request statistics..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query..."):
                assistant_message = handle_query_response(prompt)
            
            # Display response components
            display_response_components(assistant_message)
            
            # Add assistant message to chat history
            st.session_state.messages.append(assistant_message)
    
    # Handle quick actions
    if st.session_state.get('show_stats', False):
        st.session_state.show_stats = False
        
        with st.chat_message("assistant"):
            with st.spinner("📊 Generating statistics..."):
                try:
                    # Generate comprehensive statistics
                    response = st.session_state.agent.answer_query(
                        "Show me comprehensive statistics and visualizations about the Java specifications"
                    )
                    assistant_message = {
                        "role": "assistant",
                        "content": response.get('response', response.get('text_response', 'Statistics generated')),
                        "visualizations": response.get('visualizations', []),
                        "interactive_plots": response.get('interactive_plots', {}),
                        "detailed_report": response.get('detailed_report', '')
                    }
                    
                    # Display components
                    display_response_components(assistant_message)
                    
                    # Add to message history
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    st.error(f"Error generating statistics: {str(e)}")
    
    if st.session_state.get('show_report', False):
        st.session_state.show_report = False
        
        with st.chat_message("assistant"):
            with st.spinner("📋 Generating detailed report..."):
                try:
                    # Generate conflict analysis report
                    report_data = st.session_state.agent.generate_conflict_analysis_report()
                    
                    # Format the report
                    report_content = "## 📋 Comprehensive Conflict Analysis Report\n\n"
                    
                    if isinstance(report_data, dict):
                        if 'summary' in report_data:
                            report_content += f"### Summary\n{report_data['summary']}\n\n"
                        
                        if 'conflicts' in report_data:
                            report_content += f"### Detected Conflicts\n"
                            conflicts = report_data['conflicts']
                            if isinstance(conflicts, list):
                                for i, conflict in enumerate(conflicts, 1):
                                    report_content += f"**Conflict {i}:** {conflict}\n"
                            else:
                                report_content += f"{conflicts}\n"
                            report_content += "\n"
                        
                        if 'recommendations' in report_data:
                            report_content += f"### Recommendations\n{report_data['recommendations']}\n\n"
                    else:
                        report_content += str(report_data)
                    
                    st.markdown(report_content)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": report_content
                    })
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    if st.session_state.get('show_search', False):
        st.session_state.show_search = False
        
        # Knowledge base search interface
        st.subheader("🔍 Knowledge Base Search")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search term:", key="kb_search")
        with col2:
            search_btn = st.button("Search", key="search_submit")
        
        if search_btn and search_term:
            try:
                with st.spinner("Searching knowledge base..."):
                    results = st.session_state.agent.search_knowledge_base(search_term)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1}: {result.get('source', 'Unknown source')}"):
                            st.write(f"**Relevance:** {result.get('similarity', 'N/A')}")
                            st.write(f"**Content:** {result.get('content', 'No content')}")
                else:
                    st.info("No results found for your search term.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    
    # Conversation history management
    if len(st.session_state.messages) > 0:
        with st.sidebar:
            st.header("💬 Conversation")
            if st.button("📥 Export Chat"):
                chat_data = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages
                }
                st.download_button(
                    label="Download Chat History",
                    data=json.dumps(chat_data, indent=2),
                    file_name="chat_history.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
