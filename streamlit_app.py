# streamlit_app.py

# Include ALL necessary imports except those we'll import lazily
import os
import json
import time
import pickle
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
from google.colab import userdata # Needed to get secrets in Streamlit on Colab
# import torch # Keep this if other parts of your code use torch globally


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
        """Load sentence transformer model for embeddings"""
        print("📦 Loading embedding model...")
        try:
            # IMPORT SentenceTransformer *inside* the method
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            # Consider adding a fallback or raising an error
            self.embedding_model = None # Ensure it's None on failure


    def _load_knowledge_base(self):
        """Load existing knowledge base from SpecSentinel results"""
        kb_files = {
            'specifications': f'{self.config.project_path}/data/all_specifications.json',
            'processed_rules': f'{self.config.project_path}/data/processed_rules.json',
            'conflicts': f'{self.config.project_path}/results/conflict_report.json',
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
        self.knowledge_base.setdefault('conflicts', {'detailed_conflicts': []})
        self.knowledge_base.setdefault('specifications', {})

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
                        # Optional: Add a check here if the loaded documents match the current knowledge base
                        # If they don't match, rebuild the index.
                        return
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
            # Add a hash or timestamp of the knowledge base files to check for staleness later
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
                else:
                    # print(f"   Skipping empty or invalid rule document: {rule.get('id', 'N/A')}")
                    pass # Keep output cleaner


        # Process conflicts
        detailed_conflicts = self.knowledge_base.get('conflicts', {}).get('detailed_conflicts', [])
        if detailed_conflicts:
             print(f"   Preparing {len(detailed_conflicts)} conflicts for indexing...")
             for i, conflict in enumerate(detailed_conflicts):
                conflict_text = f"""
                CONFLICT: {conflict.get('type', 'Unknown')}
                Severity: {conflict.get('severity', 'Unknown')}
                Description: {conflict.get('description', '')}
                Affected Scenarios: {' '.join(conflict.get('affected_scenarios', []))}
                Resolution Needed: {conflict.get('resolution_needed', '')}
                """

                doc = {
                    'id': f"conflict_{i}_{conflict.get('rule1_version', 'unk')}_{conflict.get('rule2_version', 'unk')}", # More unique ID
                    'text': conflict_text.strip(),
                    'type': 'conflict',
                    'java_versions': [conflict.get('rule1_version'), conflict.get('rule2_version')],
                    'severity': conflict.get('severity', 'unknown'),
                    'metadata': conflict # Store original metadata
                }
                # Ensure the text is not empty before adding and is a string
                if isinstance(doc['text'], str) and doc['text'].strip():
                    documents.append(doc)
                else:
                     # print(f"   Skipping empty or invalid conflict document: {conflict.get('id', 'N/A')}")
                     pass # Keep output cleaner


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
                        else:
                             # print(f"   Skipping empty chunk from spec {spec_key}")
                             pass # Keep output cleaner

                else:
                    # print(f"   Skipping empty or invalid specification document: {spec_key}")
                    pass # Keep output cleaner


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
            "anthropic/claude-3.5-haiku-20241022",  # Fast and cheap
            "openai/gpt-4o-mini-2024-07-18",       # Cost-effective
            "deepseek/deepseek-chat-v3-0324:free", # Free backup
            "meta-llama/llama-3.2-3b-instruct:free" # Free backup
        ]

        for model in models:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/java-spec-agent",
                        "X-Title": "Java Specification Agent"
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
        """Classify user query intent"""
        query_lower = query.lower()

        # Greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in query_lower for pattern in greeting_patterns):
            return 'greeting'

        # Farewell patterns
        farewell_patterns = ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit']
        if any(pattern in query_lower for pattern in farewell_patterns):
            return 'farewell'

        # Conflict-related queries
        conflict_patterns = ['conflict', 'contradiction', 'ambiguity', 'inconsistency', 'problem', 'issue']
        if any(pattern in query_lower for pattern in conflict_patterns):
            return 'conflict_inquiry'

        # Specification queries
        spec_patterns = ['specification', 'rule', 'section', 'jls', 'java language specification']
        if any(pattern in query_lower for pattern in spec_patterns):
            return 'specification_inquiry'

        # Version comparison
        version_patterns = ['java 8', 'java 11', 'java 17', 'java 21', 'version', 'difference', 'change']
        if any(pattern in query_lower for pattern in version_patterns):
            return 'version_comparison'

        return 'general_inquiry'

    def generate_response(self, query: str) -> str:
        """Generate response using RAG and LLM"""
        intent = self.classify_query_intent(query)

        # Handle simple interactions without API calls
        if intent == 'greeting':
            return "Hello! I'm your Java Language Specification assistant. I can help you with JLS rules, conflicts between versions, and specification details. What would you like to know?"

        if intent == 'farewell':
            return "Goodbye! Feel free to come back anytime you have questions about Java specifications. Have a great day!"

        # For complex queries, use RAG
        relevant_docs = self.retrieve_relevant_context(query)

        # Build context string for the prompt
        context_parts = []
        conflict_warnings = []

        if not relevant_docs:
             context = "No relevant documents found in knowledge base."
             print("   No relevant documents found for query.")
        else:
            print(f"   Found {len(relevant_docs)} relevant documents.")
            for doc in relevant_docs:
                # Add score for debugging/analysis if needed
                # print(f"      - {doc.get('type', 'unknown')} (Score: {doc.get('similarity_score', 'N/A'):.4f})")
                if doc['type'] == 'conflict':
                    warning_text = doc['metadata'].get('description', 'Unknown conflict')
                    conflict_warnings.append(f"⚠️ CONFLICT ALERT: {warning_text}")
                    context_parts.append(f"CONFLICT (Severity: {doc['metadata'].get('severity', 'Unknown')}): {doc.get('text', '')}")
                elif doc['type'] == 'rule':
                    context_parts.append(f"RULE (Java {doc.get('java_version', 'unknown')}, Section {doc.get('section', 'unknown')}, Category: {doc.get('category', 'unknown')}): {doc.get('text', '')}")
                elif doc['type'] == 'specification':
                    # Limit specification text length in context
                    spec_text = doc.get('text', '')
                    context_parts.append(f"SPECIFICATION (Java {doc.get('java_version', 'unknown')}, Section: {doc.get('section', 'unknown')}): {spec_text[:500]}...")

            context = "\n\n".join(context_parts[:self.config.max_contexts])  # Limit context parts


        # Build messages list for LLM call (using system and user roles)
        system_prompt = f"""You are a Java Language Specification expert and helpful assistant. Answer the user's question accurately and concisely, using the provided context from the official JLS whenever relevant.

Prioritize information from 'RULE' and 'CONFLICT' contexts. If the user asks about conflicts, emphasize the conflict details.
Always mention specific Java versions or sections if they are present in the context or the user's query.
If you find conflicts in the provided context, make sure to include the conflict warnings (if any were provided) and explain the conflict in your answer.
If you don't have enough context to answer accurately, state that you can only use the available knowledge base and suggest the user refine their query or provide more context.

Relevant Context:
{context}
"""
        messages = [{"role": "system", "content": system_prompt}]

        # Optionally add conversation history (simple approach)
        # You might want to summarize or limit history for longer conversations
        # for history in self.conversation_history[-5:]: # Limit history
        #    messages.append({"role": "user", "content": history['user']})
        #    messages.append({"role": "assistant", "content": history['agent']})

        messages.append({"role": "user", "content": query})


        # Generate response
        response = self._call_llm(messages, max_tokens=800)


        # Add conflict warnings if found at the beginning of the response
        # Only add if they were not implicitly handled by the LLM based on context
        # Simple check: if response doesn't mention conflict warnings already
        if conflict_warnings and "CONFLICT ALERT" not in response:
             response = "\n".join(conflict_warnings) + "\n\n" + response

        return response


    def add_to_knowledge_base(self, content: Dict[str, Any]) -> bool:
        """Add new content to knowledge base (rules or conflicts)"""
        try:
            content_type = content.get('type', 'unknown')

            if content_type == 'rule':
                # Add to processed rules
                if 'processed_rules' not in self.knowledge_base:
                    self.knowledge_base['processed_rules'] = []

                self.knowledge_base['processed_rules'].append(content)

                # Save updated rules
                rules_file = f'{self.config.project_path}/data/processed_rules.json'
                os.makedirs(os.path.dirname(rules_file), exist_ok=True) # Ensure dir exists
                with open(rules_file, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_base['processed_rules'], f, indent=2, ensure_ascii=False) # ensure_ascii=False for non-ASCII chars


            elif content_type == 'conflict':
                # Add to conflicts
                if 'conflicts' not in self.knowledge_base:
                    self.knowledge_base['conflicts'] = {'detailed_conflicts': []}

                if 'detailed_conflicts' not in self.knowledge_base['conflicts']:
                    self.knowledge_base['conflicts']['detailed_conflicts'] = []

                self.knowledge_base['conflicts']['detailed_conflicts'].append(content)

                # Save updated conflicts
                conflicts_file = f'{self.config.project_path}/results/conflict_report.json'
                os.makedirs(os.path.dirname(conflicts_file), exist_ok=True) # Ensure dir exists
                with open(conflicts_file, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_base['conflicts'], f, indent=2, ensure_ascii=False) # ensure_ascii=False for non-ASCII chars


            else:
                 print(f"   ⚠️ Cannot add content of unknown type: {content_type}")
                 return False

            # Rebuild vector index to include new content
            # Do this in a background thread or with a delay if it's slow
            print("   Initiating background vector index rebuild...")
            import threading
            rebuild_thread = threading.Thread(target=self._build_vector_index)
            rebuild_thread.start()
            # rebuild_thread.join() # Don't block the main thread


            print(f"✅ Added new {content_type} to knowledge base")
            return True

        except Exception as e:
            print(f"❌ Error adding to knowledge base: {e}")
            return False


    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_rules': len(self.knowledge_base.get('processed_rules', [])),
            'total_conflicts': len(self.knowledge_base.get('conflicts', {}).get('detailed_conflicts', [])),
            'total_specifications': len(self.knowledge_base.get('specifications', {})),
            'java_versions': set(),
            'categories': set(),
            'last_updated': datetime.now().isoformat()
        }

        # Collect versions and categories
        for rule in self.knowledge_base.get('processed_rules', []):
            stats['java_versions'].add(rule.get('java_version', 'unknown'))
            stats['categories'].add(rule.get('category', 'unknown'))

        # Also collect versions from conflicts
        for conflict in self.knowledge_base.get('conflicts', {}).get('detailed_conflicts', []):
             stats['java_versions'].add(conflict.get('rule1_version', 'unknown'))
             stats['java_versions'].add(conflict.get('rule2_version', 'unknown'))


        stats['java_versions'] = sorted(list(stats['java_versions'])) # Sort for better display
        stats['categories'] = sorted(list(stats['categories'])) # Sort for better display


        return stats


# =============================================================================
# STREAMLIT CHAT INTERFACE
# =============================================================================

# This function will be called by streamlit when running the script
def create_streamlit_interface():
    """Create Streamlit chat interface"""
    st.set_page_config(
        page_title="Java Specification Agent",
        page_icon="☕",
        layout="wide"
    )

    st.title("☕ SpecSentinel: Java Language Specification Agent")
    st.markdown("Your AI assistant for Java specifications, conflicts, and version differences")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.info("Initializing agent and loading knowledge base...")
        # Use google.colab.userdata to get the API key in a Colab environment
        # Fallback to text input if userdata fails or is not available (e.g., local run)
        load_dotenv()
        api_key = None
        try:
            # Try to get the key from Colab secrets first
            api_key = os.getenv("OPENROUTER_API_KEY") 
            if not api_key:
                 # If userdata is available but key isn't set, raise an error to fallback
                 raise ValueError("API key not found in env")
        except Exception as e:
            # If userdata fails (not in Colab or secret not found), ask via text input
            st.error(f"Could not get API key from env: {e}")
            # Use a persistent key for the input widget
            api_key = st.text_input("Enter OpenRouter API Key:", type="password", key="api_key_input")

        # Check if API key is available BEFORE attempting agent initialization
        if api_key:
            try:
                config = AgentConfig()
                st.session_state.agent = JavaSpecAgent(config, api_key)
                st.session_state.messages = []
                st.success("Agent initialized and knowledge base loaded!")
                # Rerun to clear initialization message and password input if successful
                # st.experimental_rerun() # Use st.rerun() in newer versions
                # st.rerun() # Re-running might lose success message, let's skip immediate rerun for now
                
            except Exception as e:
                st.error(f"Error initializing agent: {e}")
                # Clear the api key input state to allow user to try again ONLY IF initialization failed
                
                if 'api_key_input' in st.session_state:
                    st.session_state.api_key_input = ""
                    # Ensure the agent state is cleared if initialization failed
                    if 'agent' in st.session_state:
                        del st.session_state.agent
                        st.stop() # Stop execution if initialization failed
                        
        else:
            st.warning("Please provide your OpenRouter API key to continue.")
            # Do not proceed further if API key is not available
            st.stop()

    # Sidebar with knowledge base stats
    with st.sidebar:
        st.header("📊 Knowledge Base")
        if hasattr(st.session_state, 'agent'):
            stats = st.session_state.agent.get_knowledge_stats()
            st.metric("Rules", stats['total_rules'])
            st.metric("Conflicts", stats['total_conflicts'])
            st.metric("Specifications", stats['total_specifications'])
            st.write("**Java Versions:**", ", ".join(stats['java_versions']))
    
    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Java specifications..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add content forms
    if st.session_state.get('show_add_rule', False):
        with st.form("add_rule_form"):
            st.subheader("Add New Rule")
            rule_text = st.text_area("Rule Text")
            java_version = st.selectbox("Java Version", ["8", "11", "17", "21", "24"])
            section = st.text_input("Section Number")
            category = st.selectbox("Category", [
                "METHOD_RESOLUTION", "TYPE_COMPATIBILITY", "INHERITANCE_RULES",
                "OVERLOADING_RULES", "ACCESS_CONTROL", "EXCEPTION_HANDLING",
                "GENERICS_RULES", "INTERFACE_RULES", "CONSTRUCTOR_RULES", "COMPILATION_RULES"
            ])
            
            if st.form_submit_button("Add Rule"):
                new_rule = {
                    'type': 'rule',
                    'text': rule_text,
                    'java_version': java_version,
                    'section_number': section,
                    'category': category,
                    'added_at': datetime.now().isoformat()
                }
                if st.session_state.agent.add_to_knowledge_base(new_rule):
                    st.success("Rule added successfully!")
                    st.session_state.show_add_rule = False

if __name__ == "__main__":
    create_streamlit_interface()
