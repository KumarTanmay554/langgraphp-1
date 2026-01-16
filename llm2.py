import asyncio
# import pyttsx3
# import speech_recognition as sr
import datetime
import time
import random
import os
import requests
# os.environ["OLLAMA_HOST"] = "http://ollama:11434"
# os.environ["OLLAMA_BASE_URL"] = "http://ollama:11434"
# os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
# os.environ["OLLAMA_API_KEY"] = ""
# import sys
import re
# import ollama
import pandas as pd
from googletrans import Translator
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from loader import load_dealer_csv, doc_load, load_employee_csv
# from embedder import build_faiss_index_csv
from retr import create_retr_csv
from chain import create_rag_chain_csv
from langdetect import detect
import json
# import whisper
# import sounddevice as sd
import numpy as np
# import scipy.io.wavfile as wav
import warnings
from typing import Any, Literal, Optional, TypedDict
# try:
# import edge_tts
#     import pygame
#     pygame.mixer.init()
ADVANCED_TTS = True
print("Using advanced TTS (edge-tts)")
# except ImportError:
#     ADVANCED_TTS = False
#     print("Advanced TTS not available, using pyttsx3")
import phonetics
from typing import Tuple, Dict, List
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH
from langchain_community.vectorstores import FAISS
warnings.filterwarnings('ignore')
from intent import get_query_intent, Intent
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# Suppress ALSA warnings
# os.environ['ALSA_SILENCE'] = '1'

# whisper model load
# whmodel = whisper.load_model("small", device="cpu")
# print("Whisper model loaded")

# language translator
translator = Translator()
tts_engine = None
lang = 'en'  # default language

async def translate_text(text, dest = "hi"):
    res = translator.translate(text,dest=dest)
    if asyncio.iscoroutine(res):
        res = await res
    return res.text

def get_lang_code(lang:str) -> str:
    map={
        "en": "en-US",
        "hi": "hi-IN",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "ta": "ta-IN",
        "te": "te-IN",
        "mr": "mr-IN",
        "bn": "bn-IN",
    }
    return map.get(lang,"en-US")

def detect_and_set_language(text: str):
    global lang
    try:
        l = safe_detect_language(text) or "en"
        lang = l
        print(f"Detected language: {lang}")
    except Exception as e:
        print(f"Language detection failed: {e}")

# Global variables for the RAG chain and memory
qa_chain = None
memory = None
vectorstore = None
dealers_df = None
emp_df = None
last_shown_dealers = []
last_topic = None

last_pincode = None 
category = None

last_requested_property = None
last_selected_dealer = None

last_name = None


# client = Client(
#     host="https://ollama.com",
#     headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
# )


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", 
    "will", "with", "the", "this", "but", "they", "have", "had", "what", "when",
    "where", "who", "which", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "did", "does", "do", "into", "if", "or"
}

STATIC_TERMS = [
    "viman", "kothrud", "baner", "hinjewadi", "wakad", "pimpri", "chinchwad","card", "shops","easy","puncture","yes", "score",
    "hadapsar", "magarpatta", "remi", "E M I", "finance", "stores", "accessories","documents","appliances", "cibil","miss","cancel",
    "service", "car", "E-bike", "two-wheeler", "authorized", "dealer", "dealership", "tyres", "nearest", "eligibility", "pincode","area","payments",
]

def mapping_category(term: str) -> str:
    term = term.lower()
    category_map = {
        "tyres": ["tyre","punture","tube","tubeless","flat","tagar", "wheel"],
        "car-accessories":["car","accesorie"]
    }
    return category_map.get(term, term.title())

CATEGORY_MAPPING = {
    "tyres": [
        "puncture", "tire", "tyre", "wheel", "tubeless", "flat", "tagar", "टायर", "vehicle", "auto", "motorcycle", "पंक्चर", "टायर्स",
        "कार", "बाइक", "गाड़ी", "वाहन", "मोटर", "ऑटो", "पहिया", "पंजर"
    ],
    "Car Accessories":["Accessories","jeep","four wheelers"],
    "Small appliances":["owen","microwave","kettle","iron"],
    "Apparels":["Clothes","shirts","pants","t shirts","kapde","clothing"],
    "Power Backup":["Battery","current","inverter","lithium"],
    "Gym and Spas":["gym","massage","spa","sauna","dumbell","weights"],
    "Paints and Hardware":["Paints","Hardware"],
    "Footwear":["slippers","chappal","sneakers","shoes","footwear","jootey"],
    "Water Purifier":["RO","water","filter","pani","water filter"],
    "REMI":["rainy","ramika","retail emi","retail emi service","lover","premicd"],
    "Vehicle Care / Servicing":["service","vehicle care","car service","bike service","vehicle servicing","repair","maintenance"],
    "E-Bike":["electeric bike", "e-bike"]
}

# category mapping
def map_category_terms(text: str) -> str:
    if not text:
        return text
        
    words = text.lower().split()
    mapped = []
    
    for word in words:
        mapped_term = word
        for category, variations in CATEGORY_MAPPING.items():
            if word in variations:
                mapped_term = category
                print(f"Category mapping: '{word}' -> '{category}'")
                break
        mapped.append(mapped_term)
        
    return " ".join(mapped)

def correct_terms_only(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False

    words = text.lower().split()
    corrected_words = []
    changed = False

    for word in words:
        # Try fuzzy matching first
        try:
            match = rf_process.extractOne(word, STATIC_TERMS, scorer=rf_fuzz.WRatio)
            if match and match[1] >= 86:  # 86% similarity threshold
                corrected_words.append(match[0])
                if match[0] != word:
                    changed = True
                continue
        except Exception:
            pass

        # Try phonetic matching if fuzzy didn't work
        try:
            word_code = dm_code(word)
            if word_code:
                for term in STATIC_TERMS:
                    if dm_code(term) == word_code:
                        corrected_words.append(term)
                        if term != word:
                            changed = True
                        break
                else:  # No phonetic match found
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        except Exception:
            corrected_words.append(word)
    return " ".join(corrected_words), changed

def correction(inp:str, words: List[str], fuzzy_threshold=80, phonetics_th=85):
    # inp = re.sub(r'[^a-zA-Z0-9\s]', '', inp.lower())
    # word = inp.split()


    # correct_words = []
    # for i in word:
    #     # fuz_mat,fuz_sc, _ = rf_process.extractOne(i, words, scorer=rf_fuzz.WRatio)
    #     try:
    #         length = len(i)
    #         len_filter = [j for j in words if abs(len(j)-length)<=2]
    #         candi = len_filter if len_filter else words

    #         match = rf_process.extractOne(i,candi, scorer=rf_fuzz.WRatio)
    #         if match and match[1]>= fuzzy_threshold:
    #             print("match found: ",match)
    #             correct_words.append(match[0])
    #             # continue
    #         else:
    #             correct_words.append(i)
    #     except Exception:
    #         pass
    #     try:
    #         user_ph = phonetics.metaphone(i)

    #         best_ph = i
    #         best_sc = 0

    #         for j in words:
    #             w_ph = phonetics.metaphone(j)
    #             # w_sc = phonetics.dmetaphone(j)
    #             score = rf_fuzz.ratio(user_ph, w_ph)
    #             if score > best_sc:
    #                 best_sc = score
    #                 best_ph = w_ph
    #                 print("Fuzzy score: ",best_sc)

    #         # if fuz_sc >= fuzzy_threshold:
    #         #     correct_words.append(fuz_mat)
    #         if best_sc >= phonetics_th:
    #             correct_words.append(best_ph)
    #         else:
    #             correct_words.append(i)
    #     except Exception:
    #         correct_words.append(i)

    # corrected_sentence = " ".join(correct_words)
    # return corrected_sentence
    if not inp:
        return inp
        
    inp = re.sub(r'[^a-zA-Z0-9\s]', '', inp.lower())
    input_words = inp.split()
    
    corrected_words = []
    for word in input_words:
        if word in STOP_WORDS:
            corrected_words.append(word)
            continue
        try:
            word_len = len(word)

            # length
            length_filtered_terms = [
                term for term in words 
                if abs(len(term.split()[0]) - word_len) <= 2
            ]
            
            # use all the terms
            candidates = length_filtered_terms if length_filtered_terms else words
            
            # fuzzy
            match = rf_process.extractOne(
                word,
                candidates,
                scorer=rf_fuzz.WRatio,
                score_cutoff=fuzzy_threshold
            )

            best_sc = 0
            best_ph = word
            mtype = None

            try:
                words_ph = phonetics.metaphone(word)
                m1,m2 = phonetics.dmetaphone(word)

                for i in candidates:
                    im1,im2 = phonetics.dmetaphone(i)

                    if(m1 and im1 and m1 == im1) or (m2 and im2 and m2 == im2):
                        best_ph = i
                        best_sc = 100
                        mtype = "dmeta"
                        break
                    term_ph = phonetics.metaphone(i)
                    if words_ph == term_ph:
                        best_ph = i
                        best_sc = 95  # Close phonetic match
                        mtype = "phonetic"
                        break
            except Exception:
                pass

            if best_sc <90:
                try:
                    match = rf_process.extractOne(word, candidates, scorer=rf_fuzz.WRatio, score_cutoff=fuzzy_threshold)
                    if match and match[1] > fuzzy_threshold:
                        best_ph = match[0]
                        best_sc = match[1]
                        mtype = "fuzzy"
                except Exception:
                    pass
            if best_sc >= fuzzy_threshold:
                if best_ph != word:
                    print(f"Match found: '{word}' -> '{best_ph}' (score: {best_sc}, type: {mtype})")
                corrected_words.append(best_ph)
            else:
                corrected_words.append(word)
        except Exception as e:
            print(f"Error correcting word '{word}': {e}")
            corrected_words.append(word)
    return " ".join(corrected_words)
    
def normalize_audio_number(text: str) -> str:
    t = str(text).strip()
    if not t:
        return t
    if re.fullmatch(r"[0-9\-\s,\.]+", t):
        return re.sub(r"\D", "", t)
    return t

def normalize_spoken_number_words(text: str) -> str:
    
    if not text:
        return text
    t = str(text).strip().lower()

    devanagari_digit_map = str.maketrans("०१२३४५६७८९", "0123456789")
    t = t.translate(devanagari_digit_map)

    t = re.sub(r"[,\-\.]", " ", t)
    
    tokens = re.findall(r"[a-zA-Z\u0900-\u097F]+|\d+", t)
    if not tokens:
        return text

    hi_words = {
        "शून्य": "0", "सिफर": "0", "जीरो": "0", "ज़ीरो": "0",
        "एक": "1",
        "दो": "2",
        "तीन": "3",
        "चार": "4",
        "पांच": "5", "पाँच": "5",
        "छह": "6",
        "सात": "7",
        "आठ": "8",
        "नौ": "9",
    }

    en_hi_words = {
        "zero": "0", "oh": "0", "o": "0", "shunya": "0", "shoonya": "0", "sunya": "0", "sifr": "0", "sifar": "0",
        "one": "1", "ek": "1",
        "two": "2", "to": "2", "too": "2", "do": "2",
        "three": "3", "teen": "3",
        "four": "4", "for": "4", "chaar": "4", "char": "4",
        "five": "5", "paanch": "5", "panch": "5",
        "six": "6", "chhe": "6", "cheh": "6", "che": "6",
        "seven": "7", "saat": "7", "saath": "7",
        "eight": "8", "aath": "8", "ath": "8", "ate": "8",
        "nine": "9", "nau": "9",
    }

    def to_digit(tok: str) -> str | None:
        if tok.isdigit():
            return tok
        if tok in hi_words:
            return hi_words[tok]
        if tok in en_hi_words:
            return en_hi_words[tok]
        return None

    out = []
    for tok in tokens:
        d = to_digit(tok)
        if d is None:
            
            return text
        out.append(d)
    return "".join(out)

# Rag Configuration
try:
    print("Initializing rag and dealers")
    try:
        df, dealer_docs = load_dealer_csv()
        # print(df.head())
        dealers_df = df
        DOCS_LOADED = True
    except Exception as e:
        print("Could not load dealers:", e)
        dealer_docs = []
        dealers_df = None

    emp_df, emp_docs = load_employee_csv()

    if emp_df is not None:
        print(f"Employee data loaded with {len(emp_df)} records")
        print(emp_df.head())
    else:
        emp_docs = []

    # Use dealer documents
    all_docs = dealer_docs 
    # print(type(all_docs))
    docs = doc_load()
    print(f"Additional documents loaded: {len(docs)}")
    all_docs.extend(docs)

    # FAISS 
    if all_docs:
        # vectorstore = build_faiss_index_csv(all_docs)
        vectorstore = FAISS.load_local(folder_path=FAISS_INDEX_PATH, embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}), allow_dangerous_deserialization=True)
        retriever = create_retr_csv(vectorstore)
    else:
        vectorstore = None
        retriever = None

    # conversational RAG chain
    if retriever:
        qa_chain, memory = create_rag_chain_csv(retriever)
    else:
        qa_chain, memory = (None, None)
    # DOMAIN_LEXICON,DOMAIN_PHONE = build_dl(dealers_df)
    # print(f"Domain lexicon built with {len(DOMAIN_LEXICON)} terms")
except Exception as e:
    print("Failed initialization of RAG components:", e)
    qa_chain = None
    memory = None
    vectorstore = None
    dealers_df = None
    emp_df = None

# LLM Configuration
class LLMClient:
    
    # def __init__(self):
    #     self.model = self._find_default_model()
    #     print(f"LLM initialized with model: {self.model}")
    def __init__(self, model:str|None = None, api_key:str|None = None):
        # self.model = self._find_default_model()
        self.model = model or os.environ.get("GEMINI_MODEL","gemini-3-pro-preview")
        self.api_key = api_key or os.getenv("API_KEY")
        try:
            self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
            print(f"Gemini intialised using {self.model}")
        except Exception as e:
            print("failed to load gemini ",e)
            self.client = None
        print(f"LLM initialized with model: {self.model}")
    
    # def _find_default_model(self):
    #     try:
    #         models = ollama.list()
    #         print(models)
    #         available_models = [model['model'] for model in models['models']]

    #         # preferred = ['llama3:latest']
    #         preferred = ['gpt-oss:120b-cloud']
            
    #         for model in preferred:
    #             if model in available_models:
    #                 return model
            
    #         if available_models:
    #             return available_models[0]
                
    #         return "gpt-oss:120b-cloud"
    #         # return "llama3:latest"
    #     except Exception as e:
    #         print(f"Error listing Ollama models - {e}")
    #         return "llama3"
    
    # def is_ollama_running(self):
    #     try:
    #         ollama.list()
    #         return True
    #     except Exception:
    #         return False
    
    def is_available(self)->bool:
        return self.client is not None
            
    def generate(self, prompt, system_prompt=None):
        # try:
        #     messages = []
        #     if system_prompt:
        #         messages.append({"role": "system", "content": system_prompt})
        #     messages.append({"role": "user", "content": prompt})
        #     start_time = time.time()
        #     response = ollama.chat(model=self.model, messages=messages)
        #     end_time = time.time()
        #     print(f"LLM response time: {end_time - start_time:.2f}s")
        #     return response['message']['content']
        # except Exception as e:
        #     return f"An error with the language model: {str(e)}"
        if not self.client:
            return "LLM client not initialized."
        contents = (system_prompt + "\n" if system_prompt else "") + prompt
        try:
            resp = self.client.models.generate_content(model=self.model, contents=contents)
            return getattr(resp, "text", str(resp))
        except Exception as e:
            return f"An error with the language model: {str(e)}"

    def generate_from_messages(self, messages):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                role = "assistant" if getattr(m, 'type', 'user') == "ai" else "user"
                content = getattr(m, 'content', str(m))
            parts.append(f"{role}:\n{content}")
        combined = "\n\n".join(parts)
        return self.generate(combined)
        # try:
        #     formatted_messages = []
        #     for msg in messages:
        #         if isinstance(msg, dict):
        #             formatted_messages.append({"role": msg.get("role"), "content": msg.get("content")})
        #         else:
        #             formatted_messages.append({"role": msg.type, "content": msg.content})
        #     response = ollama.chat(model=self.model, messages=formatted_messages)
        #     return response['message']['content']
        # except Exception as e:
        #     return f"An error with the language model: {str(e)}"

# cloud api model llm
# class LLMClient:
#     def __init__(self):
#         self.api_key = os.environ.get("OLLAMA_API_KEY")
#         self.model = "gpt-oss:120b"

#     def generate(self, prompt, system_prompt=None):
#         url = "https://ollama.com"

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }

#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})

#         messages.append({"role": "user", "content": prompt})

#         body = {
#             "model": self.model,
#             "messages": messages
#         }

#         response = requests.post(url, headers=headers, json=body)
#         data = response.json()

#         return data["choices"][0]["message"]["content"]

#     # def generate_from_messages(self, messages):
#     #     url = "https://api.ollama.com/v1/chat/completions"

#     #     headers = {
#     #         "Authorization": f"Bearer {self.api_key}",
#     #         "Content-Type": "application/json",
#     #     }

#     #     formatted = [{"role": m["role"], "content": m["content"]} for m in messages]

#     #     body = {
#     #         "model": self.model,
#     #         "messages": formatted
#     #     }

#     #     response = requests.post(url, headers=headers, json=body)
#     #     data = response.json()

#     #     return data["choices"][0]["message"]["content"]

#     def generate_from_messages(self, messages):
#         try:
#             url = "https://api.ollama.com/v1/chat/completions"

#             headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json",
#             }

#             # Format messages properly
#             formatted_messages = []
#             for msg in messages:
#                 if isinstance(msg, dict):
#                     formatted_messages.append({
#                         "role": msg.get("role", "user"),
#                         "content": msg.get("content", "")
#                     })
#                 elif hasattr(msg, 'type') and hasattr(msg, 'content'):
#                     # Handle LangChain message objects
#                     role = "assistant" if msg.type == "ai" else "user"
#                     formatted_messages.append({
#                         "role": role,
#                         "content": msg.content
#                     })
#                 else:
#                     # Fallback for unknown message format
#                     formatted_messages.append({
#                         "role": "user",
#                         "content": str(msg)
#                     })

#             body = {
#                 "model": self.model,
#                 "messages": formatted_messages
#             }

#             response = requests.post(url, headers=headers, json=body, timeout=30)
#             response.raise_for_status()
#             data = response.json()

#             return data["choices"][0]["message"]["content"]
#         except Exception as e:
#             print(f"LLM generate_from_messages error: {e}")
#             return f"I'm having trouble processing your request right now. Error: {str(e)}"


# Initialize LLM client
llm_client = None
llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY") 
                )
try:
    llm_client = LLMClient()
    if not llm_client.is_available():
        print("Warning: Ollama server not running. LLM features disabled.")
        llm_client = None
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm_client = None


# ALSA warnings suppression
# class SuppressStderr:
#     def __enter__(self):
#         self.devnull = os.open(os.devnull, os.O_WRONLY)
#         self.old_stderr = os.dup(2)
#         os.dup2(self.devnull, 2)
#         return self
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         os.dup2(self.old_stderr, 2)
#         os.close(self.old_stderr)
#         os.close(self.devnull)

# def suppress_alsa_warnings():
#     return SuppressStderr()

# Text to speech function
# async def speak(text):
#     print("Bot: ", text)
#     try:
#         # print(lang)
#         # if lang and lang.lower().startswith("en"):
#         #     translated = text
#         # else:
#         translated = await translate_text(text, "hi")
#         text = translated
#         print(f"Translated text: {translated}")

#     except Exception as e:
#         print(f"Translation error: {e}, proceeding with original text.")
#     if ADVANCED_TTS:
#         try:
#             filename = f"output_{int(time.time() * 800)}.mp3"
#             communicate = edge_tts.Communicate(
#                 text=text,
#                 voice="hi-IN-SwaraNeural",
#                 rate="+10%",
#                 pitch="+0Hz"
#             )
#             await communicate.save(filename)
#             # pygame.mixer.music.load(filename)
#             # pygame.mixer.music.play()
#             # while pygame.mixer.music.get_busy():
#             #     pygame.time.wait(100)
#             # try:
#             #     os.remove(filename)
#             # except PermissionError:
#             #     pass
#             return
#         except Exception as e:
#             print(f"Advanced TTS failed: {e}, falling back to pyttsx3")
#     try:
#         if tts_engine is None:
#             with suppress_alsa_warnings():
#                 tts_engine = pyttsx3.init()
#         tts_engine.say(text)
#         tts_engine.runAndWait()
#     except Exception as e:
#         print(f"TTS error: {e}")

# async def audio_data(text:str)->Optional[bytes]:
#     # try:
#     #     translated = await translate_text(text, "hi")
#     #     text = translated
#     #     print(f"Translated text: {translated}")

#     # except Exception as e:
#     #     print(f"Translation error: {e}, proceeding with original text.")
#     if not text:
#         return None
#     words = ["*","|","."]
#     for w in words:
#         text = text.replace(w,":")
    
#     # clean_text = text.replace("*","").strip()
#     clean_text = ' '.join(text.split())
#     try:
#         filename = f"output_{int(time.time() * 800)}.mp3"
#         communicate = edge_tts.Communicate(
#             clean_text,
#             voice="en-IN-NeerjaNeural",
#             rate="+15%",
#             pitch="+0Hz"
#         )
#         await communicate.save(filename)

#         with open(filename,"rb") as audio_file:
#             audio_bytes = audio_file.read()

#         try:
#             os.remove(filename)
#         except PermissionError:
#             pass    

#         return audio_bytes
#     except Exception as e:
#         print(f"Audio data generation error: {e}")
#         return None

# async def listen_command():
#     fs = 16000
#     duration =7 # seconds
#     print("Listening...")
#     try:
#         audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
#         sd.wait()
#         audio = audio.flatten()
#         filename = f"input_{int(time.time() * 800)}.wav"
#         wav.write(filename, fs, audio)

#         res = whmodel.transcribe(filename,
#                                  temperature=0,
#                                  beam_size=5,
#                                  best_of=5)

#         command = res['text'].strip().lower()
#         if not command:
#             print("No speech detected.")
#             return ""

#         command = normalize_spoken_number_words(command)
#         command = normalize_audio_number(command)
        
        
#         command = await translate_text(command, "en")

#         command = map_category_terms(command)
#         print("After category mapping:", command)
        
#         # detect_and_set_language(command)
#         try:
#             corrected = correction(command,STATIC_TERMS)
#             if corrected != command:
#                 print(f"Corrected terms in: '{command}' -> '{corrected}'")
#             command = corrected
#             print(f"After correction: {command}")
#         except Exception as e:
#             print(f"Correction error : {e}")

#         print("You:", command)
#         os.remove(filename)
#         return command.lower()
    
#     except Exception as e:
#         print(f"Listening error: {e}")
#         return "network error"
    
# Time-based greeting
async def time_based_greeting():
    hour = datetime.datetime.now().hour
    res = "How can I help you today?"
    # await speak("How can I help you today")
    return res
    # if 5 <= hour < 12:
    #     await speak("Good morning! How can I help you today?")
    # elif 12 <= hour < 17:
    #     await speak("Good afternoon! How can I assist you?")
    # elif 17 <= hour < 22:
    #     await speak("Good evening! What can I do for you?")
    # else:
    #     await speak("Hello! How can I help you at this hour?")

# async def query_time_async():
#     time_now = datetime.datetime.now().strftime("%I:%M %p")
#     await speak(f"The current time is {time_now}")

def parse_query_for_pincode_and_category(query, categories_list=None):
    pincode_match = re.search(r"\b(\d{6})\b", query)
    pincode = pincode_match.group(1) if pincode_match else None

    cat = None
    if categories_list is not None:
        q_lower = query.lower()
        for i in categories_list:
            if i and i.lower() in q_lower:
                cat = i
                break

    # is_dealer_query = any(w in query.lower() for w in [ "pincode", "pin code", "find", "nearest","दुकानें","dukane","dukan", "stores near me", "store near me", "dealers", "dealer near me"])
    return pincode, cat

def search_exact_pincode(df, pincode, category, top_n=10):
    print("1")
    print("pincode: ", pincode)
    print("category: ", category)
    
    if df is None or df.empty or pincode is None:
        return []
    
    # Normalize pincode
    pincode = str(pincode).strip()
    if '.' in pincode:
        pincode = pincode.split('.')[0]
    df['Dealer_Pincode_normalized'] = df['Dealer_Pincode'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    res = df[df['Dealer_Pincode_normalized'] == pincode]
    
    if category:
        category = str(category).strip()
        if 'Dealer_Category' in res.columns:
            res = res[res['Dealer_Category'].str.lower().str.contains(category.lower(), na=False, regex=False)]
        else:
            print("Dealer_Category column not found in DataFrame.")
    if res.empty:
        print("No matching dealers found.")
        return []
    
    res = res.head(top_n)
    return res.to_dict(orient='records')

def normalizeMobNum(num):
    if not num:
        return None
    return re.sub(r'\D', '', str(num).strip())

def extract_mobile_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = normalize_spoken_number_words(text)
    t = normalize_audio_number(t)
    match = re.search(r'\b(\d{10})\b', t)
    if match:
        return match.group(1)
    m2 = re.search(r'\b(\d{3}[-\s]\d{3}[-\s]\d{4})\b', t)
    if m2:
        return m2.group(1)
    return None

def is_eligibility_query(q: str) -> bool:
        if not q:
            return False
        ql = q.lower()
        kws = ["check eligibility", "am i eligible", "mobile number", "mobile no", "check"]
        return any(k in ql for k in kws)

def format_eligibility_response(found: bool, name: str, mobile: str) -> str:
    if not found:
        return "I could not find your mobile number in our records. Please check the number and try again."
    
    display_name = name if name else "Customer"
    
    if mobile is None:
        return f"Hello {display_name}, I found your record but eligibility status is not available."
    
    try:
        # elg_value = int(eligibility)
        # if elg_value == 1:
        return f"Hello {display_name}, great news! You are eligible for REMI services."
        # elif elg_value == 0:
        #     return f"Hello {display_name}, I'm sorry but you are not currently eligible for REMI services."
        # else:
        #     return f"Hello {display_name}, your eligibility status is: {eligibility}"
    except (ValueError, TypeError):
        return f"Hello {display_name}, your eligibility status is: true"

def check_remi_eligibility(mob:str):
    global emp_df, last_name
    try:
        if 'emp_df' not in globals() or emp_df is None:
            print("Employee data not available for REMI eligibility check.")
            return False, None, None
        m = normalizeMobNum(mob)
        if not m:
            return False, None, None
        # Normalize the Mob column and compare for exact match
        if 'Mob' not in emp_df.columns:
            print("Mob column not found in employee dataframe.")
            return False, None, None
        mob_col = emp_df['Mob'].astype(str).fillna("").str.replace(r'\D', '', regex=True)
        matched = emp_df[mob_col == m]
        if matched.empty:
            # exact required — no fallback
            return False, None, None
        r = matched.iloc[0]
        name = r.get('Employee_Name') or r.get('Name') or r.get('employee_name') or r.get('name')
        elg = r.get('remi_eligible') if 'remi_eligible' in r.index else (r.get('Eligibility') if 'Eligibility' in r.index else None)
        # persist last seen name for session-less flows
        try:
            if name and pd.notna(name):
                last_name = name
        except Exception:
            pass
        return True, name, elg
    except Exception as e:
        print(f"Error checking REMI eligibility: {e}")
        return False, None, None

async def req_cat(cat_res):
    global category, dealers_df
    avail = []
    if dealers_df is not None:
        avail = sorted(list(dealers_df['Dealer_Category'].dropna().unique()))

    # print("Please tell me any specific category you want")
    # cat_res = input("please enter the category ").strip()

    if not cat_res:
        print("No category detected. Please try again.")
        return None

    if avail:
        # Try exact match first
        for i in avail:
            if i and i.lower() in cat_res.lower():
                category = i
                # await speak(f"Thank you. I will now search for stores in category {category}.")
                print(f"Thank you. I will now search for stores in category {category}.")
                return category

        try:
            corrected = correction(cat_res, avail, fuzzy_threshold=80)
            if corrected != cat_res:
                category = corrected
                print(f"Corrected category from '{cat_res}' to '{corrected}'")
            else:
                category = cat_res
            # await speak(f"Thank you. I will now search for stores in category {category}.")
            print(f"Thank you. I will now search for stores in category {category}.")
            return category
        except Exception as e:
            print(f"Correction match error: {e}")
    return cat_res

async def request_pincode(pincode_response):
    global last_pincode
    
    # print("Please tell me the pincode you would like to search in.")
    
    # pincode_response = input("Please enter pincode: ").strip()
    pincode_response = normalize_spoken_number_words(pincode_response)
    pincode_response = normalize_audio_number(pincode_response)
    pincode_match = re.search(r"\b(\d{6})\b", pincode_response)
    if pincode_match:
        last_pincode = pincode_match.group(1)
        # await speak(f"Thank you. I will now search for stores in pincode {last_pincode}.")
        # print(f"Thank you. I will now search for stores in pincode {last_pincode}.")
        return last_pincode
    else:
        # await speak("I could not find a valid pincode. Please try again with a 6-digit pincode.")
        print("I could not find a valid pincode. Please try again with a 6-digit pincode.")
        return None

def map_category_term(text: str) -> str| None:
    if not text:
        print("text is none")
        return None
    
    text_lower = text.lower().strip()
    
    for standard_cat in CATEGORY_MAPPING.keys():
        if standard_cat.lower() == text_lower:
            print(f"Category mapping: '{text}' -> '{standard_cat}' (exact match)")
            return standard_cat
    
    
    for standard_cat, variations in CATEGORY_MAPPING.items():
        for keyword in variations:
            if keyword and keyword.lower() in text_lower:
                print(f"Category mapping: '{text}' -> '{standard_cat}' (keyword: '{keyword}')")
                return standard_cat
    
    print(f"Category '{text}' not found in CATEGORY_MAPPING")
    return None

async def present_dealer_results(results):
    global last_shown_dealers, last_topic, memory
    if not results:
        # await speak("No results to show.")
        print("No results to show")
        return
    
    last_shown_dealers = results
    last_topic = "dealer"
    lines = []
    for r in results:
        name = r.get('Dealer_Name') or r.get('dealer_name') or r.get('DEALER_NAME'.lower(), "")
        cat = r.get('Dealer_Category_Level_3') or r.get('Dealer_Category', "")
        pin = r.get('Dealer_Pincode') or r.get('pincode', "")
        brand = r.get('Dealer_Brand') or r.get('Brand', "")
        city = r.get('Dealer_City') or r.get('city', "")
        fosName = r.get('FOS_Name') or r.get('fos_name', "")
        addr = r.get('Dealer_Address') or r.get('dealer address', "")
        avg_txns = r.get('Dealer_Category_Avg_Txns') or r.get('avg_txns', "")
        
        parts = []
        if name: parts.append(name)
        if cat: parts.append(cat)
        if city or pin: parts.append(f"{city} {pin}".strip())
        if brand: parts.append(f"brand: {brand}")
        if addr: parts.append(f"address: {addr}")
        if fosName: parts.append(f"FOS Name: {fosName}")
        if avg_txns: parts.append(f"avg txns: {avg_txns}")

        line = "  ".join([p for p in parts if p])
        lines.append(line)

    max_to_speak = min(4, len(lines))

    print(f"I found {len(lines)} matching stores. Reading top {max_to_speak}:")
    
    for i in range(max_to_speak):
        # await speak(f"{i+1}. {lines[i]}")
        print(f"{i+1}. {lines[i]}")
    # if memory:
    #     memory.save_context({"input": query}, {"output": lines[:max_to_speak]})

async def sample_ques(chat_history):
    try:
        system_prompt = (
            "provide the sample question that can be related to the chat history provided \n\n" + chat_history
        )
        suggestions = llm_client.generate(system_prompt)
        print(suggestions)
        return suggestions
    except Exception as e:
        print(f"Error generating sample questions: {e}")
        return "Could not generate sample questions at this time."

# async def ask_llm(query,session):
#     global qa_chain, memory, llm_client, dealers_df, vectorstore, last_pincode, category, last_name

#     if session is not None:
#         last_pincode = session.get("pincode")
#         category = session.get("category")
#         customer_name = session.get("customer_name")
#         if customer_name:
#             last_name = customer_name

#     # Load existing conversation history
#     chat_history = memory.load_memory_variables({})['chat_history'] if memory else []

#     categories_list = sorted(list(dealers_df['Dealer_Category'].dropna().unique())) if dealers_df is not None else None
#     pincode, detected_category, is_dealer_query = parse_query_for_pincode_and_category(query, categories_list)

#     if not qa_chain and not llm_client:
#         print("No access to a knowledge base or language model right now.")
#         return "No access to a knowledge base or language model right now."

#     # ELIGIBILITY CHECK
#     if is_eligibility_query(query):
#         print("eligibility check")

#         mobile = extract_mobile_from_text(query)
#         if not mobile and session:
#             # if session available, maybe we already have mobile stored
#             mobile = session.get("customer_mobile")
#             # if session and session.get("customer_mobile"):
#             #     mobile = session.get("customer_mobile")
#         if not mobile:
#             # ask for mobile
#             return {
#                 "state": "need_mobile",
#                 "response_text": "Please provide your 10-digit mobile number to check REMI eligibility."
#             }
#         found, name, elg = check_remi_eligibility(mobile)
#         if found:
#             # persist in session or global
#             if session is not None:
#                 session["customer_name"] = name
#                 session["customer_mobile"] = mobile
#                 session["eligibility"] = elg
#             if name:
#                 last_name = name
#             # last_customer_name = name or last_customer_name
#             # # create response message
#             # elg_text = ""
#             # if elg is not None:
#             #     elg_text = f"Eligibility: {elg}."
#             # else:
#             #     elg_text = "Eligibility info found."
#             # display_name = name if name else "Customer"
#             response_text = format_eligibility_response(found, name, elg, mobile)

#             # response_text = f"{display_name}, I found your record. {elg_text}"
#             return {
#                 "state": "eligible_lookup",
#                 "response_text": response_text,
#                 "customer_name": name,
#                 "customer_mobile": mobile,
#                 "eligibility": elg,
#                 "found": found
#             }
#         else:
#             return {
#                 "state": "not_found",
#                 "response_text": "I could not find your mobile number in our records. Please re-check the number or try an alternate contact."
#             }
    
#     if is_dealer_query and dealers_df is not None:
#         print("dealer df")

#         if pincode:
#             if session is not None:
#                 session["pincode"] = pincode

#             else:
#                 last_pincode = pincode
#             print(f"Detected pincode: {pincode}")

#             print(f"No pincode detected. Searching based on your last request for pincode {last_pincode}.")
#         if detected_category:
#             mapped = map_category_terms(detected_category)
#             if mapped:
#                 if session is not None:
#                     session["category"] = mapped
#                 else:
#                     category = mapped
#                 print(f"Detected category '{detected_category}' mapped to '{category}'")
#             else:
#                 print(f"Detected category '{detected_category}' not in CATEGORY_MAPPING")
#                 category = None
#         current_pin = session.get("pincode") if session is not None else last_pincode
#         current_cat = session.get("category") if session is not None else category

#         if not current_pin:
#             return{
#                 "state":"need_pincode",
#                 "response_text":"Please provide a 6-digit pincode to search for stores."
#             }
#         if not current_cat:
#             return{
#                 "state":"need_cat",
#                 "response_text":"Please provide a store category to search for."
#             }

#         print(current_cat," ,",current_pin)

#         if current_pin and current_cat:
#             exact = search_exact_pincode(dealers_df, current_pin, current_cat, top_n=10)
#             print("here1")
#             if exact:
#                 response_text = f"I found {len(exact)} matching stores in pincode {current_pin} for {current_cat}."

#                 # Format dealer information
#                 dealers_info = []
#                 for idx, r in enumerate(exact[:5], 1):
#                     info = f"{idx}. {r.get('Dealer_Name', 'N/A')}"
#                     if r.get('Dealer_Address'):
#                         info += f", {r.get('Dealer_Address')}"
#                     if r.get('Dealer_Phone'):
#                         info += f", Phone: {r.get('Dealer_Phone')}"
#                     dealers_info.append(info)
                
#                 if dealers_info:
#                     response_text += " Here are the top matches: " + "; ".join(dealers_info)
                
#                 return {
#                     "response_text": response_text,
#                     "dealers": exact,
#                     "pincode": current_pin,
#                     "category": current_cat,
#                     "state": "normal_qa"
#                 }
#             else:
#                 return {
#                     "response_text": f"I could not find any matching stores in pincode {current_pin} for category {current_cat}. Please try a different pincode or category.",
#                     "pincode": current_pin,
#                     "category": current_cat,
#                     "state": "normal_qa"
#                 }

#     # Regular RAG query handling
#     if qa_chain:
        
#         print("searching knowledge base")
#         try:
#             print("chat history:", chat_history)
#             start_time = time.time()
#             res = qa_chain.invoke({"input": query, "chat_history": chat_history})
#             # print("RAW RAG OUTPUT:", res)
#             response =  (
#             res.get("answer")
#             or res.get("result")
#             or res.get("output_text")
#             or res.get("response")
#             or res.get("text")
#         )
#             # update memory
#             if memory:
#                 memory.save_context({"input": query}, {"output": response})
#             # print("Extracted RAG response:", response)
#             end_time = time.time()
#             print(f"RAG response time: {end_time - start_time:.2f}s")
            
                
#             not_found_phrases = ["i don't know", "i do not know", "no information", "cannot find", "no data", "not in the context"]
#             found_in_rag = not any(phrase in response.lower() for phrase in not_found_phrases)
            
#             if found_in_rag:
#                 # print("found in rag")
#                 # print(type(response))
#                 print("RAG Response: ", response)
#                 # if isinstance(res, dict):
#                 #     # print("Response (dict): ", res)
#                 #     print("here")
#                 # elif isinstance(res, str):
#                 #     preview = res[:100] + "..." if len(res) > 100 else res
#                 #     print("Response (text):", preview)
#                 # else:
#                 #     print("Response is None or unknown:", res)

#                 return response
#             else:
#                 print("not found in rag, using semantic fallback")
#                 return await fallback_to_llm(query)
#                 # return
#         except Exception as e:
#             print(f"RAG error: {e}")
#             # await speak("RAG failed. Let me try a more general search.")
#             print("Rag failed")
#             return await fallback_to_llm(query)
#     else:
#         print("no rag, using llm")
#         return await fallback_to_llm(query)

# Add these new functions after the existing sample_ques function

async def generate_followup_suggestions(chat_history, current_query=None, context=None):
    global llm_client
    
    if not llm_client or not llm_client.is_available():
        return get_static_suggestions(context)
    
    try:
        if not chat_history:
            return get_static_suggestions(context)
        
        # Build context
        history_text = ""
        if isinstance(chat_history, list):
            for msg in chat_history[-1:]: 
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_text += f"{role}: {content}\n"
                else:
                    role = "assistant" if getattr(msg, 'type', 'user') == "ai" else "user"
                    content = getattr(msg, 'content', str(msg))
                    history_text += f"{role}: {content}\n"
        
        if current_query:
            history_text += f"user: {current_query}\n"
        
        system_prompt = f"""
        Based on the following conversation history, generate 3-4 relevant follow-up questions that a user might ask about Bajaj Finance REMI services, dealers, or products.

        Guidelines:
        - Questions should be natural and contextually relevant
        - Questions should be very short and to the point
        - Focus on Bajaj Finance REMI products and services
        - Include questions about dealer locations, eligibility, products, or services
        - Keep questions concise and clear
        - Only 2-3 questions
        - Don't mention about giving follow up questions explicitly
        - Do NOT write things like: ‘Here are 3 things…’, ‘You could also ask…’, ‘Some follow-up questions are…’. Avoid such patterns completely.


        Conversation History:
        {history_text}"""
        
        suggestions = llm_client.generate(system_prompt)
        return parse_suggestions(suggestions)
        
    except Exception as e:
        print(f"Error generating follow-up suggestions: {e}")
        return get_static_suggestions(context)

def get_static_suggestions(context=None):
    default_suggestions = [
        "What is REMI and how does it work?",
        "Find dealers near my location",
        "Check my REMI eligibility",
        "What products are available through REMI?"
    ]
    
    dealer_suggestions = [
        "Show me more dealer details",
        "Find dealers in a different area",
        "What are the contact details?",
        "Are there dealers for other categories?"
    ]
    
    eligibility_suggestions = [
        "How can I become eligible for REMI?",
        "What documents do I need?",
        "What is my credit limit?",
        "How do I apply for REMI?"
    ]
    
    if context == "dealers":
        return dealer_suggestions
    elif context == "eligibility":
        return eligibility_suggestions
    else:
        return default_suggestions

def parse_suggestions(suggestions_text):
    if not suggestions_text:
        return get_static_suggestions()
    
    lines = suggestions_text.strip().split('\n')
    parsed_suggestions = []
    
    for line in lines:
        # cleaning
        clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
        clean_line = re.sub(r'^[-*]\s*', '', clean_line.strip())
        
        if clean_line and len(clean_line) > 10:
            parsed_suggestions.append(clean_line)
    
    return parsed_suggestions[:4] if parsed_suggestions else get_static_suggestions()


#LangGraph routing Schema
class RouteDecision(BaseModel):
    step:Literal["eligibility","dealer","faq"] = Field(
        ..., description="Decide which step to take based on user query"
    )
class GraphState(TypedDict, total=False):
    query:str
    session:Dict[str,Any]
    route:str
    result:Dict[str,Any]

async def _eligibility_flow(query:str, session:Dict[str,Any]):
    global memory
    chat_history = memory.load_memory_variables({})['chat_history'] if memory else []
    mobile = extract_mobile_from_text(query)
    if not mobile and session:
        mobile = session.get("customer_mobile")
    if not mobile:
        suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
        return {
            "state": "need_mobile",
            "response_text": "Please provide your 10-digit mobile number to check REMI eligibility.",
            "suggestions": suggestions
        }
    found, name, elg = check_remi_eligibility(mobile)
    if found:
        if session is not None:
            session["customer_name"] = name
            session["customer_mobile"] = mobile
            session["eligibility"] = elg
        if name:
            last_name = name
                
            response_text = format_eligibility_response(found, name, elg, mobile)
            suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
            
            return {
                "state": "eligible_lookup",
                "response_text": response_text,
                "customer_name": name,
                "customer_mobile": mobile,
                "eligibility": elg,
                "found": found,
                "suggestions": suggestions
            }
        else:
            suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
            return {
                "state": "not_found",
                "response_text": "I could not find your mobile number in our records. Please re-check the number or try an alternate contact.",
                "suggestions": suggestions
            }

async def _dealer_flow(query:str, session:Dict[str,Any]):
    global dealers_df, memory, last_pincode, category
    chat_history = memory.load_memory_variables({})['chat_history'] if memory else []
    categories_list = sorted(list(dealers_df['Dealer_Category'].dropna().unique())) if dealers_df is not None else None
    pincode, detected_category, is_dealer_query = parse_query_for_pincode_and_category(query, categories_list)

    if not is_dealer_query or dealers_df is None:
        return{
            "state":"normal_qa",
            "response_text":None,
            "suggestions":[]
        }
    
    if pincode:
        if session is not None:
            session["pincode"] = pincode
        else:
            last_pincode = pincode
        print(f"Detected pincode: {pincode}")

    if detected_category:
        mapped = map_category_terms(detected_category)
        if mapped:
            if session is not None:
                session["category"] = mapped
            else:
                category = mapped
            print(f"Detected category '{detected_category}' mapped to '{category}'")
    
    current_pin = session.get("pincode") if session is not None else last_pincode
    current_cat = session.get("category") if session is not None else category

    if not current_pin:
        suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
        return {
            "state": "need_pincode",
            "response_text": "Please provide a 6-digit pincode to search for stores.",
            "suggestions": suggestions
        }
    if not current_cat:
        suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
        return {
            "state": "need_cat",
            "response_text": "Please provide a store category to search for.",
            "suggestions": suggestions
        }
    
    exact = search_exact_pincode(dealers_df, current_pin, current_cat, top_n=10)
    if exact:
        response_text = f"I found {len(exact)} matching stores in pincode {current_pin} for {current_cat}."

        dealers_info = []
        for idx, r in enumerate(exact[:5], 1):
            info = f"{idx}. {r.get('Dealer_Name', 'N/A')}"
            if r.get('Dealer_Address'):
                info += f", {r.get('Dealer_Address')}"
            if r.get('Dealer_Phone'):
                info += f", Phone: {r.get('Dealer_Phone')}"
            dealers_info.append(info)
        
        if dealers_info:
            response_text += " Here are the top matches: " + "; ".join(dealers_info)
        
        suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
        
        return {
            "response_text": response_text,
            "dealers": exact,
            "pincode": current_pin,
            "category": current_cat,
            "state": "normal_qa",
            "suggestions": suggestions
        }
    else:
        suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
        return {
            "response_text": f"I could not find any matching stores in pincode {current_pin} for category {current_cat}. Please try a different pincode or category.",
            "pincode": current_pin,
            "category": current_cat,
            "state": "normal_qa",
            "suggestions": suggestions
        }

async def _faq_flow(query:str, session:Dict[str,Any]):
    global qa_chain, memory
    chat_history = memory.load_memory_variables({})['chat_history'] if memory else []

    if qa_chain:
        print("searching knowledge base")
        try:
            print("chat history:", chat_history)
            start_time = time.time()
            res = qa_chain.invoke({"input": query, "chat_history": chat_history})
            response =  (
            res.get("answer")
            or res.get("result")
            or res.get("output_text")
            or res.get("response")
            or res.get("text")
        )
            if memory:
                memory.save_context({"input": query}, {"output": response})
            end_time = time.time()
            print(f"RAG response time: {end_time - start_time:.2f}s")
            
            not_found_phrases = ["i don't know", "i do not know", "no information", "cannot find", "no data", "not in the context"]
            found_in_rag = not any(phrase in response.lower() for phrase in not_found_phrases)
            
            if found_in_rag:
                print("RAG Response: ", response)
                suggestion = await generate_followup_suggestions(chat_history, query, "faq")
                return {
                    "response_text": response,
                    "state": "normal_qa",
                    "suggestions": suggestion
                }
            else:
                print("not found in rag, using semantic fallback")
                fallback_response = await fallback_to_llm(query)
                suggestion = await generate_followup_suggestions(chat_history, query, "faq")
                return {
                    "response_text": fallback_response,
                    "state": "normal_qa",
                    "suggestions": suggestion
                }
        except Exception as e:
            print(f"RAG error: {e}")
            print("Rag failed")
            fallback_response = await fallback_to_llm(query)
            suggestion = await generate_followup_suggestions(chat_history, query, "faq")
            return {
                "response_text": fallback_response,
                "state": "normal_qa",
                "suggestions": suggestion
            }
    else:
        print("no rag, using llm")
        fallback_response = await fallback_to_llm(query)
        suggestion = await generate_followup_suggestions(chat_history, query, "faq")
        return {
            "response_text": fallback_response,
            "state": "normal_qa",
            "suggestions": suggestion
        }

# def _route_node(state:GraphState)->Dict[str,Any]:
#     global dealers_df
#     query = state["query"]

#     if is

async def ask_llm(query, session):
    global qa_chain, memory, llm_client, dealers_df, vectorstore, last_pincode, category, last_name

    if session is not None:
        last_pincode = session.get("pincode")
        category = session.get("category")
        customer_name = session.get("customer_name")
        if customer_name:
            last_name = customer_name
    
    intent_res = await get_query_intent(query)
    print("intetn result",intent_res)


    chat_history = memory.load_memory_variables({})['chat_history'] if memory else []

    categories_list = sorted(list(dealers_df['Dealer_Category'].dropna().unique())) if dealers_df is not None else None
    pincode, detected_category, is_dealer_query = parse_query_for_pincode_and_category(query, categories_list)

    if not qa_chain:
        print("No access to a knowledge base or language model right now.")
        return {
            "response_text": "No access to a knowledge base or language model right now.",
            "suggestions": get_static_suggestions()
        }

    # ELIGIBILITY CHECK
    if is_eligibility_query(query):
        print("eligibility check")

        mobile = extract_mobile_from_text(query)
        if not mobile and session:
            mobile = session.get("customer_mobile")
            
        if not mobile:
            suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
            return {
                "state": "need_mobile",
                "response_text": "Please provide your 10-digit mobile number to check REMI eligibility.",
                "suggestions": suggestions
            }
            
        found, name, elg = check_remi_eligibility(mobile)
        if found:
            if session is not None:
                session["customer_name"] = name
                session["customer_mobile"] = mobile
                session["eligibility"] = elg
            if name:
                last_name = name
                
            response_text = format_eligibility_response(found, name, elg, mobile)
            suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
            
            return {
                "state": "eligible_lookup",
                "response_text": response_text,
                "customer_name": name,
                "customer_mobile": mobile,
                "eligibility": elg,
                "found": found,
                "suggestions": suggestions
            }
        else:
            suggestions = await generate_followup_suggestions(chat_history, query, "eligibility")
            return {
                "state": "not_found",
                "response_text": "I could not find your mobile number in our records. Please re-check the number or try an alternate contact.",
                "suggestions": suggestions
            }
    
    # dealer check
    if is_dealer_query and dealers_df is not None:
        print("dealer df")

        if pincode:
            if session is not None:
                session["pincode"] = pincode
            else:
                last_pincode = pincode
            print(f"Detected pincode: {pincode}")

        if detected_category:
            mapped = map_category_terms(detected_category)
            if mapped:
                if session is not None:
                    session["category"] = mapped
                else:
                    category = mapped
                print(f"Detected category '{detected_category}' mapped to '{category}'")

        current_pin = session.get("pincode") if session is not None else last_pincode
        current_cat = session.get("category") if session is not None else category

        if not current_pin:
            suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
            return {
                "state": "need_pincode",
                "response_text": "Please provide a 6-digit pincode to search for stores.",
                "suggestions": suggestions
            }
            
        if not current_cat:
            suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
            return {
                "state": "need_cat",
                "response_text": "Please provide a store category to search for.",
                "suggestions": suggestions
            }

        if current_pin and current_cat:
            exact = search_exact_pincode(dealers_df, current_pin, current_cat, top_n=10)
            
            if exact:
                response_text = f"I found {len(exact)} matching stores in pincode {current_pin} for {current_cat}."

                dealers_info = []
                for idx, r in enumerate(exact[:5], 1):
                    info = f"{idx}. {r.get('Dealer_Name', 'N/A')}"
                    if r.get('Dealer_Address'):
                        info += f", {r.get('Dealer_Address')}"
                    if r.get('Dealer_Phone'):
                        info += f", Phone: {r.get('Dealer_Phone')}"
                    dealers_info.append(info)
                
                if dealers_info:
                    response_text += " Here are the top matches: " + "; ".join(dealers_info)
                
                suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
                
                return {
                    "response_text": response_text,
                    "dealers": exact,
                    "pincode": current_pin,
                    "category": current_cat,
                    "state": "normal_qa",
                    "suggestions": suggestions
                }
            else:
                suggestions = await generate_followup_suggestions(chat_history, query, "dealers")
                return {
                    "response_text": f"I could not find any matching stores in pincode {current_pin} for category {current_cat}. Please try a different pincode or category.",
                    "pincode": current_pin,
                    "category": current_cat,
                    "state": "normal_qa",
                    "suggestions": suggestions
                }

    # Regular RAG query handling
    if qa_chain:
        print("searching knowledge base")
        try:
            start_time = time.time()
            res = qa_chain.invoke({"input": query, "chat_history": chat_history})
            response = (
                res.get("answer")
                or res.get("result")
                or res.get("output_text")
                or res.get("response")
                or res.get("text")
            )
            
            if memory:
                memory.save_context({"input": query}, {"output": response})
                
            end_time = time.time()
            print(f"RAG response time: {end_time - start_time:.2f}s")
            
            not_found_phrases = ["i don't know", "i do not know", "no information", "cannot find", "no data", "not in the context"]
            found_in_rag = not any(phrase in response.lower() for phrase in not_found_phrases)
            
            if found_in_rag:
                print("RAG Response: ", response)
                suggestions = await generate_followup_suggestions(chat_history, query)
                return {
                    "response_text": response,
                    "suggestions": suggestions,
                    "state": "normal_qa"
                }
            else:
                print("not found in rag, using semantic fallback")
                return await fallback_to_llm(query)

        except Exception as e:
            print(f"RAG error: {e}")
            return await fallback_to_llm(query)
    else:
        print("no rag, using llm")
        return await fallback_to_llm(query)


# Update the handle_command function to display suggestions
# async def handle_command(command):
#     global last_shown_dealers, last_topic, category, last_pincode
#     global last_requested_property, last_selected_dealer
    
#     if command == "network error":
#         print("Network error occurred. Please check your connection.")
#         return
    
#     if "exit" in command or "quit" in command or "shut down" in command or "एग्जिट" in command:
#         print("Goodbye! Have a great day!")
#         return True

#     result = await ask_llm(command, session=None)
    
#     # Handle different response formats
#     if isinstance(result, dict):
#         print("Bot:", result.get("response_text", ""))
        
#         # Display suggestions if available
#         suggestions = result.get("suggestions", [])
#         if suggestions:
#             print("\n--- Follow-up Questions ---")
#             for i, suggestion in enumerate(suggestions, 1):
#                 print(f"{i}. {suggestion}")
#             print("---------------------------\n")
#     else:
#         # Handle string responses (backward compatibility)
#         print("Bot:", result)
        
#         # Generate suggestions for string responses
#         chat_history = memory.load_memory_variables({})['chat_history'] if memory else []
#         suggestions = await generate_followup_suggestions(chat_history, command)
#         if suggestions:
#             print("\n--- Follow-up Questions ---")
#             for i, suggestion in enumerate(suggestions, 1):
#                 print(f"{i}. {suggestion}")
#             print("---------------------------\n")
    
#     return False



async def fallback_to_llm(query):
    global llm_client, memory
    if llm_client:
        try:
            messages = memory.load_memory_variables({})['chat_history'] if memory else []
            # response = llm_client.generate_from_messages(messages + [{"role": "user", "content": query}])
            response = llm.invoke({"input": query})
            # print("using llm fallback response")
            
            # Save to memory
            if memory:
                memory.save_context({"input": query}, {"output": response})
                
            # Speak the response
            if len(response) > 300:
                chunks = [response[i:i+300] for i in range(0, len(response), 300)]
                for chunk in chunks:
                    # await speak(chunk)
                    print(chunk)
                    return response
            else:
                # await speak(response)
                return response
        except Exception as e:
            # await speak(f"Sorry, I couldn't find an answer to that question. {str(e)}")
            print(f"LLM fallback error: {e}")
    else:
        # await speak("LLM fallback not available right now.")
        print("LLM fallback not available right now.")

# cloud fallback
# async def fallback_to_llm(query):
#     global llm_client, memory
    
#     if not llm_client:
#         return "I'm sorry, my language model is not available right now. Please try again later."
    
#     try:
#         # Load conversation history
#         chat_history = []
#         if memory:
#             try:
#                 memory_vars = memory.load_memory_variables({})
#                 chat_history = memory_vars.get('chat_history', [])
#             except Exception as e:
#                 print(f"Memory load error: {e}")
#                 chat_history = []
        
#         # Add current query to messages
#         messages = chat_history + [{"role": "user", "content": query}]
        
#         # Get response from LLM
#         response = llm_client.generate_from_messages(messages)
        
#         if not response or response is None:
#             response = "I'm having trouble generating a response right now. Please try rephrasing your question."
        
#         # Save to memory if available
#         if memory and response:
#             try:
#                 memory.save_context({"input": query}, {"output": response})
#             except Exception as e:
#                 print(f"Memory save error: {e}")
        
#         return response
        
    # except Exception as e:
    #     print(f"LLM fallback error: {e}")
    #     return f"I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

async def text_input():
    command = input("type:").strip()
    if not command:
        print("Empty input detected. Please type something.")
        return ""

    command = await translate_text(command, "en")

    command = map_category_terms(command)
    print(f"After category mapping: {command}")
    
    try:
        corrected = correction(command, STATIC_TERMS)
        if corrected != command:
            print(f"Corrected terms in: '{command}' -> '{corrected}'")
        command = corrected
        print("After term correction:", command)
    except Exception as e:
        print(f"Term correction error: {e}")

    print("you typed:", command)
    print("translated:", command)
    return command.lower()

def safe_detect_language(text: str) -> str:
    try:
        if not text or len(text.strip()) < 2:
            # too short or empty text
            return "en"
        lang = detect(text)
        if lang == "hi":
            return lang
        else:
            return "en"
    except Exception:
        return "en"


    

async def handle_command(command):
    global last_shown_dealers, last_topic, category, last_pincode
    global last_requested_property, last_selected_dealer
    if command == "network error":
        # await speak("Network error occurred. Please check your connection.")
        print("Network error occurred. Please check your connection.")
        return
    
    if "exit" in command or "quit" in command or "shut down" in command or "एग्जिट" in command:
        # await speak("Goodbye! Have a great day!")
        print("Goodbye! Have a great day!")
        return True

    await ask_llm(command,session=None)
    return

async def advanced_querying(input_mode="audio"):
    # await time_based_greeting()
    
    while True:
        if input_mode == "audio":
            # command = await listen_command()
            print("Audio input mode not implemented, switching to text input.")
        else:
            command = await text_input()
        if command:
            print(f"Processing: {command}")
            should_exit = await handle_command(command)
            if should_exit:
                break
        elif command == "":
            print("No command detected, listening again...")
        else:
            print("Network error or other issue")

def main():
    print("- Exit: 'shut down', 'exit', 'quit'")
    
    print("\nLLM Status:")
    if qa_chain:
        print("Knowledge base is ready")
    else:
        print("knowledge base not available")
    if llm_client and llm_client.is_available():
        print("llm fallback ready")
    else:
        print("issue in llm or not running")
    
    print("\nChoose input mode (this will apply for the whole session):")
    print("1. Voice input (speak commands)")
    print("2. Text input (type commands)")
    # choice = input("Enter choice (1/2): ").strip()
    
    # input_mode = "text" if choice == "2" else "audio"
    
    try:
        input_mode = "text"
        asyncio.run(advanced_querying(input_mode))
        # asyncio.run(handle_command())
    except KeyboardInterrupt:
        print("\nAssistant stopped by user")

if __name__ == "__main__":
    main()