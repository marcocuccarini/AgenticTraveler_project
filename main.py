#!/usr/bin/env python3
"""
AgenticTraveler CLI - Command Line Interface

This CLI application implements the same enhanced functionality as the Gradio app
but provides a command-line interface using argparse for batch processing and automation.

Features:
- Enhanced error handling with retry mechanisms
- Progress indicators for CLI operations
- Image optimization and validation
- Export functionality to JSON
- Comprehensive logging
- All three analysis systems: Agent, RAG, and ARCO

Usage:
    python main.py --image path/to/image.jpg --question "What monument is this?"
    python main.py --image path/to/image.jpg --export results.json
    python main.py --help
"""

import argparse
import sys
import os
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from PIL import Image
import tempfile
import torch

# Add script directories to Python path
script_dir = Path(__file__).parent / "script"
sys.path.append(str(script_dir / "agent"))
sys.path.append(str(script_dir / "RAG system"))
sys.path.append(str(script_dir / "KG"))

try:
    # Import the three main components
    from agent_base_optim import create_agent, parse_monument_info, get_monument_coordinates, ImageRetrievalTool, Localizator
    from rag_system_smolagent import RAGSmolagent
    from ARCO_access import query_by_name
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")
    sys.exit(1)


class AgenticTravelerCLI:
    """CLI version of AgenticTraveler with enhanced features"""
    
    def __init__(self, verbose=False):
        """Initialize the CLI application"""
        self.image_retrieval_tool = None
        self.localizator = None
        self.rag_system = None
        self.verbose = verbose
        self.logger = self._setup_logging()
        self._initialize_systems()
    
    def _setup_logging(self):
        """Configure logging for the application"""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agentictravel_cli.log', encoding='utf-8')
            ]
        )
        
        # Add console handler for verbose mode
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
        
        return logging.getLogger(__name__)
    
    def _initialize_systems(self):
        """Initialize the tools and RAG systems with retry mechanisms"""
        print("🚀 Initializing AgenticTraveler CLI...")
        
        # Initialize Image Retrieval Tool
        max_retries = 3
        image_tool_initialized = False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Image Retrieval Tool (attempt {attempt + 1}/{max_retries})")
                if self.verbose or attempt > 0:
                    print(f"🔄 Initializing Image Retrieval Tool (attempt {attempt + 1}/{max_retries})...")
                
                self.image_retrieval_tool = ImageRetrievalTool()
                self.logger.info("Image Retrieval Tool initialized successfully")
                print("✅ Image Retrieval Tool initialized successfully")
                image_tool_initialized = True
                break
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Image Retrieval Tool (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize Image Retrieval Tool after {max_retries} attempts: {e}")
                    self.image_retrieval_tool = None
                else:
                    print(f"⚠️ Image tool initialization failed, retrying in 2 seconds...")
                    time.sleep(2)

        # Initialize Localization Tool
        localizator_initialized = False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Localizator (attempt {attempt + 1}/{max_retries})")
                if self.verbose or attempt > 0:
                    print(f"🔄 Initializing Localizator (attempt {attempt + 1}/{max_retries})...")
                
                self.localizator = Localizator()
                self.logger.info("Localizator initialized successfully")
                print("✅ Localizator initialized successfully")
                localizator_initialized = True
                break
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Localizator (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize Localizator after {max_retries} attempts: {e}")
                    self.localizator = None
                else:
                    print(f"⚠️ Localizator initialization failed, retrying in 2 seconds...")
                    time.sleep(2)
        
        # Initialize RAG system with retries
        rag_initialized = False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing RAG system (attempt {attempt + 1}/{max_retries})")
                if self.verbose or attempt > 0:
                    print(f"🔄 Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                
                # Usa path configurabili
                embed_model_path = os.environ.get('EMBED_MODEL_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/all-MiniLM-L6-v2")
                qwen_model_path = os.environ.get('QWEN_MODEL_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct")
                
                self.rag_system = RAGSmolagent(
                    embed_model_name=embed_model_path,
                    use_cross_encoder=False,
                    model_id=qwen_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.logger.info("RAG system initialized successfully")
                print("✅ RAG system initialized successfully")
                rag_initialized = True
                break
                
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG system (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize RAG system after {max_retries} attempts: {e}")
                    self.rag_system = None
                else:
                    print(f"⚠️ RAG initialization failed, retrying in 2 seconds...")
                    time.sleep(2)
        
        # Summary
        systems_ready = sum([image_tool_initialized, localizator_initialized, rag_initialized])
        print(f"📊 Systems Status: {systems_ready}/3 components ready + ARCO database")
        
        if systems_ready == 0:
            print("⚠️ Warning: No AI systems available. Only ARCO database queries will work.")
        
        print()  # Add spacing
    
    def optimize_image_for_processing(self, image_path, max_size=(1024, 1024), quality=85):
        """Optimize image for processing by resizing and compressing if needed"""
        try:
            with Image.open(image_path) as image:
                original_size = image.size
                self.logger.info(f"Original image size: {original_size}")
                
                if self.verbose:
                    print(f"📏 Original image size: {original_size}")
                
                # Calculate new size if resizing needed
                needs_resize = original_size[0] > max_size[0] or original_size[1] > max_size[1]
                
                if needs_resize:
                    # Maintain aspect ratio
                    ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
                    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    self.logger.info(f"Resized image to: {new_size}")
                    print(f"🔧 Resized image to: {new_size}")
                
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                    self.logger.info(f"Converted image mode from {image.mode} to RGB")
                
                # Save optimized image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    image.save(temp_file.name, format='JPEG', quality=quality, optimize=True)
                    self.logger.info(f"Saved optimized image to: {temp_file.name}")
                    return temp_file.name
                    
        except Exception as e:
            self.logger.error(f"Error optimizing image: {e}")
            print(f"⚠️ Image optimization failed, using original: {e}")
            return image_path
    
    def process_monument_recognition(self, image_path, max_retries=2):
        """Process image with monument recognition tool"""
        if not self.image_retrieval_tool:
            error_msg = "❌ Image Retrieval Tool not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🎯 Processing with Monument Recognition...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing image with monument recognition (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
                # Validate image file exists and is readable
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Check file size (limit to 10MB)
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:
                    raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
                
                if self.verbose:
                    print(f"🔍 Processing monument recognition for: {image_path}")
                
                result = self.image_retrieval_tool.forward(image_path)
                self.logger.info("Monument recognition completed successfully")
                print("✅ Monument recognition complete")
                return result
                
            except Exception as e:
                self.logger.error(f"Monument recognition failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ Monument Recognition Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check if the image file is valid and not corrupted\n"
                    error_details += "- Ensure sufficient GPU/CPU memory is available\n"
                    error_details += "- Verify all required models are properly loaded\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    print(f"⚠️ Monument recognition failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)  # Progressive backoff

    def process_localization(self, image_path, max_retries=2):
        """Process image with localization tool"""
        if not self.localizator:
            error_msg = "❌ Localizator not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🌍 Processing with Localizator...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing localization (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
                result = self.localizator.forward(image_path)
                self.logger.info("Localization completed successfully")
                print("✅ Localization complete")
                return result
                
            except Exception as e:
                self.logger.error(f"Localization failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ Localization Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check internet connectivity for GeoCLIP\n"
                    error_details += "- Verify models are properly loaded\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ Localization failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)

    def process_with_monument_rag(self, monument_name, monument_description, question, max_retries=2):
        """Process with RAG system using monument info"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🔍 Processing with RAG system...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with RAG system (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate inputs
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    raise ValueError("Monument name is empty or unknown")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                if self.verbose:
                    print(f"🏛️ Monument: {monument_name}")
                    print(f"📝 Description: {monument_description[:100]}...")
                
                # Usa il nuovo metodo per processare query sui monumenti
                result = self.rag_system.process_monument_query(monument_name, monument_description, question)
                
                print("✅ RAG processing complete")
                return result
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ RAG Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check if monument information is valid\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ RAG processing failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
    
    def extract_coordinates_from_localization(self, localization_result):
        """Estrae coordinate dal risultato della geolocalizzazione"""
        import re
        coordinates = {"lat": None, "lon": None, "source": "unknown"}
        
        try:
            # Cerca pattern di coordinate nel testo
            # Pattern per "Lat: X.XXX, Lon: Y.YYY"
            coord_pattern = r'Lat:\s*([-+]?\d*\.?\d+),\s*Lon:\s*([-+]?\d*\.?\d+)'
            matches = re.findall(coord_pattern, localization_result)
            
            if matches:
                lat, lon = matches[0]  # Prende la prima coppia trovata
                coordinates["lat"] = float(lat)
                coordinates["lon"] = float(lon)
                
                # Determina la fonte
                if "StreetCLIP" in localization_result:
                    coordinates["source"] = "StreetCLIP"
                elif "GeoCLIP" in localization_result:
                    coordinates["source"] = "GeoCLIP"
                
                print(f"✅ Coordinate estratte: {coordinates}")
            else:
                print("⚠️ Nessuna coordinata trovata nel risultato della geolocalizzazione")
                
        except Exception as e:
            print(f"❌ Errore nell'estrazione delle coordinate: {e}")
            
        return coordinates
    
    def generate_query_rewrites(self, monument_name, monument_description, question, max_retries=2):
        """Genera 2 riscritture della query usando LLM per migliorare i risultati RAG"""
        if not self.rag_system:
            self.logger.error("RAG system not available for query rewriting")
            return []
        
        print("🔄 Generating query rewrites using LLM...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Generating query rewrites (attempt {attempt + 1}/{max_retries + 1})")
                
                # Query originale che normalmente useremo per RAG
                original_query = f"Monument: {monument_name}. Description: {monument_description}"
                
                # System prompt per generare riscritture
                system_prompt = (
                    "You are an expert in information retrieval and query rewriting. "
                    "Your task is to rewrite the given monument search query into 2 different alternative versions "
                    "that could help find more relevant historical and cultural information. "
                    "Each rewrite should interpret the user's question and monument information from a different angle "
                    "to maximize information retrieval coverage. "
                    "Focus on: historical context, architectural details, cultural significance, geographical context, "
                    "construction details, historical events, and tourist information. "
                    "Respond with exactly 2 alternative queries, one per line, without numbering or formatting."
                )
                
                # User prompt con contesto completo
                user_prompt = (
                    f"Original monument query: '{original_query}'\n"
                    f"User question: '{question}'\n\n"
                    f"Generate 2 alternative search queries that could retrieve relevant information "
                    f"to answer the user's question about this monument. Each query should focus on different aspects "
                    f"(e.g., historical context, architectural style, cultural significance, construction details, etc.)."
                )
                
                if self.verbose:
                    print(f"🏛️ Original query: {original_query}")
                    print(f"❓ User question: {question}")
                
                # Genera le riscritture usando Smolagents
                try:
                    rewrite_response = self.rag_system.generate_with_smolagent(
                        system_prompt=system_prompt,
                        user_query=user_prompt,
                        context_passages=[]  # Non servono passage per questa task
                    )
                    
                    # Parse delle riscritture (una per riga)
                    rewrites = []
                    for line in rewrite_response.strip().split('\n'):
                        line = line.strip()
                        if line and len(line) > 10:  # Filtra linee vuote e troppo corte
                            # Rimuovi numerazione se presente
                            if line.startswith(('1.', '2.', '3.', '•', '-', '*')):
                                line = line.split('.', 1)[1].strip() if '.' in line else line[2:].strip()
                            rewrites.append(line)
                    
                    # Prendi solo le prime 2 riscritture
                    rewrites = rewrites[:2]
                    
                    if len(rewrites) >= 2:
                        self.logger.info(f"Successfully generated {len(rewrites)} query rewrites")
                        if self.verbose:
                            for i, rewrite in enumerate(rewrites, 1):
                                print(f"🔄 Rewrite {i}: {rewrite}")
                        
                        print(f"✅ Generated {len(rewrites)} query rewrites")
                        return rewrites
                    else:
                        raise ValueError(f"Only {len(rewrites)} rewrites generated, need 2")
                        
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents rewrite generation failed: {smolagent_error}")
                    
                    # Fallback: genera riscritture semplici basate su template
                    fallback_rewrites = [
                        f"Historical architecture and construction of {monument_name} monument cultural heritage site",
                        f"Tourist information cultural significance events {monument_name} geographical location historical context"
                    ]
                    
                    self.logger.info("Using fallback template-based rewrites")
                    print("⚠️ Using fallback template-based rewrites")
                    return fallback_rewrites
                
            except Exception as e:
                self.logger.error(f"Query rewrite generation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    self.logger.warning("All rewrite attempts failed, returning empty list")
                    print("❌ Query rewrite generation failed - will use original query only")
                    return []
                else:
                    print(f"⚠️ Rewrite generation failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
        
        return []
    
    def process_with_voting_rag(self, monument_name, monument_description, question, top_k=5, max_retries=2):
        """Process with RAG using voting mechanism with query rewrites"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🗳️ Processing with RAG system using voting mechanism...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with voting RAG (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate inputs
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    raise ValueError("Monument name is empty or unknown")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                # Database di testi separati per ogni monumento (stesso del metodo originale)
                monument_texts_database = {
                    "Colosseo": [
                        "The Colosseum is an oval amphitheatre in the centre of Rome, Italy. Built of travertine limestone, tuff, and brick-faced concrete, it was the largest amphitheatre ever built. The Colosseum is situated just east of the Roman Forum. Construction began under the emperor Vespasian in AD 72 and was completed in AD 80 under his successor and heir, Titus.",
                        "The image shows the Colosseum in Rome, Italy, under a clear blue sky. The ancient amphitheater, built of stone and concrete, stands majestically with its iconic arches and partially ruined outer walls. Tourists are gathered around the base, some taking photographs, while others listen to guides explaining the history of the site.",
                        "The Colosseum could hold an estimated 50,000 to 80,000 spectators at various points in its history, having an average audience of some 65,000. It was used for gladiatorial contests and public spectacles including animal hunts, executions, re-enactments of famous battles, and dramas based on Classical mythology.",
                        "The building ceased to be used for entertainment in the early medieval era. It was later reused for various purposes such as housing, workshops, quarters for a religious order, a fortress, a quarry, and a Christian shrine. The Colosseum is now a major tourist attraction in Rome with thousands of tourists each year."
                    ],
                    "Tour Eiffel": [
                        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals.",
                        "The Eiffel Tower stands 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and is the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world.",
                        "The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels.",
                        "The Eiffel Tower has become the global cultural icon of France and one of the most recognizable structures in the world. The tower has been visited by over 250 million people since its construction. The tower receives around 6 million visitors annually, making it the most-visited paid monument in the world."
                    ],
                    "Statua della Libertà": [
                        "The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor in New York City, United States. The copper statue, a gift from the people of France to the people of the United States, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel.",
                        "The statue is a figure of Libertas, a robed Roman liberty goddess. She holds a torch above her head with her right hand, and in her left hand carries a tabula ansata inscribed JULY IV MDCCLXXVI (July 4, 1776 in Roman numerals), the date of the U.S. Declaration of Independence.",
                        "The Statue of Liberty was dedicated on October 28, 1886. The statue became an icon of freedom and of the United States, seen as a symbol of welcome to immigrants arriving by sea. Bartholdi was inspired by a French law professor and politician, Édouard René de Laboulaye, who is said to have commented in 1865 that any monument raised to U.S. independence would properly be a joint project of the French and American peoples.",
                        "Standing 305 feet tall including its pedestal, the Statue of Liberty has welcomed millions of immigrants to the United States since 1886. The statue's torch-bearing arm was displayed at the Centennial Exposition in Philadelphia in 1876, and in Madison Square Park in Manhattan from 1876 to 1882."
                    ],
                    "Big Ben": [
                        "Big Ben is the nickname for the Great Bell of the Great Clock of Westminster, and, by extension, for the clock tower itself, which stands at the north end of the Palace of Westminster in London, England. Officially, the tower is called Elizabeth Tower, renamed to celebrate the Diamond Jubilee of Elizabeth II in 2012.",
                        "The tower was designed by Augustus Pugin in a neo-Gothic style. When completed in 1859, its clock was the largest and most accurate four-faced striking and chiming clock in the world. The tower stands 316 feet (96 m) tall, and the climb from ground level to the belfry is 334 steps.",
                        "The clock and dials were designed by Augustus Pugin. The clock dials are set in an iron framework 23 feet (7.0 m) in diameter, supporting 312 pieces of opal glass, rather like a stained-glass window. Some of the glass pieces may be removed for maintenance of the hands.",
                        "Big Ben is one of London's most famous landmarks and has appeared in many films. The sound of Big Ben chiming has become associated with London and Britain in general, often used by broadcasters as an audio symbol for London or as a time signal before news broadcasts."
                    ],
                    "Cristo Redentore": [
                        "Christ the Redeemer is an Art Deco statue of Jesus Christ in Rio de Janeiro, Brazil, created by French sculptor Paul Landowski and built by Brazilian engineer Heitor da Silva Costa, in collaboration with French engineer Albert Caquot. Romanian sculptor Gheorghe Leonida fashioned the face.",
                        "The statue is 30 metres (98 ft) tall, excluding its 8-metre (26 ft) pedestal. The arms stretch 28 metres (92 ft) wide. It is made of reinforced concrete and soapstone. Christ the Redeemer weighs 635 tonnes (625 long tons; 700 short tons), and is located at the peak of the 700-metre (2,300 ft) Corcovado mountain.",
                        "Construction of Christ the Redeemer began in 1922 and was completed in 1931. The monument was opened on October 12, 1931. The statue has become a cultural icon of both Rio de Janeiro and Brazil, and is listed as one of the New Seven Wonders of the World.",
                        "The statue overlooks the city of Rio de Janeiro from the summit of Mount Corcovado in the Tijuca National Park. It has become a symbol of Christianity across the world and is often considered the largest Art Deco statue in the world. The statue is illuminated every night and can be seen from many parts of the city."
                    ]
                }
                
                # Raccoglie tutti i testi del database
                all_monument_texts = []
                for monument, texts in monument_texts_database.items():
                    all_monument_texts.extend(texts)
                
                if self.verbose:
                    print(f"📚 Created RAG database with {len(all_monument_texts)} texts")
                
                # Build index with all monument texts (una volta sola)
                self.rag_system.build_index(all_monument_texts)
                
                # Query originale
                original_query = f"Monument: {monument_name}. Description: {monument_description}"
                
                # Genera le riscritture
                query_rewrites = self.generate_query_rewrites(monument_name, monument_description, question)
                
                # Tutte le query da testare (originale + riscritture)
                all_queries = [original_query] + query_rewrites
                
                if self.verbose:
                    print(f"🔍 Testing {len(all_queries)} queries total")
                    for i, q in enumerate(all_queries):
                        query_type = "Original" if i == 0 else f"Rewrite {i}"
                        print(f"  {query_type}: {q}")
                
                # Risultati per ogni query
                all_results = {}
                
                for i, query in enumerate(all_queries):
                    query_name = "original" if i == 0 else f"rewrite_{i}"
                    
                    try:
                        results = self.rag_system.query(query, top_k=max(5, top_k))  # Prendiamo almeno 5 o top_k se maggiore
                        all_results[query_name] = results
                        
                        if self.verbose:
                            query_type = "Original" if i == 0 else f"Rewrite {i}"
                            print(f"🎯 {query_type} found {len(results)} results")
                            
                    except Exception as query_error:
                        self.logger.warning(f"Query {query_name} failed: {query_error}")
                        all_results[query_name] = []
                
                # Sistema di voting per trovare i migliori testi
                voted_results = self.vote_for_best_passages(all_results, top_k=top_k)
                
                if not voted_results:
                    raise ValueError("No results found after voting mechanism")
                
                # Prepara informazioni per LLM
                top_k_texts = [passage for _, passage in voted_results]
                
                if self.verbose:
                    print(f"🗳️ Voting selected top {len(voted_results)} texts:")
                    for i, (score, passage) in enumerate(voted_results, 1):
                        print(f"  {i}. Score: {score:.4f} - {passage[:60]}...")
                
                # Genera la risposta finale con Smolagents
                try:
                    system_prompt = (
                        f"You are a knowledgeable tourist guide specializing in monuments and cultural heritage. "
                        f"Based on image analysis, the identified monument is: '{monument_name}' with description: '{monument_description}'. "
                        f"You have been provided with the top 3 most relevant texts selected through an advanced voting mechanism "
                        f"from multiple query interpretations. Use ALL the provided information to answer comprehensively and accurately."
                    )
                    
                    enhanced_query = (
                        f"User Question: {question}\n\n"
                        f"Monument Identified: {monument_name}\n"
                        f"Monument Description: {monument_description}\n\n"
                        f"Answer using both the monument identification and the retrieved contextual information."
                    )
                    
                    smolagent_answer = self.rag_system.generate_with_smolagent(
                        system_prompt=system_prompt,
                        user_query=enhanced_query,
                        context_passages=top_k_texts
                    )
                    
                    result = f"🗳️ **RAG Analysis with Voting Mechanism**\n\n"
                    result += f"**Identified Monument:** {monument_name}\n"
                    result += f"**Monument Description:** {monument_description}\n"
                    result += f"**User Question:** {question}\n\n"
                    result += f"**Query Rewriting Process:**\n"
                    result += f"• Original query used\n"
                    result += f"• {len(query_rewrites)} alternative interpretations generated\n"
                    result += f"• Voting mechanism applied to select best passages\n\n"
                    result += f"**Top {len(voted_results)} Selected Texts (via voting):**\n"
                    for i, (score, passage) in enumerate(voted_results, 1):
                        result += f"**{i}. Confidence: {score:.4f}**\n{passage}\n\n"
                    result += f"**🧠 AI Answer:**\n{smolagent_answer}"
                    
                    self.logger.info("Voting RAG processing completed successfully")
                    print("✅ Voting RAG processing complete")
                    return result
                    
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents generation failed: {smolagent_error}")
                    # Fallback senza Smolagents
                    result = f"🗳️ **RAG Analysis with Voting Mechanism**\n\n"
                    result += f"**Identified Monument:** {monument_name}\n"
                    result += f"**Monument Description:** {monument_description}\n"
                    result += f"**User Question:** {question}\n\n"
                    result += f"**Query Rewriting Process:**\n"
                    result += f"• Original query used\n"
                    result += f"• {len(query_rewrites)} alternative interpretations generated\n"
                    result += f"• Voting mechanism applied to select best passages\n\n"
                    result += f"**Top {len(voted_results)} Selected Texts (via voting):**\n"
                    for i, (score, passage) in enumerate(voted_results, 1):
                        result += f"**{i}. Confidence: {score:.4f}**\n{passage}\n\n"
                    result += f"⚠️ **Note**: Could not generate AI answer with Smolagents: {smolagent_error}\n"
                    result += f"💡 **Tip**: The voted texts above contain the most relevant information."
                    
                    return result
                
            except Exception as e:
                self.logger.error(f"Voting RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **Voting RAG Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check if monument information is valid\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ Voting RAG processing failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
    
    def vote_for_best_passages(self, all_results, top_k=5):
        """Implementa il sistema di voting per selezionare i migliori passaggi
        
        Priorità di ordinamento:
        1. Numero di query che hanno trovato il testo (frequenza)
        2. Posizione media nelle ricerche (qualità)
        3. Punteggio di accuratezza più alto dall'originale (tiebreaker)
        """
        # Dizionario per tracciare le statistiche di ogni passaggio
        passage_stats = {}
        
        # Raccoglie statistiche per ogni passaggio
        for query_name, results in all_results.items():
            is_original = (query_name == "original")
            
            for rank, (score, passage) in enumerate(results):
                if passage not in passage_stats:
                    passage_stats[passage] = {
                        'frequency': 0,          # Quante query l'hanno trovato
                        'positions': [],         # Posizioni in cui è stato trovato
                        'scores': [],            # Tutti i punteggi ottenuti
                        'original_score': 0.0,   # Miglior punteggio dalla query originale
                        'found_by_original': False
                    }
                
                # Incrementa frequenza
                passage_stats[passage]['frequency'] += 1
                
                # Aggiungi posizione (0-indexed)
                passage_stats[passage]['positions'].append(rank)
                
                # Aggiungi punteggio
                passage_stats[passage]['scores'].append(score)
                
                # Traccia se trovato dalla query originale e il suo punteggio
                if is_original:
                    passage_stats[passage]['found_by_original'] = True
                    passage_stats[passage]['original_score'] = max(
                        passage_stats[passage]['original_score'], score
                    )
        
        # Calcola metriche finali per ogni passaggio
        final_passages = []
        for passage, stats in passage_stats.items():
            frequency = stats['frequency']
            avg_position = sum(stats['positions']) / len(stats['positions'])
            max_score = max(stats['scores'])
            original_score = stats['original_score'] if stats['found_by_original'] else 0.0
            
            final_passages.append({
                'passage': passage,
                'frequency': frequency,
                'avg_position': avg_position,
                'max_score': max_score,
                'original_score': original_score,
                'found_by_original': stats['found_by_original']
            })
        
        # Ordinamento con le priorità richieste:
        # 1. Frequenza (decrescente) - testi trovati più volte
        # 2. Posizione media (crescente) - testi in posizioni migliori
        # 3. Punteggio originale (decrescente) - migliore accuratezza dall'originale
        sorted_passages = sorted(
            final_passages,
            key=lambda x: (-x['frequency'], x['avg_position'], -x['original_score'])
        )
        
        # Prepara i risultati finali
        final_results = []
        for item in sorted_passages[:top_k]:
            # Usa il punteggio massimo ottenuto come punteggio finale
            final_score = item['max_score']
            final_results.append((final_score, item['passage']))
        
        if self.verbose:
            print(f"🗳️ Voting results summary (top {top_k}):")
            for i, item in enumerate(sorted_passages[:top_k], 1):
                freq = item['frequency']
                avg_pos = item['avg_position']
                orig_score = item['original_score']
                found_by_orig = "✓" if item['found_by_original'] else "✗"
                print(f"  {i}. Freq: {freq}, Avg.Pos: {avg_pos:.1f}, Orig: {found_by_orig} ({orig_score:.4f})")
        
        return final_results

    def process_with_monument_rag_predefined(self, monument_name, monument_description, question, top_k=5, max_retries=2):
        """Process with RAG system using predefined texts from rag_system_smolagent.py"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🔍 Processing with RAG system using predefined monument texts...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with predefined RAG (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate inputs
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    raise ValueError("Monument name is empty or unknown")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                if self.verbose:
                    print(f"🏛️ Monument: {monument_name}")
                    print(f"📝 Using predefined texts from monument database")
                
                # Database di testi separati per ogni monumento
                monument_texts_database = {
                    "Colosseo": [
                        "The Colosseum is an oval amphitheatre in the centre of Rome, Italy. Built of travertine limestone, tuff, and brick-faced concrete, it was the largest amphitheatre ever built. The Colosseum is situated just east of the Roman Forum. Construction began under the emperor Vespasian in AD 72 and was completed in AD 80 under his successor and heir, Titus.",
                        "The image shows the Colosseum in Rome, Italy, under a clear blue sky. The ancient amphitheater, built of stone and concrete, stands majestically with its iconic arches and partially ruined outer walls. Tourists are gathered around the base, some taking photographs, while others listen to guides explaining the history of the site.",
                        "The Colosseum could hold an estimated 50,000 to 80,000 spectators at various points in its history, having an average audience of some 65,000. It was used for gladiatorial contests and public spectacles including animal hunts, executions, re-enactments of famous battles, and dramas based on Classical mythology.",
                        "The building ceased to be used for entertainment in the early medieval era. It was later reused for various purposes such as housing, workshops, quarters for a religious order, a fortress, a quarry, and a Christian shrine. The Colosseum is now a major tourist attraction in Rome with thousands of tourists each year."
                    ],
                    "Tour Eiffel": [
                        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals.",
                        "The Eiffel Tower stands 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and is the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world.",
                        "The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 m (906 ft) above the ground – the highest observation deck accessible to the public in the European Union. Tickets can be purchased to ascend by stairs or lift to the first and second levels.",
                        "The Eiffel Tower has become the global cultural icon of France and one of the most recognizable structures in the world. The tower has been visited by over 250 million people since its construction. The tower receives around 6 million visitors annually, making it the most-visited paid monument in the world."
                    ],
                    "Statua della Libertà": [
                        "The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor in New York City, United States. The copper statue, a gift from the people of France to the people of the United States, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel.",
                        "The statue is a figure of Libertas, a robed Roman liberty goddess. She holds a torch above her head with her right hand, and in her left hand carries a tabula ansata inscribed JULY IV MDCCLXXVI (July 4, 1776 in Roman numerals), the date of the U.S. Declaration of Independence.",
                        "The Statue of Liberty was dedicated on October 28, 1886. The statue became an icon of freedom and of the United States, seen as a symbol of welcome to immigrants arriving by sea. Bartholdi was inspired by a French law professor and politician, Édouard René de Laboulaye, who is said to have commented in 1865 that any monument raised to U.S. independence would properly be a joint project of the French and American peoples.",
                        "Standing 305 feet tall including its pedestal, the Statue of Liberty has welcomed millions of immigrants to the United States since 1886. The statue's torch-bearing arm was displayed at the Centennial Exposition in Philadelphia in 1876, and in Madison Square Park in Manhattan from 1876 to 1882."
                    ],
                    "Big Ben": [
                        "Big Ben is the nickname for the Great Bell of the Great Clock of Westminster, and, by extension, for the clock tower itself, which stands at the north end of the Palace of Westminster in London, England. Officially, the tower is called Elizabeth Tower, renamed to celebrate the Diamond Jubilee of Elizabeth II in 2012.",
                        "The tower was designed by Augustus Pugin in a neo-Gothic style. When completed in 1859, its clock was the largest and most accurate four-faced striking and chiming clock in the world. The tower stands 316 feet (96 m) tall, and the climb from ground level to the belfry is 334 steps.",
                        "The clock and dials were designed by Augustus Pugin. The clock dials are set in an iron framework 23 feet (7.0 m) in diameter, supporting 312 pieces of opal glass, rather like a stained-glass window. Some of the glass pieces may be removed for maintenance of the hands.",
                        "Big Ben is one of London's most famous landmarks and has appeared in many films. The sound of Big Ben chiming has become associated with London and Britain in general, often used by broadcasters as an audio symbol for London or as a time signal before news broadcasts."
                    ],
                    "Cristo Redentore": [
                        "Christ the Redeemer is an Art Deco statue of Jesus Christ in Rio de Janeiro, Brazil, created by French sculptor Paul Landowski and built by Brazilian engineer Heitor da Silva Costa, in collaboration with French engineer Albert Caquot. Romanian sculptor Gheorghe Leonida fashioned the face.",
                        "The statue is 30 metres (98 ft) tall, excluding its 8-metre (26 ft) pedestal. The arms stretch 28 metres (92 ft) wide. It is made of reinforced concrete and soapstone. Christ the Redeemer weighs 635 tonnes (625 long tons; 700 short tons), and is located at the peak of the 700-metre (2,300 ft) Corcovado mountain.",
                        "Construction of Christ the Redeemer began in 1922 and was completed in 1931. The monument was opened on October 12, 1931. The statue has become a cultural icon of both Rio de Janeiro and Brazil, and is listed as one of the New Seven Wonders of the World.",
                        "The statue overlooks the city of Rio de Janeiro from the summit of Mount Corcovado in the Tijuca National Park. It has become a symbol of Christianity across the world and is often considered the largest Art Deco statue in the world. The statue is illuminated every night and can be seen from many parts of the city."
                    ]
                }
                
                # Crea il query text dal monumento identificato
                monument_query_text = f"Monument: {monument_name}. Description: {monument_description}"
                
                # Raccoglie tutti i testi del database
                all_monument_texts = []
                for monument, texts in monument_texts_database.items():
                    all_monument_texts.extend(texts)
                
                if self.verbose:
                    print(f"📚 Created RAG database with {len(all_monument_texts)} texts")
                    print(f"🔍 Query text: {monument_query_text}")
                
                # Build index with all monument texts
                self.rag_system.build_index(all_monument_texts)
                
                self.logger.info(f"Built RAG index with {len(all_monument_texts)} monument texts")
                if self.verbose:
                    print(f"📚 Built RAG index with {len(all_monument_texts)} monument texts")
                
                # Query the RAG system using monument name + description as query
                top_results = self.rag_system.query(monument_query_text, top_k=top_k)
                
                if not top_results:
                    raise ValueError("No relevant passages found in RAG query")
                
                # Prepare all information for LLM
                top_k_texts = [passage for _, passage in top_results]
                
                if self.verbose:
                    print(f"🎯 Found top {len(top_results)} most similar texts:")
                    for i, (score, passage) in enumerate(top_results, 1):
                        print(f"  {i}. Score: {score:.4f} - {passage[:60]}...")
                
                # Try to generate answer with Smolagents
                try:
                    # Enhanced system prompt with all information
                    system_prompt = (
                        f"You are a knowledgeable tourist guide specializing in monuments and cultural heritage. "
                        f"Based on image analysis, the identified monument is: '{monument_name}' with description: '{monument_description}'. "
                        f"You have been provided with the top 3 most relevant texts from the monument database. "
                        f"Use ALL the provided information (monument identification + retrieved texts) to answer the user's question comprehensively and accurately."
                    )
                    
                    # Enhanced user query with context
                    enhanced_query = (
                        f"User Question: {question}\n\n"
                        f"Monument Identified: {monument_name}\n"
                        f"Monument Description: {monument_description}\n\n"
                        f"Please answer the user's question using both the monument identification and the retrieved contextual information."
                    )
                    
                    smolagent_answer = self.rag_system.generate_with_smolagent(
                        system_prompt=system_prompt,
                        user_query=enhanced_query,
                        context_passages=top_k_texts
                    )
                    
                    result = f"🔍 **RAG Analysis with Monument Texts Database**\n\n"
                    result += f"**Identified Monument:** {monument_name}\n"
                    result += f"**Monument Description:** {monument_description}\n"
                    result += f"**User Question:** {question}\n\n"
                    result += f"**Top {len(top_results)} Retrieved Texts:**\n"
                    for i, (score, passage) in enumerate(top_results, 1):
                        result += f"**{i}. Similarity: {score:.4f}**\n{passage}\n\n"
                    result += f"**🧠 AI Answer:**\n{smolagent_answer}"
                    
                    self.logger.info("RAG processing completed successfully with monument texts database")
                    
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents generation failed: {smolagent_error}")
                    # Fallback without Smolagents  
                    result = f"🔍 **RAG Analysis with Monument Texts Database**\n\n"
                    result += f"**Identified Monument:** {monument_name}\n"
                    result += f"**Monument Description:** {monument_description}\n"
                    result += f"**User Question:** {question}\n\n"
                    result += f"**Top {len(top_results)} Retrieved Texts:**\n"
                    for i, (score, passage) in enumerate(top_results, 1):
                        result += f"**{i}. Similarity: {score:.4f}**\n{passage}\n\n"
                    result += f"⚠️ **Note**: Could not generate AI answer with Smolagents: {smolagent_error}\n"
                    result += f"💡 **Tip**: The retrieved texts above contain relevant information to answer your question."
                
                print("✅ RAG processing complete")
                return result
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ RAG Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check if monument information is valid\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ RAG processing failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
    
    def process_with_arco_and_coordinates(self, monument_name, coordinates, max_retries=2):
        """Process with ARCO using monument name, fallback to coordinates if needed"""
        print("🏛️ Processing with ARCO database (with coordinate fallback)...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with ARCO database (attempt {attempt + 1}/{max_retries + 1})")
                
                arco_output = "🏛️ **ARCO Database Results with Coordinate Fallback**\n\n"
                
                # Prima prova con il nome del monumento
                if monument_name and monument_name.strip() != "" and monument_name != "Unknown":
                    self.logger.info(f"Querying ARCO database for monument: {monument_name}")
                    arco_output += f"**Primary Search: {monument_name}**\n"
                    
                    try:
                        results = query_by_name(monument_name)
                        
                        if results:
                            arco_output += f"✅ Found {len(results)} results by monument name:\n"
                            
                            for i, result in enumerate(results[:5], 1):  # Limit to 5 results
                                try:
                                    entity = result.get("entity", {}).get("value", "N/A")
                                    label = result.get("label", {}).get("value", "N/A")
                                    arco_output += f"  {i}. **{label}**\n"
                                    arco_output += f"     URI: {entity}\n"
                                except Exception as result_error:
                                    self.logger.warning(f"Error processing ARCO result {i}: {result_error}")
                                    arco_output += f"  {i}. **Error processing result**\n"
                            arco_output += "\n"
                            
                            print("✅ ARCO processing complete (found by monument name)")
                            return arco_output
                            
                        else:
                            arco_output += "❌ No results found by monument name\n\n"
                            
                    except Exception as query_error:
                        self.logger.warning(f"ARCO query failed for {monument_name}: {query_error}")
                        arco_output += f"⚠️ Monument name query failed: {str(query_error)}\n\n"
                
                # Fallback: prova con le coordinate
                if coordinates.get("lat") is not None and coordinates.get("lon") is not None:
                    arco_output += f"**Fallback Search: Geographic Coordinates**\n"
                    arco_output += f"Latitude: {coordinates['lat']}, Longitude: {coordinates['lon']}\n"
                    arco_output += f"Source: {coordinates['source']}\n\n"
                    
                    # Determina la città/regione dalle coordinate (approximazione)
                    location_name = self.get_location_name_from_coordinates(coordinates)
                    
                    if location_name:
                        self.logger.info(f"Querying ARCO database for location: {location_name}")
                        arco_output += f"**Searching ARCO for location: {location_name}**\n"
                        
                        try:
                            location_results = query_by_name(location_name)
                            
                            if location_results:
                                arco_output += f"✅ Found {len(location_results)} results by location:\n"
                                
                                for i, result in enumerate(location_results[:3], 1):  # Limit to 3 results
                                    try:
                                        entity = result.get("entity", {}).get("value", "N/A")
                                        label = result.get("label", {}).get("value", "N/A")
                                        arco_output += f"  {i}. **{label}**\n"
                                        arco_output += f"     URI: {entity}\n"
                                    except Exception as result_error:
                                        self.logger.warning(f"Error processing location result {i}: {result_error}")
                                        arco_output += f"  {i}. **Error processing result**\n"
                                arco_output += "\n"
                            else:
                                arco_output += "❌ No results found by geographic location\n\n"
                                
                        except Exception as location_error:
                            self.logger.warning(f"ARCO location query failed for {location_name}: {location_error}")
                            arco_output += f"⚠️ Location query failed: {str(location_error)}\n\n"
                    else:
                        arco_output += "❌ Could not determine location name from coordinates\n\n"
                else:
                    arco_output += "❌ No valid coordinates available for fallback search\n\n"
                
                # Se arriviamo qui, nessuna strategia ha funzionato
                arco_output += "💡 **Suggestions:**\n"
                arco_output += "- Verify monument name spelling\n"
                arco_output += "- Check internet connectivity\n"
                arco_output += "- Try manual search in ARCO database\n"
                
                self.logger.info("ARCO processing completed (no results found)")
                print("⚠️ ARCO processing complete (no results found)")
                return arco_output
                
            except Exception as e:
                self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **ARCO Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check internet connectivity for ARCO database access\n"
                    error_details += "- Verify SPARQL endpoints are accessible\n"
                    error_details += "- Try again later if database is temporarily unavailable\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ ARCO processing failed, retrying in {2 + attempt} seconds...")
                    time.sleep(2 + attempt)
    
    def get_location_name_from_coordinates(self, coordinates):
        """Determina il nome della località dalle coordinate"""
        lat, lon = coordinates.get("lat"), coordinates.get("lon")
        
        # Mapping approssimativo di famosi monumenti per coordinate
        famous_locations = [
            ((41.8902, 12.4922), "Roma"),       # Colosseo
            ((48.8584, 2.2945), "Parigi"),      # Tour Eiffel  
            ((40.6892, -74.0445), "New York"), # Statua della Libertà
            ((51.5007, -0.1246), "Londra"),    # Big Ben
            ((-22.9519, -43.2105), "Rio de Janeiro")  # Cristo Redentore
        ]
        
        # Trova la località più vicina (con tolleranza di ~50km)
        tolerance = 0.5  # Circa 50km
        
        for (ref_lat, ref_lon), location in famous_locations:
            if (abs(lat - ref_lat) < tolerance and abs(lon - ref_lon) < tolerance):
                return location
        
        # Fallback generico basato su continente/regione
        if 40 <= lat <= 50 and -10 <= lon <= 30:
            return "Italia"
        elif 45 <= lat <= 55 and -5 <= lon <= 10:
            return "Francia" 
        elif 50 <= lat <= 60 and -10 <= lon <= 2:
            return "Regno Unito"
        elif 35 <= lat <= 45 and -80 <= lon <= -70:
            return "New York"
        elif -25 <= lat <= -20 and -45 <= lon <= -40:
            return "Brasile"
        
        return None
    
    def generate_final_response(self, monument_name, monument_description, coordinates, 
                               rag_result, arco_result, user_question, max_retries=2):
        """Genera una risposta finale combinando tutte le informazioni trovate"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available for final response generation"
            self.logger.error(error_msg)
            return error_msg
        
        print("🧠 Generating final AI response using all collected information...")
        
        # Controlla se ARCO ha avuto successo
        arco_success = self.check_arco_success(arco_result)
        
        if not arco_success:
            print("⚠️ ARCO search was not successful - skipping final response generation")
            return ("❌ **Final Response Skipped**\n\n"
                   "The final AI response is only generated when ARCO database search is successful.\n"
                   "ARCO search did not find sufficient results, so no final synthesis will be provided.\n\n"
                   "💡 **Tip**: The individual results from each step above contain detailed information about your query.")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Generating final response (attempt {attempt + 1}/{max_retries + 1})")
                
                # Estrai informazioni dalla localizzazione
                location_info = self.extract_location_info(coordinates)
                
                # Estrai testi brevi dall'image RAG (monument recognition)
                short_texts = self.extract_short_texts_from_monument_result(monument_description)
                
                # Estrai testi lunghi dal text RAG
                long_texts = self.extract_long_texts_from_rag_result(rag_result)
                
                # Estrai informazioni ARCO
                arco_info = self.extract_arco_info(arco_result)
                
                if self.verbose:
                    print(f"🏛️ Monument: {monument_name}")
                    print(f"📍 Location: {location_info}")
                    print(f"📝 Short texts: {len(short_texts)} items")
                    print(f"📚 Long texts: {len(long_texts)} items") 
                    print(f"🏛️ ARCO info: {len(arco_info)} items")
                
                # Costruisci il prompt completo per il final LLM
                system_prompt = self.build_final_system_prompt()
                user_prompt = self.build_final_user_prompt(
                    monument_name, monument_description, location_info,
                    short_texts, long_texts, arco_info, user_question
                )
                
                # Genera la risposta finale
                try:
                    final_answer = self.rag_system.generate_with_smolagent(
                        system_prompt=system_prompt,
                        user_query=user_prompt,
                        context_passages=long_texts  # Usa i testi lunghi come context principale
                    )
                    
                    result = f"🤖 **FINAL AI RESPONSE**\n\n"
                    result += f"**Question:** {user_question}\n"
                    result += f"**Monument:** {monument_name} ({location_info})\n\n"
                    result += f"**🎯 Comprehensive Answer:**\n{final_answer}\n\n"
                    result += f"**📋 Information Sources Used:**\n"
                    result += f"• Monument Recognition: {monument_name}\n"
                    result += f"• Geolocation: {location_info}\n"
                    result += f"• Text Database: {len(long_texts)} relevant passages\n"
                    result += f"• ARCO Database: {len(arco_info)} cultural heritage entries\n"
                    
                    self.logger.info("Final response generated successfully")
                    print("✅ Final response generation complete")
                    return result
                    
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents generation failed for final response: {smolagent_error}")
                    # Fallback response
                    result = f"🤖 **FINAL RESPONSE (Structured Summary)**\n\n"
                    result += f"**Question:** {user_question}\n"
                    result += f"**Monument:** {monument_name} ({location_info})\n\n"
                    result += f"**Key Information Found:**\n"
                    result += f"• Monument: {monument_description}\n"
                    result += f"• Location: {location_info}\n"
                    
                    if long_texts:
                        result += f"• Historical Details: {len(long_texts)} detailed texts retrieved\n"
                        result += f"  - {long_texts[0][:100]}...\n"
                    
                    if arco_info:
                        result += f"• Related Heritage Sites: {len(arco_info)} entries from ARCO database\n"
                        result += f"  - {'; '.join(arco_info[:2])}\n"
                    
                    result += f"\n⚠️ **Note**: AI synthesis unavailable ({smolagent_error})\n"
                    result += f"💡 **Tip**: The information above provides comprehensive details to answer your question."
                    
                    return result
                
            except Exception as e:
                self.logger.error(f"Final response generation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **Final Response Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Available Information:**\n"
                    error_details += f"• Monument: {monument_name}\n"
                    error_details += f"• Location: {location_info if 'location_info' in locals() else 'Unknown'}\n"
                    error_details += "• Individual step results are available above\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ Final response generation failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
    
    def check_arco_success(self, arco_result):
        """Controlla se la ricerca ARCO ha avuto successo"""
        if not arco_result or "❌" in arco_result:
            return False
        
        # Cerca indicatori di successo
        success_indicators = ["✅ Found", "results by monument name", "results by location"]
        return any(indicator in arco_result for indicator in success_indicators)
    
    def extract_location_info(self, coordinates):
        """Estrae informazioni sulla localizzazione"""
        if coordinates.get("lat") and coordinates.get("lon"):
            location_name = self.get_location_name_from_coordinates(coordinates)
            if location_name:
                return f"{location_name} (Lat: {coordinates['lat']:.4f}, Lon: {coordinates['lon']:.4f})"
            else:
                return f"Lat: {coordinates['lat']:.4f}, Lon: {coordinates['lon']:.4f}"
        return "Location unknown"
    
    def extract_short_texts_from_monument_result(self, monument_description):
        """Estrae testi brevi dal riconoscimento monumenti"""
        return [monument_description] if monument_description else []
    
    def extract_long_texts_from_rag_result(self, rag_result):
        """Estrae testi lunghi dal risultato RAG"""
        import re
        long_texts = []
        
        # Cerca pattern dei testi nel risultato RAG
        # Pattern per "**1. Similarity: 0.XXXX**\nTESTO"
        pattern = r'\*\*\d+\.\s*Similarity:\s*[\d.]+\*\*\n(.*?)(?=\n\*\*\d+\.|$)'
        matches = re.findall(pattern, rag_result, re.DOTALL)
        
        for match in matches:
            text = match.strip()
            if len(text) > 50:  # Solo testi significativi
                long_texts.append(text)
        
        return long_texts[:3]  # Max 3 testi
    
    def extract_arco_info(self, arco_result):
        """Estrae informazioni ARCO (nomi di monumenti correlati)"""
        import re
        arco_info = []
        
        # Cerca pattern "**Nome Monumento**" nel risultato ARCO
        pattern = r'\*\*(.*?)\*\*'
        matches = re.findall(pattern, arco_result)
        
        for match in matches:
            # Filtra solo i nomi di monumenti/luoghi (non headers)
            if (not match.startswith("Primary Search") and 
                not match.startswith("Fallback Search") and
                not match.startswith("ARCO") and
                len(match) > 5 and len(match) < 100):
                arco_info.append(match.strip())
        
        return list(set(arco_info))  # Rimuovi duplicati
    
    def build_final_system_prompt(self):
        """Costruisce il system prompt per la risposta finale"""
        return (
            "You are an expert tourist guide and cultural heritage specialist. "
            "You have been provided with comprehensive information about a monument from multiple sources: "
            "image recognition, geolocation, historical texts, and cultural heritage databases. "
            "Your task is to provide a complete, engaging answer to the user's question using all available information. "
            "Use the ARCO database findings sparingly and as 'curious extras' to intrigue the user to ask more questions. "
            "Focus primarily on the monument identification and historical texts, with location context. "
            "Be informative, accurate, and slightly enticing about related discoveries."
        )
    
    def build_final_user_prompt(self, monument_name, monument_description, location_info,
                               short_texts, long_texts, arco_info, user_question):
        """Costruisce il prompt utente completo per la risposta finale"""
        prompt = f"""User Question: {user_question}

MONUMENT IDENTIFIED:
- Name: {monument_name}
- Description: {monument_description}
- Location: {location_info}

SHORT TEXTS (from image recognition):
"""
        for i, text in enumerate(short_texts, 1):
            prompt += f"{i}. {text}\n"
        
        prompt += f"\nLONG TEXTS (from historical database):\n"
        for i, text in enumerate(long_texts, 1):
            prompt += f"{i}. {text}\n"
        
        if arco_info:
            prompt += f"\nRELATED HERITAGE SITES (from ARCO - use sparingly as curious extras):\n"
            for i, info in enumerate(arco_info, 1):
                prompt += f"{i}. {info}\n"
        
        prompt += f"""
Please provide a comprehensive answer to the user's question using:
1. PRIMARY FOCUS: Monument identification + historical texts + location
2. SECONDARY: Mention ARCO findings briefly as "interesting related sites" to spark curiosity
3. Make the response engaging and informative
4. End with a subtle hint about the related sites that might interest them for further exploration
"""
        
        return prompt
    
    def process_with_monument_arco(self, monument_name, max_retries=2):
        """Process with ARCO knowledge graph using monument name"""
        print("🏛️ Processing with ARCO database...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with ARCO database (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate input
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    self.logger.warning("Monument name is empty or unknown")
                    return ("❌ ARCO Results: Monument name is empty or unknown\n\n"
                           "💡 Tip: Ensure the monument recognition step successfully identified a monument")
                
                self.logger.info(f"Querying ARCO database for: {monument_name}")
                if self.verbose:
                    print(f"🏛️ Searching ARCO for: {monument_name}")
                
                arco_output = "🏛️ **ARCO Database Results**\n\n"
                arco_output += f"**Searching for: {monument_name}**\n"
                
                try:
                    results = query_by_name(monument_name)
                    
                    if results:
                        arco_output += f"✅ Found {len(results)} results:\n"
                        
                        for i, result in enumerate(results[:5], 1):  # Limit to 5 results
                            try:
                                entity = result.get("entity", {}).get("value", "N/A")
                                label = result.get("label", {}).get("value", "N/A")
                                arco_output += f"  {i}. **{label}**\n"
                                arco_output += f"     URI: {entity}\n"
                            except Exception as result_error:
                                self.logger.warning(f"Error processing ARCO result {i}: {result_error}")
                                arco_output += f"  {i}. **Error processing result**\n"
                        arco_output += "\n"
                    else:
                        arco_output += "❌ No results found in ARCO database\n\n"
                        arco_output += "💡 **Suggestions:**\n"
                        arco_output += "- Try different search terms\n"
                        arco_output += "- Check internet connectivity\n"
                        
                except Exception as query_error:
                    self.logger.warning(f"ARCO query failed for {monument_name}: {query_error}")
                    arco_output += f"⚠️ Query failed: {str(query_error)}\n\n"
                
                self.logger.info("ARCO processing completed")
                print("✅ ARCO processing complete")
                return arco_output
                
            except Exception as e:
                self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ ARCO Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check internet connectivity for ARCO database access\n"
                    error_details += "- Verify SPARQL endpoints are accessible\n"
                    error_details += "- Try again later if database is temporarily unavailable\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    print(f"⚠️ ARCO processing failed, retrying in {2 + attempt} seconds...")
                    time.sleep(2 + attempt)

    # def OLD_process_with_agent(self, image_path, question, max_retries=2):
    #     """Process image with the agent system with enhanced error handling"""
    #     if not self.agent:
    #         error_msg = "❌ Agent system not available - initialization failed"
    #         self.logger.error(error_msg)
    #         return error_msg
        
    #     print("🤖 Processing with Agent system...")
        
    #     for attempt in range(max_retries + 1):
    #         try:
    #             self.logger.info(f"Processing image with agent (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
    #             # Validate image file exists and is readable
    #             if not os.path.exists(image_path):
    #                 raise FileNotFoundError(f"Image file not found: {image_path}")
                
    #             # Check file size (limit to 10MB)
    #             file_size = os.path.getsize(image_path)
    #             if file_size > 10 * 1024 * 1024:
    #                 raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
                
    #             # The agent expects a command that includes both localization and info extraction
    #             command = f"Localize the image: {image_path} and then give some info about it."
                
    #             if self.verbose:
    #                 print(f"🔍 Executing agent command: {command}")
                
    #             result = self.agent.run(command)
    #             self.logger.info("Agent processing completed successfully")
    #             print("✅ Agent analysis complete")
    #             return result
                
    #         except Exception as e:
    #             self.logger.error(f"Agent processing failed (attempt {attempt + 1}): {str(e)}")
                
    #             if attempt == max_retries:
    #                 # Final attempt failed
    #                 error_details = f"❌ Agent Error (after {max_retries + 1} attempts)\n"
    #                 error_details += f"Error Type: {type(e).__name__}\n"
    #                 error_details += f"Error Message: {str(e)}\n\n"
    #                 error_details += "Possible Solutions:\n"
    #                 error_details += "- Check if the image file is valid and not corrupted\n"
    #                 error_details += "- Ensure sufficient GPU/CPU memory is available\n"
    #                 error_details += "- Verify all required models are properly loaded\n"
    #                 if self.verbose:
    #                     error_details += f"\nTraceback:\n{traceback.format_exc()}"
    #                 return error_details
    #             else:
    #                 # Wait before retry
    #                 print(f"⚠️ Agent processing failed, retrying in {1 + attempt} seconds...")
    #                 time.sleep(1 + attempt)  # Progressive backoff
    
    # # METODI VECCHI (NON PIU' UTILIZZATI NEL NUOVO WORKFLOW)
    # def OLD_process_with_rag(self, agent_output, question, max_retries=2):
    #     """Process with RAG system using agent output as context with enhanced error handling"""
    #     if not self.rag_system:
    #         error_msg = "❌ RAG system not available - initialization failed"
    #         self.logger.error(error_msg)
    #         return error_msg
        
    #     print("🔍 Processing with RAG system...")
        
    #     for attempt in range(max_retries + 1):
    #         try:
    #             self.logger.info(f"Processing with RAG system (attempt {attempt + 1}/{max_retries + 1})")
                
    #             # Validate inputs
    #             if not agent_output or agent_output.strip() == "":
    #                 raise ValueError("Agent output is empty or invalid")
                
    #             if not question or question.strip() == "":
    #                 raise ValueError("Question is empty or invalid")
                
    #             # Extract meaningful text from agent output for RAG processing
    #             clean_agent_output = self._clean_text_for_rag(agent_output)
                
    #             if len(clean_agent_output.strip()) < 10:
    #                 raise ValueError("Cleaned agent output is too short for meaningful RAG processing")
                
    #             # Create passages from agent output
    #             passages = self.rag_system.split_text(clean_agent_output, chunk_size=200, overlap=50)
                
    #             if not passages:
    #                 raise ValueError("No valid text passages found from agent output")
                
    #             self.logger.info(f"Created {len(passages)} text passages for RAG indexing")
    #             if self.verbose:
    #                 print(f"📚 Created {len(passages)} text passages for RAG indexing")
                
    #             # Build index with the passages
    #             self.rag_system.build_index(passages)
                
    #             # Query the RAG system
    #             top_results = self.rag_system.query(question, top_k=3)
                
    #             if not top_results:
    #                 raise ValueError("No relevant passages found in RAG query")
                
    #             # Format results
    #             rag_output = f"🔍 RAG System Results\n\n"
    #             rag_output += f"Query: {question}\n"
    #             rag_output += f"Processed {len(passages)} passages from agent output\n\n"
    #             rag_output += "Top Retrieved Passages:\n\n"
                
    #             for i, (score, passage) in enumerate(top_results, 1):
    #                 rag_output += f"{i}. Similarity: {score:.4f}\n"
    #                 rag_output += f"{passage}\n\n"
                
    #             # Try to generate answer with Ollama if available
    #             try:
    #                 context_passages = [passage for _, passage in top_results]
    #                 system_prompt = ("You are a helpful assistant specializing in monuments and cultural heritage. "
    #                                "Use the provided context to answer the user's question about the monument or landmark.")
                    
    #                 smolagent_answer = self.rag_system.generate_with_smolagent(
    #                     system_prompt=system_prompt,
    #                     user_query=question,
    #                     context_passages=context_passages
    #                 )
                    
    #                 rag_output += f"🧠 Generated Answer:\n{smolagent_answer}"
    #                 self.logger.info("RAG processing completed successfully with Smolagents generation")
                    
    #             except Exception as smolagent_error:
    #                 self.logger.warning(f"Smolagents generation failed: {smolagent_error}")
    #                 rag_output += f"\n⚠️ Note: Could not generate answer with Smolagents: {smolagent_error}"
    #                 rag_output += f"\n💡 Tip: Ensure HF_TOKEN is set and Smolagents models are available"
                
    #             print("✅ RAG processing complete")
    #             return rag_output
                
    #         except Exception as e:
    #             self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
    #             if attempt == max_retries:
    #                 # Final attempt failed
    #                 error_details = f"❌ RAG Error (after {max_retries + 1} attempts)\n"
    #                 error_details += f"Error Type: {type(e).__name__}\n"
    #                 error_details += f"Error Message: {str(e)}\n\n"
    #                 error_details += "Possible Solutions:\n"
    #                 error_details += "- Check if agent output contains meaningful text\n"
    #                 error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
    #                 error_details += "- Verify sentence-transformers models are available\n"
    #                 error_details += "- Check available memory for FAISS indexing\n"
    #                 if self.verbose:
    #                     error_details += f"\nTraceback:\n{traceback.format_exc()}"
    #                 return error_details
    #             else:
    #                 # Wait before retry
    #                 print(f"⚠️ RAG processing failed, retrying in {1 + attempt} seconds...")
    #                 time.sleep(1 + attempt)
    
    # def OLD_process_with_arco(self, agent_output, max_retries=2):
    #     """Process with ARCO knowledge graph with enhanced error handling"""
    #     print("🏛️ Processing with ARCO database...")
        
    #     for attempt in range(max_retries + 1):
    #         try:
    #             self.logger.info(f"Processing with ARCO database (attempt {attempt + 1}/{max_retries + 1})")
                
    #             # Validate input
    #             if not agent_output or agent_output.strip() == "":
    #                 raise ValueError("Agent output is empty or invalid for ARCO processing")
                
    #             # Extract monument/landmark names from agent output for ARCO query
    #             monument_names = self._extract_monument_names(agent_output)
                
    #             if not monument_names:
    #                 self.logger.warning("No monument names found in agent output")
    #                 return ("❌ ARCO Results: No monument names found in agent output\n\n"
    #                        "💡 Tip: Ensure the agent successfully identified monuments or landmarks in the image")
                
    #             self.logger.info(f"Extracted {len(monument_names)} monument names: {monument_names[:3]}")
    #             if self.verbose:
    #                 print(f"🏛️ Found monument names: {', '.join(monument_names[:3])}")
                
    #             arco_output = "🏛️ ARCO Database Results\n\n"
    #             total_results = 0
                
    #             for monument_name in monument_names[:3]:  # Limit to top 3 names
    #                 arco_output += f"Searching for: {monument_name}\n"
    #                 self.logger.info(f"Querying ARCO database for: {monument_name}")
                    
    #                 # Add timeout and connection handling
    #                 try:
    #                     results = query_by_name(monument_name)
                        
    #                     if results:
    #                         arco_output += f"✅ Found {len(results)} results:\n"
    #                         total_results += len(results)
                            
    #                         for i, result in enumerate(results[:5], 1):  # Limit to 5 results per monument
    #                             try:
    #                                 entity = result.get("entity", {}).get("value", "N/A")
    #                                 label = result.get("label", {}).get("value", "N/A")
    #                                 arco_output += f"  {i}. {label}\n"
    #                                 arco_output += f"     URI: {entity}\n"
    #                             except Exception as result_error:
    #                                 self.logger.warning(f"Error processing ARCO result {i}: {result_error}")
    #                                 arco_output += f"  {i}. Error processing result\n"
    #                         arco_output += "\n"
    #                     else:
    #                         arco_output += "❌ No results found in ARCO database\n\n"
                            
    #                 except Exception as query_error:
    #                     self.logger.warning(f"ARCO query failed for {monument_name}: {query_error}")
    #                     arco_output += f"⚠️ Query failed: {str(query_error)}\n\n"
                
    #             if total_results == 0:
    #                 arco_output += "\n💡 Suggestions:\n"
    #                 arco_output += "- Try different search terms or monument names\n"
    #                 arco_output += "- Check internet connectivity for ARCO database access\n"
    #                 arco_output += "- Verify ARCO endpoints are accessible\n"
                
    #             self.logger.info(f"ARCO processing completed. Total results found: {total_results}")
    #             print("✅ ARCO processing complete")
    #             return arco_output
                
    #         except Exception as e:
    #             self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
    #             if attempt == max_retries:
    #                 # Final attempt failed
    #                 error_details = f"❌ ARCO Error (after {max_retries + 1} attempts)\n"
    #                 error_details += f"Error Type: {type(e).__name__}\n"
    #                 error_details += f"Error Message: {str(e)}\n\n"
    #                 error_details += "Possible Solutions:\n"
    #                 error_details += "- Check internet connectivity for ARCO database access\n"
    #                 error_details += "- Verify SPARQL endpoints are accessible\n"
    #                 error_details += "- Ensure agent output contains monument/landmark names\n"
    #                 error_details += "- Try again later if database is temporarily unavailable\n\n"
                    
    #                 # Add endpoint status information
    #                 error_details += "ARCO Endpoints:\n"
    #                 error_details += "- Primary: http://wit.istc.cnr.it/arco/virtuoso/sparql\n"
    #                 error_details += "- Backup: https://dati.beniculturali.it/sparql\n"
    #                 if self.verbose:
    #                     error_details += f"\nTraceback:\n{traceback.format_exc()}"
    #                 return error_details
    #             else:
    #                 # Wait before retry
    #                 print(f"⚠️ ARCO processing failed, retrying in {2 + attempt} seconds...")
    #                 time.sleep(2 + attempt)
    
    def _clean_text_for_rag(self, text):
        """Clean text by removing emojis and excessive formatting for RAG processing"""
        import re
        
        # Remove emoji patterns
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)
        
        # Clean the text
        cleaned = emoji_pattern.sub(' ', text)
        
        # Remove markdown formatting
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Bold
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Italic
        cleaned = re.sub(r'#{1,6}\s', '', cleaned)          # Headers
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)        # Code
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _extract_monument_names(self, text):
        """Extract potential monument/landmark names from text"""
        import re
        
        # Common patterns for monument names
        patterns = [
            r'(?:monument|landmark|building|structure|tower|bridge|castle|palace|church|cathedral|basilica|temple|mosque|statue|memorial|arch|gate|fountain|square|plaza|piazza)[\s\w]*',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Tower|Bridge|Castle|Palace|Church|Cathedral|Basilica|Temple|Mosque|Statue|Memorial|Arch|Gate|Fountain|Square|Plaza|Piazza))',
            r'(?:Colosseum|Colosseo|Tower of Pisa|Eiffel Tower|Statue of Liberty|Big Ben|Notre Dame|Sagrada Familia|Taj Mahal|Machu Picchu|Christ the Redeemer)'
        ]
        
        names = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and validate the match
                name = match.strip()
                if len(name) > 3 and name not in names:
                    names.append(name)
        
        # If no specific names found, try to extract capitalized words/phrases
        if not names:
            capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            matches = re.findall(capitalized_pattern, text)
            for match in matches:
                if len(match.split()) >= 2 and len(match) > 5:  # Multi-word names
                    names.append(match)
        
        return names[:5]  # Return top 5 names
    
    def export_results(self, agent_result, rag_result, arco_result, filename=None):
        """Export analysis results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if filename is None:
                filename = f"agentictravel_cli_results_{timestamp}.json"
            elif not filename.endswith('.json'):
                filename += '.json'
            
            export_data = {
                "timestamp": timestamp,
                "export_source": "CLI",
                "analysis_results": {
                    "agent_analysis": agent_result,
                    "rag_system": rag_result,
                    "arco_database": arco_result
                },
                "metadata": {
                    "export_version": "1.0",
                    "app_version": "AgenticTraveler CLI Enhanced",
                    "components": ["Agent System", "RAG System", "ARCO Database"]
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results exported to: {filename}")
            print(f"💾 Results exported to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            print(f"❌ Export failed: {e}")
            return None
    
    def analyze_image(self, image_path, question, export_file=None, use_rewrite=False, top_k=5):
        """Main analysis function using the NEW WORKFLOW"""
        start_time = time.time()
        
        print(f"🖼️ Analyzing image: {image_path}")
        print(f"❓ Question: {question}")
        if use_rewrite:
            print("🔄 Mode: Enhanced RAG with Query Rewriting & Voting")
        else:
            print("📖 Mode: Standard RAG with Predefined Texts")
        print("=" * 80)
        
        # Validate image exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None
        
        # Optimize image
        print("🔧 Optimizing image...")
        optimized_image_path = self.optimize_image_for_processing(image_path)
        
        try:
            # === NUOVO WORKFLOW ===
            # Step 1: Segmentazione e riconoscimento monumento
            print("\n" + "=" * 80)
            print("🎯 STEP 1: Monument Recognition")
            print("-" * 40)
            monument_result = self.process_monument_recognition(optimized_image_path)
            print(f"📊 MONUMENT RECOGNITION RESULTS:")
            print(monument_result)
            
            # Parse monument info
            monument_info = parse_monument_info(monument_result)
            monument_name = monument_info.get('monument_name', 'Unknown')
            monument_description = monument_info.get('monument_description', 'No description available')
            
            # Step 2: Geolocation
            print("\n" + "=" * 80)
            print("🎯 STEP 2: Geolocation")
            print("-" * 40)
            localization_result = self.process_localization(optimized_image_path)
            print(f"📊 GEOLOCATION RESULTS:")
            print(localization_result)
            
            # Extract coordinates from localization result
            coordinates = self.extract_coordinates_from_localization(localization_result)
            
            # Step 3: RAG Processing con informazioni monumento
            print("\n" + "=" * 80)
            if use_rewrite:
                print(f"🎯 STEP 3: RAG Analysis with Query Rewriting & Voting (top-{top_k})")
                print("-" * 40)
                rag_result = self.process_with_voting_rag(monument_name, monument_description, question, top_k=top_k)
            else:
                print(f"🎯 STEP 3: RAG Analysis with Predefined Texts (top-{top_k})")
                print("-" * 40)
                rag_result = self.process_with_monument_rag_predefined(monument_name, monument_description, question, top_k=top_k)
            print(f"📊 RAG SYSTEM RESULTS:")
            print(rag_result)
            
            # Step 4: ARCO Database query con fallback coordinate
            print("\n" + "=" * 80)
            print("🎯 STEP 4: ARCO Database Query (with coordinate fallback)")
            print("-" * 40)
            arco_result = self.process_with_arco_and_coordinates(monument_name, coordinates)
            print(f"📊 ARCO DATABASE RESULTS:")
            print(arco_result)
            
            # Step 5: Final AI Response (solo se ARCO ha successo)
            print("\n" + "=" * 80)
            print("🎯 STEP 5: Final AI Response")
            print("-" * 40)
            final_response = self.generate_final_response(
                monument_name, monument_description, coordinates, 
                rag_result, arco_result, question
            )
            print(f"📊 FINAL AI RESPONSE:")
            print(final_response)
            
            # Export results if requested
            if export_file:
                print("\n" + "=" * 80)
                exported_file = self.export_results(monument_result, rag_result, arco_result, export_file)
                if exported_file:
                    print(f"📁 Results available in: {exported_file}")
            
            # Processing summary
            end_time = time.time()
            processing_time = end_time - start_time
            print("\n" + "=" * 80)
            print(f"✅ Analysis completed in {processing_time:.1f} seconds")
            print(f"🏛️ Monument: {monument_name}")
            print(f"🧠 Systems used: Image: {'✅' if self.image_retrieval_tool else '❌'} | RAG: {'✅' if self.rag_system else '❌'} | ARCO: ✅ | Geo: {'✅' if self.localizator else '❌'}")
            
            return {
                "monument_result": monument_result,
                "rag_result": rag_result,
                "arco_result": arco_result,
                "localization_result": localization_result,
                "final_response": final_response,
                "processing_time": processing_time
            }
            
        finally:
            # Clean up optimized image if it was created
            if optimized_image_path != image_path:
                try:
                    os.unlink(optimized_image_path)
                    self.logger.info(f"Cleaned up temporary file: {optimized_image_path}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AgenticTraveler CLI - AI-powered monument and landmark analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --image colosseum.jpg
    python main.py --image tower.png --question "What is the architectural style?"
    python main.py --image landmark.jpg --export results.json
    python main.py --image monument.jpg --question "Tell me about its history" --export analysis.json --verbose
    python main.py --image monument.jpg --question "Tell me about its history" --rewrite --verbose
    python main.py --image monument.jpg --question "Tell me about its history" --rewrite --top-k 7
    
For more information, visit: https://github.com/your-repo/AgenticTraveler
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--image', '-i',
        required=True,
        type=str,
        help='Path to the image file to analyze'
    )
    
    # Optional arguments
    parser.add_argument(
        '--question', '-q',
        type=str,
        default="What monument or landmark is shown in this image?",
        help='Question to ask about the image (default: "What monument or landmark is shown in this image?")'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export results to JSON file (provide filename or use auto-generated name)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    
    parser.add_argument(
        '--rewrite', '-r',
        action='store_true',
        help='Enable query rewriting with LLM to generate alternative interpretations and improve RAG results'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of top passages to retrieve and use in RAG analysis (default: 5)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AgenticTraveler CLI v1.0 - Enhanced Edition'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize CLI application
        app = AgenticTravelerCLI(verbose=args.verbose)
        
        # Process the image
        results = app.analyze_image(
            image_path=args.image,
            question=args.question,
            export_file=args.export,
            use_rewrite=args.rewrite,
            top_k=args.top_k
        )
        
        if results:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            print(f"Traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())