"""
AgenticTraveler Gradio App - Enhanced Version

This Gradio application integrates the AgenticTraveler system components:
1. Agent System (agent_base_optim.py) - Image analysis and geolocation
2. RAG System (rag_sytem.py) - Document retrieval using semantic search
3. ARCO Knowledge Graph (ARCO_access.py) - Italian cultural heritage database queries

The app takes an image and a question as input, then provides results from all three systems
in separate output boxes for comprehensive monument/landmark analysis.

Enhanced Features:
- Improved error handling with retry mechanisms
- Progress indicators for long operations
- Optimized image processing with auto-resize
- Example images for quick testing
- Better output formatting
- Export functionality for results
"""

import gradio as gr
import os
import sys
import tempfile
import traceback
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO

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
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")

class AgenticTravelerApp:
    def __init__(self):
        """Initialize the AgenticTraveler application"""
        self.image_retrieval_tool = None
        self.localizator = None
        self.rag_system = None
        self.logger = self._setup_logging()
        self._initialize_systems()
    
    def _setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('agentictravel_app.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_systems(self):
        """Initialize the tools and RAG systems with retry mechanisms"""
        max_retries = 3
        
        # Initialize Image Retrieval Tool
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Image Retrieval Tool (attempt {attempt + 1}/{max_retries})...")
                print(f"🔄 Initializing Image Retrieval Tool (attempt {attempt + 1}/{max_retries})...")
                self.image_retrieval_tool = ImageRetrievalTool()
                self.logger.info("Image Retrieval Tool initialized successfully")
                print("✅ Image Retrieval Tool initialized successfully")
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
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Localizator (attempt {attempt + 1}/{max_retries})...")
                print(f"🔄 Initializing Localizator (attempt {attempt + 1}/{max_retries})...")
                self.localizator = Localizator()
                self.logger.info("Localizator initialized successfully")
                print("✅ Localizator initialized successfully")
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
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                print(f"🔄 Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                
                # Usa path configurabili
                import os
                embed_model_path = os.environ.get('EMBED_MODEL_PATH', "all-MiniLM-L6-v2")
                qwen_model_path = os.environ.get('QWEN_MODEL_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct")
                
                self.rag_system = RAGSmolagent(
                    embed_model_name=embed_model_path,
                    use_cross_encoder=False,
                    model_id=qwen_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self.logger.info("RAG system initialized successfully")
                print("✅ RAG system initialized successfully")
                break
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG system (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize RAG system after {max_retries} attempts: {e}")
                    self.rag_system = None
                else:
                    print(f"⚠️ RAG initialization failed, retrying in 2 seconds...")
                    time.sleep(2)
    
    # === NUOVO WORKFLOW METHODS ===
    
    def process_monument_recognition(self, image_path, max_retries=2):
        """Process image with monument recognition tool"""
        if not self.image_retrieval_tool:
            error_msg = "❌ Image Retrieval Tool not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing monument recognition (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:
                    raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
                
                result = self.image_retrieval_tool.forward(image_path)
                self.logger.info("Monument recognition completed successfully")
                return f"🎯 **Monument Recognition Results**\n\n{result}"
                
            except Exception as e:
                self.logger.error(f"Monument recognition failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **Monument Recognition Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check if the image file is valid and not corrupted\n"
                    error_details += "- Ensure sufficient GPU/CPU memory is available\n"
                    error_details += "- Verify all required models are properly loaded\n"
                    return error_details
                else:
                    time.sleep(1 + attempt)
    
    def process_localization(self, image_path, max_retries=2):
        """Process image with localization tool"""
        if not self.localizator:
            error_msg = "❌ Localizator not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing localization (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
                result = self.localizator.forward(image_path)
                self.logger.info("Localization completed successfully")
                return f"🌍 **Geolocation Results**\n\n{result}"
                
            except Exception as e:
                self.logger.error(f"Localization failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **Localization Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check internet connectivity for GeoCLIP\n"
                    error_details += "- Verify models are properly loaded\n"
                    return error_details
                else:
                    time.sleep(1 + attempt)

    def process_with_monument_rag(self, monument_name, monument_description, question, max_retries=2):
        """Process with RAG system using monument info"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with RAG system (attempt {attempt + 1}/{max_retries + 1})")
                
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    raise ValueError("Monument name is empty or unknown")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                # Usa il nuovo metodo per processare query sui monumenti
                result = self.rag_system.process_monument_query(monument_name, monument_description, question)
                
                self.logger.info("RAG processing completed successfully")
                return result
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **RAG Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check if monument information is valid\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    return error_details
                else:
                    time.sleep(1 + attempt)
    
    def process_with_monument_arco(self, monument_name, max_retries=2):
        """Process with ARCO knowledge graph using monument name"""
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with ARCO database (attempt {attempt + 1}/{max_retries + 1})")
                
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    self.logger.warning("Monument name is empty or unknown")
                    return ("❌ **ARCO Results**: Monument name is empty or unknown\n\n"
                           "💡 **Tip**: Ensure the monument recognition step successfully identified a monument")
                
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
                    return error_details
                else:
                    time.sleep(2 + attempt)
    
    def extract_coordinates_from_localization(self, localization_result):
        """Estrae coordinate dal risultato della geolocalizzazione"""
        import re
        coordinates = {"lat": None, "lon": None, "source": "unknown"}
        
        try:
            # Cerca pattern di coordinate nel testo
            coord_pattern = r'Lat:\s*([-+]?\d*\.?\d+),\s*Lon:\s*([-+]?\d*\.?\d+)'
            matches = re.findall(coord_pattern, localization_result)
            
            if matches:
                lat, lon = matches[0]
                coordinates["lat"] = float(lat)
                coordinates["lon"] = float(lon)
                
                if "StreetCLIP" in localization_result:
                    coordinates["source"] = "StreetCLIP"
                elif "GeoCLIP" in localization_result:
                    coordinates["source"] = "GeoCLIP"
                    
        except Exception as e:
            self.logger.error(f"Error extracting coordinates: {e}")
            
        return coordinates

    def process_with_monument_rag_predefined(self, monument_name, monument_description, question, max_retries=2):
        """Process with RAG system using predefined texts"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with predefined RAG (attempt {attempt + 1}/{max_retries + 1})")
                
                if not monument_name or monument_name.strip() == "" or monument_name == "Unknown":
                    raise ValueError("Monument name is empty or unknown")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
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
                
                # Build index with all monument texts
                self.rag_system.build_index(all_monument_texts)
                
                # Query using monument info as query text
                top_results = self.rag_system.query(monument_query_text, top_k=3)
                
                if not top_results:
                    raise ValueError("No relevant passages found in RAG query")
                
                # Prepare all information for LLM
                top_3_texts = [passage for _, passage in top_results]
                
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
                        context_passages=top_3_texts
                    )
                    
                    result = f"🔍 **RAG Analysis with Monument Texts Database**\n\n"
                    result += f"**Identified Monument:** {monument_name}\n"
                    result += f"**Monument Description:** {monument_description}\n"
                    result += f"**User Question:** {question}\n\n"
                    result += f"**Top 3 Retrieved Texts:**\n"
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
                    result += f"**Top 3 Retrieved Texts:**\n"
                    for i, (score, passage) in enumerate(top_results, 1):
                        result += f"**{i}. Similarity: {score:.4f}**\n{passage}\n\n"
                    result += f"⚠️ **Note**: Could not generate AI answer with Smolagents: {smolagent_error}\n"
                    result += f"💡 **Tip**: The retrieved texts above contain relevant information to answer your question."
                
                return result
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **RAG Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    return error_details
                else:
                    time.sleep(1 + attempt)

    def process_with_arco_and_coordinates(self, monument_name, coordinates, max_retries=2):
        """Process with ARCO using monument name, fallback to coordinates if needed"""
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
                            
                            for i, result in enumerate(results[:5], 1):
                                try:
                                    entity = result.get("entity", {}).get("value", "N/A")
                                    label = result.get("label", {}).get("value", "N/A")
                                    arco_output += f"  {i}. **{label}**\n"
                                    arco_output += f"     URI: {entity}\n"
                                except Exception as result_error:
                                    self.logger.warning(f"Error processing ARCO result {i}: {result_error}")
                                    arco_output += f"  {i}. **Error processing result**\n"
                            arco_output += "\n"
                            
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
                    
                    location_name = self.get_location_name_from_coordinates(coordinates)
                    
                    if location_name:
                        self.logger.info(f"Querying ARCO database for location: {location_name}")
                        arco_output += f"**Searching ARCO for location: {location_name}**\n"
                        
                        try:
                            location_results = query_by_name(location_name)
                            
                            if location_results:
                                arco_output += f"✅ Found {len(location_results)} results by location:\n"
                                
                                for i, result in enumerate(location_results[:3], 1):
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
                
                # Nessuna strategia ha funzionato
                arco_output += "💡 **Suggestions:**\n"
                arco_output += "- Verify monument name spelling\n"
                arco_output += "- Check internet connectivity\n"
                
                return arco_output
                
            except Exception as e:
                self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    error_details = f"❌ **ARCO Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n"
                    return error_details
                else:
                    time.sleep(2 + attempt)
    
    def get_location_name_from_coordinates(self, coordinates):
        """Determina il nome della località dalle coordinate"""
        lat, lon = coordinates.get("lat"), coordinates.get("lon")
        
        famous_locations = [
            ((41.8902, 12.4922), "Roma"),       # Colosseo
            ((48.8584, 2.2945), "Parigi"),      # Tour Eiffel  
            ((40.6892, -74.0445), "New York"), # Statua della Libertà
            ((51.5007, -0.1246), "Londra"),    # Big Ben
            ((-22.9519, -43.2105), "Rio de Janeiro")  # Cristo Redentore
        ]
        
        tolerance = 0.5  # Circa 50km
        
        for (ref_lat, ref_lon), location in famous_locations:
            if (abs(lat - ref_lat) < tolerance and abs(lon - ref_lon) < tolerance):
                return location
        
        # Fallback generico
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

    # === METODI VECCHI (NON PIU' UTILIZZATI) ===
    def OLD_process_with_agent(self, image_path, question, max_retries=2):
        """Process image with the agent system with enhanced error handling"""
        if not self.agent:
            error_msg = "❌ Agent system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing image with agent (attempt {attempt + 1}/{max_retries + 1}): {image_path}")
                
                # Validate image file exists and is readable
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Check file size (limit to 10MB)
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:
                    raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
                
                # The agent expects a command that includes both localization and info extraction
                command = f"Localize the image: {image_path} and then give some info about it."
                
                result = self.agent.run(command)
                self.logger.info("Agent processing completed successfully")
                return f"🤖 **Agent Analysis Results**\n\n{result}"
                
            except Exception as e:
                self.logger.error(f"Agent processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ **Agent Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check if the image file is valid and not corrupted\n"
                    error_details += "- Ensure sufficient GPU/CPU memory is available\n"
                    error_details += "- Verify all required models are properly loaded\n\n"
                    error_details += f"**Traceback:**\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    time.sleep(1 + attempt)  # Progressive backoff
    
    def process_with_rag(self, agent_output, question, max_retries=2):
        """Process with RAG system using agent output as context with enhanced error handling"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with RAG system (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate inputs
                if not agent_output or agent_output.strip() == "":
                    raise ValueError("Agent output is empty or invalid")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                # Extract meaningful text from agent output for RAG processing
                # Remove emoji and formatting to get clean text for embeddings
                clean_agent_output = self._clean_text_for_rag(agent_output)
                
                if len(clean_agent_output.strip()) < 10:
                    raise ValueError("Cleaned agent output is too short for meaningful RAG processing")
                
                # Create passages from agent output
                passages = self.rag_system.split_text(clean_agent_output, chunk_size=200, overlap=50)
                
                if not passages:
                    raise ValueError("No valid text passages found from agent output")
                
                self.logger.info(f"Created {len(passages)} text passages for RAG indexing")
                
                # Build index with the passages
                self.rag_system.build_index(passages)
                
                # Query the RAG system
                top_results = self.rag_system.query(question, top_k=3)
                
                if not top_results:
                    raise ValueError("No relevant passages found in RAG query")
                
                # Format results
                rag_output = "🔍 **RAG System Results**\n\n"
                rag_output += f"**Query**: {question}\n\n"
                rag_output += f"**Processed {len(passages)} passages from agent output**\n\n"
                rag_output += "**Top Retrieved Passages:**\n\n"
                
                for i, (score, passage) in enumerate(top_results, 1):
                    rag_output += f"**{i}. Similarity: {score:.4f}**\n"
                    rag_output += f"{passage}\n\n"
                
                # Try to generate answer with Ollama if available
                try:
                    context_passages = [passage for _, passage in top_results]
                    system_prompt = ("You are a helpful assistant specializing in monuments and cultural heritage. "
                                   "Use the provided context to answer the user's question about the monument or landmark.")
                    
                    smolagent_answer = self.rag_system.generate_with_smolagent(
                        system_prompt=system_prompt,
                        user_query=question,
                        context_passages=context_passages
                    )
                    
                    rag_output += f"**🧠 Generated Answer:**\n{smolagent_answer}"
                    self.logger.info("RAG processing completed successfully with Smolagents generation")
                    
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents generation failed: {smolagent_error}")
                    rag_output += f"\n⚠️ **Note**: Could not generate answer with Smolagents: {smolagent_error}"
                    rag_output += f"\n💡 **Tip**: Ensure HF_TOKEN is set and Smolagents models are available"
                
                return rag_output
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ **RAG Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check if agent output contains meaningful text\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    error_details += "- Check available memory for FAISS indexing\n\n"
                    error_details += f"**Traceback:**\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    time.sleep(1 + attempt)
    
    def process_with_arco(self, agent_output, max_retries=2):
        """Process with ARCO knowledge graph with enhanced error handling"""
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with ARCO database (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate input
                if not agent_output or agent_output.strip() == "":
                    raise ValueError("Agent output is empty or invalid for ARCO processing")
                
                # Extract monument/landmark names from agent output for ARCO query
                monument_names = self._extract_monument_names(agent_output)
                
                if not monument_names:
                    self.logger.warning("No monument names found in agent output")
                    return ("❌ **ARCO Results**: No monument names found in agent output\n\n"
                           "💡 **Tip**: Ensure the agent successfully identified monuments or landmarks in the image")
                
                self.logger.info(f"Extracted {len(monument_names)} monument names: {monument_names[:3]}")
                arco_output = "🏛️ **ARCO Database Results**\n\n"
                total_results = 0
                
                for monument_name in monument_names[:3]:  # Limit to top 3 names
                    arco_output += f"**Searching for: {monument_name}**\n"
                    self.logger.info(f"Querying ARCO database for: {monument_name}")
                    
                    # Add timeout and connection handling
                    try:
                        results = query_by_name(monument_name)
                        
                        if results:
                            arco_output += f"✅ Found {len(results)} results:\n"
                            total_results += len(results)
                            
                            for i, result in enumerate(results[:5], 1):  # Limit to 5 results per monument
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
                            
                    except Exception as query_error:
                        self.logger.warning(f"ARCO query failed for {monument_name}: {query_error}")
                        arco_output += f"⚠️ Query failed: {str(query_error)}\n\n"
                
                if total_results == 0:
                    arco_output += "\n💡 **Suggestions:**\n"
                    arco_output += "- Try different search terms or monument names\n"
                    arco_output += "- Check internet connectivity for ARCO database access\n"
                    arco_output += "- Verify ARCO endpoints are accessible\n"
                
                self.logger.info(f"ARCO processing completed. Total results found: {total_results}")
                return arco_output
                
            except Exception as e:
                self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ **ARCO Error** (after {max_retries + 1} attempts)\n\n"
                    error_details += f"**Error Type:** {type(e).__name__}\n"
                    error_details += f"**Error Message:** {str(e)}\n\n"
                    error_details += "**Possible Solutions:**\n"
                    error_details += "- Check internet connectivity for ARCO database access\n"
                    error_details += "- Verify SPARQL endpoints are accessible\n"
                    error_details += "- Ensure agent output contains monument/landmark names\n"
                    error_details += "- Try again later if database is temporarily unavailable\n\n"
                    
                    # Add endpoint status information
                    error_details += "**ARCO Endpoints:**\n"
                    error_details += "- Primary: http://wit.istc.cnr.it/arco/virtuoso/sparql\n"
                    error_details += "- Backup: https://dati.beniculturali.it/sparql\n\n"
                    error_details += f"**Traceback:**\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    time.sleep(2 + attempt)
    
    def _clean_text_for_rag(self, text):
        """Clean text by removing emojis and excessive formatting for RAG processing"""
        import re
        
        # Remove common emojis
        emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002700-\U000027BF"  # dingbats
                                 u"\U0001f926-\U0001f937"
                                 u"\U00010000-\U0010ffff"
                                 u"\u2640-\u2642"
                                 u"\u2600-\u2B55"
                                 u"\u200d"
                                 u"\u23cf"
                                 u"\u23e9"
                                 u"\u231a"
                                 u"\ufe0f"
                                 u"\u3030"
                                 "]+", flags=re.UNICODE)
        
        # Remove emojis and clean formatting
        cleaned = emoji_pattern.sub('', text)
        cleaned = re.sub(r'\*+', '', cleaned)  # Remove asterisks
        cleaned = re.sub(r'#+', '', cleaned)   # Remove hash marks
        cleaned = re.sub(r'\n+', ' ', cleaned) # Replace multiple newlines with space
        cleaned = re.sub(r'\s+', ' ', cleaned) # Replace multiple spaces with single space
        
        return cleaned.strip()
    
    def _extract_monument_names(self, agent_output):
        """Extract potential monument names from agent output"""
        import re
        
        # Look for matches in the agent output
        # Common patterns: "Colosseo", "Big Ben", etc.
        matches = re.findall(r'Matches: ([^(]+)', agent_output)
        
        monument_names = []
        for match in matches:
            # Split by comma and clean each name
            names = [name.strip() for name in match.split(',') if name.strip() and 'above similarity threshold' not in name]
            monument_names.extend(names)
        
        # Also look for explicit monument mentions
        lines = agent_output.split('\n')
        for line in lines:
            if 'Description:' in line:
                # Extract the description part
                desc = line.split('Description:')[-1].strip()
                if desc and len(desc.split()) <= 4:  # Likely a monument name
                    monument_names.append(desc)
        
        # Remove duplicates and return unique names
        return list(set(monument_names))

# Initialize the app
app_instance = AgenticTravelerApp()

def optimize_image_for_processing(image, max_size=(1024, 1024), quality=85):
    """Optimize image for processing by resizing and compressing if needed"""
    try:
        # Get original size
        original_size = image.size
        app_instance.logger.info(f"Original image size: {original_size}")
        
        # Calculate new size if resizing needed
        if original_size[0] > max_size[0] or original_size[1] > max_size[1]:
            # Maintain aspect ratio
            ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            app_instance.logger.info(f"Resized image to: {new_size}")
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        app_instance.logger.error(f"Error optimizing image: {e}")
        return image

def process_image_and_question(image, question, progress=gr.Progress()):
    """Main processing function for Gradio interface with NEW WORKFLOW"""
    if image is None:
        return "❌ Please upload an image", "❌ No image provided", "❌ No image provided", "❌ No image provided"
    
    if not question or question.strip() == "":
        question = "What monument or landmark is shown in this image?"
    
    progress(0, desc="Starting analysis...")
    
    # Optimize image before processing
    try:
        progress(0.1, desc="Optimizing image...")
        optimized_image = optimize_image_for_processing(image)
    except Exception as e:
        app_instance.logger.error(f"Image optimization failed: {e}")
        optimized_image = image
    
    # Save uploaded image to temporary file
    progress(0.15, desc="Preparing image file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        optimized_image.save(temp_file.name, format='JPEG', quality=85, optimize=True)
        temp_image_path = temp_file.name
        app_instance.logger.info(f"Saved optimized image to: {temp_image_path}")
    
    try:
        # === NUOVO WORKFLOW ===
        # Step 1: Monument Recognition
        progress(0.2, desc="🎯 Step 1: Monument Recognition...")
        monument_result = app_instance.process_monument_recognition(temp_image_path)
        progress(0.4, desc="Monument recognition complete")
        
        # Parse monument info
        monument_info = parse_monument_info(monument_result)
        monument_name = monument_info.get('monument_name', 'Unknown')
        monument_description = monument_info.get('monument_description', 'No description available')
        
        # Step 2: Geolocation
        progress(0.4, desc="🎯 Step 2: Geolocation...")
        localization_result = app_instance.process_localization(temp_image_path)
        progress(0.6, desc="Geolocation complete")
        
        # Extract coordinates from localization result
        coordinates = app_instance.extract_coordinates_from_localization(localization_result)
        
        # Step 3: RAG Processing with predefined texts
        progress(0.6, desc="🎯 Step 3: RAG Analysis...")
        rag_result = app_instance.process_with_monument_rag_predefined(monument_name, monument_description, question)
        progress(0.8, desc="RAG processing complete")
        
        # Step 4: ARCO Database Query with coordinate fallback
        progress(0.8, desc="🎯 Step 4: ARCO Database...")
        arco_result = app_instance.process_with_arco_and_coordinates(monument_name, coordinates)
        progress(1.0, desc="All processing complete!")
        
        app_instance.logger.info("Successfully completed all four processing stages")
        return monument_result, rag_result, arco_result, localization_result
    
    except Exception as e:
        error_msg = f"❌ **Processing Error**\n\nUnexpected error during processing: {str(e)}"
        app_instance.logger.error(f"Processing failed: {e}")
        return error_msg, error_msg, error_msg, error_msg
    
    finally:
        # Clean up temporary file
        progress(1.0, desc="Cleaning up...")
        try:
            os.unlink(temp_image_path)
            app_instance.logger.info(f"Cleaned up temporary file: {temp_image_path}")
        except Exception as cleanup_error:
            app_instance.logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")

# Create Gradio interface
with gr.Blocks(title="AgenticTraveler - AI Tourist Guide", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🗺️ AgenticTraveler - AI Tourist Guide
    
    Upload an image of a monument or landmark and ask a question about it. The system will analyze the image using four sequential steps:
    
    1. **🎯 Monument Recognition**: Image segmentation and embedding matching to identify monuments
    2. **🌍 Geolocation**: Geographic location detection using StreetCLIP and GeoCLIP
    3. **🔍 RAG System**: Semantic search using predefined monument texts and identified information  
    4. **🏛️ ARCO Database**: Query the Italian cultural heritage database (with coordinate fallback)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📸 Upload Monument/Landmark Image", 
                type="pil",
                height=300
            )
            question_input = gr.Textbox(
                label="❓ Your Question", 
                placeholder="What monument is this? Tell me about its history...",
                value="What monument or landmark is shown in this image?",
                lines=2
            )
            submit_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            # Add example images section
            gr.Markdown("### 📸 Try Example Images")
            example_images = []
            example_descriptions = [
                "Colosseum in Rome",
                "Eiffel Tower in Paris", 
                "Statue of Liberty in New York",
                "Big Ben in London"
            ]
            
            # Try to load example images from data_test directory if available
            example_dir = Path("data_test")
            if example_dir.exists():
                for img_file in example_dir.glob("*.{jpg,jpeg,png,gif}"):
                    if len(example_images) < 4:  # Limit to 4 examples
                        example_images.append(str(img_file))
            
            # Create example image buttons if we have images
            if example_images:
                with gr.Row():
                    for i, (img_path, desc) in enumerate(zip(example_images, example_descriptions)):
                        if i < 4:  # Max 4 examples
                            ex_btn = gr.Button(f"📷 {desc}", size="sm")
                            ex_btn.click(
                                lambda path=img_path: Image.open(path),
                                outputs=image_input
                            )
            else:
                gr.Markdown("💡 **Tip**: Add example images to `data_test/` directory for quick testing")
    
    with gr.Column(scale=2):
        gr.Markdown("## 📊 Analysis Results")
        
        with gr.Row():
            agent_output = gr.Textbox(
                label="🤖 Agent Analysis (Computer Vision + Geolocation)",
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
        
        with gr.Row():
            rag_output = gr.Textbox(
                label="🔍 RAG System (Semantic Document Search)",
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
        
        with gr.Row():
            arco_output = gr.Textbox(
                label="🏛️ ARCO Database (Cultural Heritage Information)",
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
        
        with gr.Row():
            localization_output = gr.Textbox(
                label="🌍 Geolocation (Geographic Information)",
                lines=8,
                max_lines=12,
                show_copy_button=True
            )
    
    # Export functionality section
    with gr.Row():
        gr.Markdown("### 💾 Export Results")
        export_btn = gr.Button("📄 Export to JSON", variant="secondary")
        export_file = gr.File(label="Download Results", visible=False)
        
    def export_results(monument_res, rag_res, arco_res, location_res):
        """Export analysis results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_data = {
                "timestamp": timestamp,
                "analysis_results": {
                    "monument_recognition": monument_res,
                    "rag_system": rag_res,
                    "arco_database": arco_res,
                    "geolocation": location_res
                },
                "metadata": {
                    "export_version": "2.0",
                    "app_version": "AgenticTraveler Enhanced - New Workflow",
                    "components": ["Monument Recognition", "RAG System", "ARCO Database", "Geolocation"]
                }
            }
            
            filename = f"agentictravel_results_{timestamp}.json"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            app_instance.logger.info(f"Results exported to: {filepath}")
            return gr.File(value=filepath, visible=True)
            
        except Exception as e:
            app_instance.logger.error(f"Export failed: {e}")
            return gr.File(visible=False)
    
    export_btn.click(
        fn=export_results,
        inputs=[agent_output, rag_output, arco_output, localization_output],
        outputs=export_file
    )
    
    # Connect the interface
    submit_btn.click(
        fn=process_image_and_question,
        inputs=[image_input, question_input],
        outputs=[agent_output, rag_output, arco_output, localization_output],
        show_progress=True
    )
    
    # Enhanced examples and help section
    gr.Markdown("## 📋 Example Usage & Tips")
    
    with gr.Accordion("💡 Usage Tips & Best Practices", open=False):
        gr.Markdown("""
        ### 🎯 **Optimal Results Tips:**
        - **Image Quality**: Use clear, well-lit photos of monuments/landmarks
        - **Image Size**: Images are automatically optimized (max 1024x1024px) for processing
        - **File Formats**: Supports JPG, PNG, GIF formats
        - **File Size**: Maximum 10MB per image
        
        ### ❓ **Example Questions:**
        - "What monument is this and where is it located?"
        - "Tell me about the history of this landmark"
        - "What architectural style is this building?"
        - "When was this monument built?"
        - "What cultural significance does this place have?"
        - "Who designed or built this monument?"
        - "What materials were used in construction?"
        
        ### 🔧 **System Components:**
        1. **🤖 Agent System**: Computer vision analysis with monument recognition and geolocation
        2. **🔍 RAG System**: Semantic search through relevant documents using AI embeddings
        3. **🏛️ ARCO Database**: Query Italian cultural heritage database for official information
        
        ### ⚠️ **Troubleshooting:**
        - If processing fails, check image quality and try again
        - ARCO database requires internet connectivity
        - RAG system works best with detailed agent output
        - Large images are automatically resized for optimal processing
        """)
    
    with gr.Accordion("📊 System Status & Requirements", open=False):
        gr.Markdown(f"""
        ### 🔄 **Current Status:**
        - Monument Recognition: {'✅ Ready' if app_instance.image_retrieval_tool else '❌ Not Available'}
        - RAG System: {'✅ Ready' if app_instance.rag_system else '❌ Not Available'}
        - ARCO Database: ✅ Available (requires internet)
        - Geolocation: {'✅ Ready' if app_instance.localizator else '❌ Not Available'}
        
        ### 🛠️ **Requirements:**
        - Python dependencies installed (PyTorch, Transformers, Smolagents, etc.)
        - HF_TOKEN environment variable set for Smolagents (optional)
        - Internet connection for ARCO database and geolocation features
        - GPU recommended for optimal performance (CPU fallback available)
        
        ### 📁 **Logging:**
        - Application logs are saved to `agentictravel_app.log`
        - Detailed error information available in logs
        - Export functionality creates timestamped JSON files
        """)
        
    gr.Markdown("---")
    gr.Markdown("🚀 **AgenticTraveler** - Powered by AI for Cultural Heritage Discovery")

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting AgenticTraveler Gradio App...")
    print("Make sure you have:")
    print("  - All Python dependencies installed (gradio, torch, transformers, smolagents, etc.)")
    print("  - HF_TOKEN environment variable set (for Smolagents RAG system)")
    print("  - Models available at the specified paths")
    print("  - Internet connection (for ARCO database and some geolocation features)")
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True
    )