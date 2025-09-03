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
    from agent_base_optim import create_agent
    from rag_sytem import RAG
    from ARCO_access import query_by_name
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")

class AgenticTravelerApp:
    def __init__(self):
        """Initialize the AgenticTraveler application"""
        self.agent = None
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
        """Initialize the agent and RAG systems with retry mechanisms"""
        # Initialize Agent system with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Agent system (attempt {attempt + 1}/{max_retries})...")
                print(f"🔄 Initializing Agent system (attempt {attempt + 1}/{max_retries})...")
                self.agent = create_agent()
                self.logger.info("Agent system initialized successfully")
                print("✅ Agent system initialized successfully")
                break
            except Exception as e:
                self.logger.error(f"Failed to initialize Agent system (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"❌ Failed to initialize Agent system after {max_retries} attempts: {e}")
                    self.agent = None
                else:
                    print(f"⚠️ Agent initialization failed, retrying in 2 seconds...")
                    time.sleep(2)
        
        # Initialize RAG system with retries
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                print(f"🔄 Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                self.rag_system = RAG(
                    embed_model_name="all-MiniLM-L6-v2",
                    use_cross_encoder=False,
                    ollama_url="http://localhost:11434/api/generate"
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
    
    def process_with_agent(self, image_path, question, max_retries=2):
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
                    
                    ollama_answer = self.rag_system.generate_with_ollama(
                        system_prompt=system_prompt,
                        user_query=question,
                        context_passages=context_passages,
                        model="llama3.1"
                    )
                    
                    rag_output += f"**🧠 Generated Answer:**\n{ollama_answer}"
                    self.logger.info("RAG processing completed successfully with Ollama generation")
                    
                except Exception as ollama_error:
                    self.logger.warning(f"Ollama generation failed: {ollama_error}")
                    rag_output += f"\n⚠️ **Note**: Could not generate answer with Ollama: {ollama_error}"
                    rag_output += f"\n💡 **Tip**: Ensure Ollama is running locally at http://localhost:11434"
                
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
                    error_details += "- Ensure Ollama is running locally for answer generation\n"
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
    """Main processing function for Gradio interface with progress indicators"""
    if image is None:
        return "❌ Please upload an image", "❌ No image provided", "❌ No image provided"
    
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
        # Process with Agent system
        progress(0.2, desc="🤖 Analyzing image with AI agent...")
        agent_result = app_instance.process_with_agent(temp_image_path, question)
        progress(0.5, desc="Agent analysis complete")
        
        # Process with RAG system
        progress(0.5, desc="🔍 Processing with RAG system...")
        rag_result = app_instance.process_with_rag(agent_result, question)
        progress(0.8, desc="RAG processing complete")
        
        # Process with ARCO database
        progress(0.8, desc="🏛️ Querying ARCO database...")
        arco_result = app_instance.process_with_arco(agent_result)
        progress(1.0, desc="All processing complete!")
        
        app_instance.logger.info("Successfully completed all three processing stages")
        return agent_result, rag_result, arco_result
    
    except Exception as e:
        error_msg = f"❌ **Processing Error**\n\nUnexpected error during processing: {str(e)}"
        app_instance.logger.error(f"Processing failed: {e}")
        return error_msg, error_msg, error_msg
    
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
    
    Upload an image of a monument or landmark and ask a question about it. The system will analyze the image using three different approaches:
    
    1. **🤖 Agent Analysis**: Computer vision analysis with monument recognition and geolocation
    2. **🔍 RAG System**: Semantic search through relevant documents using AI embeddings  
    3. **🏛️ ARCO Database**: Query the Italian cultural heritage database for official information
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
    
    # Export functionality section
    with gr.Row():
        gr.Markdown("### 💾 Export Results")
        export_btn = gr.Button("📄 Export to JSON", variant="secondary")
        export_file = gr.File(label="Download Results", visible=False)
        
    def export_results(agent_res, rag_res, arco_res):
        """Export analysis results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_data = {
                "timestamp": timestamp,
                "analysis_results": {
                    "agent_analysis": agent_res,
                    "rag_system": rag_res,
                    "arco_database": arco_res
                },
                "metadata": {
                    "export_version": "1.0",
                    "app_version": "AgenticTraveler Enhanced",
                    "components": ["Agent System", "RAG System", "ARCO Database"]
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
        inputs=[agent_output, rag_output, arco_output],
        outputs=export_file
    )
    
    # Connect the interface
    submit_btn.click(
        fn=process_image_and_question,
        inputs=[image_input, question_input],
        outputs=[agent_output, rag_output, arco_output],
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
        - Agent System: {'✅ Ready' if app_instance.agent else '❌ Not Available'}
        - RAG System: {'✅ Ready' if app_instance.rag_system else '❌ Not Available'}
        - ARCO Database: ✅ Available (requires internet)
        
        ### 🛠️ **Requirements:**
        - Python dependencies installed (PyTorch, Transformers, etc.)
        - Ollama running locally for RAG answer generation (optional)
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
    print("  - All Python dependencies installed (gradio, torch, transformers, etc.)")
    print("  - Ollama running locally (for RAG system)")
    print("  - Models available at the specified paths")
    print("  - Internet connection (for ARCO database and some geolocation features)")
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True
    )