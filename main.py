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

# Add script directories to Python path
script_dir = Path(__file__).parent / "script"
sys.path.append(str(script_dir / "agent"))
sys.path.append(str(script_dir / "RAG system"))
sys.path.append(str(script_dir / "KG"))

try:
    # Import the three main components
    from agent_base_optim import create_agent
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
        self.agent = None
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
        """Initialize the agent and RAG systems with retry mechanisms"""
        print("🚀 Initializing AgenticTraveler CLI...")
        
        # Initialize Agent system with retries
        max_retries = 3
        agent_initialized = False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing Agent system (attempt {attempt + 1}/{max_retries})")
                if self.verbose or attempt > 0:
                    print(f"🔄 Initializing Agent system (attempt {attempt + 1}/{max_retries})...")
                
                self.agent = create_agent()
                self.logger.info("Agent system initialized successfully")
                print("✅ Agent system initialized successfully")
                agent_initialized = True
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
        rag_initialized = False
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing RAG system (attempt {attempt + 1}/{max_retries})")
                if self.verbose or attempt > 0:
                    print(f"🔄 Initializing RAG system (attempt {attempt + 1}/{max_retries})...")
                
                self.rag_system = RAGSmolagent(
                    embed_model_name="all-MiniLM-L6-v2",
                    use_cross_encoder=False,
                    model_id="Qwen/Qwen2.5-Coder-32B-Instruct"
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
        systems_ready = sum([agent_initialized, rag_initialized])
        print(f"📊 Systems Status: {systems_ready}/3 components ready (ARCO database available with internet)")
        
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
    
    def process_with_agent(self, image_path, question, max_retries=2):
        """Process image with the agent system with enhanced error handling"""
        if not self.agent:
            error_msg = "❌ Agent system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🤖 Processing with Agent system...")
        
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
                
                if self.verbose:
                    print(f"🔍 Executing agent command: {command}")
                
                result = self.agent.run(command)
                self.logger.info("Agent processing completed successfully")
                print("✅ Agent analysis complete")
                return result
                
            except Exception as e:
                self.logger.error(f"Agent processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ Agent Error (after {max_retries + 1} attempts)\n"
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
                    print(f"⚠️ Agent processing failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)  # Progressive backoff
    
    def process_with_rag(self, agent_output, question, max_retries=2):
        """Process with RAG system using agent output as context with enhanced error handling"""
        if not self.rag_system:
            error_msg = "❌ RAG system not available - initialization failed"
            self.logger.error(error_msg)
            return error_msg
        
        print("🔍 Processing with RAG system...")
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Processing with RAG system (attempt {attempt + 1}/{max_retries + 1})")
                
                # Validate inputs
                if not agent_output or agent_output.strip() == "":
                    raise ValueError("Agent output is empty or invalid")
                
                if not question or question.strip() == "":
                    raise ValueError("Question is empty or invalid")
                
                # Extract meaningful text from agent output for RAG processing
                clean_agent_output = self._clean_text_for_rag(agent_output)
                
                if len(clean_agent_output.strip()) < 10:
                    raise ValueError("Cleaned agent output is too short for meaningful RAG processing")
                
                # Create passages from agent output
                passages = self.rag_system.split_text(clean_agent_output, chunk_size=200, overlap=50)
                
                if not passages:
                    raise ValueError("No valid text passages found from agent output")
                
                self.logger.info(f"Created {len(passages)} text passages for RAG indexing")
                if self.verbose:
                    print(f"📚 Created {len(passages)} text passages for RAG indexing")
                
                # Build index with the passages
                self.rag_system.build_index(passages)
                
                # Query the RAG system
                top_results = self.rag_system.query(question, top_k=3)
                
                if not top_results:
                    raise ValueError("No relevant passages found in RAG query")
                
                # Format results
                rag_output = f"🔍 RAG System Results\n\n"
                rag_output += f"Query: {question}\n"
                rag_output += f"Processed {len(passages)} passages from agent output\n\n"
                rag_output += "Top Retrieved Passages:\n\n"
                
                for i, (score, passage) in enumerate(top_results, 1):
                    rag_output += f"{i}. Similarity: {score:.4f}\n"
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
                    
                    rag_output += f"🧠 Generated Answer:\n{smolagent_answer}"
                    self.logger.info("RAG processing completed successfully with Smolagents generation")
                    
                except Exception as smolagent_error:
                    self.logger.warning(f"Smolagents generation failed: {smolagent_error}")
                    rag_output += f"\n⚠️ Note: Could not generate answer with Smolagents: {smolagent_error}"
                    rag_output += f"\n💡 Tip: Ensure HF_TOKEN is set and Smolagents models are available"
                
                print("✅ RAG processing complete")
                return rag_output
                
            except Exception as e:
                self.logger.error(f"RAG processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ RAG Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check if agent output contains meaningful text\n"
                    error_details += "- Ensure HF_TOKEN is set and Smolagents models are available\n"
                    error_details += "- Verify sentence-transformers models are available\n"
                    error_details += "- Check available memory for FAISS indexing\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    print(f"⚠️ RAG processing failed, retrying in {1 + attempt} seconds...")
                    time.sleep(1 + attempt)
    
    def process_with_arco(self, agent_output, max_retries=2):
        """Process with ARCO knowledge graph with enhanced error handling"""
        print("🏛️ Processing with ARCO database...")
        
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
                    return ("❌ ARCO Results: No monument names found in agent output\n\n"
                           "💡 Tip: Ensure the agent successfully identified monuments or landmarks in the image")
                
                self.logger.info(f"Extracted {len(monument_names)} monument names: {monument_names[:3]}")
                if self.verbose:
                    print(f"🏛️ Found monument names: {', '.join(monument_names[:3])}")
                
                arco_output = "🏛️ ARCO Database Results\n\n"
                total_results = 0
                
                for monument_name in monument_names[:3]:  # Limit to top 3 names
                    arco_output += f"Searching for: {monument_name}\n"
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
                                    arco_output += f"  {i}. {label}\n"
                                    arco_output += f"     URI: {entity}\n"
                                except Exception as result_error:
                                    self.logger.warning(f"Error processing ARCO result {i}: {result_error}")
                                    arco_output += f"  {i}. Error processing result\n"
                            arco_output += "\n"
                        else:
                            arco_output += "❌ No results found in ARCO database\n\n"
                            
                    except Exception as query_error:
                        self.logger.warning(f"ARCO query failed for {monument_name}: {query_error}")
                        arco_output += f"⚠️ Query failed: {str(query_error)}\n\n"
                
                if total_results == 0:
                    arco_output += "\n💡 Suggestions:\n"
                    arco_output += "- Try different search terms or monument names\n"
                    arco_output += "- Check internet connectivity for ARCO database access\n"
                    arco_output += "- Verify ARCO endpoints are accessible\n"
                
                self.logger.info(f"ARCO processing completed. Total results found: {total_results}")
                print("✅ ARCO processing complete")
                return arco_output
                
            except Exception as e:
                self.logger.error(f"ARCO processing failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == max_retries:
                    # Final attempt failed
                    error_details = f"❌ ARCO Error (after {max_retries + 1} attempts)\n"
                    error_details += f"Error Type: {type(e).__name__}\n"
                    error_details += f"Error Message: {str(e)}\n\n"
                    error_details += "Possible Solutions:\n"
                    error_details += "- Check internet connectivity for ARCO database access\n"
                    error_details += "- Verify SPARQL endpoints are accessible\n"
                    error_details += "- Ensure agent output contains monument/landmark names\n"
                    error_details += "- Try again later if database is temporarily unavailable\n\n"
                    
                    # Add endpoint status information
                    error_details += "ARCO Endpoints:\n"
                    error_details += "- Primary: http://wit.istc.cnr.it/arco/virtuoso/sparql\n"
                    error_details += "- Backup: https://dati.beniculturali.it/sparql\n"
                    if self.verbose:
                        error_details += f"\nTraceback:\n{traceback.format_exc()}"
                    return error_details
                else:
                    # Wait before retry
                    print(f"⚠️ ARCO processing failed, retrying in {2 + attempt} seconds...")
                    time.sleep(2 + attempt)
    
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
    
    def analyze_image(self, image_path, question, export_file=None):
        """Main analysis function that processes image through all three systems"""
        start_time = time.time()
        
        print(f"🖼️ Analyzing image: {image_path}")
        print(f"❓ Question: {question}")
        print("=" * 80)
        
        # Validate image exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None
        
        # Optimize image
        print("🔧 Optimizing image...")
        optimized_image_path = self.optimize_image_for_processing(image_path)
        
        try:
            # Step 1: Agent Analysis
            print("\n" + "=" * 80)
            agent_result = self.process_with_agent(optimized_image_path, question)
            print(f"\n📊 AGENT ANALYSIS RESULTS:")
            print("-" * 40)
            print(agent_result)
            
            # Step 2: RAG Processing
            print("\n" + "=" * 80)
            rag_result = self.process_with_rag(agent_result, question)
            print(f"\n📊 RAG SYSTEM RESULTS:")
            print("-" * 40)
            print(rag_result)
            
            # Step 3: ARCO Database
            print("\n" + "=" * 80)
            arco_result = self.process_with_arco(agent_result)
            print(f"\n📊 ARCO DATABASE RESULTS:")
            print("-" * 40)
            print(arco_result)
            
            # Export results if requested
            if export_file:
                print("\n" + "=" * 80)
                exported_file = self.export_results(agent_result, rag_result, arco_result, export_file)
                if exported_file:
                    print(f"📁 Results available in: {exported_file}")
            
            # Processing summary
            end_time = time.time()
            processing_time = end_time - start_time
            print("\n" + "=" * 80)
            print(f"✅ Analysis completed in {processing_time:.1f} seconds")
            print(f"🧠 Systems used: Agent: {'✅' if self.agent else '❌'} | RAG: {'✅' if self.rag_system else '❌'} | ARCO: ✅")
            
            return {
                "agent_result": agent_result,
                "rag_result": rag_result,
                "arco_result": arco_result,
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
            export_file=args.export
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