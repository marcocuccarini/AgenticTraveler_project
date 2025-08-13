# Tourist Guide AI Assistant

An advanced AI-powered tourist guide system that combines computer vision, natural language processing, and knowledge graph integration to analyze monuments, landmarks, and provide detailed information about them.

## 🏛️ Project Overview

This system enables tourists to:
- Upload photos of monuments and landmarks
- Get automatic identification and analysis of architectural elements
- Receive detailed historical and cultural information
- Access data from cultural heritage knowledge graphs (ARCO)
- Get geographical location information for monuments

The system leverages optimized AI models, efficient memory management, and cultural heritage databases to provide comprehensive tourist guidance.

## 🚀 Key Features

- **Image Analysis**: Advanced monument recognition and segmentation
- **Geographical Location**: Multi-method location detection
- **ARCO Integration**: Access to Italian cultural heritage data
- **Memory Optimized**: Efficient handling of AI models
- **Multi-Model Architecture**: Combines CLIP, StreetCLIP, and Segformer models

## �️ System Components

### Core Files

#### `agent_base_optim.py`
The main agent implementation with optimized memory management:
- Model loading and unloading optimization
- GPU memory management
- Image processing and segmentation
- Geographical location detection
- Integration of multiple AI models:
  - CLIP for image understanding
  - StreetCLIP for location detection
  - Segformer for image segmentation

Key features:
- Lazy model loading
- Memory-efficient processing
- GPU/CPU compatibility
- Automatic model offloading
- Optimized image resizing

#### `ARCO_access.py`
Integration with the ARCO (Architecture of Knowledge) cultural heritage system:
- SPARQL endpoint connections
- Cultural heritage data querying
- Multiple endpoint fallback system
- Italian monuments and landmarks data access

Features:
- Redundant endpoints for reliability
- Error handling and fallback mechanisms
- Configurable query limits
- Rich metadata retrieval

## � Installation

### Prerequisites
```bash
# Required Python version
Python 3.8+

# Required CUDA (for GPU acceleration)
CUDA 11.0+
```

### Environment Setup

1. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies using uv (recommended for faster installation):
```bash
pip install uv
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Image Analysis and Monument Recognition

```python
from script.agent.agent_base_optim import create_agent

# Create an optimized agent
agent = create_agent()

# Analyze an image
image_path = "data_test/colosseo.jpg"
result = agent.run(f"Localize the image: {image_path} and then give some info about it.")
print(result)
```

### ARCO Knowledge Graph Queries

```python
from script.KG.ARCO_access import query_by_name

# Search for a monument
results = query_by_name("Colosseo")
for result in results:
    print(f"Entity: {result['entity']['value']}")
    print(f"Label: {result['label']['value']}")
```

## � Project Structure

```
AgenticTraveler_project/
├── data_test/              # Test images
│   ├── colosseo.jpg
│   ├── cristo_redentor.jpg
│   └── ...
├── script/
│   ├── agent/
│   │   └── agent_base_optim.py  # Main agent implementation
│   └── KG/
│       └── ARCO_access.py       # ARCO integration
├── requirements.txt        # Project dependencies
└── README.md
```

## 🔍 Technical Details

### Model Management
The system uses a sophisticated model management system that:
- Loads models only when needed
- Automatically offloads models to save memory
- Handles GPU/CPU transitions seamlessly
- Optimizes memory usage for large models

### Image Processing Pipeline
1. Image Loading and Optimization
2. Monument Segmentation
3. Location Detection
   - StreetCLIP analysis
   - Geographical coordinates extraction
4. Cultural Information Retrieval
   - ARCO knowledge graph queries
   - Monument information synthesis

### ARCO Integration
The system connects to multiple ARCO endpoints:
- Primary: http://wit.istc.cnr.it/arco/virtuoso/sparql
- Backup: https://dati.beniculturali.it/sparql

Features:
- Automatic endpoint fallback
- Error handling
- Query rate limiting
- Rich metadata extraction

## 🎯 Performance Optimization

### Memory Management
- Lazy model loading
- Automatic GPU memory cleanup
- Model offloading when not in use
- Optimized image processing

### GPU Acceleration
- CUDA support for faster processing
- Automatic CPU fallback
- Mixed precision operations
- Batch processing capabilities

## 🔧 Troubleshooting

### Common Issues

#### Memory Issues
- Ensure sufficient RAM (8GB+ recommended)
- Enable GPU support if available
- Check for memory leaks with monitoring tools

#### ARCO Connection Issues
- Verify internet connectivity
- Check endpoint availability
- Try alternative endpoints
- Verify query syntax

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations