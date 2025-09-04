from smolagents import Tool, CodeAgent, TransformersModel
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, SegformerForSemanticSegmentation, AutoImageProcessor
import numpy as np
import os
from geoclip import GeoCLIP
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import reverse_geocoder as rg
import socket
import gc
from contextlib import contextmanager


# Configurabile per diversi ambienti
ROOT_PATH = os.environ.get('AGENTIC_ROOT_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/src/")

# ==== CONFIGURAZIONE OTTIMIZZATA ====
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configurazione per ottimizzare la VRAM
torch.cuda.empty_cache() if torch.cuda.is_available() else None
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ==== CONTEXT MANAGER PER GESTIONE MEMORIA ====
@contextmanager
def model_context(model, processor=None):
    """Context manager per caricare temporaneamente i modelli"""
    try:
        model = model.to(device) if hasattr(model, 'to') else model
        yield model, processor
    finally:
        if hasattr(model, 'cpu'):
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ModelManager:
    """Gestisce il caricamento lazy e l'offloading dei modelli"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        # Path configurabili tramite variabili d'ambiente
        base_model_path = os.environ.get('AGENTIC_MODEL_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model")
        self.model_paths = {
            'clip': os.path.join(base_model_path, "clip-vit-base-patch32"),
            'streetclip': os.path.join(base_model_path, "StreetCLIP"),
            'segformer': os.path.join(base_model_path, "segformer")
        }
        
    def get_model(self, model_name):
        """Carica il modello solo quando necessario"""
        if model_name not in self.models:
            self._load_model(model_name)
        return self.models[model_name], self.processors.get(model_name)
    
    def _load_model(self, model_name):
        """Carica specificamente il modello richiesto"""
        if model_name == 'clip':
            self.models['clip'] = CLIPModel.from_pretrained(
                self.model_paths['clip'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).cpu()  # Inizialmente su CPU
            self.processors['clip'] = CLIPProcessor.from_pretrained(self.model_paths['clip'])
            
        elif model_name == 'streetclip':
            self.models['streetclip'] = CLIPModel.from_pretrained(
                self.model_paths['streetclip'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).cpu()
            self.processors['streetclip'] = CLIPProcessor.from_pretrained(self.model_paths['streetclip'])
            
        elif model_name == 'segformer':
            self.processors['segformer'] = AutoImageProcessor.from_pretrained(self.model_paths['segformer'])
            self.models['segformer'] = SegformerForSemanticSegmentation.from_pretrained(
                self.model_paths['segformer'],
                # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
    
    def unload_model(self, model_name):
        """Rimuove il modello dalla memoria"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.processors:
                del self.processors[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


# Inizializza il gestore dei modelli
model_manager = ModelManager()

# Mappa id → label (sarà popolata quando necessario)
id2label = None

def get_id2label():
    """Carica id2label solo quando necessario"""
    global id2label
    if id2label is None:
        model, _ = model_manager.get_model('segformer')
        id2label = model.config.id2label
    return id2label


# ==== FUNZIONI OTTIMIZZATE ====
def internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def initialize_geoclip():
    """Inizializza GeoCLIP solo quando necessario e se c'è connessione"""
    if internet_available():
        try:
            geoclip = GeoCLIP()
            if torch.cuda.is_available():
                geoclip = geoclip.half()  # Usa half precision
            return geoclip.to(device)
        except Exception as e:
            print(f"Errore nell'inizializzazione di GeoCLIP: {e}")
            return None
    return None


# Geopy per conversione coordinate <-> stato
geolocator = Nominatim(user_agent="LocalizatorGeoAgent")

# Lista label StreetCLIP
labels = [
    "Albania", "Andorra", "Argentina", "Australia", "Austria", "Bangladesh", "Belgium",
    "Bermuda", "Bhutan", "Bolivia", "Botswana", "Brazil", "Bulgaria", "Cambodia", "Canada",
    "Chile", "China", "Colombia", "Croatia", "Czech Republic", "Denmark", "Dominican Republic",
    "Ecuador", "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Greenland", "Guam",
    "Guatemala", "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Israel", "Italy", "Japan",
    "Jordan", "Kenya", "Kyrgyzstan", "Laos", "Latvia", "Lesotho", "Lithuania", "Luxembourg",
    "Macedonia", "Madagascar", "Malaysia", "Malta", "Mexico", "Monaco", "Mongolia", "Montenegro",
    "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Palestine", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Romania", "Russia", "Rwanda", "Senegal", "Serbia", "Singapore",
    "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Swaziland", "Sweden",
    "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay"
]


# ==== DATABASE OTTIMIZZATO ====
class ImageDatabase:
    """Database ottimizzato per embeddings delle immagini"""
    
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        
    def add_image(self, path, text, embedding):
        # Converti a float16 per risparmiare memoria
        embedding_fp16 = embedding.astype(np.float16) if embedding.dtype != np.float16 else embedding
        self.embeddings.append(embedding_fp16)
        self.metadata.append({"path": path, "text": text})
    
    def search(self, query_embedding, threshold=0.6):
        query_emb = query_embedding.astype(np.float16) if query_embedding.dtype != np.float16 else query_embedding
        similarities = []
        
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            if sim >= threshold:
                similarities.append((float(sim), self.metadata[i]["text"]))
        
        return sorted(similarities, reverse=True)


# Inizializza database ottimizzato
image_db = ImageDatabase()


def embed_image_optimized(image: Image.Image):
    """Embedding ottimizzato con gestione memoria"""
    with model_context(*model_manager.get_model('clip')) as (model, processor):
        # Ridimensiona l'immagine se troppo grande
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        return emb.cpu().numpy().flatten()


# ==== INIZIALIZZAZIONE MONUMENTI OTTIMIZZATA ====
def initialize_monuments():
    """Inizializza il database dei monumenti in modo lazy"""
    base_data_path = os.environ.get('AGENTIC_DATA_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data")
    
    monuments = [
        ("big_ben.png", "Big Ben", "Big Ben is a famous clock tower in London, England. It's part of the Palace of Westminster and is one of London's most iconic landmarks."),
        ("statue_liberty.png", "Statua della Libertà", "The Statue of Liberty is a neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France to the United States."),
        ("tour_eiffel.png", "Tour Eiffel", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was built in 1889 for the World's Fair."),
        ("colosseo.png", "Colosseo", "The Colosseum is an oval amphitheatre in the centre of Rome, Italy. Built of travertine limestone, it was the largest amphitheatre ever built."),
        ("cristo_redentor.png", "Cristo Redentore", "Christ the Redeemer is an Art Deco statue of Jesus Christ in Rio de Janeiro, Brazil. It overlooks the city from atop Mount Corcovado."),
    ]

    # Prova diversi path per le immagini di esempio
    possible_paths = [
        os.path.join(base_data_path, "test.jpg"),
        os.path.join(base_data_path, "statua_liberta.jpg"),
        os.path.join(base_data_path, "tour_eiffel.jpg"),
        os.path.join(base_data_path, "colosseo.jpg"),
        os.path.join(base_data_path, "cristo_redentor.jpg")
    ]
    
    # Fallback per directory locali
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_paths = [
        os.path.join(current_dir, "..", "..", "..", "data_test", "colosseum.jpg"),
        os.path.join(current_dir, "..", "..", "..", "data_test", "statue_liberty.jpg"),
        os.path.join(current_dir, "..", "..", "..", "data_test", "eiffel_tower.jpg"),
        os.path.join(current_dir, "..", "..", "..", "data_test", "big_ben.jpg"),
        os.path.join(current_dir, "..", "..", "..", "data_test", "christ_redeemer.jpg")
    ]

    for i, (filename, name, description) in enumerate(monuments):
        # Prova prima il path originale
        image_path = possible_paths[i] if i < len(possible_paths) else None
        
        # Se non esiste, prova i fallback
        if not (image_path and os.path.exists(image_path)):
            if i < len(fallback_paths):
                image_path = fallback_paths[i]
        
        # Se l'immagine esiste, crea l'embedding
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                emb = embed_image_optimized(img)
                # Usa la descrizione invece del solo nome
                image_db.add_image(filename, f"{name}: {description}", emb)
                print(f"✅ Caricato esempio: {name} da {image_path}")
            except Exception as e:
                print(f"⚠️ Errore caricamento {name}: {e}")
        else:
            print(f"⚠️ Immagine non trovata per {name}")
    
    # Forza la pulizia della memoria dopo l'inizializzazione
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"📊 Database monumenti inizializzato con {len(image_db.embeddings)} esempi")


# ==== TOOL OTTIMIZZATO ====
class ImageRetrievalTool(Tool):
    name = "image_retrieval"
    description = """Tool for monument recognition through image segmentation and embedding matching. Returns the best monument match with its description."""
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to analyze and retrieve monument information."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        # Inizializza database se necessario
        if len(image_db.embeddings) == 0:
            initialize_monuments()
            
        output_dir = "tmp_segments"
        os.makedirs(output_dir, exist_ok=True)
        segments_info = self.segment_image_and_get_paths(image_path, output_dir, conf_threshold=0.7)

        print(f"🔍 Segmenti trovati: {len(segments_info)}")
        
        # Lista per memorizzare tutti i match trovati
        all_matches = []
        
        for seg_path, label_text, seg_conf in segments_info:
            try:
                # Costruisci il path completo considerando se è già assoluto o relativo
                if os.path.isabs(seg_path):
                    full_seg_path = seg_path
                else:
                    full_seg_path = os.path.join(ROOT_PATH, seg_path) if ROOT_PATH else seg_path
                
                if os.path.exists(full_seg_path):
                    query_img = Image.open(full_seg_path).convert("RGB")
                    query_emb = embed_image_optimized(query_img)

                    matches = image_db.search(query_emb, threshold=0.6)  # Soglia più permissiva
                    
                    for score, monument_info in matches:
                        # Dividi nome e descrizione
                        if ": " in monument_info:
                            monument_name, monument_desc = monument_info.split(": ", 1)
                        else:
                            monument_name = monument_info
                            monument_desc = f"Informazioni su {monument_name}"
                        
                        all_matches.append({
                            'score': score,
                            'segment': label_text,
                            'segment_conf': seg_conf,
                            'monument_name': monument_name,
                            'monument_description': monument_desc,
                            'segment_path': seg_path
                        })
                        print(f"✅ Match trovato: {monument_name} (score: {score:.3f})")
                else:
                    print(f"⚠️ File segmento non trovato: {full_seg_path}")
            except Exception as e:
                print(f"❌ Errore processamento segmento {seg_path}: {e}")

        # Pulizia memoria dopo l'elaborazione
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Seleziona il miglior match
        if all_matches:
            best_match = max(all_matches, key=lambda x: x['score'])
            
            result = {
                'monument_name': best_match['monument_name'],
                'monument_description': best_match['monument_description'],
                'confidence_score': best_match['score'],
                'segment_type': best_match['segment'],
                'segment_confidence': best_match['segment_conf'],
                'total_segments': len(segments_info),
                'total_matches': len(all_matches)
            }
            
            print(f"🏆 Miglior match: {result['monument_name']} (score: {result['confidence_score']:.3f})")
            return str(result)  # Restituisce dizionario come stringa per l'agent
        else:
            result = {
                'monument_name': 'Unknown',
                'monument_description': 'No monument matches found in the database',
                'confidence_score': 0.0,
                'segment_type': 'N/A',
                'segment_confidence': 0.0,
                'total_segments': len(segments_info),
                'total_matches': 0
            }
            print("❌ Nessun match trovato nel database")
            return str(result)

    def segment_image_and_get_paths(self, image_path, output_dir="output_segments", conf_threshold=0.5):
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        
        # Ridimensiona se troppo grande
        # max_size = 1024
        # if max(image.size) > max_size:
            # image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        orig_width, orig_height = image.size

        with model_context(*model_manager.get_model('segformer')) as (model, processor):
            inputs = processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model(**inputs)
                
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(orig_height, orig_width), mode="bilinear", align_corners=False
        )
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
        seg = torch.argmax(probs, dim=0).cpu().numpy()
        unique_segments = np.unique(seg)
        unique_segments = unique_segments[unique_segments != 0]

        def get_position_name(mask):
            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                return "unknown"
            cx, cy = xs.mean(), ys.mean()
            if cx < orig_width / 3:
                horizontal_pos = "left"
            elif cx < 2 * orig_width / 3:
                horizontal_pos = "center"
            else:
                horizontal_pos = "right"
            if cy < orig_height / 3:
                vertical_pos = "top"
            elif cy < 2 * orig_height / 3:
                vertical_pos = "middle"
            else:
                vertical_pos = "bottom"
            return f"{vertical_pos} {horizontal_pos}" if vertical_pos != "middle" else horizontal_pos

        segment_paths = []
        current_id2label = get_id2label()
        
        for seg_id in unique_segments:
            mask = (seg == seg_id).astype(np.uint8)
            seg_conf = probs[seg_id][mask == 1].mean().item()
            if seg_conf < conf_threshold:
                continue

            position_name = get_position_name(mask)
            class_name = current_id2label.get(int(seg_id), f"class_{seg_id}")
            label_text = f"{class_name} - {position_name} ({seg_conf:.2f})"

            img_np = np.array(image)
            alpha = (mask * 255).astype(np.uint8)
            rgba = np.dstack((img_np, alpha))
            seg_img = Image.fromarray(rgba, mode="RGBA")

            seg_path = os.path.join(output_dir, f"{class_name}_{position_name}.png")
            seg_img.save(seg_path)
            segment_paths.append((str(seg_path), label_text, seg_conf))
        
        # print(segment_paths)

        return segment_paths


class Localizator(Tool):
    name = "localizator"
    description = """Comprehensive geolocation tool that determines the geographical location of images."""
    inputs = {
        "image_path": {"type": "string", "description": "Path of the image to geolocate."}
    }
    output_type = "string"

    def forward(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        
        # Ridimensiona se necessario
        if max(img.size) > 512:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        has_internet = self.internet_available()

        # ==== Metodo 1: StreetCLIP (sempre disponibile) ====
        with model_context(*model_manager.get_model('streetclip')) as (model, processor):
            street_inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                street_inputs = {k: v.to(device) for k, v in street_inputs.items()}
                
            with torch.no_grad():
                street_outputs = model(**street_inputs)
                
        logits_per_image = street_outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)
        sorted_confidences = sorted(
            {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}.items(),
            key=lambda item: item[1], reverse=True
        )
        street_country, street_conf = sorted_confidences[0]
        street_lat, street_lon = self.get_country_coordinates(street_country, has_internet)
        street_city, street_country_name = self.reverse_geocode(street_lat, street_lon, has_internet)

        result_text = (
            "🌍 **Risultati Localizzazione** 🌍\n"
            f"📌 **StreetCLIP**: {street_city}, {street_country_name} ({street_conf*100:.2f}%)\n"
            f"   Lat: {street_lat:.6f}, Lon: {street_lon:.6f}\n"
        )

        # ==== Metodo 2: GeoCLIP (solo se disponibile e c'è internet) ====
        if has_internet:
            geoclip_model = initialize_geoclip()
            if geoclip_model is not None:
                try:
                    with model_context(geoclip_model) as (model, _):
                        top_pred_gps, _ = model.predict(image_path, top_k=1)
                    geo_lat, geo_lon = top_pred_gps[0]
                    geo_city, geo_country = self.reverse_geocode(geo_lat, geo_lon, has_internet)
                    
                    result_text += (
                        f"\n📌 **GeoCLIP**: {geo_city}, {geo_country}\n"
                        f"   Lat: {geo_lat:.6f}, Lon: {geo_lon:.6f}"
                    )
                except Exception as e:
                    result_text += f"\n⚠️ **GeoCLIP**: Errore durante la localizzazione - {str(e)}"
            else:
                result_text += "\n⚠️ **GeoCLIP**: Non inizializzato correttamente"
        else:
            result_text += "\n⚠️ **GeoCLIP**: Non disponibile (nessuna connessione internet)"

        # Pulizia memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result_text

    def internet_available(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def get_country_coordinates(self, country_name, has_internet=None):
        if has_internet is None:
            has_internet = self.internet_available()
            
        if has_internet:
            try:
                location = geolocator.geocode(country_name, timeout=10)
                if location:
                    return location.latitude, location.longitude
            except GeocoderTimedOut:
                pass
                
        # FALLBACK OFFLINE
        try:
            country_code_map = {
                'United States': 'US', 'United Kingdom': 'GB', 'France': 'FR', 'Germany': 'DE',
                'Italy': 'IT', 'Spain': 'ES', 'Japan': 'JP', 'China': 'CN', 'Brazil': 'BR',
                'Canada': 'CA', 'Australia': 'AU', 'Russia': 'RU', 'India': 'IN'
            }
            
            target_cc = country_code_map.get(country_name, country_name[:2].upper())
            
            country_coords = {
                'US': (39.8283, -98.5795), 'GB': (55.3781, -3.4360), 'FR': (46.6034, 2.2137),
                'DE': (51.1657, 10.4515), 'IT': (41.8719, 12.5674), 'ES': (40.4637, -3.7492),
                'JP': (36.2048, 138.2529), 'CN': (35.8617, 104.1954), 'BR': (-14.2350, -51.9253),
                'CA': (56.1304, -106.3468), 'AU': (-25.2744, 133.7751), 'RU': (61.5240, 105.3188),
                'IN': (20.5937, 78.9629)
            }
            
            if target_cc in country_coords:
                return country_coords[target_cc]
                
        except Exception as e:
            print(f"Errore nel fallback offline: {e}")
            
        return (0.0, 0.0)

    def reverse_geocode(self, lat, lon, has_internet=None):
        if has_internet is None:
            has_internet = self.internet_available()
            
        if has_internet:
            try:
                location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
                if location and location.raw.get("address"):
                    city = location.raw["address"].get("city") or location.raw["address"].get("town") or location.raw["address"].get("village") or "Unknown"
                    country = location.raw["address"].get("country", "Unknown")
                    return city, country
            except GeocoderTimedOut:
                pass

        # FALLBACK OFFLINE con reverse_geocoder
        try:
            results = rg.search((lat, lon))
            if results:
                entry = results[0]
                return entry["name"], entry["cc"]
        except Exception as e:
            print(f"Errore nel reverse geocoding offline: {e}")
            
        return "Unknown", "Unknown"


# ==== INIZIALIZZAZIONE AGENTE OTTIMIZZATA ====
def create_agent():
    """Crea l'agente con configurazione ottimizzata per VRAM"""
    retrieval_tool = ImageRetrievalTool()
    localizator = Localizator()
    
    qwen_model_path = os.environ.get('QWEN_MODEL_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/model/Qwen2.5-Coder-7B-Instruct")
    
    agent_model = TransformersModel(
        model_id=qwen_model_path,
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # max_memory={0: "6GB"} if torch.cuda.is_available() else None  # Limita memoria GPU
    )

    agent = CodeAgent(
        tools=[localizator, retrieval_tool],
        model=agent_model,
        additional_authorized_imports=["PIL", "torch", "transformers", "numpy", "io", "gc"],
        planning_interval=3,
        max_steps=5
    )
    
    return agent


# ==== FUNZIONE DI TEST OTTIMIZZATA ====
# ==== UTILITY FUNCTIONS ====
def parse_monument_info(agent_result_str):
    """Parse del risultato dell'agent per estrarre informazioni sul monumento"""
    import ast
    try:
        # L'agent restituisce una stringa che rappresenta un dizionario
        result_dict = ast.literal_eval(agent_result_str.strip())
        return result_dict
    except:
        # Fallback parsing se il formato non è un dizionario valido
        return {
            'monument_name': 'Unknown',
            'monument_description': agent_result_str,
            'confidence_score': 0.0
        }

def get_monument_coordinates(monument_name):
    """Ottiene coordinate approssimative di monumenti famosi"""
    monument_coords = {
        'Big Ben': (51.5007, -0.1246),
        'Statue of Liberty': (40.6892, -74.0445), 
        'Statua della Libertà': (40.6892, -74.0445),
        'Tour Eiffel': (48.8584, 2.2945),
        'Eiffel Tower': (48.8584, 2.2945),
        'Colosseo': (41.8902, 12.4922),
        'Colosseum': (41.8902, 12.4922),
        'Cristo Redentore': (-22.9519, -43.2105),
        'Christ the Redeemer': (-22.9519, -43.2105)
    }
    
    return monument_coords.get(monument_name, (0.0, 0.0))

def run_optimized_test():
    """Esegue un test ottimizzato"""
    # Pulizia iniziale
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    agent = create_agent()
    base_data_path = os.environ.get('AGENTIC_DATA_PATH', "/leonardo_work/uTS25_Pinna/phd_proj/TurismAgent/TurismAgent/data")
    file_path = os.path.join(base_data_path, "new_test.jpg")
    
    # Fallback per percorsi locali
    if not os.path.exists(file_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_path = os.path.join(current_dir, "..", "..", "..", "data_test", "colosseum.jpg")
        if os.path.exists(fallback_path):
            file_path = fallback_path
    
    try:
        # Usa solo l'ImageRetrievalTool per il riconoscimento
        result = agent.run(f"Use the image_retrieval tool to identify the monument in this image: {file_path}")
        return result
    finally:
        # Pulizia finale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    result = run_optimized_test()
    print(result)