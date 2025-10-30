import os
import sys
from typing import Optional, Tuple
import numpy as np
import torch
import trimesh
import gradio as gr
from huggingface_hub import snapshot_download
from PIL import Image
import gc
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.image_process import prepare_image
from scripts.briarmbg import BriaRMBG
import pymeshlab
from mmgp import offload, profile_type


# ============================================================================
# TODO:
# - memory leak while generating textures
# - automatic download of model and weights
# - better path organisation
# - add Hunyuan3D-2 to requirement
# - image no bg appear at the end not when it's finish
# - may crash and stop generating out of nowhere, maybe due to my memory leak

# ============================================================================


# ============================================================================
# 🎯 CONFIGURATION DES CHEMINS DES MODÈLES - MODIFIER ICI
# ============================================================================

# Chemin vers TripoSG (génération de mesh)
TRIPOSG_MODEL_PATH = "pretrained_weights/TripoSG"

# Chemin vers RMBG (suppression arrière-plan)
RMBG_MODEL_PATH = "pretrained_weights/RMBG-1.4"

# Chemin vers Hunyuan3D pour la TEXTURE (le plus important à configurer)
# Option 1: Laisser None pour auto-détection
# Option 2: Spécifier le chemin exact vers hunyuan3d-delight-v2-0
snapshot_id = "9cd649ba691...."# something that look like this
HUNYUAN_TEXTURE_PATH = f"F:\\.cache\\huggingface\\hub\\models--tencent--Hunyuan3D-2\\snapshots\\{snapshot_id}\\"
# --model_path tencent/Hunyuan3D-2 --subfolder Hunyuan3D-Paint-v2-0-Turbo --texgen_model_path tencent/Hunyuan3D-2

# ============================================================================


class TripoSGApp:
    def __init__(self, device="cuda", dtype=torch.float16, profile=4):
        self.device = device
        self.dtype = dtype
        self.profile = profile
        self.pipe = None
        self.rmbg_net = None
        self.texgen_worker = None
        self.face_reduce_worker = None
        self.hunyuan_texture_path = HUNYUAN_TEXTURE_PATH  # ✅ prend directement ton path global
        self.hunyuan_available = False

    def initialize_models(self, enable_texture=False):
        """Initialise les modèles"""
        if self.pipe is None or self.rmbg_net is None:
            # TripoSG
            if not os.path.exists(TRIPOSG_MODEL_PATH):
                print(f"📥 Téléchargement TripoSG vers {TRIPOSG_MODEL_PATH}...")
                snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=TRIPOSG_MODEL_PATH)

            # RMBG
            if not os.path.exists(RMBG_MODEL_PATH):
                print(f"📥 Téléchargement RMBG vers {RMBG_MODEL_PATH}...")
                snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=RMBG_MODEL_PATH)

            print(f"📂 Chargement RMBG depuis: {RMBG_MODEL_PATH}")
            self.rmbg_net = BriaRMBG.from_pretrained(RMBG_MODEL_PATH).to(self.device)
            self.rmbg_net.eval()

            print(f"📂 Chargement TripoSG depuis: {TRIPOSG_MODEL_PATH}")
            self.pipe = TripoSGPipeline.from_pretrained(TRIPOSG_MODEL_PATH).to(self.device, self.dtype)
            offload.profile(self.pipe, profile_type.LowRAM_LowVRAM)

        # Initialiser FaceReducer
        if self.face_reduce_worker is None:
            try:
                from hy3dgen.shapegen import FaceReducer
                self.face_reduce_worker = FaceReducer()
                print("✅ FaceReducer chargé")
            except Exception:
                print("⚠️ FaceReducer non disponible, utilisation de pymeshlab")

        # Initialiser la texture Hunyuan3D
        if enable_texture and self.texgen_worker is None:
            self._initialize_texture_models()

    def _initialize_texture_models(self):
        """Initialise Hunyuan3D avec ton chemin global défini en haut du fichier"""
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            texgen_model_path = self.hunyuan_texture_path

            # ✅ Vérifie que le chemin existe
            if texgen_model_path and os.path.exists(texgen_model_path):
                print(f"✅ Chargement Hunyuan3D depuis ton chemin: {texgen_model_path}")
            else:
                print(f"⚠️ Chemin Hunyuan3D invalide: {texgen_model_path}")
                print("💡 Tentative de chargement via le cache Hugging Face...")
                texgen_model_path = "tencent/Hunyuan3D-2"

            # ✅ Vérification que le dossier contient bien un modèle HuggingFace
            if os.path.isdir(texgen_model_path):
                if not os.path.exists(os.path.join(texgen_model_path, "config.json")):
                    print(f"⚠️ Aucun 'config.json' trouvé dans {texgen_model_path}.")
                    print("💡 Vérifie que le dossier correspond bien au snapshot complet du modèle.")

            print(f"🔄 Initialisation de Hunyuan3D depuis: {texgen_model_path}")
            self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(texgen_model_path)

            # Optimisation mémoire
            if hasattr(self.texgen_worker, 'models'):
                mv = self.texgen_worker.models.get("multiview_model")
                if mv and hasattr(mv.pipeline, "vae"):
                    mv.pipeline.vae.use_slicing = True

            self.hunyuan_available = True
            print("✅ Hunyuan3D initialisé avec succès!")

        except Exception as e:
            import traceback
            print(f"❌ Erreur lors de l'initialisation de Hunyuan3D: {e}")
            traceback.print_exc()
            self.texgen_worker = None
            self.hunyuan_available = False       

    def clear_memory(self):
        """Libère la mémoire"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def save_temp_image(self, image: Image.Image) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name)
        return temp_file.name
    
    def mesh_to_pymesh(self, vertices, faces):
        mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        return ms
    
    def pymesh_to_trimesh(self, mesh):
        return trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix())
    
    def simplify_mesh(self, mesh: trimesh.Trimesh, n_faces: int):
        """Simplifie le mesh (fallback si FaceReducer non disponible)"""
        if n_faces > 0 and mesh.faces.shape[0] > n_faces:
            ms = self.mesh_to_pymesh(mesh.vertices, mesh.faces)
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
            simplified = self.pymesh_to_trimesh(ms.current_mesh())
            del ms
            return simplified
        return mesh
    
    def reduce_faces(self, mesh: trimesh.Trimesh, target_faces: int = 10000):
        """Réduit les faces avec FaceReducer si disponible, sinon pymeshlab"""
        if mesh.faces.shape[0] <= target_faces:
            return mesh
        
        if self.face_reduce_worker is not None:
            try:
                return self.face_reduce_worker(mesh, target_faces)
            except:
                print("⚠️ FaceReducer a échoué, utilisation de pymeshlab")
                return self.simplify_mesh(mesh, target_faces)
        else:
            return self.simplify_mesh(mesh, target_faces)
    
    @torch.no_grad()
    def generate_3d(self, image, seed, num_inference_steps, guidance_scale, faces, progress):
        temp_image_path = None
        try:
            self.initialize_models()
            
            progress(0.05, desc="Sauvegarde temporaire...")
            temp_image_path = self.save_temp_image(image)
            
            progress(0.1, desc="Suppression de l'arrière-plan...")
            img_pil = prepare_image(temp_image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg_net)
            
            yield None, img_pil, "⏳ Image traitée, génération 3D...", None
            
            progress(0.3, desc="Génération du mesh 3D...")
            outputs = self.pipe(
                image=img_pil,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).samples[0]
            
            progress(0.8, desc="Construction du mesh...")
            mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
            
            del outputs
            self.clear_memory()
            
            if faces > 0:
                progress(0.9, desc="Simplification...")
                mesh = self.simplify_mesh(mesh, faces)
            
            output_path = "output_temp.glb"
            mesh.export(output_path)
            
            stats = f"✅ Mesh généré!\n📊 Vertices: {len(mesh.vertices):,} | Faces: {len(mesh.faces):,}\n💾 Taille: {os.path.getsize(output_path)/1024:.2f} KB"
            
            del mesh
            self.clear_memory()
            
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            progress(1.0, desc="Terminé!")
            yield output_path, img_pil, stats, output_path
            
        except Exception as e:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            self.clear_memory()
            yield None, None, f"❌ Erreur: {str(e)}", None
    
    def apply_texture(self, processed_image, mesh_path, reduce_faces_before, target_faces, progress):
        """Applique la texture avec Hunyuan3D (méthode officielle comme dans gradio_app.py)"""
        
        # Initialiser Hunyuan3D si pas déjà fait
        if self.texgen_worker is None:
            progress(0.1, desc="Initialisation de Hunyuan3D...")
            self.initialize_models(enable_texture=True)
        
        if not self.hunyuan_available or self.texgen_worker is None:
            yield None, "❌ Hunyuan3D non disponible. Vérifiez l'installation."
            return
        
        try:
            progress(0.2, desc="Chargement du mesh...")
            
            # Charger le mesh
            loaded = trimesh.load(mesh_path, force='mesh')
            
            if isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) == 0:
                    yield None, "❌ Aucune géométrie trouvée"
                    return
                meshes_list = list(loaded.geometry.values())
                mesh = meshes_list[0] if len(meshes_list) == 1 else trimesh.util.concatenate(meshes_list)
            else:
                mesh = loaded
            
            if not isinstance(mesh, trimesh.Trimesh):
                yield None, f"❌ Type non supporté: {type(mesh)}"
                return
            
            original_faces = len(mesh.faces)
            progress(0.3, desc=f"Mesh: {len(mesh.vertices):,} vertices, {original_faces:,} faces")
            
            # Réduction des faces (comme dans le code original)
            if reduce_faces_before:
                progress(0.35, desc=f"Réduction des faces à {target_faces:,}...")
                mesh = self.reduce_faces(mesh, target_faces)
                print(f"Faces réduites: {original_faces:,} → {len(mesh.faces):,}")
            
            progress(0.4, desc="🎨 Génération de texture avec Hunyuan3D...")
            
            # APPEL HUNYUAN3D (méthode officielle)
            # textured_mesh = texgen_worker(mesh, image)
            textured_mesh = self.texgen_worker(mesh, processed_image)
            
            progress(0.8, desc="Sauvegarde du mesh texturé...")
            
            textured_glb_path = "textured_mesh.glb"
            textured_mesh.export(textured_glb_path, include_normals=True)
            
            stats = f"✅ Texture appliquée avec Hunyuan3D!\n📊 Vertices: {len(textured_mesh.vertices):,} | Faces: {len(textured_mesh.faces):,}\n💾 Taille: {os.path.getsize(textured_glb_path)/1024:.2f} KB\n🎨 Méthode: Hunyuan3D Paint Pipeline"
            
            del mesh, textured_mesh, loaded
            self.clear_memory()
            
            progress(1.0, desc="Terminé!")
            yield textured_glb_path, stats
            
        except Exception as e:
            import traceback
            self.clear_memory()
            error_trace = traceback.format_exc()
            yield None, f"❌ Erreur Hunyuan3D:\n{str(e)}\n\n📋 Traceback:\n{error_trace}"


# Initialiser l'application
LOCAL_MODEL_PATHS = {}
app = TripoSGApp()


def generate_mesh(image, seed, steps, guidance, faces, progress=gr.Progress()):
    if image is None:
        yield None, None, "⚠️ Veuillez charger une image", None
        return
    yield from app.generate_3d(image, seed, steps, guidance, faces, progress)



def apply_texture_to_mesh(processed_image, mesh_path, reduce_faces, target_faces, progress=gr.Progress()):
    if mesh_path is None:
        yield None, "⚠️ Veuillez d'abord générer un mesh 3D"
        return
    if processed_image is None:
        yield None, "⚠️ Image traitée manquante"
        return

    # Force les types corrects et log utile pour debug
    reduce_faces_bool = bool(reduce_faces)
    try:
        target_faces_int = int(target_faces)
    except Exception:
        target_faces_int = 10000

    print(f"[DEBUG] apply_texture_to_mesh: reduce_faces={reduce_faces_bool}, target_faces={target_faces_int}")

    yield from app.apply_texture(processed_image, mesh_path, reduce_faces_bool, target_faces_int, progress)



# Interface Gradio
with gr.Blocks(title="TripoSG + Hunyuan3D", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
        # 🎨 TripoSG + Hunyuan3D - Texture Officielle
        Génération 3D avec texture haute qualité via **Hunyuan3D Paint Pipeline**
        
        ⚙️ Utilise la méthode officielle de texture de Hunyuan3D
    """)
    
    mesh_state = gr.State(None)
    processed_image_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Image d'entrée")
            input_image = gr.Image(label="Glissez-déposez votre image", type="pil", height=400)
            
            gr.Markdown("### ⚙️ Paramètres de génération")
            
            seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, value=42, step=1)
            num_steps = gr.Slider(label="Étapes d'inférence", minimum=20, maximum=100, value=50, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.0, step=0.5)
            faces = gr.Slider(label="Nombre de faces (-1 = auto)", minimum=-1, maximum=100000, value=-1, step=1000)
            
            generate_btn = gr.Button("🚀 Générer le modèle 3D", variant="primary", size="lg")
            
            gr.Markdown("### 🎨 Paramètres de texture")
            
            reduce_faces = gr.Checkbox(
                label="Réduire les faces avant texture",
                value=True,
                info="Recommandé pour optimiser la mémoire (comme dans Hunyuan3D)"
            )
            
            target_faces = gr.Slider(
                label="Faces cible pour texture",
                minimum=5000,
                maximum=50000,
                value=10000,
                step=1000,
                info="Valeur par défaut Hunyuan3D: 10000"
            )
            
            texture_btn = gr.Button("🎨 Appliquer Texture (Hunyuan3D)", variant="secondary", size="lg", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Image sans arrière-plan")
            processed_image = gr.Image(label="Aperçu", type="pil", height=300)
            
            gr.Markdown("### 🎭 Modèle 3D")
            
            with gr.Tabs():
                with gr.Tab("Sans texture"):
                    output_3d = gr.Model3D(label="Mesh", height=400, clear_color=[1.0, 1.0, 1.0, 1.0])
                    status_3d = gr.Textbox(label="Statut", interactive=False, lines=3)
                
                with gr.Tab("Avec texture"):
                    textured_3d = gr.Model3D(label="Mesh Texturé", height=400, clear_color=[1.0, 1.0, 1.0, 1.0])
                    status_texture = gr.Textbox(label="Statut Texture", interactive=False, lines=4)
    
    gr.Markdown("""
        ### 💡 Guide d'utilisation:
        1. **Charger une image** → Cliquez "Générer le modèle 3D"
        2. **Attendre la génération** du mesh
        3. **Cliquez "Appliquer Texture"** → Hunyuan3D génère la texture
        
        ### 🎨 À propos de Hunyuan3D:
        - **Méthode officielle**: Utilise `Hunyuan3DPaintPipeline` de Tencent
        - **Haute qualité**: Génération de texture réaliste avec modèles de diffusion
        - **Optimisé**: Réduction automatique des faces pour économiser la VRAM
        
        ### 📦 Installation requise:
        Hunyuan3D doit être installé avec custom_rasterizer compilé
        ```bash
        # Installer Hunyuan3D
        git clone https://github.com/Tencent/Hunyuan3D-2.git
        cd Hunyuan3D-2
        pip install -e .
        
        # Compiler custom_rasterizer
        cd hy3dgen/texgen/custom_rasterizer
        python setup.py install
        ```
        
        ### ⚠️ Notes importantes:
        - La compilation de `custom_rasterizer` nécessite Visual Studio sur Windows
        - Si custom_rasterizer échoue, la texture ne fonctionnera pas
        - Minimum 8 GB VRAM recommandé pour la texture
    """)
    
    # Événements
    generate_result = generate_btn.click(
        fn=generate_mesh,
        inputs=[input_image, seed, num_steps, guidance_scale, faces],
        outputs=[output_3d, processed_image, status_3d, mesh_state]
    )
    
    generate_result.then(
        fn=lambda img: img,
        inputs=[processed_image],
        outputs=[processed_image_state]
    ).then(
        fn=lambda mesh: gr.Button(interactive=mesh is not None),
        inputs=[mesh_state],
        outputs=[texture_btn]
    )
    
    texture_btn.click(
    fn=apply_texture_to_mesh,
    # app.apply_texture(processed_image, mesh_path, reduce_faces, target_faces, progress)
    inputs=[processed_image_state, mesh_state, reduce_faces, target_faces],
    outputs=[textured_3d, status_texture]
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TripoSG with Official Hunyuan3D Texture")
    parser.add_argument("--texgen_model_path", type=str, default=None, help="Chemin Hunyuan3D")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    
    args = parser.parse_args()
    
    if args.texgen_model_path:
        LOCAL_MODEL_PATHS['texgen_model_path'] = args.texgen_model_path
        print(f"📁 Chemin Hunyuan3D: {args.texgen_model_path}")
    
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)



