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


class TripoSGApp:
    def __init__(self, device="cuda", dtype=torch.float16, profile=4):
        self.device = device
        self.dtype = dtype
        self.profile = profile
        self.pipe = None
        self.rmbg_net = None
        
    def initialize_models(self):
        """Load the models if they aren’t already initialized."""
        if self.pipe is None or self.rmbg_net is None:
            # download pre-ttrained weight
            triposg_weights_dir = "pretrained_weights/TripoSG"
            rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
            
            if not os.path.exists(triposg_weights_dir):
                snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
            if not os.path.exists(rmbg_weights_dir):
                snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
            
            # Init RMBG
            self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(self.device)
            self.rmbg_net.eval()
            
            # Init TripoSG
            self.pipe = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(self.device, self.dtype)
            
            # Apply mmgp for memmory optimisation
            offload.profile(self.pipe, profile_type.LowRAM_LowVRAM)
            
    def clear_memory(self):
        """Libère la mémoire GPU et CPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def save_temp_image(self, image: Image.Image) -> str:
        """Sauvegarde temporairement l'image PIL et retourne le chemin"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name)
        return temp_file.name
    
    def mesh_to_pymesh(self, vertices, faces):
        """Convert mesh in pymeshlab MeshSet"""
        mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        return ms
    
    def pymesh_to_trimesh(self, mesh):
        """Convert pymeshlab mesh in trimesh"""
        verts = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        return trimesh.Trimesh(vertices=verts, faces=faces)
    
    def simplify_mesh(self, mesh: trimesh.Trimesh, n_faces: int):
        """Simplify the mesh to the desired number of faces."""
        if n_faces > 0 and mesh.faces.shape[0] > n_faces:
            ms = self.mesh_to_pymesh(mesh.vertices, mesh.faces)
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
            simplified = self.pymesh_to_trimesh(ms.current_mesh())
            del ms
            return simplified
        return mesh
    
    @torch.no_grad()
    def generate_3d(
        self,
        image: Image.Image,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        faces: int = -1,
        progress=gr.Progress()
    ):
        """Generate 3D mesh from image with progressive display."""
        temp_image_path = None
        try:
            self.initialize_models()
            
            progress(0.05, desc="Saving image temporarily...")
            
            # temp PIL save
            temp_image_path = self.save_temp_image(image)
            
            progress(0.1, desc="Removing background...")
            
            # Prepare image with bg removal
            img_pil = prepare_image(
                temp_image_path, 
                bg_color=np.array([1.0, 1.0, 1.0]), 
                rmbg_net=self.rmbg_net
            )
            
            # ✨ YIELD 1: Display image with transparent background immediately
            yield None, img_pil, "⏳ Image processed, generating 3D model..."
            
            progress(0.3, desc="Generating 3D mesh...")
            
            # Generate mesh
            outputs = self.pipe(
                image=img_pil,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).samples[0]
            
            progress(0.8, desc="Building mesh...")
            
            # Create Trimesh mesh
            mesh = trimesh.Trimesh(
                outputs[0].astype(np.float32), 
                np.ascontiguousarray(outputs[1])
            )
            
            # Clean outputs
            del outputs
            self.clear_memory()
            
            # Simplify if needed
            if faces > 0:
                progress(0.9, desc="Simplification du mesh...")
                mesh = self.simplify_mesh(mesh, faces)
            
            # Save the mesh
            output_path = "output_temp.glb"
            mesh.export(output_path)
            
            # got data from mesh
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            file_size = os.path.getsize(output_path) / 1024  # KB
            
            # Clean
            del mesh
            self.clear_memory()
            
            # Clean temp files
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            progress(1.0, desc="Finish!")
            
            success_msg = f"✅ 3D mesh generated successfully!\n📊 Vertices: {num_vertices:,} | Faces: {num_faces:,}\n💾 Size: {file_size:.2f} KB"
          
            
            # ✨ YIELD 2: Show finale result
            yield output_path, img_pil, success_msg
            
        except Exception as e:
            # Nettoyer en cas d'erreur
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            self.clear_memory()
            yield None, None, f"❌ Erreur lors de la génération: {str(e)}"


# Init app
app = TripoSGApp()


def generate_mesh(image, seed, steps, guidance, faces, progress=gr.Progress()):
    """Generate 3D mesh with automatic background removal and progressive display"""
    if image is None:
        yield None, None, "⚠️ Please upload an image"
        return
    
    # Use yield from to propagate progressive updates
    yield from app.generate_3d(image, seed, steps, guidance, faces, progress)


# Gradio Interface
with gr.Blocks(title="TripoSG - Image to 3D", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 TripoSG – 3D Generation from Image
        Turn your images into high-quality 3D models.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Inupt Image")
            input_image = gr.Image(
                label="Drag & drop your image here",
                type="pil",
                height=400
            )
            
            gr.Markdown("### ⚙️ Settings")
            
            seed = gr.Slider(
                label="Seed (for reproducibility)",
                minimum=0,
                maximum=1000000,
                value=42,
                step=1
            )
            
            num_steps = gr.Slider(
                label="Number of inference steps",
                minimum=20,
                maximum=100,
                value=50,
                step=1
            )
            
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=15.0,
                value=7.0,
                step=0.5
            )
            
            faces = gr.Slider(
                label="Number of faces (simplification, -1 to disable)",
                minimum=-1,
                maximum=100000,
                value=-1,
                step=1000
            )
            
            generate_btn = gr.Button("🚀 Generate 3D model", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Image without background")
            processed_image = gr.Image(label="Aperçu", type="pil", height=300)
            
            gr.Markdown("### 🎭 Generated 3D model")
            output_3d = gr.Model3D(
                label="3D visualization",
                height=400,
                clear_color=[1.0, 1.0, 1.0, 1.0]
            )
            status_3d = gr.Textbox(label="Statut", interactive=False, lines=3)
    
    gr.Markdown(
        """
        ### 💡 Usage tips:
        - Use images with a well-defined subject
        - Isolated objects work best
        - Increase the number of steps for higher quality (50–100)
        - Mesh simplification reduces file size
        - The background is automatically removed during generation
        """
    )
    
    # Événements
    generate_btn.click(
        fn=generate_mesh,
        inputs=[input_image, seed, num_steps, guidance_scale, faces],
        outputs=[output_3d, processed_image, status_3d]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
