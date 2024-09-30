import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

def capture_view(vis, angle):
    """
    Captures the view from a given camera angle (front or back).
    """
    ctr = vis.get_view_control()
    ctr.rotate(angle[0], angle[1])
    vis.poll_events()
    vis.update_renderer()
    
    # Capture screen and convert to numpy array
    image = vis.capture_screen_float_buffer(do_render=True)
    return np.asarray(image)

def render_ply_to_image(ply_file, output_folder):
    # Load PLY file
    mesh = o3d.io.read_triangle_mesh(ply_file)
    
    # Check if the mesh has vertex normals, if not, compute them
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Get the filename without extension
    file_name = os.path.basename(ply_file).split('.')[0]
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    
    # Set camera parameters and render front view (no rotation)
    front_image = capture_view(vis, (0, 0))
    
    # Save front view as an image
    front_output_file = os.path.join(output_folder, f"{file_name}_front.png")
    plt.imsave(front_output_file, front_image)
    
    # Render back view (180-degree rotation along y-axis)
    back_image = capture_view(vis, (0, 180))
    
    # Save back view as an image
    back_output_file = os.path.join(output_folder, f"{file_name}_back.png")
    plt.imsave(back_output_file, back_image)
    
    # Destroy the visualizer window
    vis.destroy_window()

def process_ply_folder(ply_folder, output_folder):
    """
    Process all PLY files in a folder and save images in output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the ply_folder
    for file_name in os.listdir(ply_folder):
        if file_name.endswith(".ply"):
            ply_file = os.path.join(ply_folder, file_name)
            print(f"Processing {file_name}...")
            render_ply_to_image(ply_file, output_folder)
    
    print("All PLY files processed.")

# Example usage
ply_folder = r"C:\SURANADI\UCF\CV_Systems\Project\images\shapE_outputs"  
output_folder = r"C:\SURANADI\UCF\CV_Systems\Project\view_2d\output_images"
process_ply_folder(ply_folder, output_folder)

print("Images saved.")
