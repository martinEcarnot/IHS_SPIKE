import tkinter as tk
from tkinter import messagebox
import subprocess

# Valeurs par défaut (les tiennes)
default_data = {
    'input_path': 'data/in',
    'output_path': 'data/out',
}

default_segment_kernels = {
    'model_path': 'models/sam_b.pt',
    'output_mask_path': 'None',
    'device': 'cpu',  # ou 'cuda'
    'model_type': 'vit_b',  # ou 'vit_l'
    'area_min': 4000,
    'area_max': 15000,
    'crop_x_left': 690,
    'crop_x_right': 3470,
    'save_kernels': False
}

# Lancer le script main.py
def run_script():
    try:
        # Tu peux ajouter ici les arguments ou les variables du script si besoin
        subprocess.run(['python', 'main.py'])
        messagebox.showinfo("Exécution", "Le script main.py a été exécuté avec succès!")
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de l'exécution: {e}")

# Créer l'interface graphique
root = tk.Tk()
root.title("IHS_SPIKE")

# Variables
save_kernels_var = tk.BooleanVar(value=default_segment_kernels['save_kernels'])

tk.Label(root, text="input_path:").grid(row=1, column=0, sticky="e")
input_path_entry = tk.Entry(root)
input_path_entry.grid(row=1, column=1)
input_path_entry.insert(0, default_data['input_path'])

tk.Label(root, text="output_path:").grid(row=2, column=0, sticky="e")
output_path_entry = tk.Entry(root)
output_path_entry.grid(row=2, column=1)
output_path_entry.insert(0, default_data['output_path'])

# Champs pour la section "segment_kernels"
tk.Label(root, text="model_path:").grid(row=3, column=0, sticky="e")
model_path_entry = tk.Entry(root)
model_path_entry.grid(row=3, column=1)
model_path_entry.insert(0, default_segment_kernels['model_path'])

tk.Label(root, text="output_mask_path:").grid(row=4, column=0, sticky="e")
output_mask_entry = tk.Entry(root)
output_mask_entry.grid(row=4, column=1)
output_mask_entry.insert(0, default_segment_kernels['output_mask_path'])

tk.Label(root, text="device:").grid(row=5, column=0, sticky="e")
device_entry = tk.Entry(root)
device_entry.grid(row=5, column=1)
device_entry.insert(0, default_segment_kernels['device'])

tk.Label(root, text="model_type:").grid(row=6, column=0, sticky="e")
model_type_entry = tk.Entry(root)
model_type_entry.grid(row=6, column=1)
model_type_entry.insert(0, default_segment_kernels['model_type'])

tk.Label(root, text="area_min:").grid(row=7, column=0, sticky="e")
area_min_entry = tk.Entry(root)
area_min_entry.grid(row=7, column=1)
area_min_entry.insert(0, default_segment_kernels['area_min'])

tk.Label(root, text="area_max:").grid(row=8, column=0, sticky="e")
area_max_entry = tk.Entry(root)
area_max_entry.grid(row=8, column=1)
area_max_entry.insert(0, default_segment_kernels['area_max'])

tk.Label(root, text="crop_x_left:").grid(row=9, column=0, sticky="e")
crop_x_left_entry = tk.Entry(root)
crop_x_left_entry.grid(row=9, column=1)
crop_x_left_entry.insert(0, default_segment_kernels['crop_x_left'])

tk.Label(root, text="crop_x_right:").grid(row=10, column=0, sticky="e")
crop_x_right_entry = tk.Entry(root)
crop_x_right_entry.grid(row=10, column=1)
crop_x_right_entry.insert(0, default_segment_kernels['crop_x_right'])

tk.Label(root, text="save_kernels:").grid(row=11, column=0, sticky="e")
save_kernels_checkbox = tk.Checkbutton(root, variable=save_kernels_var)
save_kernels_checkbox.grid(row=11, column=1)

# Bouton pour lancer le script
run_button = tk.Button(root, text="Run", command=run_script)
run_button.grid(row=12, column=1)

root.mainloop()
