import gradio as gr
import numpy as np
from PIL import Image

def get_feature_vectors(image, grid_rows=5, grid_cols=4):
    if image is None:
        return "Завантажте зображення.", ""
    
    img_array = np.array(image.convert('L'))
    binary_img = img_array < 128
    
    h, w = binary_img.shape
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    vec = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = binary_img[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
            vec.append(int(np.sum(cell)))
            
    m = max(vec) if max(vec) > 0 else 1
    norm_vec = [round(x / m, 3) for x in vec]
    
    return str(vec), str(norm_vec)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Завантажте графічне зображення")
        with gr.Column():
            abs_out = gr.Textbox(label="Абсолютний вектор ознак")
            norm_out = gr.Textbox(label="Нормований вектор ознак")
             
    img_in.change(get_feature_vectors, inputs=[img_in], outputs=[abs_out, norm_out])

if __name__ == "__main__":
    demo.launch()