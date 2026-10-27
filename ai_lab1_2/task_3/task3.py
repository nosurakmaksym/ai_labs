import gradio as gr
import numpy as np
import os
from PIL import Image

def get_norm_vec(image, r=5, c=4):
    img_array = np.array(image.convert('L')) < 128
    h, w = img_array.shape
    ch, cw = h // r, w // c
    vec = [int(np.sum(img_array[i*ch:(i+1)*ch, j*cw:(j+1)*cw])) for i in range(r) for j in range(c)]
    m = max(vec) if max(vec) > 0 else 1
    return np.array([x / m for x in vec])

def calc_distance(v1, v2, metric):
    if metric == "Евклідова": return np.sqrt(np.sum((v1 - v2)**2))
    elif metric == "Манхетенська": return np.sum(np.abs(v1 - v2))
    elif metric == "Чебишевська": return np.max(np.abs(v1 - v2))
    return 0
 
def classify_cluster(img_unk, metric):
    if img_unk is None: return "Завантажте невідомий образ."
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "references")
    
    if not os.path.exists(base_dir):
        return f"Помилка\n{base_dir}"

    clusters = {}
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            vectors = []
            for f in files:
                img = Image.open(os.path.join(class_dir, f))
                vectors.append(get_norm_vec(img))
            if vectors:
                clusters[class_name] = np.mean(vectors, axis=0)

    if not clusters: return "Помилка"

    unk_vec = get_norm_vec(img_unk)
    distances = {name: calc_distance(center, unk_vec, metric) for name, center in clusters.items()}
    best_match = min(distances, key=distances.get)

    
    res = f"Відстані: ({metric}):\n"
    for k, d in distances.items():
        res += f"Клас '{k}': {d:.3f}\n"
    res += f"\nРезультат класифікації: {best_match}"
    return res

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Невідомий образ")
        with gr.Column():
            metric = gr.Dropdown(choices=["Евклідова", "Манхетенська", "Чебишевська"], value="Евклідова", label="Варіант норми")
            btn = gr.Button("Класифікувати")
            out = gr.Textbox(label="Результати обчислень", lines=7)
            
    btn.click(classify_cluster, inputs=[img_in, metric], outputs=[out])

if __name__ == "__main__":
    demo.launch()