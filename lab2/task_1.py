import gradio as gr
import numpy as np
import random
from PIL import Image

def get_features(image, grid_rows=5, grid_cols=4):
    if image is None:
        return None, None
    
    img_array = np.array(image.convert('L')) < 128
    h, w = img_array.shape
    ch, cw = h // grid_rows, w // grid_cols
    
    vec = [
        int(np.sum(img_array[i*ch:(i+1)*ch, j*cw:(j+1)*cw]))
        for i in range(grid_rows)
        for j in range(grid_cols)
    ]
    
    m = max(vec) if max(vec) > 0 else 1
    norm_vec = [round(x / m, 4) for x in vec]
    
    return vec, norm_vec


def init_perceptron(num_features=20):
    w = [round(random.uniform(0.05, 0.5), 4) for _ in range(num_features)]
    b = round(random.uniform(0.05, 0.5), 4)
    return w, b, 0


def train_step(image, true_class, lr, weights, bias, success_streak):
    if image is None:
        return (
            "Завантажте зображення", "", "", "", "", "",
            "Завантажте зображення",
            weights, bias, success_streak
        )
    
    abs_vec, norm_vec = get_features(image)
    
    d = 1 if true_class == "Клас А (+1)" else -1
    
    S = sum(w * x for w, x in zip(weights, norm_vec)) - bias
    y = 1 if S >= 0 else -1
    
    old_weights = [round(w, 4) for w in weights]
    old_bias = round(bias, 4)
    
    if y == d:
        status = f"Правильно! y = {y}, d = {d}"
        correction = "Ваги не змінювалися"
        success_streak += 1
    else:
        status = f"Помилка! y = {y}, d = {d}"
        success_streak = 0
        
        delta = d - y
        for i in range(len(weights)):
            weights[i] = round(weights[i] + lr * delta * norm_vec[i], 4)
        
        bias = round(bias - lr * delta, 4)
        correction = "Ваги та зсув скориговано"
    
    new_weights = [round(w, 4) for w in weights]
    new_bias = round(bias, 4)
    
    if success_streak >= 8:
        convergence = f"Мережа успішно навчена! (правильних відповідей поспіль: {success_streak})"
    elif success_streak >= 4:
        convergence = f"Добре просувається (правильних поспіль: {success_streak})"
    else:
        convergence = f"Навчання триває... (правильних поспіль: {success_streak})"
    
    abs_str = str(abs_vec)
    norm_str = str(norm_vec)
    
    s_info = f"S = {round(S, 4)}  ->  y = {y}"
    
    weights_info = (
        f"ДО корекції:\n"
        f"w = {old_weights}\n"
        f"b = {old_bias}\n\n"
        f"ПІСЛЯ корекції:\n"
        f"w = {new_weights}\n"
        f"b = {new_bias}"
    )
    
    return (
        abs_str,
        norm_str,
        s_info,
        status,
        correction,
        weights_info,
        convergence,
        weights, bias, success_streak
    )


def reset_perceptron():
    w, b, streak = init_perceptron()
    return (
        "", "", "", "", "", "",
        "Ваги скинуто",
        w, b, streak
    )


with gr.Blocks(theme=gr.themes.Soft(), title="") as demo:
    
    init_w, init_b, init_streak = init_perceptron()
    state_w = gr.State(init_w)
    state_b = gr.State(init_b)
    state_streak = gr.State(init_streak)
    
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="")
            
            true_class = gr.Radio(
                choices=["Клас А", "Клас Б"],
                value="Клас А",
                label="Вчитель: правильний клас"
            )
            
            lr = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, value=0.1,
                label="Швидкість навчання"
            )
            
            with gr.Row():
                btn_train = gr.Button("Виконати крок навчання", variant="primary")
                btn_reset = gr.Button("Скинути ваги", variant="secondary")
        
        with gr.Column(scale=2):
            gr.Markdown("Вектори ознак")
            out_abs = gr.Textbox(label="Абсолютний вектор", lines=2)
            out_norm = gr.Textbox(label="Нормований вектор", lines=2)
            
            gr.Markdown("Результат класифікації")
            out_S = gr.Textbox(label="Значення S та вихід y")
            out_status = gr.Textbox(label="Вердикт")
            out_corr = gr.Textbox(label="Корекція")
            
            gr.Markdown("Стан перцептрона")
            out_weights = gr.Textbox(label="Ваги та зсув (до / після)", lines=8)
            
            gr.Markdown("Збіжність навчання")
            out_conv = gr.Textbox(label="Оцінка збіжності")
    
    btn_train.click(
        fn=train_step,
        inputs=[img_in, true_class, lr, state_w, state_b, state_streak],
        outputs=[
            out_abs, out_norm, out_S, out_status, out_corr,
            out_weights, out_conv,
            state_w, state_b, state_streak
        ]
    )
    
    btn_reset.click(
        fn=reset_perceptron,
        inputs=[],
        outputs=[
            out_abs, out_norm, out_S, out_status, out_corr,
            out_weights, out_conv,
            state_w, state_b, state_streak
        ]
    )

if __name__ == "__main__":
    demo.launch()