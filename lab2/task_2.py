import gradio as gr
import numpy as np
from PIL import Image

def get_abs_vector(image, grid_rows=5, grid_cols=4):
    if image is None:
        return None
    img_array = np.array(image.convert('L')) < 128
    h, w = img_array.shape
    ch, cw = max(1, h // grid_rows), max(1, w // grid_cols)

    vec = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = img_array[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            vec.append(int(np.sum(cell)))
    return np.array(vec, dtype=float)


def binarize(vec):
    return np.where(vec > 0, 1, -1).astype(int)


def normalize(vec):
    m = np.max(vec) if np.max(vec) > 0 else 1
    return np.round(vec / m, 4)


def hopfield_network(v1, v2, v3, v_unk):
    N = len(v1)

    W = np.zeros((N, N))
    for v in (v1, v2, v3):
        W += np.outer(v, v)
    np.fill_diagonal(W, 0)

    matrix_str = f"Матриця ваг Хопфілда W ({N}×{N}):\n{np.array2string(W, max_line_width=120)}"

    Y = v_unk.copy().astype(float)
    history = "Початковий стан Y(0) = невідомий вектор:\n" + str(Y.astype(int).tolist()) + "\n\n"

    for t in range(1, 16):
        S = W @ Y
        Y_new = np.where(S >= 0, 1, -1)

        history += f"Ітерація {t}:\n"
        history += f"S = {np.round(S, 2).tolist()}\n"
        history += f"Y = {Y_new.tolist()}\n\n"

        if np.array_equal(Y, Y_new):
            history += f"Стабілізація на ітерації {t}\n"
            break
        Y = Y_new.copy()
    else:
        history += "Досягнуто ліміту ітерацій\n"

    if np.array_equal(Y, v1):
        result = "Результат: Клас 1"
    elif np.array_equal(Y, v2):
        result = "Результат: Клас 2"
    elif np.array_equal(Y, v3):
        result = "Результат: Клас 3"
    else:
        result = "Нерозпізнано"

    return matrix_str, history, result



def hamming_network(v1, v2, v3, v_unk):
    N = len(v1)
    m = 3

    W = np.column_stack((v1, v2, v3)) / 2.0
    b = N / 2.0

    matrix_str = (f"Ваги першого шару W (розмір {N}×3):\n"
                  f"{np.round(W, 4)}\n\n"
                  f"Пороги b для всіх нейронів = {b}")

    Y1 = v_unk @ W + b
    history = (f"Виходи першого шару:\n"
               f"Y1 (Клас 1) = {Y1[0]:.4f}\n"
               f"Y2 (Клас 2) = {Y1[1]:.4f}\n"
               f"Y3 (Клас 3) = {Y1[2]:.4f}\n\n")

    eps = 0.2
    Y2 = Y1.copy()

    history += f"Другий шар, e = {eps}\n"
    history += f"Y(0): {np.round(Y2, 4).tolist()}\n\n"

    for t in range(1, 40):
        S = np.zeros(m)
        for j in range(m):
            sum_others = np.sum(Y2) - Y2[j]
            S[j] = Y2[j] - eps * sum_others

        Y2_new = np.maximum(S, 0)

        history += f"Ітерація {t}: {np.round(Y2_new, 4).tolist()}\n"

        if np.allclose(Y2, Y2_new, atol=1e-5):
            history += f"\nСтабілізація на ітерації {t}\n"
            break
        Y2 = Y2_new.copy()
    else:
        history += "\nДосягнуто ліміту ітерацій\n"

    best = int(np.argmax(Y2))
    result = f"Результат: Клас {best + 1}"

    return matrix_str, history, result


def run_network(img1, img2, img3, img_unk, net_type):
    if None in (img1, img2, img3, img_unk):
        return ("Завантажте всі 4 зображення",
                "", "", "", "", "")

    abs1 = get_abs_vector(img1)
    abs2 = get_abs_vector(img2)
    abs3 = get_abs_vector(img3)
    abs_unk = get_abs_vector(img_unk)

    abs_str = (f"Клас 1: {abs1.astype(int).tolist()}\n"
               f"Клас 2: {abs2.astype(int).tolist()}\n"
               f"Клас 3: {abs3.astype(int).tolist()}")

    if net_type == "Мережа Хопфілда":
        v1 = binarize(abs1)
        v2 = binarize(abs2)
        v3 = binarize(abs3)
        v_unk = binarize(abs_unk)

        feat_str = (f"Клас 1: {v1.tolist()}\n"
                    f"Клас 2: {v2.tolist()}\n"
                    f"Клас 3: {v3.tolist()}")
        unk_str = str(v_unk.tolist())

        matrix_str, history, result = hopfield_network(v1, v2, v3, v_unk)

    else:
        v1 = normalize(abs1)
        v2 = normalize(abs2)
        v3 = normalize(abs3)
        v_unk = normalize(abs_unk)

        feat_str = (f"Клас 1: {v1.tolist()}\n"
                    f"Клас 2: {v2.tolist()}\n"
                    f"Клас 3: {v3.tolist()}")
        unk_str = str(v_unk.tolist())

        matrix_str, history, result = hamming_network(v1, v2, v3, v_unk)

    return abs_str, feat_str, matrix_str, unk_str, history, result



with gr.Blocks(theme=gr.themes.Soft(), title="") as demo:
    
    with gr.Row():
        img1 = gr.Image(type="pil", label="Еталон Класу 1", height=220)
        img2 = gr.Image(type="pil", label="Еталон Класу 2", height=220)
        img3 = gr.Image(type="pil", label="Еталон Класу 3", height=220)

    with gr.Row():
        with gr.Column(scale=1):
            img_unk = gr.Image(type="pil", label="Невідомий образ", height=220)
            net_type = gr.Radio(
                choices=["Мережа Хопфілда", "Мережа Хеммінга"],
                value="Мережа Хопфілда",
                label="Тип мережі"
            )
            btn = gr.Button("Обчислити та класифікувати", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("Абсолютні та підготовлені вектори", open=True):
                out_abs = gr.Textbox(label="Абсолютні вектори еталонів", lines=4)
                out_feat = gr.Textbox(label="Підготовлені еталони (бінарні / нормовані)", lines=4)
                out_unk = gr.Textbox(label="Ознаковий вектор невідомого образу", lines=2)

            out_matrix = gr.Textbox(label="Матриця коефіцієнтів / ваги 1-го шару", lines=8)
            out_history = gr.Textbox(label="Аналіз процесу класифікації (ітерації)", lines=12)
            out_res = gr.Textbox(label="Фінальний результат", lines=2)

    btn.click(
        fn=run_network,
        inputs=[img1, img2, img3, img_unk, net_type],
        outputs=[out_abs, out_feat, out_matrix, out_unk, out_history, out_res]
    )

if __name__ == "__main__":
    demo.launch()