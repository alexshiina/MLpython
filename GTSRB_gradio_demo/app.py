import gradio as gr
import os
import torch
import torchvision
from torchvision import transforms

from model import Vit_b16_model

from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = {
    0: 'Ograniczenie do 20',
    1: 'Ograniczenie do 30',
    2: 'Ograniczenie do 50',
    3: 'Ograniczenie do 60',
    4: 'Ograniczenie do 70',
    5: 'Ograniczenie do 80',
    6: 'Koniec ograniczenia prędkości',
    7: 'Ograniczenie do 100',
    8: 'Ograniczenie do 120',
    9: 'Zakaz wyprzedzania',
    10: 'Zakaz wyprzedzania przez samochody ciężarowe',
    11: 'Skrzyżowanie z drogą podporządkowaną występującą po obu stronach',
    12: 'Pierwszeństwo',
    13: 'Ustąp pierszeństwa',
    14: 'STOP',
    15: 'Zakaz ruchu w obu kierunkach',
    16: 'Zakaz wjazdu ciężarówek',
    17: 'Zakaz wjazdu',
    18: 'Inne niebezpieczeństwo',
    19: 'Niebezpieczny zakręt w lewo',
    20: 'Niebezpieczny zakręt w prawo',
    21: 'Niebezpieczne zakręty - pierwszy w lewo',
    22: 'Nierówna droga',
    23: 'Śliska jezdnia',
    24: 'Zwężenie jezdni prawostronne',
    25: 'Roboty drogowe',
    26: 'Sygnały świetlne',
    27: 'Piesi',
    28: 'Dzieci',
    29: 'Nie umiem rozpoznac',
    30: 'Oszronienie jezdni',
    31: 'Zwierzęta dzikie',
    32: 'Koniec zakazów',
    33: 'Nakaz skrętu w prawo za znakiem',
    34: 'Skręt w prawo za znakiem',
    35: 'Nakaz jazdy prosto',
    36: 'Nakaz jazdy prosto lub w prawo',
    37: 'Nakaz jazdy prosto lub w lewo',
    38: 'Obowiązkowe obejście w prawo',
    39: 'Obowiązkowe obejście w lewo',
    40: 'Ruch okrężny',
    41: 'Nie umiem rozpoznać - 2',
    42: 'Koniec zakazu wyprzedzania'
}

vitb16, vitb16_transforms = Vit_b16_model()

vitb16.load_state_dict(
    torch.load(
        f="model.pth",
        map_location = torch.device("cpu")
    )
)


def predict(img) -> Tuple[Dict,float]:
  start_time = timer()
  image = vitb16_transforms(img).unsqueeze(0) #batch dimension

  vitb16.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(vitb16(image),dim=1)

  pred_labels_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  pred_time = round(timer() - start_time,5)

  return pred_labels_probs,pred_time

title = "Demo of GTSRB 90+% "
description = "I made this demo to showcase my skills. I hope it can be somehow useful to you :) Aleksander"
article = """Created with help of mrdbourkes tutorial. Available class names:
    0: 'Ograniczenie do 20',
    1: 'Ograniczenie do 30',
    2: 'Ograniczenie do 50',
    3: 'Ograniczenie do 60',
    4: 'Ograniczenie do 70',
    5: 'Ograniczenie do 80',
    6: 'Koniec ograniczenia prędkości',
    7: 'Ograniczenie do 100',
    8: 'Ograniczenie do 120',
    9: 'Zakaz wyprzedzania',
    10: 'Zakaz wyprzedzania przez samochody ciężarowe',
    11: 'Skrzyżowanie z drogą podporządkowaną występującą po obu stronach',
    12: 'Pierwszeństwo',
    13: 'Ustąp pierszeństwa',
    14: 'STOP',
    15: 'Zakaz ruchu w obu kierunkach',
    16: 'Zakaz wjazdu ciężarówek',
    17: 'Zakaz wjazdu',
    18: 'Inne niebezpieczeństwo',
    19: 'Niebezpieczny zakręt w lewo',
    20: 'Niebezpieczny zakręt w prawo',
    21: 'Niebezpieczne zakręty - pierwszy w lewo',
    22: 'Nierówna droga',
    23: 'Śliska jezdnia',
    24: 'Zwężenie jezdni prawostronne',
    25: 'Roboty drogowe',
    26: 'Sygnały świetlne',
    27: 'Piesi',
    28: 'Dzieci',
    29: 'Nie umiem rozpoznac',
    30: 'Oszronienie jezdni',
    31: 'Zwierzęta dzikie',
    32: 'Koniec zakazów',
    33: 'Nakaz skrętu w prawo za znakiem',
    34: 'Skręt w prawo za znakiem',
    35: 'Nakaz jazdy prosto',
    36: 'Nakaz jazdy prosto lub w prawo',
    37: 'Nakaz jazdy prosto lub w lewo',
    38: 'Obowiązkowe obejście w prawo',
    39: 'Obowiązkowe obejście w lewo',
    40: 'Ruch okrężny',
    41: 'Nie umiem rozpoznać - 2',
    42: 'Koniec zakazu wyprzedzania'
"""
example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type = "pil"),
    outputs=[gr.Label(num_top_classes = 43, label = "Predictions"),
             gr.Number(label = "Prediction time(s)")],
    examples = example_list,
    title = title,
    description = description,
    article= article
)
demo.launch()