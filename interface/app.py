# interface/app.py

import streamlit as st
from models.full_system import TarotSystem
from utils.visualizer import visualize_card_circle
import torch

# Инициализация модели
model = TarotSystem(embedding_dim=64)

# Заголовок интерфейса
st.title("Таро Расклад")

# Выбор карт расклада
cards = st.multiselect("Выберите карты (по индексу):", list(range(78)), default=[0, 1, 2])

# Пример контекста
context = st.text_area("Контекст расклада", "Что мне нужно понять в своей жизни?")

# Кнопка для генерации трактовок
if st.button("Генерировать трактовку"):
    if cards:
        logits = model(cards)
        topk = torch.topk(logits, k=5)
        st.write(f"Лучшие трактовки (ID слов): {topk.indices.tolist()}")
        visualize_card_circle(model.model.cards[cards[0]], card_name="Первая Карта")
    else:
        st.warning("Пожалуйста, выберите хотя бы одну карту.")
