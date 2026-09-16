# Assistive Vision & Edge AI Prototype 👁️🤖

Protótipo de percepção ambiental desenvolvido em Python utilizando OpenCV, focado em tecnologias assistivas vestíveis para pessoas com deficiência visual.

## 🎯 Objetivo do Projeto
Este projeto é a base algorítmica de uma pesquisa voltada para o desenvolvimento de sistemas vestíveis assistivos. O objetivo é extrair informações semanticamente ricas do ambiente para apoiar a mobilidade e a detecção de obstáculos. O foco principal é manter um rigoroso equilíbrio entre acurácia, tempo de resposta e consumo de hardware, visando a implantação em dispositivos de baixo custo (Edge AI).

## 🛠️ Tecnologias e Técnicas Aplicadas
* **Linguagem:** Python
* **Visão Computacional:** OpenCV e NumPy
* **Percepção Ambiental:** Processamento de máscaras de cor (HSV) para segmentação de regiões navegáveis e identificação de obstáculos.
* **Lógica de Decisão:** Extração de contornos e cálculo de proximidade matemática via caixas delimitadoras (Bounding Boxes).

## 🚀 Próximos Passos (Roadmap)
* Integração de modelos de Deep Learning compactos (YOLOv8n-seg) para segmentação semântica de superfícies.
* Otimização da resolução de inferência para ganho de FPS em hardwares restritos.
* Migração e integração da lógica de percepção para aplicativos Companion em Kotlin, utilizando o Meta Wearables Device Access Toolkit (Smart Glasses), visando processamento on-device.
