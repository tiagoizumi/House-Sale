# O projeto possui dois passos:
1. Implementação da rede neural
2. Construção de um mapa interativo

# 1. Rede neural
Nesse passo é criado uma rede neural que tem como objetivo predizer o valor das casas com base nos atributos
Para achar a melhor combinação de número de camadas e número de neurônios em cada camada é implementado um grid search junto a um k-fold cross-validation

# 2. Mapa
Nesse passo é construido um mapa, que para cada casa tem um marcador de uma cor que indica a precisão da predição do preço:
- Vermelho: preço real > predição
- Branco: preço real = predição
- Verde: preço real < preidção
 