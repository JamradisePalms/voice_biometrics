# ECAPA-TDNN Voice Biometrics

## Описание
Данный репозиторий содержит пример обучения модели для биометрической идентификации по голосу. В качестве модели используется ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time-Delay Neural Network), разработанная с нуля.

## Содержание
- [Описание модели](#описание-модели)
- [TODO](#todo)
- [Ссылки на исследования](#ссылки-на-исследования)
- [Связанные работы](#связанные-работы)

## Описание модели
ECAPA-TDNN является современным подходом к задаче биометрической идентификации по голосу. Модель учитывает временные характеристики и эффективно обрабатывает различные голоса. Для обучения использовался набор данных VoxCeleb.

![ECAPA-TDNN Model](https://raw.githubusercontent.com/JamradisePalms/voice_biometrics/main/ecapa.jpg)

## TODO
- Реализовать Triplet Loss для улучшения качества модели.

## Ссылки на исследования
- **Основная работа**: [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143.pdf)

### Связанные работы
- **SE-Blocks**: [Статья](https://arxiv.org/pdf/1709.01507.pdf)
- **Res2Net**: [Статья](https://arxiv.org/pdf/1904.01169.pdf)
- **Attentive Stats Pooling**: [Статья](https://arxiv.org/pdf/1803.10963.pdf)
- **AAM Softmax**: [Статья](https://arxiv.org/pdf/1906.07317.pdf)
