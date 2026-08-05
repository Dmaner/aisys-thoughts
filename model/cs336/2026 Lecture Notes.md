> Lecture : https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
> Slides: https://cs336.stanford.edu/lectures/?trace=lecture_17
# Lecture 1 Overview, Tokenization

## Tokenization
BPE
# **Lecture 2: PyTorch (einops)**

$$
\text{actual multiply-add throughput} = \frac{\text{advertised FLOPS}}{2}
$$

现在有哪些数据类型fp32 等, 列举出来, 表格化说明数据的表示范围上下限


| 类型       | bit | sign | exponent | fraction | 最大值     | 最小正数     |
| -------- | --- | ---- | -------- | -------- | ------- | -------- |
| FP64     | 64  | 1    | 11       | 52       | 1.8e308 | 2.2e-308 |
| FP32     | 32  | 1    | 8        | 23       | 3.4e38  | 1.18e-38 |
| FP16     | 16  | 1    | 5        | 10       | 65504   | 6.1e-5   |
| BF16     | 16  | 1    | 8        | 7        | 3.4e38  | 1.18e-38 |
| FP8 E4M3 | 8   | 1    | 4        | 3        | 448     | 0.00195  |
| FP8 E5M2 | 8   | 1    | 5        | 2        | 57344   | 1.52e-5  |


FLOPs: 总计算量

FLOP/s: 每秒可以做多少浮点计算次数

为什么FLOP/s和硬件以及datatype有关系

- 不同类型的硬件电路设计不同

MFU = actual FLOPS/s / promised FLOP/s

# Lecture 17 Alignment - Multimodality

## Keywords

> CLIP, LLAVA, QWEN VL, Multimodality

## Lectures Notes

课程大纲

- Encoding images
- Injecting image encodings into LLMs
- Towards Omni models

### ViT

![arch](assets/vit.png)

### CLIP


自然语言可以作为通用的视觉监督

> https://arxiv.org/abs/2103.00020


![CLIP](assets/clip.png)

pseudocode

![CLIP](assets/code-clip.png)

### llava

Image -> Vision Encoder: CLIP -> Projection -> LM

![llava](assets/llava.png)

llava onevision

![llava_onevision](assets/llava_onevision.png)

### Qwen-VL

Qwen2-VL

![Qwen2-VL](assets/qwen2-vl.png)

Qwen3-VL

![Qwen3-VL](assets/qwen3-vl.png)

## Explore

### Omni Model

Qwen3-Omni

![arch](assets/qwen3-omni.png)

AuT

![arch](assets/aut.png)
