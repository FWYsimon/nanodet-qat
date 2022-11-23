# NanoDet-QAT
The inplement of the qat version of nanodet. Here is the offical repo of [nanodet](https://github.com/RangiLyu/nanodet) and the offical tutorial of [Pytorch QAT](https://pytorch.org/docs/1.12/quantization.html).

## Code
The QAT code is inherited from the origin code. 

You can check the main quantization code in:

`model/backbone/quantization_mobilenetv2.py`

`model/fpn/quantization_ghost_pan.py`

`model/head/quantization_nanodet_plus_head.py`

## How to run
Just modify the yaml file in ./config. The model already converted was trained by `config/nanodet-plus-m_qat_320.yml`.

* Train
```bash
python tools/train_qat.py image --config CONFIG_PATH
