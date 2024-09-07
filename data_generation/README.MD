# :sushi:Reproduce SALMon!:sushi:
For each script, you need to download custom data. Some scripts also require AZURE credentials to generate speech.

## Data
In order to reproduce SALMon, you first need to download the data.
Some datasets have been filtered by us, and download links are specified,
and some are the publicly available datasets that you can download by yourself.

|      Dataset       |                        Cite                         |                                              Download Link                                              |
|:------------------:|:---------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:| 
|     LJ Speech      |    [LJ](https://keithito.com/LJ-Speech-Dataset/)    | [google drive](https://drive.google.com/drive/folders/1gR-tKR6R9qy1rx-Icfb-cOmcPEr5Ft7k?usp=share_link) |
| Filtered Bg Noises | [FSD50K](https://paperswithcode.com/dataset/fsd50k) | [google drive](https://drive.google.com/drive/folders/1EXWHrvAoPsyCQ_Bng7miRO5xwPsSFX4f?usp=share_link) |
|    Filtered Rir    |  [EchoThief](https://www.echothief.com/downloads/)  | [google drive](https://drive.google.com/drive/folders/1sTxylBHk_jKh5CXtJh9kJtWtioMKHwec?usp=share_link) |
|        VCTK        |   [VCTK](https://paperswithcode.com/dataset/vctk)   |                             [VCTK](https://datashare.ed.ac.uk/download/DS_10283_2651.zip)                             |
|      EXPRESSO      |  [Expresso](https://speechbot.github.io/expresso/)  |                            [Expresso](https://speechbot.github.io/expresso/)                            |
|       Texts        |                       GPT-4o                        | [google drive](https://drive.google.com/drive/folders/17AFHMJhO3b2di9_Hg_i3VybsL9tw1OZL?usp=share_link) |



## Generate

Each task can be easily generated using its specified python script.
Note that speaker consistency and gender consistency both created using `speaker_consistency.py`,
and both background consistency versions are created using `background_consistency.py`. Pay attention for each script arguments.

For example in order to recreate the background consistency dataset, you need to run the following command:

```bash
python bg_consistency --distract-method random --num-samples 200 --output-path /path/to/output
```



