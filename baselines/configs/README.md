## Segmentor Configs Explanation

This readme will be used to explain config files format.

We use to config files to have an easy and flexible way to set choose how the segmentor will work and make it flexiable.

As the paper states, most approaches used in the paper use 3 parts:
1. sentencer
2. scorer
3. span selector

We create a segmentor mixing and matching different configs for each of those parts.

this gives us a format for a segmentor:

```
{
    "sentencer": {
        ...
    },
    "scorer": {
        ...
    },
    "sselector":{
        ...
    },
    "default_sample_rate": 16000,
    "type": <type>
}

```

if we don't need some part in the pipeline we can omit it (this of course depends on the `type` of the segmentor, e.g. equal length segmentor doesn't need a scorer,unsupseg doesn't need any of those)

### Segmentor Config

this is the full config given to [audio_segment.py](../../audio_segment.py).

The required keys/arguments needed for the segmentor are:
- `type`: type of segmentor to use. currently available (i) `"speech_pmi"`(ii) `'equal_length'` (iii) `'next_frame'`
- `default_sample_rate`: default sample rate used. 

### Sentencer Config

This is a config used for the sentencer in the pipeline.

required keys/arguments:
- `type`: type of sentencer. currently the only type available is `"constant"`

for more information and arguments for the sentencer look at [speech_sentencer.py](../speech_sentencer.py)

### Scorer Config

This is a config used for the scorer in the pipeline.

required keys/arguments:

- `type`: type of scorer to use. currently the only one available is `"speech_pmi`.
for `speech_pmi` you also need the arguments `model_type` with option `TWIST-(350M/1-3B/7B)` and a tokenizer that needs 5 arguments

this gives us the following config 

```
"scorer": {
    "type" : "speech_pmi",
    "model_type" : "TWIST-(350M/1-3B/7B)",
    "tokenizer": {
        ...
    }
}
```

#### tokenizer config

the tokenizer uses textless. it has 5 arguments needed

- `dense_model_name`: model used for embedding
- `quantizer_model_name` : quantizer used to convert embeddings into tokens
- `encoder_vocab_size`: vocab size
- `deduplicate`: deduplicate the tokens
- `need_f0`: get f0 as well from the tokenizer

for more information look at [textless-lib](https://github.com/facebookresearch/textlesslib)


all expirements used in the paper use the same tokenizer config

```
"tokenizer": {
    "dense_model_name": "mhubert-base-25hz",
    "quantizer_model_name" : "kmeans",
    "encoder_vocab_size": 500,
    "deduplicate": true,
    "need_f0": false
}
```


### Span Selector Config

This is a config used for the span selector in the pipeline. we use the name sselector

required keys/arguments:
- `type`: type of span selector. currently available (i) constant (ii) adaptive (iii) threshold

each span selector has it's own keys/arguments. I recommend looking at the file [span_selector.py](../spans_selector.py) for more info


### All configs files used in the paper are available and can be directly used using the file [audio_segment.py](../../audio_segment.py)


