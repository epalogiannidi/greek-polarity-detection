##  Polarity detection for the Greek language

Implements two methods for computing polarity in Greek: 
1. Affective lexicon based
2. LLM based

### Affective Lexicon used:

@inproceedings{palogiannidi2016affective,
  title={Affective lexicon creation for the Greek language},
  author={Palogiannidi, Elisavet and Koutsakis, Polychronis and Iosif, Elias and Potamianos, Alexandros},
  booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
  pages={2867--2872},
  year={2016}
}

The lexicon can be found under data folder or at the repository: https://github.com/epalogiannidi/Affect-of-words/tree/master/data/anew_el


### Greek Dataset used for polarity detection
https://www.kaggle.com/datasets/nikosfragkis/skroutz-shop-reviews-sentiment-analysis

Exectute:
```commandline
python -m greek_polarity_detection
```

