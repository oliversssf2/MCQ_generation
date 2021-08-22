# The changes made to the question_generation library

1. The old huggingface nlp library is replaced by the latest huggingface datasets library. (nlp.xxxx -> datasets.xxxx) This allow the qg library to access newer functionality of the datasets library (e.g. more modeling saving options).
2. 