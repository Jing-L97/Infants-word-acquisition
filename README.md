# Infants-word-acquisition
The current project aims to model infants' acquisition of vacabulary knowledge from raw speech signals. 
In Experiment 1, we test different types of segmentation models(Accumulator; STELA; DP-Parsse) based on different levels of units(words; unsegmented pohonemes;raw speech) as inputs. 
In Experiment 2, episodic memory modules are incoporated in the LSTM model in order to simulate the top-down influence in langauge acquisition.  

Project progress:
https://docs.google.com/spreadsheets/d/1u6DDvlb4E5FxxtPDV3KRQj4AjZ3wvxzYkExOt0gxv_I/edit?usp=sharing

![image](https://user-images.githubusercontent.com/84009338/220105295-c06dca1c-1db4-4ff6-a1ef-4f68fb9414d7.png)

For model details, see:

Experiment 1

Accumulator: 
Hidaka, S. (2013). A computational model associating learning process, word attributes, and age of acquisition. PLOS one, 8(11), e76242. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0076242

DP-Parse:
Algayres, R., Ricoul, T., Karadayi, J., Laurençon, H., Zaiem, S., Mohame, A., ... & Dupoux, E. (2022). DP-Parse: Finding Word Boundaries from Raw Speech with an Instance Lexicon. Transactions of the Association for Computational Linguistics, 10, 1051-1065. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00505/113018/DP-Parse-Finding-Word-Boundaries-from-Raw-Speech

STELA:
Nguyen, T. A., de Seyssel, M., Algayres, R., Roze, P., Dunbar, E., & Dupoux, E. (2022). Are word boundaries useful for unsupervised language learning?. arXiv preprint arXiv:2210.02956. https://arxiv.org/abs/2210.02956

The model settings replicate the architecture in https://github.com/MarvinLvn/InfTrain


Experiment 2

Episodic memory: 
Fortunato, M., Tan, M., Faulkner, R., Hansen, S., Puigdomènech Badia, A., Buttimore, G., ... & Blundell, C. (2019). Generalization of reinforcement learners with working and episodic memory. Advances in neural information processing systems, 32. https://proceedings.neurips.cc/paper/2019/hash/02ed812220b0705fabb868ddbf17ea20-Abstract.html
