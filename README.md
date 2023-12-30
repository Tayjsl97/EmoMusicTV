# EmoMusicTV
This is the official implementation of EmoMusicTV, which  is a transformer-based variational autoencoder (VAE) that contains a hierarchical latent variable structure to explore the impact of time-varying emotional conditions on multiple music generation tasks and to capture the rich variability of musical sequences. <br>
- [Paper link](https://ieeexplore.ieee.org/abstract/document/10124351)
- Check our [demo page](https://tayjsl97.github.io/demos/tmm) and listen!<br>
<br>

<img src="img/model.jpg" width="300" height="350" alt="model"/><img src="img/instantiation.jpg" width="300" height="250" alt="model"/>

# Data Interpretation
👇Interpretation of indices in melody.data 
Index | Definition
-------|----------
0 | bar
1-61 | pitch (1 for rest)
62-98 | duration
99-106 | time signature
Consequently, each melody event is represented by a 107-D vector.

👇Interpretation of indices in chord.data 
Index | Definition
-------|----------
0-6 | chord mode (0 for rest)
0-40 | root tone 
Consequently, each chord event is represented by a 48-D vector (concatenation of 7-D and 41-D).

# Reference
If you find the code useful for your research, please consider citing
```bib
@article{ji2023emomusictv,
  title={EmoMusicTV: Emotion-conditioned Symbolic Music Generation with Hierarchical Transformer VAE},
  author={Ji, Shulei and Yang, Xinyu},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```
