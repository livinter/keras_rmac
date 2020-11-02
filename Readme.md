# Detect Videos in Videos

Train some videos:

`python3 train.py [input video file] `

Then test new videos if they contain parts of the already trained videos.

`python3 test.py [input video file] `




Image detection based on Keras RMAC
Re-implementation of Regional Maximum Activations of Convolutions (RMAC) feature extractor for Keras, based on (Tolias et al. 2016) and (Gordo et al. 2016). The architecture of the model is as in the image below:

![rmac](https://github.com/noagarcia/keras_rmac/blob/master/data/model.png?raw=true)

RoiPooling code from: https://github.com/yhenon/keras-spp

## Prerequisites 
This code requires Tensorflow version 2.1 or greater.
- [Python][1] (3.6+)
- [Keras][2] (2.1.2)
- [Tensorflow] (2.1.0)

## References

- Tolias, G., Sicre, R., & Jégou, H. Particular object retrieval with integral max-pooling of CNN activations. ICLR 2016.

- Gordo, A., Almazán, J., Revaud, J., & Larlus, D. Deep image retrieval: Learning global representations for image search. ECCV 2016. 


## Citation

This code is a re-implementation of RMAC for Keras. 

If using this code, please cite the paper where the re-implementation is used and the original RMAC paper:

```
@article{garcia2018asymmetric,
   author    = {Noa Garcia and George Vogiatzis},
   title     = {Asymmetric Spatio-Temporal Embeddings for Large-Scale Image-to-Video Retrieval},
   booktitle = {Proceedings of the British Machine Vision Conference},
   year      = {2018},
}
``` 
```
@article{tolias2016particular,
   author    = {Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
   title     = {Particular object retrieval with integral max-pooling of CNN activations},
   booktitle = {Proceedings of the International Conference on Learning Representations},
   year      = {2016},
}
``` 

[1]: https://www.python.org/download/releases/2.7/
[2]: https://keras.io/
