# Novel class discovery meets foundation models for 3D semantic segmentation
Official implementation of the IJCV-SI paper "Novel class discovery meets foundation models for 3D semantic segmentation"

![teaser](assets/teaser.png)

## Introduction
The task of Novel Class Discovery (NCD) in semantic segmentation involves training a model to accurately segment unlabelled (novel) classes, using the supervision available from annotated (base) classes. 
The NCD task within the 3D point cloud domain is novel, and it is characterised by assumptions and challenges absent in its 2D counterpart. 
This paper advances the analysis of point cloud data in four directions. 
Firstly, it introduces the novel task of NCD for point cloud semantic segmentation. 
Secondly, it demonstrates that directly applying an existing NCD method for 2D image semantic segmentation to 3D data yields limited results. 
Thirdly, it presents a new NCD approach based on online clustering, uncertainty estimation, and semantic distillation. 
Lastly, it proposes a novel evaluation protocol to rigorously assess the performance of NCD in point cloud semantic segmentation. 
Through comprehensive evaluations on the SemanticKITTI, SemanticPOSS, and S3DIS datasets, our approach show superior performance compared to the considered baselines.

Authors: 
        [Luigi Riz](https://scholar.google.com/citations?user=djO2pVUAAAAJ&hl),
        [Cristiano Saltori](https://scholar.google.com/citations?user=PID7Z4oAAAAJ&hl),
        [Yiming Wang](https://scholar.google.co.uk/citations?user=KBZ3zrEAAAAJ),
        [Elisa Ricci](https://scholar.google.ca/citations?user=xf1T870AAAAJ&hl),
        [Fabio Poiesi](https://scholar.google.co.uk/citations?user=BQ7li6AAAAAJ&hl)


Camera ready and code will be released soon!